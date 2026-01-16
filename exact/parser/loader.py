"""Parser loading utilities.

Provides a unified interface for loading trained parsers or falling back
to mock parsers for testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, Union

import numpy as np
import torch


class ParserProtocol(Protocol):
    """Protocol defining the parser interface."""
    
    def parse(self, motion: np.ndarray) -> str:
        """Parse a motion sequence into a program string.
        
        Args:
            motion: Motion sequence of shape (seq_len, motion_dim)
            
        Returns:
            Program string in ExAct grammar format
        """
        ...
    
    def parse_batch(self, motions: list[np.ndarray]) -> list[str]:
        """Parse multiple motion sequences.
        
        Args:
            motions: List of motion sequences
            
        Returns:
            List of program strings
        """
        ...


class TrainedParser:
    """Wrapper for trained MotionConditionedParser models.
    
    Loads a trained parser from a checkpoint directory and provides
    a simple parse() interface.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        use_grammar_constraint: bool = True,
        max_new_tokens: int = 256,
    ):
        """Load a trained parser from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory containing:
                - adapter_model.safetensors (LoRA weights)
                - encoder.pt (trajectory encoder weights)
                - config.yaml (training config)
            device: Device to run on ('auto', 'cuda', 'cpu')
            use_grammar_constraint: Whether to use grammar-constrained decoding
            max_new_tokens: Maximum tokens to generate
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.max_new_tokens = max_new_tokens
        self.use_grammar_constraint = use_grammar_constraint
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load components
        self._load_model()
    
    def _load_model(self):
        """Load model components from checkpoint."""
        from omegaconf import OmegaConf
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        from exact.parser import (
            MotionConditionedParser,
            TrajectoryEncoder,
            TemporalTrajectoryEncoder,
            create_grammar_processor,
        )
        
        # Load config
        config_path = self.checkpoint_path / "config.yaml"
        if not config_path.exists():
            # Try parent directory (checkpoint might be in a subdirectory)
            config_path = self.checkpoint_path.parent / "config.yaml"
        
        if config_path.exists():
            self.config = OmegaConf.load(config_path)
        else:
            raise FileNotFoundError(f"Config not found at {config_path}")
        
        # Load tokenizer
        model_name = self.config.get("model_name", "meta-llama/Llama-3.2-3B")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.config.get("bf16", True) else torch.float32,
            device_map=self.device,
        )
        
        # Load LoRA adapter
        adapter_path = self.checkpoint_path
        if (self.checkpoint_path / "adapter_model.safetensors").exists():
            adapter_path = self.checkpoint_path
        elif (self.checkpoint_path / "checkpoint-best").exists():
            adapter_path = self.checkpoint_path / "checkpoint-best"
        
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Create trajectory encoder
        encoder_type = self.config.get("encoder_type", "temporal")
        motion_dim = self.config.get("motion_dim", 72)
        hidden_size = self._get_model_hidden_size(model)
        
        if encoder_type == "temporal":
            encoder = TemporalTrajectoryEncoder(
                motion_dim=motion_dim,
                hidden_dim=self.config.get("motion_hidden_dim", 256),
                output_dim=hidden_size,
                num_encoder_layers=self.config.get("motion_num_encoder_layers", 4),
                num_decoder_layers=self.config.get("motion_num_decoder_layers", 2),
                num_queries=self.config.get("num_queries", 32),
                nhead=self.config.get("encoder_nhead", 8),
                dropout=self.config.get("encoder_dropout", 0.1),
            )
        else:
            encoder = TrajectoryEncoder(
                input_dim=motion_dim,
                output_dim=hidden_size,
            )
        
        # Load encoder weights
        encoder_path = self.checkpoint_path / "encoder.pt"
        if encoder_path.exists():
            encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        
        encoder = encoder.to(self.device)
        
        # Create parser
        self.parser = MotionConditionedParser(
            model=model,
            trajectory_encoder=encoder,
            tokenizer=self.tokenizer,
        )
        self.parser.eval()
        
        # Create grammar processor if needed
        self.grammar_processor = None
        if self.use_grammar_constraint:
            try:
                self.grammar_processor = create_grammar_processor(self.tokenizer)
            except Exception as e:
                print(f"Warning: Could not create grammar processor: {e}")
    
    def _get_model_hidden_size(self, model) -> int:
        """Get hidden size from model config."""
        config = model.config
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return config.text_config.hidden_size
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        raise AttributeError(f"Cannot find hidden_size in model config")
    
    @torch.no_grad()
    def parse(self, motion: np.ndarray) -> str:
        """Parse a motion sequence into a program.
        
        Args:
            motion: Motion sequence of shape (seq_len, motion_dim)
            
        Returns:
            Program string
        """
        from exact.parser.utils import post_process_program
        
        # Prepare input
        motion_tensor = torch.tensor(motion, dtype=torch.float32)
        if motion_tensor.dim() == 2:
            motion_tensor = motion_tensor.unsqueeze(0)  # Add batch dim
        motion_tensor = motion_tensor.to(self.device)
        
        # Generate
        generated_ids = self.parser.generate(
            motion=motion_tensor,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            grammar_processor=self.grammar_processor,
        )
        
        # Decode and post-process
        raw_program = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        program, _ = post_process_program(raw_program, repair=True)
        
        return program
    
    def parse_batch(self, motions: list[np.ndarray]) -> list[str]:
        """Parse multiple motion sequences.
        
        Args:
            motions: List of motion sequences
            
        Returns:
            List of program strings
        """
        return [self.parse(m) for m in motions]


def load_parser(
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    use_mock: bool = False,
    **kwargs,
) -> Union[TrainedParser, "MockParser"]:
    """Load a parser from checkpoint or return mock parser.
    
    This is the main entry point for loading parsers. It handles:
    - Loading trained parsers from checkpoints
    - Falling back to mock parsers for testing
    - Automatic detection of checkpoint availability
    
    Args:
        checkpoint_path: Path to trained checkpoint (None for mock)
        device: Device to run on ('auto', 'cuda', 'cpu')
        use_mock: Force use of mock parser (for testing)
        **kwargs: Additional arguments passed to parser constructor
        
    Returns:
        Parser instance (TrainedParser or MockParser)
        
    Example:
        >>> # Load trained parser
        >>> parser = load_parser("results/parser/20251215_123456")
        >>> program = parser.parse(motion_data)
        
        >>> # Use mock parser for testing
        >>> parser = load_parser(use_mock=True)
        >>> program = parser.parse(motion_data)
    """
    from exact.models.mock_parser import MockParser
    
    if use_mock or checkpoint_path is None:
        return MockParser(**kwargs)
    
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using mock parser")
        return MockParser(**kwargs)
    
    # Check for required files
    has_adapter = (checkpoint / "adapter_model.safetensors").exists()
    has_config = (checkpoint / "config.yaml").exists() or (checkpoint.parent / "config.yaml").exists()
    
    if not has_adapter or not has_config:
        print(f"Warning: Incomplete checkpoint at {checkpoint_path}, using mock parser")
        return MockParser(**kwargs)
    
    return TrainedParser(checkpoint_path, device=device, **kwargs)
