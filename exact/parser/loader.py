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
    """Wrapper for trained MotionPrefixParser models.
    
    Loads a trained parser from a checkpoint directory and provides
    a simple parse() interface.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        use_grammar_constraint: bool = False,
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
        
        from exact.parser.parser import MotionPrefixParser
        from exact.encoder import STGCNEncoder
        from .utils import create_grammar_processor
        
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
        model_name = self.config.get("model_name", "/pvc/Qwen/Qwen2.5-Coder-3B")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model_dtype = torch.bfloat16 if self.config.get("bf16", True) else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.model_dtype,
            device_map=self.device,
        )
        
        # Load LoRA adapter
        adapter_path = self.checkpoint_path
        if (self.checkpoint_path / "adapter_model.safetensors").exists():
            adapter_path = self.checkpoint_path
        elif (self.checkpoint_path / "lora_adapter" / "adapter_model.safetensors").exists():
            adapter_path = self.checkpoint_path / "lora_adapter"
        elif (self.checkpoint_path / "checkpoint-best").exists():
            adapter_path = self.checkpoint_path / "checkpoint-best"
        
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Create ST-GCN encoder from config
        hidden_size = self._get_model_hidden_size(model)
        motion_dim = self.config.get("motion_dim", 72)
        num_nodes = motion_dim // 3
        
        encoder = STGCNEncoder(
            num_nodes=num_nodes,
            input_channels=3,
            hidden_channels=self.config.get("stgcn_hidden_channels", 64),
            output_dim=hidden_size,
            num_blocks=self.config.get("stgcn_num_blocks", 4),
            num_temporal_tokens=self.config.get("stgcn_num_temporal_tokens", 16),
            temporal_kernel_size=self.config.get("stgcn_temporal_kernel", 9),
            spatial_kernel_size=self.config.get("stgcn_spatial_kernel", 3),
            dropout=self.config.get("stgcn_dropout", 0.1),
            graph_strategy=self.config.get("graph_strategy", "spatial"),
            joint_embedding=self.config.get("stgcn_joint_embedding", False),
        )
        
        # Load encoder weights
        encoder_path = self.checkpoint_path / "trajectory_encoder.pt"
        if not encoder_path.exists():
            encoder_path = self.checkpoint_path / "encoder.pt"
        if encoder_path.exists():
            encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        
        encoder = encoder.to(self.device)
        
        # Create MotionPrefixParser
        self.parser = MotionPrefixParser(
            model=model,
            trajectory_encoder=encoder,
            tokenizer=self.tokenizer,
            encoder_dim=hidden_size,
            alignment_weight=self.config.get("alignment_weight", 0.1),
            alignment_dim=self.config.get("alignment_dim", 256),
            alignment_temperature=self.config.get("alignment_temperature", 0.07),
            joint_loss_weight=self.config.get("joint_loss_weight", 0.5),
        )
        
        # Load parser weights (projection + alignment + joint heads)
        parser_weights_path = self.checkpoint_path / "prefix_parser.pt"
        if parser_weights_path.exists():
            weights = torch.load(parser_weights_path, map_location=self.device)
            self.parser.motion_projection.load_state_dict(weights["motion_projection"])
            if "motion_align_head" in weights:
                self.parser.motion_align_head.load_state_dict(weights["motion_align_head"])
            if "program_align_head" in weights:
                self.parser.program_align_head.load_state_dict(weights["program_align_head"])
            if "logit_scale" in weights:
                self.parser.logit_scale.data = weights["logit_scale"]
            if "joint_head" in weights:
                self.parser.joint_head.load_state_dict(weights["joint_head"])
        
        self.parser.to(self.device)
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
        from exact.data.dataset import PROMPT_PREFIX
        raw_program = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        # Strip prompt prefix if present
        if raw_program.startswith(PROMPT_PREFIX):
            raw_program = raw_program[len(PROMPT_PREFIX):]
        program, _ = post_process_program(raw_program, repair=True)
        
        return program
    
    @torch.no_grad()
    def parse_batch(self, motions: list[np.ndarray]) -> list[str]:
        """Parse multiple motion sequences in a single batched forward pass.
        
        Pads variable-length motions to the longest in the batch,
        runs encoder + LLM generation once, then decodes each.
        
        Args:
            motions: List of motion sequences, each (seq_len_i, motion_dim)
            
        Returns:
            List of program strings
        """
        if len(motions) == 0:
            return []
        if len(motions) == 1:
            return [self.parse(motions[0])]
        
        from exact.parser.utils import post_process_program
        from exact.data.dataset import PROMPT_PREFIX
        
        # Pad motions to max length in batch
        max_len = max(m.shape[0] for m in motions)
        motion_dim = motions[0].shape[1]
        B = len(motions)
        
        padded = np.zeros((B, max_len, motion_dim), dtype=np.float32)
        for i, m in enumerate(motions):
            padded[i, :m.shape[0]] = m
        
        motion_tensor = torch.tensor(padded, dtype=torch.float32, device=self.device)
        
        # Batched generation
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
        
        # Decode each sequence
        programs = []
        for i in range(B):
            raw_program = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True).strip()
            if raw_program.startswith(PROMPT_PREFIX):
                raw_program = raw_program[len(PROMPT_PREFIX):]
            program, _ = post_process_program(raw_program, repair=True)
            programs.append(program)
        
        return programs


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
    
    # Check for required files (adapter can be in lora_adapter/ subfolder)
    has_adapter = (
        (checkpoint / "adapter_model.safetensors").exists() or
        (checkpoint / "lora_adapter" / "adapter_model.safetensors").exists()
    )
    has_config = (checkpoint / "config.yaml").exists() or (checkpoint.parent / "config.yaml").exists()
    
    if not has_adapter or not has_config:
        print(f"Warning: Incomplete checkpoint at {checkpoint_path}, using mock parser")
        return MockParser(**kwargs)
    
    return TrainedParser(checkpoint_path, device=device, **kwargs)
