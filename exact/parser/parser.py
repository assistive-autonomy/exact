import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from syncode import SyncodeLogitsProcessor

from .encoder import TrajectoryEncoder

# Default system prompt describing the motion-to-program task
# Keep it minimal to avoid confusing the model with Python-like language
DEFAULT_SYSTEM_PROMPT = """Output a motion program. Format: [start,end]joint.axis(value)*joint.axis(value);[start,end]...
Example: [0,30]head.y(1.65)*rwrist.z(0.45);[30,60]pelvis.y(0.80)
Program:"""


class MotionConditionedParser(nn.Module):
    """LLM with motion prefix conditioning for program generation."""

    def __init__(
        self,
        model,
        trajectory_encoder: TrajectoryEncoder,
        tokenizer: PreTrainedTokenizer = None,
    ):
        super().__init__()

        self.model = model
        self.trajectory_encoder = trajectory_encoder
        self.tokenizer = tokenizer
        self.system_prompt = DEFAULT_SYSTEM_PROMPT

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the underlying model."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the underlying model."""
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def _get_system_prompt_embeds(self, batch_size: int, device: torch.device):
        """Get system prompt embeddings.

        Returns:
            system_embeds: [batch_size, prompt_len, hidden_dim]
            system_mask: [batch_size, prompt_len]
        """
        if self.tokenizer is None or not self.system_prompt:
            return None, None

        # Tokenize system prompt
        encoded = self.tokenizer(
            self.system_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        prompt_ids = encoded["input_ids"].to(device)
        prompt_mask = encoded["attention_mask"].to(device)

        # Get embeddings and expand for batch
        prompt_embeds = self.model.get_input_embeddings()(prompt_ids)
        prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
        prompt_mask = prompt_mask.expand(batch_size, -1)

        return prompt_embeds, prompt_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        motion: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """Forward pass with motion conditioning.

        Args:
            input_ids: [batch_size, seq_len] token ids
            attention_mask: [batch_size, seq_len] attention mask
            motion: [batch_size, motion_len, motion_dim] motion sequence
            labels: [batch_size, seq_len] target labels (optional)

        Returns:
            Model outputs with loss if labels provided
        """
        batch_size = motion.shape[0]
        device = motion.device

        # Get embeddings
        motion_embeddings = self.trajectory_encoder(motion)
        token_embeds = self.model.get_input_embeddings()(input_ids)
        system_embeds, system_mask = self._get_system_prompt_embeds(batch_size, device)

        # Build inputs: [system_prompt] + [motion] + [tokens]
        if system_embeds is not None:
            inputs_embeds = torch.cat(
                [system_embeds, motion_embeddings, token_embeds], dim=1
            )
            prefix_len = system_embeds.shape[1] + motion_embeddings.shape[1]
        else:
            inputs_embeds = torch.cat([motion_embeddings, token_embeds], dim=1)
            prefix_len = motion_embeddings.shape[1]

        # Build attention mask
        motion_mask = torch.ones(
            batch_size,
            motion_embeddings.shape[1],
            dtype=torch.long,
            device=device,
        )
        if system_mask is not None:
            full_attention_mask = torch.cat(
                [system_mask, motion_mask, attention_mask], dim=1
            )
        else:
            full_attention_mask = torch.cat([motion_mask, attention_mask], dim=1)

        # Prepare labels: mask out system prompt and motion prefix tokens
        if labels is not None:
            prefix_labels = torch.full(
                (batch_size, prefix_len),
                -100,
                dtype=torch.long,
                device=device,
            )
            full_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            full_labels = None

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        motion: torch.Tensor,
        max_new_tokens: int = 128,
        grammar_processor: SyncodeLogitsProcessor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate program from motion sequence.

        Args:
            motion: [batch_size, motion_len, motion_dim] motion sequence
            max_new_tokens: maximum tokens to generate
            grammar_processor: SyncodeLogitsProcessor for grammar-constrained decoding

        Returns:
            generated_ids: [batch_size, generated_len]
        """
        batch_size = motion.shape[0]
        device = motion.device

        motion_embeddings = self.trajectory_encoder(motion)
        system_embeds, system_mask = self._get_system_prompt_embeds(batch_size, device)

        # Build inputs: [system_prompt] + [motion]
        if system_embeds is not None:
            inputs_embeds = torch.cat([system_embeds, motion_embeddings], dim=1)
            motion_mask = torch.ones(
                batch_size,
                motion_embeddings.shape[1],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.cat([system_mask, motion_mask], dim=1)
        else:
            inputs_embeds = motion_embeddings
            attention_mask = torch.ones(
                batch_size,
                motion_embeddings.shape[1],
                dtype=torch.long,
                device=device,
            )

        # Reset grammar processor state if provided
        if grammar_processor is not None:
            grammar_processor.reset()
            kwargs["logits_processor"] = kwargs.get("logits_processor", []) + [
                grammar_processor
            ]

        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
