import torch
import torch.nn as nn

from syncode import SyncodeLogitsProcessor

from .encoder import TrajectoryEncoder

class MotionConditionedParser(nn.Module):
    """LLM with motion prefix conditioning for program generation."""

    def __init__(
        self,
        model,
        trajectory_encoder: TrajectoryEncoder,
    ):
        super().__init__()

        self.model = model
        self.trajectory_encoder = trajectory_encoder

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
        motion_embeddings = self.trajectory_encoder(motion)
        token_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([motion_embeddings, token_embeds], dim=1)

        motion_mask = torch.ones(
            motion_embeddings.shape[0],
            motion_embeddings.shape[1],
            dtype=torch.long,
            device=motion.device,
        )
        full_attention_mask = torch.cat([motion_mask, attention_mask], dim=1)

        # Prepare labels: mask out motion prefix tokens
        if labels is not None:
            prefix_labels = torch.full(
                (motion_embeddings.shape[0], motion_embeddings.shape[1]),
                -100,
                dtype=torch.long,
                device=motion.device,
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
        motion_embeddings = self.trajectory_encoder(motion)

        # Create attention mask for motion embeddings (all ones since no padding)
        attention_mask = torch.ones(
            motion_embeddings.shape[0],
            motion_embeddings.shape[1],
            dtype=torch.long,
            device=motion.device,
        )

        # Reset grammar processor state if provided
        if grammar_processor is not None:
            grammar_processor.reset()
            kwargs["logits_processor"] = kwargs.get("logits_processor", []) + [
                grammar_processor
            ]

        return self.model.generate(
            inputs_embeds=motion_embeddings,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
