from pathlib import Path

import torch
import torch.nn as nn

from syncode import SyncodeLogitsProcessor, Grammar

DEFAULT_GRAMMAR_PATH = Path(__file__).parent / "programs" / "grammar.lark"


def create_grammar_processor(
    tokenizer,
    grammar_path: str | Path | None = None,
) -> SyncodeLogitsProcessor:
    """Create a SynCode logits processor for grammar-constrained decoding.

    Args:
        tokenizer: HuggingFace tokenizer
        grammar_path: Path to .lark grammar file (default: exact/programs/grammar.lark)

    Returns:
        SyncodeLogitsProcessor configured with the grammar
    """
    if grammar_path is None:
        grammar_path = DEFAULT_GRAMMAR_PATH

    grammar_str = Path(grammar_path).read_text()
    grammar = Grammar(grammar_str)

    return SyncodeLogitsProcessor(
        grammar=grammar,
        tokenizer=tokenizer,
        parse_output_only=True,
    )


class TrajectoryEncoder(nn.Module):
    """Encodes motion trajectories into prefix embeddings for the LLM."""

    def __init__(
        self,
        trajectory_dim: int = 214,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_layers: int = 4,
        num_prefix_tokens: int = 8,
    ):
        super().__init__()

        self.num_prefix_tokens = num_prefix_tokens
        self.output_dim = output_dim

        self.input_projection = nn.Linear(trajectory_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(hidden_dim, output_dim * num_prefix_tokens)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        """Encode motion sequence into prefix embeddings.

        Args:
            motion: [batch_size, seq_len, trajectory_dim]

        Returns:
            embeddings: [batch_size, num_prefix_tokens, output_dim]
        """
        batch_size = motion.shape[0]

        x = self.input_projection(motion)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(-1)
        x = self.output_projection(x)
        embeddings = x.view(batch_size, self.num_prefix_tokens, self.output_dim)

        return embeddings


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

        # Reset grammar processor state if provided
        if grammar_processor is not None:
            grammar_processor.reset()
            kwargs["logits_processor"] = kwargs.get("logits_processor", []) + [
                grammar_processor
            ]

        return self.model.generate(
            inputs_embeds=motion_embeddings,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
