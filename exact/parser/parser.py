"""Motion Parser: ST-GCN motion tokens as LLM prefix conditioning.

Architecture:
    1. ST-GCN encodes motion [B, T, 72] -> [B, num_temporal_tokens, encoder_dim]
    2. Learned projection maps motion tokens to LLM embedding space
    3. Motion tokens concatenated as prefix to text token embeddings
    4. LLM processes [motion_tok_1, ..., motion_tok_N, Program: <program> <eos>]
    5. LoRA adapts the LLM's self-attention to attend to motion tokens
    6. Labels masked for motion prefix + prompt; loss only on program tokens

Auxiliary losses:
    - InfoNCE contrastive loss aligns motion encoder latent space with
      LLM program embedding space.
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import ModelOutput

from exact.data.dataset import PROMPT_PREFIX
from exact.encoder import STGCNEncoder


@dataclass
class ParserOutput(ModelOutput):
    """Output from MotionPrefixParser."""

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    alignment_loss: Optional[torch.Tensor] = None


class MotionPrefixParser(nn.Module):
    """
    Motion-to-program parser using prefix conditioning + LoRA.

    Architecture:
        1. ST-GCN encodes motion [B, T, 72] → [B, N_motion, encoder_dim]
        2. Motion projection: encoder_dim → hidden_dim (+ LayerNorm)
        3. Text embeddings from LLM's embedding layer
        4. Concatenate: [motion_tokens, text_tokens]
        5. LLM forward with inputs_embeds (self-attention handles conditioning)
        6. Loss only on program tokens (motion prefix, prompt, padding masked)

    Args:
        model: LoRA-adapted LLM (PeftModel or CausalLM)
        trajectory_encoder: ST-GCN motion encoder
        tokenizer: HuggingFace tokenizer
        encoder_dim: ST-GCN output dimension
        alignment_weight: Weight for InfoNCE loss (0 to disable)
        alignment_dim: Shared latent space dimension for alignment
        alignment_temperature: InfoNCE temperature
    """

    def __init__(
        self,
        model,
        trajectory_encoder: STGCNEncoder,
        tokenizer: PreTrainedTokenizer,
        encoder_dim: int = 2048,
        alignment_weight: float = 0.1,
        alignment_dim: int = 256,
        alignment_temperature: float = 0.07,
    ):
        super().__init__()

        self.model = model
        self.trajectory_encoder = trajectory_encoder
        self.tokenizer = tokenizer

        # Get LLM hidden dimension
        hidden_dim = model.config.hidden_size
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim

        # Motion projection to LLM embedding space
        # Even when dims match, we use a linear + LN to learn proper scaling.
        # NOTE: We add a learned scaling factor after LayerNorm to match the
        # text embedding L2 norm (~1.0). LayerNorm produces unit-variance features,
        # giving L2 norm ≈ sqrt(hidden_dim) ≈ 45 for dim=2048, which completely
        # overwhelms the text embeddings at ~1.0 L2 norm.
        self.motion_projection = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Learnable scale factor to match text embedding magnitude.
        # Initialized to 1/sqrt(hidden_dim) so that after LayerNorm (which
        # produces std≈1 → L2≈sqrt(d)), the resulting L2 norm ≈ 1.0,
        # matching the LLM's native text embedding scale.
        self._motion_scale = nn.Parameter(
            torch.tensor(1.0 / (hidden_dim ** 0.5))
        )

        # ── InfoNCE alignment ───────────────────────────────────────────
        self.alignment_weight = alignment_weight
        self.alignment_temperature = alignment_temperature

        self.motion_align_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, alignment_dim),
        )

        self.program_align_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, alignment_dim),
        )

        # Learnable log-temperature for InfoNCE (CLIP-style)
        self.logit_scale = nn.Parameter(
            torch.tensor(1.0 / alignment_temperature).log()
        )

        # Cache prompt info
        self._prompt_prefix = PROMPT_PREFIX
        self._prompt_ids = None
        self._prompt_len = len(
            tokenizer(PROMPT_PREFIX, add_special_tokens=False)["input_ids"]
        )

        # For logging
        self._last_lm_loss = None
        self._last_alignment_loss = None

    @property
    def dtype(self):
        """Return model dtype."""
        if hasattr(self.model, "dtype"):
            return self.model.dtype
        for p in self.model.parameters():
            return p.dtype
        return torch.float32

    @property
    def device(self):
        """Return model device."""
        for p in self.model.parameters():
            return p.device
        return torch.device("cpu")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def _get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from the LLM (handles PEFT wrapping)."""
        embed_layer = self.model.get_input_embeddings()
        return embed_layer(input_ids)

    def _encode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        """
        Encode motion to prefix token embeddings (with scale normalization).

        Applies the learned ``_motion_scale`` factor so that the resulting
        embeddings have L2 norms matching the LLM's text embeddings (~1.0).

        Args:
            motion: [B, T, 72] raw motion sequence

        Returns:
            motion_features: [B, num_temporal_tokens, hidden_dim] (scaled)
        """
        # ST-GCN encoding (float32 for BatchNorm stability)
        motion_features = self.trajectory_encoder(motion)

        # Cast to projection dtype and project to LLM embedding space
        proj_param = next(self.motion_projection.parameters())
        target_dtype = proj_param.dtype
        target_device = proj_param.device
        motion_features = motion_features.to(device=target_device, dtype=target_dtype)
        motion_features = self.motion_projection(motion_features)

        return motion_features * self._motion_scale

    def _get_prompt_ids(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get tokenized 'Program: ' prompt IDs for generation seeding."""
        if self._prompt_ids is None or self._prompt_ids.device != device:
            encoded = self.tokenizer(
                self._prompt_prefix,
                return_tensors="pt",
                add_special_tokens=False,
            )
            self._prompt_ids = encoded["input_ids"].to(device)
        return self._prompt_ids.expand(batch_size, -1)

    def _compute_alignment_loss(
        self,
        motion_features: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss between motion and program embeddings.

        Args:
            motion_features: [B, num_temporal_tokens, hidden_dim]
            hidden_states: [B, seq_len, hidden_dim] (text portion only)
            attention_mask: [B, seq_len] text attention mask

        Returns:
            Scalar InfoNCE loss
        """
        # Pool motion: mean over temporal tokens → [B, hidden_dim]
        motion_pooled = motion_features.mean(dim=1)

        # Pool program: mean over non-padding tokens → [B, hidden_dim]
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        program_pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(
            min=1
        )

        # Project to shared alignment space
        motion_embed = F.normalize(self.motion_align_head(motion_pooled), dim=-1)
        program_embed = F.normalize(self.program_align_head(program_pooled), dim=-1)

        # Compute scaled cosine similarity matrix [B, B]
        # Tight clamp (~1/0.07 ≈ 14.3) prevents logit_scale runaway that
        # collapses the contrastive loss to zero (observed in previous runs
        # where scale grew to 24.8+, killing alignment gradients).
        logit_scale = self.logit_scale.exp().clamp(max=14.3)
        logits = logit_scale * motion_embed @ program_embed.t()

        # Symmetric cross-entropy (motion→program + program→motion)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_m2p = F.cross_entropy(logits, labels)
        loss_p2m = F.cross_entropy(logits.t(), labels)

        return (loss_m2p + loss_p2m) / 2

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        motion: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> ParserOutput:
        """
        Training forward pass with motion prefix conditioning.

        Motion features are encoded, projected, and prepended as prefix tokens
        to the text embeddings. The LLM's self-attention naturally conditions
        on the motion prefix at every layer (no monkey-patching needed).

        Args:
            input_ids: [B, seq_len] token IDs for "Program: <program><eos>"
            attention_mask: [B, seq_len]
            motion: [B, T, 72] motion sequence (None disables conditioning)
            labels: [B, seq_len] target labels (-100 for prompt/padding)
        """
        prefix_len = 0
        motion_features = None

        if motion is not None:
            # Encode motion → prefix embeddings
            motion_features = self._encode_motion(motion)
            prefix_len = motion_features.shape[1]
            B = motion_features.shape[0]

            # Get text token embeddings
            text_embeds = self._get_input_embeddings(input_ids)

            # Concatenate [motion_prefix, text_tokens]
            inputs_embeds = torch.cat([motion_features, text_embeds], dim=1)

            # Create combined attention mask
            motion_mask = torch.ones(
                B, prefix_len, device=attention_mask.device, dtype=attention_mask.dtype
            )
            combined_mask = torch.cat([motion_mask, attention_mask], dim=1)

            # Create combined labels (motion prefix positions masked)
            combined_labels = None
            if labels is not None:
                prefix_labels = torch.full(
                    (B, prefix_len), -100, device=labels.device, dtype=labels.dtype
                )
                combined_labels = torch.cat([prefix_labels, labels], dim=1)

            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=combined_mask,
                labels=combined_labels,
                return_dict=True,
                output_hidden_states=True,
            )
        else:
            # No motion — pure text forward (for testing/debugging)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                output_hidden_states=True,
            )

        lm_loss = outputs.loss
        total_loss = lm_loss
        alignment_loss = None

        # Compute InfoNCE alignment loss during training
        if (
            motion_features is not None
            and self.alignment_weight > 0
            and self.training
        ):
            last_hidden = outputs.hidden_states[-1]
            # Extract only the text portion of hidden states
            text_hidden = last_hidden[:, prefix_len:]
            alignment_loss = self._compute_alignment_loss(
                motion_features, text_hidden, attention_mask
            )
            total_loss = total_loss + self.alignment_weight * alignment_loss

        # Store loss breakdown for logging callback
        self._last_lm_loss = lm_loss.item() if lm_loss is not None else None
        self._last_alignment_loss = (
            alignment_loss.item() if alignment_loss is not None else None
        )

        return ParserOutput(
            loss=total_loss,
            logits=outputs.logits,
            lm_loss=lm_loss,
            alignment_loss=alignment_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        motion: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.95,
        do_sample: bool = True,
        grammar_processor=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate program from motion using prefix conditioning.

        Motion features are encoded and prepended as prefix tokens. The LLM
        then generates conditioned on both the motion prefix and the
        "Program: " text prompt via its native self-attention.

        Args:
            motion: [B, T, 72] motion sequence
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or greedy decode
            grammar_processor: Optional grammar-constrained logits processor

        Returns:
            generated_ids: [B, prompt_len + generated_len] token IDs
                          (motion prefix stripped, starts with prompt tokens)
        """
        B = motion.shape[0]
        device = motion.device

        # Encode motion → prefix embeddings
        motion_features = self._encode_motion(motion)
        prefix_len = motion_features.shape[1]

        # Get "Program: " prompt embeddings
        prompt_ids = self._get_prompt_ids(B, device)
        prompt_embeds = self._get_input_embeddings(prompt_ids)

        # Combined embeddings: [motion_prefix, prompt_text]
        # Cast motion_features to match prompt_embeds dtype (e.g. bfloat16)
        motion_features = motion_features.to(dtype=prompt_embeds.dtype)
        inputs_embeds = torch.cat([motion_features, prompt_embeds], dim=1)

        # Create dummy input_ids for generate() sequence tracking
        # Motion positions use pad_token_id (unused for embedding — inputs_embeds
        # overrides), but generate() uses input_ids to track the running sequence
        pad_ids = torch.full(
            (B, prefix_len),
            self.tokenizer.pad_token_id,
            device=device,
            dtype=torch.long,
        )
        input_ids = torch.cat([pad_ids, prompt_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        # Setup logits processors
        logits_processor = kwargs.pop("logits_processor", [])
        if grammar_processor is not None:
            grammar_processor.reset()
            logits_processor.append(grammar_processor)

        generated = self.model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            logits_processor=logits_processor if logits_processor else None,
            **kwargs,
        )

        # Strip motion prefix (keep prompt + generated for downstream decoding)
        generated = generated[:, prefix_len:]

        return generated

    def save_pretrained(self, save_directory: str):
        """Save trainable components (encoder + projection + alignment).

        The LoRA adapter is saved separately by PEFT's save_pretrained.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save ST-GCN encoder
        torch.save(
            self.trajectory_encoder.state_dict(),
            os.path.join(save_directory, "trajectory_encoder.pt"),
        )

        # Save motion projection + alignment heads
        save_dict = {
            "motion_projection": self.motion_projection.state_dict(),
            "motion_align_head": self.motion_align_head.state_dict(),
            "program_align_head": self.program_align_head.state_dict(),
            "logit_scale": self.logit_scale.data,
            "motion_scale": self._motion_scale.data,
            "encoder_dim": self.encoder_dim,
            "hidden_dim": self.hidden_dim,
        }

        torch.save(save_dict, os.path.join(save_directory, "prefix_parser.pt"))
