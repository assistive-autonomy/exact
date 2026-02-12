"""Cross-Attention Motion Parser with InfoNCE Alignment.

Architecture: ST-GCN motion encoder + gated cross-attention layers injected into
a LoRA fine-tuned code LLM. Motion features serve as key/value for cross-attention
at selected decoder layers, providing strong conditioning at every level of the LLM.

An auxiliary InfoNCE contrastive loss aligns the motion encoder's latent space
with the LLM's program embedding space. Matching (motion, program) pairs are
pulled together while non-matching pairs are pushed apart, ensuring the encoder
learns representations that the LLM can meaningfully attend to.

Key design decisions:
    - Gated cross-attention (gate init=0) for stable training start
    - Cross-attention injected every N decoder layers (configurable)
    - InfoNCE alignment between pooled motion features and LLM program embeddings
    - Generation seeded with "Program: " text prompt (not just embeddings)
    - Grammar-constrained decoding compatible via logits processors
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
from exact.parser.cross_attention import GatedCrossAttention


@dataclass
class ParserOutput(ModelOutput):
    """Output from CrossAttentionParser."""

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    alignment_loss: Optional[torch.Tensor] = None


class CrossAttentionParser(nn.Module):
    """
    Motion-to-program parser using cross-attention conditioning + LoRA.

    Architecture:
        1. ST-GCN encodes motion [B, T, 72] → [B, num_temporal_tokens, encoder_dim]
        2. Motion features projected to LLM hidden dim and normalized
        3. Gated cross-attention layers injected at every Nth LLM decoder layer
           - Query = text hidden states, K/V = motion features
           - Gated residual: h' = h + tanh(gate) * CrossAttn(LN(h), motion)
        4. LoRA adapts the LLM self-attention weights
        5. Generation seeded with "Program: " text prompt

    Cross-attention provides much stronger conditioning than prefix-only approaches
    because the motion signal is refreshed at every injected decoder layer, not just
    at the input embedding level.

    Args:
        model: LoRA-adapted LLM (PeftModel or CausalLM)
        trajectory_encoder: ST-GCN motion encoder
        tokenizer: HuggingFace tokenizer
        encoder_dim: ST-GCN output dimension
        cross_attn_every_n: Insert cross-attention every N decoder layers
        cross_attn_num_heads: Number of attention heads in cross-attention
        cross_attn_dropout: Dropout in cross-attention layers
    """

    def __init__(
        self,
        model,
        trajectory_encoder: STGCNEncoder,
        tokenizer: PreTrainedTokenizer,
        encoder_dim: int = 2048,
        cross_attn_every_n: int = 4,
        cross_attn_num_heads: int = 8,
        cross_attn_dropout: float = 0.1,
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

        # Motion projection to LLM hidden dim (identity if dims match)
        if encoder_dim != hidden_dim:
            self.motion_projection = nn.Linear(encoder_dim, hidden_dim)
        else:
            self.motion_projection = nn.Identity()

        self.motion_norm = nn.LayerNorm(hidden_dim)

        # ── InfoNCE alignment ───────────────────────────────────────────
        # Projection heads map motion/program embeddings to shared latent space.
        # The contrastive loss aligns the encoder with the LLM's representations.
        self.alignment_weight = alignment_weight
        self.alignment_temperature = alignment_temperature

        # Motion projection head: pool temporal tokens → single vector → latent
        self.motion_align_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, alignment_dim),
        )

        # Program projection head: pool LLM hidden states → latent
        self.program_align_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, alignment_dim),
        )

        # Learnable log-temperature for InfoNCE (CLIP-style)
        self.logit_scale = nn.Parameter(
            torch.tensor(1.0 / alignment_temperature).log()
        )

        # Determine which decoder layers get cross-attention
        num_layers = model.config.num_hidden_layers
        self.cross_attn_layer_indices = list(
            range(cross_attn_every_n - 1, num_layers, cross_attn_every_n)
        )

        # Create gated cross-attention modules
        self.cross_attn_layers = nn.ModuleDict(
            {
                str(idx): GatedCrossAttention(
                    hidden_dim=hidden_dim,
                    num_heads=cross_attn_num_heads,
                    dropout=cross_attn_dropout,
                )
                for idx in self.cross_attn_layer_indices
            }
        )

        # Motion features cache (set during forward/generate, cleared after)
        self._motion_features: Optional[torch.Tensor] = None

        # Cached prompt token IDs for generation
        self._prompt_prefix = PROMPT_PREFIX
        self._prompt_ids = None

        # Inject cross-attention into LLM decoder layers
        self._inject_cross_attention()

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

    def _get_decoder_layers(self):
        """Navigate through PEFT/model wrappers to find decoder layers."""
        candidates = [
            # PeftModel → LoraModel → CausalLM → Model → layers
            lambda m: m.base_model.model.model.layers,
            # PeftModel → CausalLM → Model → layers
            lambda m: m.base_model.model.layers,
            # CausalLM → Model → layers
            lambda m: m.model.layers,
            # Direct access
            lambda m: m.layers,
        ]
        for get_layers in candidates:
            try:
                layers = get_layers(self.model)
                if isinstance(layers, nn.ModuleList):
                    return layers
            except AttributeError:
                continue
        raise RuntimeError(f"Cannot find decoder layers in {type(self.model)}")

    def _inject_cross_attention(self):
        """Inject gated cross-attention into selected LLM decoder layers.

        Monkey-patches the forward method of each selected decoder layer to
        apply cross-attention after the layer's normal self-attn + FFN pass.
        The motion features are accessed via self._motion_features which is
        set before forward/generate and cleared after.

        This approach is compatible with gradient checkpointing because
        nn.Module.__call__ invokes the patched forward method.
        """
        decoder_layers = self._get_decoder_layers()

        for idx_str, xattn in self.cross_attn_layers.items():
            idx = int(idx_str)
            original_forward = decoder_layers[idx].forward

            # Closure captures the correct references for each layer
            def make_forward(orig_fn, xattn_mod, parser_ref):
                def new_forward(*args, **kwargs):
                    outputs = orig_fn(*args, **kwargs)
                    if parser_ref._motion_features is not None:
                        hidden_states = outputs[0]
                        motion = parser_ref._motion_features.to(hidden_states.dtype)
                        hidden_states = xattn_mod(hidden_states, motion)
                        outputs = (hidden_states,) + outputs[1:]
                    return outputs

                return new_forward

            decoder_layers[idx].forward = make_forward(
                original_forward, xattn, self
            )

    def _encode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        """
        Encode motion to cross-attention features.

        Args:
            motion: [B, T, 72] raw motion sequence

        Returns:
            motion_features: [B, num_temporal_tokens, hidden_dim]
        """
        # ST-GCN encoding (float32 for BatchNorm stability)
        motion_features = self.trajectory_encoder(motion)

        # Cast to model dtype, project and normalize to LLM hidden dim
        target_dtype = self.motion_norm.weight.dtype
        target_device = self.motion_norm.weight.device
        motion_features = motion_features.to(device=target_device, dtype=target_dtype)
        motion_features = self.motion_projection(motion_features)
        motion_features = self.motion_norm(motion_features)

        return motion_features

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

        Pools motion temporal tokens and LLM hidden states to single vectors,
        projects both into a shared latent space, and computes symmetric
        cross-entropy (CLIP-style) over cosine similarities.

        Args:
            motion_features: [B, num_temporal_tokens, hidden_dim]
            hidden_states: [B, seq_len, hidden_dim] LLM last hidden states
            attention_mask: [B, seq_len] to mask padding when pooling

        Returns:
            Scalar InfoNCE loss
        """
        # Pool motion: mean over temporal tokens → [B, hidden_dim]
        motion_pooled = motion_features.mean(dim=1)

        # Pool program: mean over non-padding tokens → [B, hidden_dim]
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, T, 1]
        program_pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Project to shared alignment space
        motion_embed = F.normalize(self.motion_align_head(motion_pooled), dim=-1)
        program_embed = F.normalize(self.program_align_head(program_pooled), dim=-1)

        # Compute scaled cosine similarity matrix [B, B]
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
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
        Training forward pass: LM loss + InfoNCE alignment loss.

        The motion features are encoded once and cached. Each injected decoder
        layer applies cross-attention using these cached features. After the
        LM forward pass, an InfoNCE contrastive loss aligns the motion encoder's
        latent space with the LLM's program representations.

        Total loss = LM_loss + alignment_weight * InfoNCE_loss

        Args:
            input_ids: [B, seq_len] token IDs for "Program: <program><eos>"
            attention_mask: [B, seq_len]
            motion: [B, T, 72] motion sequence (None disables conditioning)
            labels: [B, seq_len] target labels for LM loss
        """
        # Encode and cache motion features for cross-attention layers
        motion_features = None
        if motion is not None:
            motion_features = self._encode_motion(motion)
            self._motion_features = motion_features

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                output_hidden_states=True,
            )
        finally:
            self._motion_features = None

        lm_loss = outputs.loss
        total_loss = lm_loss
        alignment_loss = None

        # Compute InfoNCE alignment loss during training
        if motion_features is not None and self.alignment_weight > 0 and self.training:
            # Use last hidden state from the LLM for program embeddings
            last_hidden = outputs.hidden_states[-1]
            alignment_loss = self._compute_alignment_loss(
                motion_features, last_hidden, attention_mask
            )
            total_loss = lm_loss + self.alignment_weight * alignment_loss

        # Store loss breakdown for logging callback
        self._last_lm_loss = lm_loss.item() if lm_loss is not None else None
        self._last_alignment_loss = alignment_loss.item() if alignment_loss is not None else None

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
        Generate program from motion, seeded with 'Program: ' prompt.

        The motion features are encoded and cached for the entire generation
        loop. The LLM's generate() is called with "Program: " as the input,
        and cross-attention to motion features happens at every injected layer
        for each generated token.

        Args:
            motion: [B, T, 72] motion sequence
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or greedy decode
            grammar_processor: Optional grammar-constrained logits processor

        Returns:
            generated_ids: [B, prompt_len + generated_len] token IDs
        """
        batch_size = motion.shape[0]
        device = motion.device

        # Encode and cache motion features for entire generation
        self._motion_features = self._encode_motion(motion)

        # Seed generation with "Program: " prompt
        prompt_ids = self._get_prompt_ids(batch_size, device)
        prompt_mask = torch.ones_like(prompt_ids)

        # Setup logits processors
        logits_processor = kwargs.pop("logits_processor", [])
        if grammar_processor is not None:
            grammar_processor.reset()
            logits_processor.append(grammar_processor)

        try:
            generated = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                logits_processor=logits_processor if logits_processor else None,
                **kwargs,
            )
        finally:
            self._motion_features = None

        return generated

    def save_pretrained(self, save_directory: str):
        """Save trainable components (encoder + cross-attention) for loading.

        The LoRA adapter is saved separately by PEFT's save_pretrained.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save ST-GCN encoder
        torch.save(
            self.trajectory_encoder.state_dict(),
            os.path.join(save_directory, "trajectory_encoder.pt"),
        )

        # Save cross-attention layers + motion projection + norm + alignment heads
        save_dict = {
            "cross_attn_layer_indices": self.cross_attn_layer_indices,
            "cross_attn_layers": self.cross_attn_layers.state_dict(),
            "motion_norm": self.motion_norm.state_dict(),
            "motion_align_head": self.motion_align_head.state_dict(),
            "program_align_head": self.program_align_head.state_dict(),
            "logit_scale": self.logit_scale.data,
            "encoder_dim": self.encoder_dim,
            "hidden_dim": self.hidden_dim,
        }
        if not isinstance(self.motion_projection, nn.Identity):
            save_dict["motion_projection"] = self.motion_projection.state_dict()

        torch.save(save_dict, os.path.join(save_directory, "cross_attention.pt"))
