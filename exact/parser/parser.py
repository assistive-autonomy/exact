import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional

from syncode import SyncodeLogitsProcessor

from exact.encoder import STGCNEncoder

# Default system prompt describing the motion-to-program task
# Keep it minimal to avoid confusing the model with Python-like language
DEFAULT_SYSTEM_PROMPT = """Convert motion to a program. Grammar:
start: motion (";" motion)*
motion: "[" NUMBER "," NUMBER "]" sensor ("*" sensor)*
sensor: JOINT "." AXIS "(" NUMBER ")"
JOINT: "pelvis"|"torso"|"spine"|"chest"|"neck"|"head"|"lhip"|"lknee"|"lankle"|"ltoe"|"rhip"|"rknee"|"rankle"|"rtoe"|"lthorax"|"lshoulder"|"lelbow"|"lwrist"|"lhand"|"rthorax"|"rshoulder"|"relbow"|"rwrist"|"rhand"
AXIS: "x"|"y"|"z"
NUMBER: integer or decimal
[start_frame,end_frame] defines temporal segment. Sensors describe joint movements.
Example: [0,512]head.y(1.5)*rwrist.z(0.4);[512,1024]pelvis.y(0.8)
Motion embedding follows. Output program:"""


@dataclass
class MotionParserOutput(ModelOutput):
    """Output from MotionConditionedParser with auxiliary losses.

    Inherits from ModelOutput to be compatible with HuggingFace Trainer.
    """

    loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    alignment_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class CrossModalAttention(nn.Module):
    """Cross-attention layer for attending to motion embeddings from decoder states.
    
    This forces the decoder to explicitly ground its predictions in the motion sequence
    by computing attention over the encoder outputs at each decoding step.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query comes from decoder hidden states
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # Key and value come from motion encoder
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Gating mechanism to blend cross-attention with original hidden states
        # Starts at 0 so initially the model behavior is unchanged
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,  # [B, seq_len, hidden_dim]
        motion_embeddings: torch.Tensor,  # [B, num_motion_tokens, hidden_dim]
        motion_mask: Optional[torch.Tensor] = None,  # [B, num_motion_tokens]
    ) -> torch.Tensor:
        """Compute cross-attention from decoder to motion encoder.
        
        Returns:
            Updated decoder hidden states with motion context blended in.
        """
        batch_size, seq_len, _ = decoder_hidden.shape
        num_motion_tokens = motion_embeddings.shape[1]
        
        # Project queries from decoder, keys/values from motion
        q = self.q_proj(decoder_hidden)  # [B, seq_len, hidden_dim]
        k = self.k_proj(motion_embeddings)  # [B, num_motion_tokens, hidden_dim]
        v = self.v_proj(motion_embeddings)  # [B, num_motion_tokens, hidden_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_motion_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_motion_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # q: [B, num_heads, seq_len, head_dim]
        # k, v: [B, num_heads, num_motion_tokens, head_dim]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        # attn_weights: [B, num_heads, seq_len, num_motion_tokens]
        
        # Apply motion mask if provided
        if motion_mask is not None:
            # Expand mask for heads: [B, 1, 1, num_motion_tokens]
            mask = motion_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: [B, num_heads, seq_len, head_dim]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attn_output = self.out_proj(attn_output)
        
        # Gated residual connection - gate starts at 0, learned during training
        gate = torch.sigmoid(self.gate)
        output = decoder_hidden + gate * self.layer_norm(attn_output)
        
        return output


class MotionProjectionHead(nn.Module):
    """Projects motion embeddings to a latent space for alignment loss.
    
    Used to align encoder output with decoder embeddings of the target program.
    """
    
    def __init__(self, hidden_dim: int, latent_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Layer norm for stability
        self.norm = nn.LayerNorm(latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and normalize embeddings.
        
        Args:
            x: [B, seq_len, hidden_dim] or [B, hidden_dim]
            
        Returns:
            Projected embeddings: [B, latent_dim] (pooled) or [B, seq_len, latent_dim]
        """
        projected = self.projection(x)
        return self.norm(projected)


def info_nce_loss(
    motion_latent: torch.Tensor,
    program_latent: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute bidirectional InfoNCE contrastive loss.
    
    This loss encourages motion embeddings to be similar to their corresponding
    program embeddings while being dissimilar to other programs in the batch.
    
    Args:
        motion_latent: [B, latent_dim] motion embeddings
        program_latent: [B, latent_dim] program embeddings  
        temperature: Temperature for softmax (lower = sharper distribution)
        
    Returns:
        Scalar contrastive loss
    """
    batch_size = motion_latent.shape[0]
    
    # L2 normalize embeddings
    motion_latent = F.normalize(motion_latent, p=2, dim=-1)
    program_latent = F.normalize(program_latent, p=2, dim=-1)
    
    # Compute similarity matrix [B, B]
    # Each row i contains similarity of motion_i with all programs
    similarity = torch.matmul(motion_latent, program_latent.T) / temperature
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(batch_size, device=motion_latent.device)
    
    # Bidirectional loss: motion->program and program->motion
    loss_m2p = F.cross_entropy(similarity, labels)  # motion to program
    loss_p2m = F.cross_entropy(similarity.T, labels)  # program to motion
    
    return (loss_m2p + loss_p2m) / 2


class MotionConditionedParser(nn.Module):
    """LLM with ST-GCN motion prefix conditioning and cross-modal attention for program generation.
    
    Key architectural features:
    1. Motion prefix: ST-GCN encoded motion tokens prepended to input sequence
    2. Latent alignment: InfoNCE contrastive loss between motion and program embeddings
    """

    def __init__(
        self,
        model,
        trajectory_encoder: STGCNEncoder,
        tokenizer: PreTrainedTokenizer = None,
        motion_dim: int = 72,
        use_cross_attention: bool = False,  # Deprecated, kept for compatibility
        use_alignment_loss: bool = True,
        alignment_weight: float = 0.3,
        alignment_latent_dim: int = 256,
        alignment_temperature: float = 0.07,
        cross_attention_heads: int = 8,  # Deprecated, kept for compatibility
    ):
        super().__init__()

        self.model = model
        self.trajectory_encoder = trajectory_encoder
        self.tokenizer = tokenizer
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.alignment_temperature = alignment_temperature
        self.use_cross_attention = False  # Disabled - use alignment loss instead
        
        # Get hidden dimension from model config
        if hasattr(model, "config"):
            hidden_dim = model.config.hidden_size
        else:
            hidden_dim = 4096  # Default for DeepSeek-Coder-6.7B
        
        # Latent space alignment (InfoNCE contrastive loss)
        self.use_alignment_loss = use_alignment_loss
        self.alignment_weight = alignment_weight
        if use_alignment_loss:
            self.motion_projection = MotionProjectionHead(
                hidden_dim=hidden_dim, 
                latent_dim=alignment_latent_dim,
            )
            self.text_projection = MotionProjectionHead(
                hidden_dim=hidden_dim, 
                latent_dim=alignment_latent_dim,
            )

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
    ) -> MotionParserOutput:
        """Forward pass with motion conditioning, cross-attention, and alignment loss.

        Args:
            input_ids: [batch_size, seq_len] token ids
            attention_mask: [batch_size, seq_len] attention mask
            motion: [batch_size, motion_len, motion_dim] motion sequence
            labels: [batch_size, seq_len] target labels (optional)

        Returns:
            MotionParserOutput with loss, alignment_loss, and logits
        """
        batch_size = motion.shape[0]
        device = motion.device

        # Get embeddings
        token_embeds = self.model.get_input_embeddings()(input_ids)
        motion_embeddings = self.trajectory_encoder(motion)
        # Cast motion embeddings to match token embeddings dtype (e.g., bf16)
        motion_embeddings = motion_embeddings.to(token_embeds.dtype)

        system_embeds, system_mask = self._get_system_prompt_embeds(batch_size, device)

        # Build inputs: [system_prompt] + [motion] + [tokens]
        if system_embeds is not None:
            inputs_embeds = torch.cat(
                [system_embeds, motion_embeddings, token_embeds], dim=1
            )
            prefix_len = system_embeds.shape[1] + motion_embeddings.shape[1]
            system_len = system_embeds.shape[1]
        else:
            inputs_embeds = torch.cat([motion_embeddings, token_embeds], dim=1)
            prefix_len = motion_embeddings.shape[1]
            system_len = 0

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

        # Forward through model with output_hidden_states for alignment loss
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            output_hidden_states=self.use_alignment_loss,
        )
        
        lm_loss = outputs.loss
        alignment_loss = None
        total_loss = lm_loss
        
        # Compute latent space alignment loss if enabled
        # This forces the motion encoder to produce embeddings that align with program embeddings
        if self.use_alignment_loss and labels is not None and outputs.hidden_states is not None:
            # Get encoder output: pool motion embeddings
            motion_pooled = motion_embeddings.mean(dim=1)  # [B, hidden_dim]
            motion_latent = self.motion_projection(motion_pooled)  # [B, latent_dim]
            
            # Get decoder output: use hidden states for program tokens
            last_hidden = outputs.hidden_states[-1]  # [B, full_seq_len, hidden_dim]
            program_hidden = last_hidden[:, prefix_len:, :]  # [B, seq_len, hidden_dim]
            
            # Masked mean pooling (exclude padding tokens)
            valid_mask = (labels != -100) & (attention_mask == 1)  # [B, seq_len]
            valid_mask = valid_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
            
            masked_hidden = program_hidden * valid_mask
            sum_hidden = masked_hidden.sum(dim=1)  # [B, hidden_dim]
            count = valid_mask.sum(dim=1).clamp(min=1)  # [B, 1]
            program_pooled = sum_hidden / count  # [B, hidden_dim]
            
            # Project to shared latent space
            program_latent = self.text_projection(program_pooled)  # [B, latent_dim]
            
            # Bidirectional contrastive loss (InfoNCE)
            alignment_loss = info_nce_loss(
                motion_latent, 
                program_latent,
                temperature=self.alignment_temperature,
            )
            
            total_loss = total_loss + self.alignment_weight * alignment_loss

        return MotionParserOutput(
            loss=total_loss,
            lm_loss=lm_loss,
            alignment_loss=alignment_loss,
            logits=outputs.logits,
        )

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
        # Cast motion embeddings to match model dtype (e.g., bf16)
        motion_embeddings = motion_embeddings.to(self.model.dtype)
        
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
