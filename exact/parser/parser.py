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


class MotionConditionedParser(nn.Module):
    """LLM with ST-GCN motion prefix conditioning and cross-modal attention for program generation.
    
    Key architectural features:
    1. Motion prefix: ST-GCN encoded motion tokens prepended to input sequence
    2. Cross-modal attention: Explicit attention from decoder to motion embeddings
    3. Latent alignment: MSE loss between motion embeddings and target program embeddings
    """

    def __init__(
        self,
        model,
        trajectory_encoder: STGCNEncoder,
        tokenizer: PreTrainedTokenizer = None,
        motion_dim: int = 72,
        use_cross_attention: bool = True,
        use_alignment_loss: bool = True,
        alignment_weight: float = 0.1,
        alignment_latent_dim: int = 256,
        cross_attention_heads: int = 8,
    ):
        super().__init__()

        self.model = model
        self.trajectory_encoder = trajectory_encoder
        self.tokenizer = tokenizer
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        
        # Get hidden dimension from model config
        if hasattr(model, "config"):
            hidden_dim = model.config.hidden_size
        else:
            hidden_dim = 4096  # Default for DeepSeek-Coder-6.7B
        
        # Cross-modal attention
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attention = CrossModalAttention(
                hidden_dim=hidden_dim,
                num_heads=cross_attention_heads,
            )
        
        # Latent space alignment
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

        # Forward through model with output_hidden_states for cross-attention and alignment
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            output_hidden_states=(self.use_cross_attention or self.use_alignment_loss),
        )
        
        lm_loss = outputs.loss
        alignment_loss = None
        total_loss = lm_loss
        
        # Apply cross-modal attention if enabled
        if self.use_cross_attention and outputs.hidden_states is not None:
            # Get the last hidden states
            last_hidden = outputs.hidden_states[-1]  # [B, full_seq_len, hidden_dim]
            
            # Extract the decoder portion (after system prompt and motion)
            decoder_hidden = last_hidden[:, prefix_len:, :]  # [B, seq_len, hidden_dim]
            
            # Apply cross-attention to motion embeddings
            # Note: This is a "late fusion" approach - we process the cross-attention output
            # but don't modify the forward pass itself. For deeper integration, you would
            # need to modify the model architecture or use hooks.
            cross_attended = self.cross_attention(
                decoder_hidden=decoder_hidden,
                motion_embeddings=motion_embeddings,
                motion_mask=motion_mask,
            )
            
            # The cross-attention output can be used to compute an auxiliary loss
            # that encourages the decoder to use motion information
            # Here we use a simple reconstruction objective on the motion embeddings
            if labels is not None:
                # Pool decoder hidden states to predict motion embedding
                # This creates a learning signal that forces decoder to encode motion info
                pooled_decoder = cross_attended.mean(dim=1)  # [B, hidden_dim]
                pooled_motion = motion_embeddings.mean(dim=1)  # [B, hidden_dim]
                
                if self.use_alignment_loss:
                    # Project both to latent space and compute MSE
                    projected_decoder = self.motion_projection(pooled_decoder)
                    projected_motion = self.motion_projection(pooled_motion)
                    cross_attn_loss = F.mse_loss(projected_decoder, projected_motion.detach())
                    total_loss = total_loss + 0.01 * cross_attn_loss  # Small weight
        
        # Compute latent space alignment loss if enabled
        if self.use_alignment_loss and labels is not None and outputs.hidden_states is not None:
            # Get encoder output: pool motion embeddings
            motion_pooled = motion_embeddings.mean(dim=1)  # [B, hidden_dim]
            motion_latent = self.motion_projection(motion_pooled)  # [B, latent_dim]
            
            # Get decoder output: embed target tokens and pool
            # Use the hidden states corresponding to the program tokens
            last_hidden = outputs.hidden_states[-1]  # [B, full_seq_len, hidden_dim]
            
            # Get hidden states for program tokens only (after prefix)
            program_hidden = last_hidden[:, prefix_len:, :]  # [B, seq_len, hidden_dim]
            
            # Mask out padding (where labels == -100 or attention_mask == 0)
            if labels is not None:
                # Create mask for valid program tokens (not padding, not -100)
                valid_mask = (labels != -100) & (attention_mask == 1)  # [B, seq_len]
                valid_mask = valid_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
                
                # Masked mean pooling
                masked_hidden = program_hidden * valid_mask
                sum_hidden = masked_hidden.sum(dim=1)  # [B, hidden_dim]
                count = valid_mask.sum(dim=1).clamp(min=1)  # [B, 1]
                program_pooled = sum_hidden / count  # [B, hidden_dim]
            else:
                program_pooled = program_hidden.mean(dim=1)
            
            # Project program embedding to latent space
            program_latent = self.text_projection(program_pooled)  # [B, latent_dim]
            
            # MSE loss to align motion and program representations
            alignment_loss = F.mse_loss(motion_latent, program_latent.detach())
            
            # Add to total loss with weight
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
