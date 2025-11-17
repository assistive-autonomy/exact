from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import wandb

from exact.bm import BehaviourModel
from exact.programs import RewardBuilder
from exact.utils import wasserstein_distance
from exact.motion_encoder import MotionEncoder


class InverseBehaviorModelTrainer(Trainer):
    """Inverse Behavior Model (IBM) Trainer (supervised) that maps motion to programs.
    
    Uses PEFT/LoRA for efficient fine-tuning of the LLM while training a motion encoder
    to map motion sequences to the LLM's input space.
    
    Trainable parameters:
    1. LoRA adapters on the base LLM (parameter-efficient)
    2. Motion encoder (maps motion sequences to prefix embeddings)
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        motion_encoder: MotionEncoder,
        behaviour_model: BehaviourModel,
        args: TrainingArguments = None,
        train_dataset=None,
        eval_dataset=None,
        # Loss weights
        ce_weight: float = 1.0,
        ot_weight: float = 1.0,
        # Training config
        max_program_length: int = 128,
        # WandB
        wandb_enabled: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs,
        )

        self.motion_encoder = motion_encoder
        self.behaviour_model = behaviour_model
        self.ce_weight = ce_weight
        self.ot_weight = ot_weight
        self.max_program_length = max_program_length
        self.wandb_enabled = wandb_enabled

        # Freeze behaviour model parameters
        for param in self.behaviour_model.model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute combined loss: CE + OT.
        
        Expected inputs format from dataloader:
        - poses: [B, T, 214] - motion poses only
        - input_ids: [B, seq_len] - tokenized programs
        - attention_mask: [B, seq_len] - attention mask for programs
        """
        
        # Extract inputs from the batch
        poses = inputs["poses"]  # [B, T, 214]
        input_ids = inputs["input_ids"]  # [B, seq_len]
        attention_mask = inputs["attention_mask"]  # [B, seq_len]
        
        # Encode poses into prefix embeddings using the motion encoder
        motion_embeddings = self.motion_encoder(poses)  # [B, num_prefix_tokens, hidden_dim]

        # Get token embeddings for the programs
        token_embeds = model.get_input_embeddings()(input_ids)  # [B, seq_len, hidden_dim]
        
        # Concatenate motion embeddings as prefix to token embeddings
        inputs_embeds = torch.cat([motion_embeddings, token_embeds], dim=1)  # [B, num_prefix_tokens + seq_len, hidden_dim]

        # Prepare attention mask: add ones for motion prefix tokens
        motion_mask = torch.ones(
            motion_embeddings.shape[0],
            motion_embeddings.shape[1],
            dtype=torch.long,
            device=model.device,
        )
        full_attention_mask = torch.cat([motion_mask, attention_mask], dim=1)  # [B, num_prefix_tokens + seq_len]

        # Prepare labels: -100 for motion prefix tokens (don't compute loss on them)
        prefix_labels = torch.full(
            (motion_embeddings.shape[0], motion_embeddings.shape[1]),
            -100,
            dtype=torch.long,
            device=model.device,
        )
        labels = torch.cat([prefix_labels, input_ids], dim=1)  # [B, num_prefix_tokens + seq_len]

        # Forward pass
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
        )

        # 1. Cross-entropy loss (from model)
        ce_loss = outputs.loss
        
        # 2. Optimal Transport loss (optional, only if behavior model provided)
        ot_loss = 0.0
        ot_count = 0
        
        if self.behaviour_model is not None and self.ot_weight > 0:
            # Generate programs from motion embeddings
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs_embeds=motion_embeddings,
                    max_new_tokens=64,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,  # Greedy for deterministic evaluation
                )
                generated_programs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Decode reference programs from input_ids
            reference_programs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            # Compute OT distance between original and reconstructed motions
            for pose_seq, gen_prog, ref_prog in zip(poses, generated_programs, reference_programs):
                try:
                    # Generate motion from generated program
                    reward_fn = RewardBuilder.reward_from_name(gen_prog)
                    generated_poses, _ = self.behaviour_model.generate(
                        reward_fn,
                        steps=min(100, pose_seq.shape[0]),
                        render=False,
                    )
                    
                    # Extract poses from the sequence
                    num_steps = min(generated_poses.shape[0], pose_seq.shape[0])
                    original_poses = pose_seq[:num_steps]  # Already just poses [T, 214]
                    
                    # Compute OT distance
                    ot_dist = wasserstein_distance(
                        original_poses.cpu(),
                        generated_poses.cpu(),
                    )
                    ot_loss += ot_dist
                    ot_count += 1
                    
                except Exception:
                    # If generation fails, add penalty
                    ot_loss += 10.0
                    ot_count += 1
        
        if ot_count > 0:
            ot_loss = ot_loss / ot_count

        # Combined loss
        total_loss = self.ce_weight * ce_loss
        if self.ot_weight > 0 and ot_count > 0:
            total_loss += self.ot_weight * ot_loss

        # Log to wandb
        if self.wandb_enabled and self.state.global_step % self.args.logging_steps == 0:
            log_dict = {
                "loss/total": total_loss.item() if torch.is_tensor(total_loss) else total_loss,
                "loss/ce": ce_loss.item(),
            }
            if self.ot_weight > 0 and ot_count > 0:
                log_dict["loss/ot"] = ot_loss if isinstance(ot_loss, float) else ot_loss.item()
            wandb.log(log_dict, step=self.state.global_step)

        return (total_loss, outputs) if return_outputs else total_loss

   