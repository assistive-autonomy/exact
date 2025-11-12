"""
Trainer for finetuning an LLM to generate programs from motion tensors using GRPO.
"""

import torch
from typing import Optional, Dict, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from exact.bm import BehaviourModel
from exact.programs import RewardBuilder
from exact.utils import wasserstein_distance, MotionProgramDataset, MotionEncoder


class MotionConditionedGRPOTrainer(GRPOTrainer):
    """Custom GRPO trainer that conditions generation on motion sequences."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        motion_encoder: MotionEncoder,
        behaviour_model: Optional[BehaviourModel],
        config: GRPOConfig,
        ot_weight: float = 1.0,
        validity_weight: float = 0.1,
        wandb_enabled: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            config=config,
            **kwargs,
        )
        
        self.motion_encoder = motion_encoder
        self.behaviour_model = behaviour_model
        self.ot_weight = ot_weight
        self.validity_weight = validity_weight
        self.wandb_enabled = wandb_enabled and WANDB_AVAILABLE
        
        if self.behaviour_model is not None:
            for param in self.behaviour_model.model.parameters():
                param.requires_grad = False
    
    def compute_reward(
        self,
        original_motion: torch.Tensor,
        generated_program: str,
        num_steps: int = 100,
    ) -> float:
        reward = 0.0
        is_valid = False
        ot_dist = None
        
        try:
            reward_fn = RewardBuilder.reward_from_name(generated_program)
            reward += self.validity_weight
            is_valid = True
            
            if self.behaviour_model is not None:
                try:
                    generated_poses, _ = self.behaviour_model.generate(
                        reward_fn,
                        steps=num_steps,
                        render=False,
                    )
                    
                    original_poses = original_motion[:num_steps, :214]
                    
                    ot_dist = wasserstein_distance(
                        original_poses,
                        generated_poses.squeeze(),
                    )
                    
                    reward += -self.ot_weight * ot_dist
                    
                except Exception:
                    reward += -10.0
                    
        except (ValueError, RuntimeError, SyntaxError):
            reward += -10.0
        
        # Log to wandb if enabled
        if self.wandb_enabled and self.state.global_step % self.args.logging_steps == 0:
            log_dict = {
                "reward/total": reward,
                "reward/is_valid": float(is_valid),
            }
            if ot_dist is not None:
                log_dict["reward/ot_distance"] = ot_dist
            wandb.log(log_dict, step=self.state.global_step)
        
        return reward
    
    def prepare_model_inputs(
        self,
        motion_embeddings: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if input_ids is not None:
            token_embeds = self.model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([motion_embeddings, token_embeds], dim=1)
            
            motion_mask = torch.ones(
                motion_embeddings.shape[0],
                motion_embeddings.shape[1],
                dtype=torch.long,
                device=motion_embeddings.device,
            )
            token_mask = torch.ones_like(input_ids)
            attention_mask = torch.cat([motion_mask, token_mask], dim=1)
        else:
            inputs_embeds = motion_embeddings
            attention_mask = torch.ones(
                motion_embeddings.shape[0],
                motion_embeddings.shape[1],
                dtype=torch.long,
                device=motion_embeddings.device,
            )
        
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }


def create_grpo_config(
    output_dir: str = "outputs",
    num_train_epochs: int = 10,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-6,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 100,
    num_generations: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    **kwargs,
) -> GRPOConfig:
    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        warmup_steps=warmup_steps,
        num_generations=num_generations,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        **kwargs,
    )
    
    return config


def train_motion_to_program_grpo(
    train_motions: list[torch.Tensor],
    train_programs: list[str],
    val_motions: list[torch.Tensor] = None,
    val_programs: list[str] = None,
    behaviour_model: Optional[BehaviourModel] = None,
    output_dir: str = "outputs",
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    training_config: dict = None,
    motion_encoder_config: dict = None,
    wandb_enabled: bool = False,
    device: str = None,
):
    """Train a motion-to-program model using GRPO with Hydra config support."""
    
    # Set default configs if not provided
    if motion_encoder_config is None:
        motion_encoder_config = {
            "motion_dim": 256,
            "hidden_dim": 512,
            "num_layers": 4,
            "num_prefix_tokens": 8,
        }
    
    if training_config is None:
        training_config = {
            "num_train_epochs": 10,
            "per_device_train_batch_size": 4,
            "learning_rate": 5e-6,
            "ot_weight": 1.0,
            "validity_weight": 0.1,
        }
    
    accelerator = Accelerator()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    motion_encoder = MotionEncoder(
        motion_dim=motion_encoder_config.get("motion_dim", 256),
        hidden_dim=motion_encoder_config.get("hidden_dim", 512),
        output_dim=model.config.hidden_size,
        num_layers=motion_encoder_config.get("num_layers", 4),
        num_prefix_tokens=motion_encoder_config.get("num_prefix_tokens", 8),
    )
    
    train_dataset = MotionProgramDataset(train_motions, train_programs, tokenizer)
    eval_dataset = None
    if val_motions is not None and val_programs is not None:
        eval_dataset = MotionProgramDataset(val_motions, val_programs, tokenizer)
    
    config = create_grpo_config(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_train_epochs", 10),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 5e-6),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        warmup_steps=training_config.get("warmup_steps", 100),
        num_generations=training_config.get("num_generations", 4),
        max_new_tokens=training_config.get("max_new_tokens", 64),
        temperature=training_config.get("temperature", 0.8),
        logging_steps=training_config.get("logging_steps", 10),
        save_strategy=training_config.get("save_strategy", "epoch"),
        evaluation_strategy=training_config.get("evaluation_strategy", "epoch"),
        save_total_limit=training_config.get("save_total_limit", 3),
        fp16=training_config.get("fp16", torch.cuda.is_available()),
        report_to=["wandb"] if wandb_enabled else [],
    )
    
    trainer = MotionConditionedGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        motion_encoder=motion_encoder,
        behaviour_model=behaviour_model,
        config=config,
        ot_weight=training_config.get("ot_weight", 1.0),
        validity_weight=training_config.get("validity_weight", 0.1),
        wandb_enabled=wandb_enabled,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.model, trainer.motion_encoder = accelerator.prepare(
        trainer.model, trainer.motion_encoder
    )
    
    trainer.train()
    
    return trainer


