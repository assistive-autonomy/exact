from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from accelerate import Accelerator
import wandb

from exact.bm import BehaviourModel
from exact.programs import RewardBuilder
from exact.utils import wasserstein_distance, MotionProgramDataset


class InverseBehaviorTrainer(Trainer):
    """Trainer for Inverse Behavior Model (IBM) that maps motion to programs."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        behaviour_model: Optional[BehaviourModel] = None,
        args: TrainingArguments = None,
        train_dataset=None,
        eval_dataset=None,
        # Loss weights
        ce_weight: float = 1.0,
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

        self.behaviour_model = behaviour_model
        self.ce_weight = ce_weight
        self.max_program_length = max_program_length
        self.wandb_enabled = wandb_enabled

        # Freeze behavior model if provided
        if self.behaviour_model is not None:
            for param in self.behaviour_model.model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss for inverse behavior modeling."""
        # Extract inputs
        motions = inputs["motion"]  # [B, T, 256]
        reference_programs = inputs["reference_program"]  # List[str]

        # Tokenize reference programs
        tokenized = self.tokenizer(
            reference_programs,
            padding="max_length",
            truncation=True,
            max_length=self.max_program_length,
            return_tensors="pt",
        ).to(model.device)

        # Prepare inputs: use a learned projection to map motion to initial hidden states
        batch_size = motions.size(0)

        # Average pool the motion sequence to get a fixed-size representation
        motion_embeddings = motions.mean(dim=1)  # [B, 256]

        # Project motion embeddings to match the model's hidden size
        if not hasattr(self, "motion_projection"):
            self.motion_projection = nn.Linear(
                256, model.config.hidden_size, device=model.device  # motion feature dim
            )

        # Get initial hidden states from motion
        hidden_states = self.motion_projection(motion_embeddings).unsqueeze(
            1
        )  # [B, 1, hidden_size]

        # Forward pass
        outputs = model(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            labels=tokenized["input_ids"].clone(),
            decoder_inputs_embeds=hidden_states,
        )

        # Cross-entropy loss
        ce_loss = outputs.loss

        # Log to wandb
        if self.wandb_enabled and self.state.global_step % self.args.logging_steps == 0:
            log_dict = {
                "loss/total": ce_loss.item(),
                "loss/ce": ce_loss.item(),
            }
            wandb.log(log_dict, step=self.state.global_step)

        return (ce_loss, outputs) if return_outputs else ce_loss

    def generate(self, motions: torch.Tensor, **generation_kwargs) -> List[str]:
        """Generate programs from motion sequences."""
        if not hasattr(self, "motion_projection"):
            raise RuntimeError("Model must be trained before generation")

        self.model.eval()

        # Process motion inputs
        motion_embeddings = motions.mean(dim=1)  # [B, 256]
        hidden_states = self.motion_projection(motion_embeddings).unsqueeze(
            1
        )  # [B, 1, hidden_size]

        # Default generation config
        default_kwargs = {
            "max_length": self.max_program_length,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        generation_kwargs = {**default_kwargs, **generation_kwargs}

        # Generate programs
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=hidden_states, **generation_kwargs
            )

        # Decode and clean up
        programs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return programs


def train_inverse_behavior_model(
    train_motions: List[torch.Tensor],
    train_programs: List[str],
    val_motions: Optional[List[torch.Tensor]] = None,
    val_programs: Optional[List[str]] = None,
    behaviour_model: Optional[BehaviourModel] = None,
    output_dir: str = "outputs",
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    training_config: dict = None,
    wandb_enabled: bool = False,
    device: str = None,
):
    """Train an inverse behavior model that maps motion to programs.

    Args:
        train_motions: List of training motion sequences [T, 256]
        train_programs: List of training program strings
        val_motions: Optional validation motions
        val_programs: Optional validation programs
        behaviour_model: Optional behavior model for evaluation
        output_dir: Directory to save checkpoints
        model_name: HuggingFace model name
        training_config: Training hyperparameters
        wandb_enabled: Enable WandB logging
        device: Device to use (cuda/cpu)

    Returns:
        Trained InverseBehaviorTrainer
    """
    # Set default configs if not provided
    if training_config is None:
        training_config = {
            "num_train_epochs": 10,
            "per_device_train_batch_size": 8,
            "learning_rate": 3e-5,
            "ce_weight": 1.0,
            "max_program_length": 128,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
        }

    # Initialize accelerator
    accelerator = Accelerator()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize model with appropriate config
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create datasets
    train_dataset = MotionProgramDataset(
        train_motions,
        train_programs,
        tokenizer,
        max_length=training_config.get("max_program_length", 128),
    )

    eval_dataset = None
    if val_motions is not None and val_programs is not None:
        eval_dataset = MotionProgramDataset(
            val_motions,
            val_programs,
            tokenizer,
            max_length=training_config.get("max_program_length", 128),
        )

    # Calculate warmup steps
    total_steps = (
        len(train_dataset)
        * training_config["num_train_epochs"]
        / (
            training_config["per_device_train_batch_size"]
            * training_config.get("gradient_accumulation_steps", 1)
        )
    )
    warmup_steps = int(training_config.get("warmup_ratio", 0.1) * total_steps)

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_train_epochs", 10),
        per_device_train_batch_size=training_config.get(
            "per_device_train_batch_size", 8
        ),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=training_config.get(
            "gradient_accumulation_steps", 1
        ),
        learning_rate=training_config.get("learning_rate", 3e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        warmup_steps=warmup_steps,
        logging_steps=training_config.get("logging_steps", 10),
        save_strategy=training_config.get("save_strategy", "epoch"),
        evaluation_strategy=training_config.get("evaluation_strategy", "epoch")
        if eval_dataset
        else "no",
        save_total_limit=training_config.get("save_total_limit", 3),
        fp16=training_config.get("fp16", torch.cuda.is_available()),
        report_to=["wandb"] if wandb_enabled else [],
        remove_unused_columns=False,  # Keep our custom columns
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
    )

    # Create trainer
    trainer = InverseBehaviorTrainer(
        model=model,
        tokenizer=tokenizer,
        behaviour_model=behaviour_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        ce_weight=training_config.get("ce_weight", 1.0),
        max_program_length=training_config.get("max_program_length", 128),
        wandb_enabled=wandb_enabled,
    )

    # Prepare with accelerator
    trainer.model = accelerator.prepare(trainer.model)

    # Train
    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer
