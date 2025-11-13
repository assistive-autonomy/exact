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
from peft import (
    LoraConfig,
    get_peft_model,
)
from accelerate import Accelerator
import wandb

from exact.bm import BehaviourModel
from exact.programs import RewardBuilder
from exact.utils import wasserstein_distance, MotionProgramDataset
from exact.motion_encoder import MotionEncoder


class MotionProgramDataCollator:
    """Custom data collator for motion-program pairs."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, padding: bool = True):
        self.tokenizer = tokenizer
        self.padding = padding
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate motion tensors and labels from text features
        motions = [f.pop("motion") for f in features]
        labels = [f.pop("labels") for f in features] if "labels" in features[0] else None
        
        # Pad and collate text features using the tokenizer
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Stack motion tensors
        batch["motion"] = torch.stack(motions)
        
        # Pad labels separately with -100 (ignore index)
        if labels is not None:
            max_length = max(len(l) for l in labels)
            padded_labels = []
            for label_ids in labels:
                padding_length = max_length - len(label_ids)
                # Pad with -100 (standard ignore index for CrossEntropyLoss)
                padded = label_ids + [-100] * padding_length
                padded_labels.append(padded)
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        return batch


class InverseBehaviorTrainer(Trainer):
    """Trainer for Inverse Behavior Model (IBM) that maps motion to programs.
    
    Uses PEFT/LoRA for efficient fine-tuning of the LLM while training a motion encoder
    to map motion sequences to the LLM's input space.
    
    Trainable parameters:
    1. LoRA adapters on the base LLM (parameter-efficient)
    2. Motion encoder (maps motion sequences to prefix embeddings)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        motion_encoder: MotionEncoder,
        behaviour_model: Optional[BehaviourModel] = None,
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

        # Freeze behavior model if provided
        if self.behaviour_model is not None:
            for param in self.behaviour_model.model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute combined loss: CE + OT."""
        
        # Extract inputs
        motions = inputs["motion"]  # [B, T, 256]
        reference_programs = inputs["reference_program"]  # List[str]

        # Encode motions into prefix embeddings using the motion encoder
        motion_embeddings = self.motion_encoder(motions)  # [B, num_prefix_tokens, hidden_dim]

        # Tokenize reference programs
        tokenized = self.tokenizer(
            reference_programs,
            padding=True,
            truncation=True,
            max_length=self.max_program_length,
            return_tensors="pt",
        ).to(model.device)

        # Prepare inputs: concatenate motion embeddings with token embeddings
        token_embeds = model.get_input_embeddings()(tokenized["input_ids"])
        inputs_embeds = torch.cat([motion_embeddings, token_embeds], dim=1)

        # Prepare attention mask
        motion_mask = torch.ones(
            motion_embeddings.shape[0],
            motion_embeddings.shape[1],
            dtype=torch.long,
            device=model.device,
        )
        attention_mask = torch.cat([motion_mask, tokenized["attention_mask"]], dim=1)

        # Prepare labels: -100 for motion prefix tokens (don't compute loss on them)
        labels = tokenized["input_ids"].clone()
        prefix_labels = torch.full(
            (motion_embeddings.shape[0], motion_embeddings.shape[1]),
            -100,
            dtype=torch.long,
            device=model.device,
        )
        labels = torch.cat([prefix_labels, labels], dim=1)

        # Forward pass
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        # 1. Cross-entropy loss (from model)
        ce_loss = outputs.loss
        
        # 2. Generate programs for OT loss (optional, only if behavior model provided)
        ot_loss = 0.0
        ot_count = 0
        
        if self.behaviour_model is not None and self.ot_weight > 0:
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs_embeds=motion_embeddings,
                    max_new_tokens=64,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,  # Greedy for deterministic evaluation
                )
                generated_programs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for motion, gen_prog, ref_prog in zip(motions, generated_programs, reference_programs):
                try:
                    # Generate motion from generated program
                    reward_fn = RewardBuilder.reward_from_name(gen_prog)
                    generated_poses, _ = self.behaviour_model.generate(
                        reward_fn,
                        steps=min(100, motion.shape[0]),
                        render=False,
                    )
                    
                    # Extract original poses
                    num_steps = min(generated_poses.shape[0], motion.shape[0])
                    original_poses = motion[:num_steps, :214]
                    
                    # Compute OT distance
                    ot_dist = wasserstein_distance(
                        original_poses.cpu(),
                        generated_poses.squeeze().cpu(),
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

    def generate(self, motions: torch.Tensor, **generation_kwargs) -> List[str]:
        """Generate programs from motion sequences."""
        self.model.eval()

        # Encode motions using the motion encoder
        motion_embeddings = self.motion_encoder(motions)  # [B, num_prefix_tokens, hidden_size]

        # Default generation config
        default_kwargs = {
            "max_new_tokens": 64,
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
                inputs_embeds=motion_embeddings, **generation_kwargs
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
    motion_encoder_config: dict = None,
    lora_config: dict = None,
    wandb_enabled: bool = False,
    device: str = None,
):
    """Train an inverse behavior model that maps motion to programs using LoRA.

    The model has two types of trainable parameters:
    1. LoRA adapters: Parameter-efficient fine-tuning of the base LLM
    2. Motion encoder: Maps motion sequences to prefix embeddings for the LLM

    Args:
        train_motions: List of training motion sequences [T, 256]
        train_programs: List of training program strings
        val_motions: Optional validation motions
        val_programs: Optional validation programs
        behaviour_model: Optional behavior model for OT loss evaluation
        output_dir: Directory to save checkpoints
        model_name: HuggingFace model name
        training_config: Training hyperparameters
        motion_encoder_config: Motion encoder configuration
        lora_config: LoRA configuration (rank, alpha, dropout, target_modules)
        wandb_enabled: Enable WandB logging
        device: Device to use (cuda/cpu)

    Returns:
        Trained InverseBehaviorTrainer
    """


    # Initialize accelerator
    accelerator = Accelerator()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    
    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
    )
    
    # Apply LoRA to model
    print(f"Applying LoRA with rank={peft_config.r}, alpha={peft_config.lora_alpha}")
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Initialize motion encoder
    print("Initializing motion encoder...")
    motion_encoder = MotionEncoder(
        motion_dim=motion_encoder_config.get("motion_dim", 256),
        hidden_dim=motion_encoder_config.get("hidden_dim", 512),
        output_dim=model.config.hidden_size,
        num_layers=motion_encoder_config.get("num_layers", 4),
        num_prefix_tokens=motion_encoder_config.get("num_prefix_tokens", 8),
    )
    
    # Move motion encoder to device
    if device:
        motion_encoder = motion_encoder.to(device)
    elif torch.cuda.is_available():
        motion_encoder = motion_encoder.to("cuda")
    
    print(f"Motion encoder has {sum(p.numel() for p in motion_encoder.parameters() if p.requires_grad):,} trainable parameters")

    # Create datasets
    train_dataset = MotionProgramDataset(
        train_motions,
        train_programs,
        tokenizer,
    )

    eval_dataset = None
    if val_motions is not None and val_programs is not None:
        eval_dataset = MotionProgramDataset(
            val_motions,
            val_programs,
            tokenizer,
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
    save_strategy = training_config.get("save_strategy", "epoch")
    eval_strategy = training_config.get("eval_strategy", save_strategy) if eval_dataset else "no"
    
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
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        save_total_limit=training_config.get("save_total_limit", 3),
        fp16=training_config.get("fp16", torch.cuda.is_available()),
        report_to=["wandb"] if wandb_enabled else [],
        remove_unused_columns=False,  # Keep our custom columns
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
    )

    # Create custom data collator
    data_collator = MotionProgramDataCollator(tokenizer=tokenizer, padding=True)

    # Create trainer
    trainer = InverseBehaviorTrainer(
        model=model,
        tokenizer=tokenizer,
        motion_encoder=motion_encoder,
        behaviour_model=behaviour_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        ce_weight=training_config.get("ce_weight", 1.0),
        ot_weight=training_config.get("ot_weight", 1.0),
        max_program_length=training_config.get("max_program_length", 128),
        wandb_enabled=wandb_enabled,
    )

    # Prepare with accelerator
    trainer.model, trainer.motion_encoder = accelerator.prepare(
        trainer.model, trainer.motion_encoder
    )

    # Train
    print("\nStarting training...")
    print(f"Trainable parameters:")
    print(f"  - LoRA adapters: {sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'lora' in n):,}")
    print(f"  - Motion encoder: {sum(p.numel() for p in motion_encoder.parameters() if p.requires_grad):,}")
    trainer.train()

    # Save the final model (LoRA adapters) and motion encoder
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save motion encoder separately
    motion_encoder_path = f"{output_dir}/motion_encoder.pt"
    torch.save(motion_encoder.state_dict(), motion_encoder_path)
    print(f"Saved motion encoder to {motion_encoder_path}")

    return trainer
