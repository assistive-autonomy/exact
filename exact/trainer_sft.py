"""
Trainer for supervised fine-tuning (SFT) of LLM to generate programs from motion tensors.

Uses a dual loss function:
1. Cross-entropy loss for program token prediction (standard SFT)
2. Optimal transport loss for motion reconstruction quality
3. Structural program loss for predicate and argument matching
"""

import re
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from accelerate import Accelerator

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from exact.bm import BehaviourModel
from exact.programs import RewardBuilder
from exact.utils import wasserstein_distance, MotionProgramDataset, MotionEncoder


@dataclass
class ProgramStructure:
    """Parsed structure of a program."""
    predicates: List[str]  # e.g., ["lhip", "rknee", "head"]
    arguments: List[float]  # e.g., [0.5, 1.2, 1.4]
    
    @staticmethod
    def parse(program: str) -> Optional["ProgramStructure"]:
        """Parse a program string into its structural components.
        
        Example: "lhip(0.5)*rknee(1.2)*head(1.4)" -> 
                 predicates=["lhip", "rknee", "head"], arguments=[0.5, 1.2, 1.4]
        """
        try:
            parts = program.split("*")
            predicates = []
            arguments = []
            
            for part in parts:
                match = re.match(r"^(\w+)\(([\d.]+)\)$", part.strip())
                if match:
                    predicates.append(match.group(1))
                    arguments.append(float(match.group(2)))
                else:
                    return None
            
            return ProgramStructure(predicates=predicates, arguments=arguments)
        except Exception:
            return None


def compute_structural_loss(
    reference_program: str,
    generated_program: str,
    predicate_weight: float = 1.0,
    argument_weight: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Compute structural similarity between reference and generated programs.
    
    Measures:
    1. Predicate overlap (Jaccard similarity)
    2. Argument similarity (MSE for matching predicates)
    
    Args:
        reference_program: Ground truth program string
        generated_program: Generated program string
        predicate_weight: Weight for predicate matching loss
        argument_weight: Weight for argument matching loss
        
    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains individual components
    """
    ref_struct = ProgramStructure.parse(reference_program)
    gen_struct = ProgramStructure.parse(generated_program)
    
    # If either program is invalid, return high loss
    if ref_struct is None or gen_struct is None:
        return 10.0, {
            "structural/predicate_loss": 1.0,
            "structural/argument_loss": 1.0,
            "structural/is_valid": 0.0,
        }
    
    # Predicate matching: Jaccard distance (1 - Jaccard similarity)
    ref_preds = set(ref_struct.predicates)
    gen_preds = set(gen_struct.predicates)
    
    if len(ref_preds.union(gen_preds)) == 0:
        predicate_loss = 0.0
    else:
        jaccard_sim = len(ref_preds.intersection(gen_preds)) / len(ref_preds.union(gen_preds))
        predicate_loss = 1.0 - jaccard_sim
    
    # Argument matching: MSE for predicates that appear in both
    matching_predicates = ref_preds.intersection(gen_preds)
    
    if len(matching_predicates) == 0:
        argument_loss = 1.0  # No matching predicates
    else:
        # Build dictionaries for quick lookup
        ref_dict = {pred: arg for pred, arg in zip(ref_struct.predicates, ref_struct.arguments)}
        gen_dict = {pred: arg for pred, arg in zip(gen_struct.predicates, gen_struct.arguments)}
        
        # Compute MSE for matching predicates
        squared_errors = [(ref_dict[pred] - gen_dict[pred]) ** 2 for pred in matching_predicates]
        argument_loss = sum(squared_errors) / len(matching_predicates)
    
    total_loss = predicate_weight * predicate_loss + argument_weight * argument_loss
    
    return total_loss, {
        "structural/predicate_loss": predicate_loss,
        "structural/argument_loss": argument_loss,
        "structural/is_valid": 1.0,
        "structural/num_matching_predicates": len(matching_predicates),
        "structural/jaccard_similarity": 1.0 - predicate_loss,
    }


class MotionConditionedSFTTrainer(Trainer):
    """Custom SFT trainer with motion encoder and dual loss function."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        motion_encoder: MotionEncoder,
        behaviour_model: Optional[BehaviourModel] = None,
        args: TrainingArguments = None,
        train_dataset = None,
        eval_dataset = None,
        # Loss weights
        ce_weight: float = 1.0,
        ot_weight: float = 1.0,
        structural_weight: float = 0.5,
        predicate_weight: float = 1.0,
        argument_weight: float = 1.0,
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
        self.structural_weight = structural_weight
        self.predicate_weight = predicate_weight
        self.argument_weight = argument_weight
        self.wandb_enabled = wandb_enabled and WANDB_AVAILABLE
        
        # Freeze behavior model if provided
        if self.behaviour_model is not None:
            for param in self.behaviour_model.model.parameters():
                param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute combined loss: CE + OT + Structural."""
        
        # Extract inputs
        motions = inputs["motion"]  # [B, T, 256]
        reference_programs = inputs["reference_program"]  # List[str]
        
        # Encode motions into prefix embeddings
        motion_embeddings = self.motion_encoder(motions)  # [B, num_prefix_tokens, hidden_dim]
        
        # Tokenize reference programs
        tokenized = self.tokenizer(
            reference_programs,
            padding=True,
            truncation=True,
            max_length=128,
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
        
        # 2. Generate programs for OT and structural loss
        with torch.no_grad():
            generated_ids = model.generate(
                inputs_embeds=motion_embeddings,
                max_new_tokens=64,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,  # Greedy for deterministic evaluation
            )
            generated_programs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 3. Optimal transport loss
        ot_loss = 0.0
        ot_count = 0
        
        if self.behaviour_model is not None and self.ot_weight > 0:
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
        
        # 4. Structural loss
        structural_loss = 0.0
        struct_metrics = {
            "structural/predicate_loss": 0.0,
            "structural/argument_loss": 0.0,
            "structural/is_valid": 0.0,
        }
        
        if self.structural_weight > 0:
            for ref_prog, gen_prog in zip(reference_programs, generated_programs):
                loss, metrics = compute_structural_loss(
                    ref_prog,
                    gen_prog,
                    predicate_weight=self.predicate_weight,
                    argument_weight=self.argument_weight,
                )
                structural_loss += loss
                for key, value in metrics.items():
                    struct_metrics[key] += value
            
            structural_loss = structural_loss / len(reference_programs)
            for key in struct_metrics:
                struct_metrics[key] = struct_metrics[key] / len(reference_programs)
        
        # Combined loss
        total_loss = (
            self.ce_weight * ce_loss +
            self.ot_weight * ot_loss +
            self.structural_weight * structural_loss
        )
        
        # Log to wandb
        if self.wandb_enabled and self.state.global_step % self.args.logging_steps == 0:
            log_dict = {
                "loss/total": total_loss.item() if torch.is_tensor(total_loss) else total_loss,
                "loss/ce": ce_loss.item(),
                "loss/ot": ot_loss if isinstance(ot_loss, float) else ot_loss.item(),
                "loss/structural": structural_loss if isinstance(structural_loss, float) else structural_loss.item(),
                **struct_metrics,
            }
            wandb.log(log_dict, step=self.state.global_step)
        
        return (total_loss, outputs) if return_outputs else total_loss


def train_motion_to_program_sft(
    train_motions: List[torch.Tensor],
    train_programs: List[str],
    val_motions: Optional[List[torch.Tensor]] = None,
    val_programs: Optional[List[str]] = None,
    behaviour_model: Optional[BehaviourModel] = None,
    output_dir: str = "outputs",
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    training_config: dict = None,
    motion_encoder_config: dict = None,
    wandb_enabled: bool = False,
    device: str = None,
):
    """Train a motion-to-program model using supervised fine-tuning with dual loss.
    
    Args:
        train_motions: List of training motion sequences [T, 256]
        train_programs: List of training program strings
        val_motions: Optional validation motions
        val_programs: Optional validation programs
        behaviour_model: Behavior model for motion generation (for OT loss)
        output_dir: Directory to save checkpoints
        model_name: HuggingFace model name
        training_config: Training hyperparameters
        motion_encoder_config: Motion encoder configuration
        wandb_enabled: Enable WandB logging
        device: Device to use (cuda/cpu)
        
    Returns:
        Trained MotionConditionedSFTTrainer
    """
    
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
            "ce_weight": 1.0,
            "ot_weight": 1.0,
            "structural_weight": 0.5,
            "predicate_weight": 1.0,
            "argument_weight": 1.0,
        }
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Initialize motion encoder
    motion_encoder = MotionEncoder(
        motion_dim=motion_encoder_config.get("motion_dim", 256),
        hidden_dim=motion_encoder_config.get("hidden_dim", 512),
        output_dim=model.config.hidden_size,
        num_layers=motion_encoder_config.get("num_layers", 4),
        num_prefix_tokens=motion_encoder_config.get("num_prefix_tokens", 8),
    )
    
    # Create datasets
    train_dataset = MotionProgramDataset(train_motions, train_programs, tokenizer)
    eval_dataset = None
    if val_motions is not None and val_programs is not None:
        eval_dataset = MotionProgramDataset(val_motions, val_programs, tokenizer)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_train_epochs", 10),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 5e-6),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        warmup_steps=training_config.get("warmup_steps", 100),
        logging_steps=training_config.get("logging_steps", 10),
        save_strategy=training_config.get("save_strategy", "epoch"),
        evaluation_strategy=training_config.get("evaluation_strategy", "epoch") if eval_dataset else "no",
        save_total_limit=training_config.get("save_total_limit", 3),
        fp16=training_config.get("fp16", torch.cuda.is_available()),
        report_to=["wandb"] if wandb_enabled else [],
        remove_unused_columns=False,  # Important: keep our custom columns
    )
    
    # Create trainer
    trainer = MotionConditionedSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        motion_encoder=motion_encoder,
        behaviour_model=behaviour_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        ce_weight=training_config.get("ce_weight", 1.0),
        ot_weight=training_config.get("ot_weight", 1.0),
        structural_weight=training_config.get("structural_weight", 0.5),
        predicate_weight=training_config.get("predicate_weight", 1.0),
        argument_weight=training_config.get("argument_weight", 1.0),
        wandb_enabled=wandb_enabled,
    )
    
    # Prepare with accelerator
    trainer.model, trainer.motion_encoder = accelerator.prepare(
        trainer.model, trainer.motion_encoder
    )
    
    # Train
    trainer.train()
    
    return trainer
