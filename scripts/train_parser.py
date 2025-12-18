"""Train the motion-conditioned parser using HuggingFace Trainer."""
import os

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import wandb
from exact.data import TrajectoryGenerationDataset
from exact.parser import MotionConditionedParser, TrajectoryEncoder


def get_collate_fn(tokenizer):
    """Create a collate function for the dataloader."""
    def collate_fn(batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        obs = torch.stack([x["obs"] for x in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "motion": obs,
            "labels": input_ids.clone(),
        }
    return collate_fn


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """Train the motion-conditioned parser."""
    logger.info("Training Configuration")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Set random seeds
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = cfg.get("bf16", True) and torch.cuda.is_bf16_supported() and device == "cuda"
    model_dtype = torch.bfloat16 if use_bf16 else torch.float32
    logger.info(f"Using device: {device}, dtype: {model_dtype}")

    # Initialize wandb
    if cfg.wandb_mode != "disabled":
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb_mode,
        )

    # Load tokenizer and model
    logger.info("[1/4] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=model_dtype,
        device_map="auto",
    )
    model_hidden_size = base_model.config.hidden_size
    logger.info(f"Model: {cfg.model_name}, hidden_size: {model_hidden_size}")

    # Apply LoRA
    logger.info("[2/4] Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Initialize trajectory encoder
    logger.info("[3/4] Initializing trajectory encoder...")
    trajectory_encoder = TrajectoryEncoder(
        trajectory_dim=cfg.motion_dim,
        hidden_dim=cfg.motion_hidden_dim,
        output_dim=model_hidden_size,
        num_layers=cfg.motion_num_layers,
        num_prefix_tokens=cfg.num_prefix_tokens,
    ).to(device, dtype=model_dtype)

    # Create motion-conditioned parser
    model = MotionConditionedParser(
        model=base_model,
        trajectory_encoder=trajectory_encoder,
    )

    # Load datasets
    logger.info("[4/4] Loading datasets...")
    if not os.path.exists(cfg.train_data):
        raise FileNotFoundError(f"Training data not found: {cfg.train_data}")

    train_dataset = TrajectoryGenerationDataset(
        path=cfg.train_data,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
    )
    logger.info(f"Loaded {len(train_dataset)} training samples")

    eval_dataset = None
    if cfg.eval_data and os.path.exists(cfg.eval_data):
        eval_dataset = TrajectoryGenerationDataset(
            path=cfg.eval_data,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_seq_length,
        )
        logger.info(f"Loaded {len(eval_dataset)} evaluation samples")

    # Setup training arguments
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy if eval_dataset else "no",
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end if eval_dataset else False,
        metric_for_best_model=cfg.metric_for_best_model,
        bf16=use_bf16,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=device == "cuda",
        report_to="wandb" if cfg.wandb_mode != "disabled" else "none",
        seed=cfg.seed,
        remove_unused_columns=False,  # Important: keep motion column
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=get_collate_fn(tokenizer),
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model components
    logger.info("Saving model...")
    trainer.save_model()
    
    # Save trajectory encoder separately for inference
    torch.save(
        trajectory_encoder.state_dict(),
        os.path.join(output_dir, "trajectory_encoder.pt"),
    )

    # Save LoRA adapter
    base_model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    tokenizer.save_pretrained(os.path.join(output_dir, "lora_adapter"))

    logger.success(f"Training complete! Models saved to: {output_dir}")

    if cfg.wandb_mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()


