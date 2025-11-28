"""Train the motion-conditioned parser."""
import os

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from exact.data import TrajectoryGenerationDataset
from exact.parser import MotionConditionedParser, TrajectoryEncoder, create_grammar_processor
from exact.trainer import ParserTrainer


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    """Train the motion-conditioned parser."""
    logger.info("Training Configuration")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Set random seeds
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Initialize wandb
    wandb_enabled = cfg.wandb.mode != "disabled"
    if wandb_enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )

    # Load tokenizer and model
    logger.info("[1/5] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.training.model.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create grammar processor for constrained decoding
    grammar_path = cfg.training.get("grammar_path", None)
    grammar_processor = create_grammar_processor(tokenizer, grammar_path)
    logger.info(f"Grammar processor created (path: {grammar_path or 'default'})")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.training.model.name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model_hidden_size = model.config.hidden_size
    logger.info(f"Model: {cfg.training.model.name}, hidden_size: {model_hidden_size}")

    # Apply LoRA
    logger.info("[2/5] Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.training.lora.r,
        lora_alpha=cfg.training.lora.alpha,
        lora_dropout=cfg.training.lora.dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Initialize trajectory encoder
    logger.info("[3/5] Initializing trajectory encoder...")
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    trajectory_encoder = TrajectoryEncoder(
        trajectory_dim=cfg.training.motion_encoder.motion_dim,
        hidden_dim=cfg.training.motion_encoder.hidden_dim,
        output_dim=model_hidden_size,
        num_layers=cfg.training.motion_encoder.num_layers,
        num_prefix_tokens=cfg.training.motion_encoder.num_prefix_tokens,
    ).to(device, dtype=model_dtype)

    # Create motion-conditioned parser
    parser = MotionConditionedParser(
        model=model,
        trajectory_encoder=trajectory_encoder,
    )

    # Load datasets
    logger.info("[4/5] Loading datasets...")
    train_data_path = cfg.training.train_data_path
    eval_data_path = cfg.training.get("eval_data_path", None)

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found: {train_data_path}")

    train_dataset = TrajectoryGenerationDataset(
        path=train_data_path,
        tokenizer=tokenizer,
        max_seq_length=cfg.training.max_seq_length,
    )
    logger.info(f"Loaded {len(train_dataset)} training samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    eval_loader = None
    if eval_data_path and os.path.exists(eval_data_path):
        eval_dataset = TrajectoryGenerationDataset(
            path=eval_data_path,
            tokenizer=tokenizer,
            max_seq_length=cfg.training.max_seq_length,
        )
        logger.info(f"Loaded {len(eval_dataset)} evaluation samples")

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )

    # Setup optimizer and scheduler
    logger.info("[5/5] Setting up training...")
    optimizer = torch.optim.AdamW(
        [
            {"params": parser.trajectory_encoder.parameters()},
            {"params": parser.model.parameters()},
        ],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.get("weight_decay", 0.01),
    )

    num_training_steps = len(train_loader) * cfg.training.num_train_epochs
    warmup_steps = cfg.training.warmup_steps

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # Create trainer
    trainer = ParserTrainer(
        model=parser,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        dtype=model_dtype,
        gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 4),
        max_grad_norm=cfg.training.max_grad_norm,
    )

    # Training loop
    logger.info("Starting training...")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    best_eval_loss = float("inf")

    for epoch in range(1, cfg.training.num_train_epochs + 1):
        train_metrics = trainer.train_epoch(
            train_loader,
            epoch,
            logger=wandb if wandb_enabled else None,
        )
        logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")

        if eval_loader is not None:
            eval_metrics = trainer.evaluate(
                eval_loader,
                tokenizer=tokenizer,
                grammar_processor=grammar_processor,
                logger=wandb if wandb_enabled else None,
            )
            logger.info(f"Epoch {epoch} - Eval Loss: {eval_metrics['loss']:.4f}")
            if "validity_rate" in eval_metrics:
                logger.info(f"Epoch {epoch} - Validity Rate: {eval_metrics['validity_rate']:.2%}")

            # Save best model
            if eval_metrics["loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["loss"]
                trainer.save_checkpoint(
                    os.path.join(checkpoints_dir, "best_model.pt"),
                    epoch,
                )
                logger.success("New best model saved!")

        # Save periodic checkpoint
        if epoch % cfg.training.get("save_every", 5) == 0:
            trainer.save_checkpoint(
                os.path.join(checkpoints_dir, f"epoch_{epoch}.pt"),
                epoch,
            )

    # Save final model
    trainer.save_checkpoint(
        os.path.join(checkpoints_dir, "final_model.pt"),
        cfg.training.num_train_epochs,
    )

    # Save trajectory encoder separately for inference
    torch.save(
        trajectory_encoder.state_dict(),
        os.path.join(output_dir, "trajectory_encoder.pt"),
    )

    # Save LoRA adapter
    model.save_pretrained(os.path.join(output_dir, "lora_adapter"))

    logger.success(f"Training complete! Models saved to: {output_dir}")

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()