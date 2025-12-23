#!/usr/bin/env python
"""Motion-conditioned Parser training and evaluation."""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Suppress DataParallel scalar gather warning
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exact.data import TrajectoryGenerationDataset
from exact.parser import MotionConditionedParser, TrajectoryEncoder

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_config(config_path: str, overrides: list[str] = None) -> DictConfig:
    """Load config file and apply CLI overrides."""
    base_cfg = OmegaConf.load(config_path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(base_cfg, override_cfg)
    else:
        cfg = base_cfg

    return cfg


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """Get torch device from config string."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


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


def load_eval_samples(h5_path: str, n_samples: int = 8):
    """Load a subset of samples for evaluation."""
    samples = []
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())[:n_samples]
        for key in keys:
            motion = torch.tensor(f[key]["motion"][()], dtype=torch.float32)
            program = f[key].attrs["program"]
            samples.append({"key": key, "motion": motion, "target": program})
    return samples


@torch.no_grad()
def evaluate_samples(
    model: MotionConditionedParser,
    tokenizer,
    samples: list,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int = 256,
    grammar_processor=None,
) -> dict:
    """Evaluate model on sample batch and return metrics."""
    model.eval()

    results = []
    exact_matches = 0

    for sample in tqdm(samples, desc="Evaluating samples"):
        motion = sample["motion"].unsqueeze(0).to(device=device, dtype=dtype)

        generated_ids = model.generate(
            motion=motion,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            grammar_processor=grammar_processor,
        )

        predicted = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        target = sample["target"]
        is_match = predicted == target

        if is_match:
            exact_matches += 1

        results.append(
            {
                "key": sample["key"],
                "target": target,
                "predicted": predicted,
                "exact_match": is_match,
            }
        )

    accuracy = exact_matches / len(samples) if samples else 0

    return {
        "accuracy": accuracy,
        "exact_matches": exact_matches,
        "total": len(samples),
        "samples": results,
    }


def log_samples_to_wandb(eval_results: dict, step: int = None):
    """Log sample predictions to wandb as a table."""
    if not WANDB_AVAILABLE:
        return

    # Create wandb table
    table = wandb.Table(columns=["key", "target", "predicted", "exact_match"])

    for sample in eval_results["samples"]:
        table.add_data(
            sample["key"],
            sample["target"],
            sample["predicted"],
            "✓" if sample["exact_match"] else "✗",
        )

    log_data = {
        "eval/sample_predictions": table,
        "eval/accuracy": eval_results["accuracy"],
        "eval/exact_matches": eval_results["exact_matches"],
    }

    if step is not None:
        wandb.log(log_data, step=step)
    else:
        wandb.log(log_data)


def print_eval_results(eval_results: dict):
    """Pretty print evaluation results."""
    print("\n" + "=" * 70)
    print("SAMPLE EVALUATION RESULTS")
    print("=" * 70)
    print(
        f"Accuracy: {eval_results['exact_matches']}/{eval_results['total']} ({eval_results['accuracy']:.1%})"
    )
    print("-" * 70)

    for sample in eval_results["samples"]:
        status = "✓" if sample["exact_match"] else "✗"
        print(f"\n[{status}] {sample['key']}")
        print(f"  Target:    {sample['target']}")
        print(f"  Predicted: {sample['predicted']}")

    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Motion-conditioned Parser")
    parser.add_argument(
        "--config", type=str, default="configs/parser.yaml", help="Config file"
    )
    parser.add_argument(
        "--eval-only", type=str, default=None, help="Checkpoint dir for eval only"
    )
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config, args.overrides)

    # Setup
    set_seed(cfg.seed)
    device = get_device(cfg.get("device", "auto"))
    model_dtype = torch.float32

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.get("output_dir", "results/parser")) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    OmegaConf.save(cfg, output_dir / "config.yaml")

    # Initialize wandb
    use_wandb = cfg.wandb_mode != "disabled" and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"parser_{timestamp}",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb_mode,
        )

    print(f"\n{'='*70}")
    print("Motion-conditioned Parser")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    if use_wandb:
        print(f"Wandb: {cfg.wandb_project}")
    print(f"{'='*70}\n")

    # Eval-only mode
    if args.eval_only:
        logger.info(f"Eval-only mode: {args.eval_only}")
        model, tokenizer, _, model_dtype = load_checkpoint(args.eval_only, device)

        # Load eval samples
        eval_samples = load_eval_samples(
            cfg.eval_data, n_samples=cfg.get("eval_samples", 8)
        )

        # Create grammar processor
        try:
            from exact.parser.utils import create_grammar_processor

            grammar_processor = create_grammar_processor(tokenizer)
        except Exception as e:
            logger.warning(f"Could not create grammar processor: {e}")
            grammar_processor = None

        # Evaluate
        eval_results = evaluate_samples(
            model,
            tokenizer,
            eval_samples,
            device,
            model_dtype,
            grammar_processor=grammar_processor,
        )
        print_eval_results(eval_results)

        if use_wandb:
            log_samples_to_wandb(eval_results)
            wandb.finish()
        return

    # Training mode
    logger.info("[1/4] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=model_dtype,
    ).to(device)
    model_hidden_size = base_model.config.hidden_size
    logger.info(f"Model: {cfg.model_name}, hidden_size: {model_hidden_size}")

    # Apply LoRA
    logger.info("[2/4] Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
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
        tokenizer=tokenizer,
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
    training_args = TrainingArguments(
        output_dir=str(output_dir),
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
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=device.type == "cuda",
        report_to="wandb" if use_wandb else "none",
        seed=cfg.seed,
        remove_unused_columns=False,
        save_safetensors=False,
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

    # Save trajectory encoder
    torch.save(
        trajectory_encoder.state_dict(),
        output_dir / "trajectory_encoder.pt",
    )

    # Save LoRA adapter
    base_model.save_pretrained(output_dir / "lora_adapter")
    tokenizer.save_pretrained(output_dir / "lora_adapter")

    # Post-training evaluation on samples
    logger.info("Running sample evaluation...")
    eval_samples = load_eval_samples(
        cfg.eval_data, n_samples=cfg.get("eval_samples", 8)
    )

    try:
        from exact.parser.utils import create_grammar_processor

        grammar_processor = create_grammar_processor(tokenizer)
    except Exception as e:
        logger.warning(f"Could not create grammar processor: {e}")
        grammar_processor = None

    eval_results = evaluate_samples(
        model,
        tokenizer,
        eval_samples,
        device,
        model_dtype,
        grammar_processor=grammar_processor,
    )
    print_eval_results(eval_results)

    # Save results
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    if use_wandb:
        log_samples_to_wandb(eval_results)
        wandb.summary["final_accuracy"] = eval_results["accuracy"]
        wandb.finish()

    logger.success(f"Training complete! Results saved to: {output_dir}")


def load_checkpoint(checkpoint_dir: str, device: torch.device):
    """Load trained model from checkpoint directory."""
    import yaml

    config_path = os.path.join(checkpoint_dir, "config.yaml")
    if not os.path.exists(config_path):
        # Try hydra config location
        config_path = os.path.join(checkpoint_dir, ".hydra", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found in: {checkpoint_dir}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_dtype = torch.float32

    # Load tokenizer and base model
    logger.info(f"Loading base model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=model_dtype,
    ).to(device)
    model_hidden_size = base_model.config.hidden_size

    # Load LoRA adapter
    lora_path = os.path.join(checkpoint_dir, "lora_adapter")
    if os.path.exists(lora_path):
        logger.info(f"Loading LoRA adapter from: {lora_path}")
        base_model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        logger.warning(f"LoRA adapter not found at {lora_path}")

    # Load trajectory encoder
    encoder_path = os.path.join(checkpoint_dir, "trajectory_encoder.pt")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Trajectory encoder not found: {encoder_path}")

    logger.info(f"Loading trajectory encoder from: {encoder_path}")
    trajectory_encoder = TrajectoryEncoder(
        trajectory_dim=config["motion_dim"],
        hidden_dim=config["motion_hidden_dim"],
        output_dim=model_hidden_size,
        num_layers=config["motion_num_layers"],
        num_prefix_tokens=config["num_prefix_tokens"],
    ).to(device, dtype=model_dtype)
    trajectory_encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    # Create motion-conditioned parser
    model = MotionConditionedParser(
        model=base_model,
        trajectory_encoder=trajectory_encoder,
        tokenizer=tokenizer,
    )
    model.eval()

    return model, tokenizer, config, model_dtype


if __name__ == "__main__":
    main()
