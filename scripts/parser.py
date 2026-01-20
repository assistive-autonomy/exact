#!/usr/bin/env python
"""Motion-conditioned Parser training and evaluation."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Disable tokenizers parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import h5py
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exact.data import TrajectoryGenerationDataset
from exact.parser import MotionConditionedParser
from exact.parser.utils import create_grammar_processor
from exact.encoder import STGCNEncoder

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


def get_model_hidden_size(model) -> int:
    """Get hidden size from model config (handles Gemma and Llama)."""
    config = model.config
    # Gemma 3 uses text_config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    # Standard models (Llama, etc.) use hidden_size directly
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    raise AttributeError(f"Cannot find hidden_size in model config: {type(config)}")


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
    num_retries: int = 2,
    retry_temperature: float = 0.8,
    retry_top_p: float = 0.9,
    use_constrained_decoding: bool = True,
    log_attempts: bool = False,
) -> dict:
    """Evaluate model on sample batch and return metrics."""
    from exact.parser.utils import post_process_program
    from exact.programs.edit_distance import program_edit_distance, parse_to_tree
    from lark.exceptions import LarkError

    model.eval()
    
    # Create grammar processor for constrained decoding
    grammar_processor = create_grammar_processor(tokenizer) if use_constrained_decoding else None

    results = []
    exact_matches = 0
    valid_programs = 0
    edit_distances = []

    def _generate_with_retries(motion_batch: torch.Tensor):
        attempts = []

        for attempt in range(num_retries + 1):
            use_sampling = attempt > 0
            temperature = retry_temperature if use_sampling else None
            top_p = retry_top_p if use_sampling else None
            
            # Use constrained decoding for first attempts, fallback to unconstrained + repair
            use_grammar = grammar_processor is not None and attempt < num_retries

            generated_ids = model.generate(
                motion=motion_batch,
                max_new_tokens=max_new_tokens,
                do_sample=use_sampling,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                grammar_processor=grammar_processor if use_grammar else None,
                use_cache=True,
            )

            raw_predicted = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
            predicted, is_valid = post_process_program(raw_predicted, repair=True)

            attempts.append(
                {
                    "attempt": attempt + 1,
                    "sampling": use_sampling,
                    "temperature": temperature,
                    "top_p": top_p,
                    "constrained": use_grammar,
                    "raw": raw_predicted,
                    "predicted": predicted,
                    "is_valid": is_valid,
                }
            )

            logger.debug(
                f"Attempt {attempt + 1} ({'sample' if use_sampling else 'greedy'}, "
                f"{'constrained' if use_grammar else 'unconstrained'}): "
                f"valid={is_valid}, predicted='{predicted}'"
            )

            if is_valid:
                break

        return attempts[-1], attempts

    for sample in tqdm(samples, desc="Evaluating samples"):
        motion = sample["motion"].unsqueeze(0).to(device=device, dtype=dtype)

        best_attempt, attempts = _generate_with_retries(motion)

        raw_predicted = best_attempt["raw"]
        predicted = best_attempt["predicted"]
        is_valid = best_attempt["is_valid"]

        target = sample["target"]
        is_match = predicted == target

        if is_match:
            exact_matches += 1
        if is_valid:
            valid_programs += 1

        # Compute edit distance if both programs are valid
        edit_dist = None
        normalized_edit_dist = None
        if is_valid and predicted:
            try:
                target_tree = parse_to_tree(target)
                pred_tree = parse_to_tree(predicted)
                edit_dist = program_edit_distance(target_tree, pred_tree)
                # Normalize by max tree size for comparability
                max_size = max(len(target_tree), len(pred_tree))
                normalized_edit_dist = edit_dist / max_size if max_size > 0 else 0.0
                edit_distances.append(normalized_edit_dist)
            except LarkError:
                # If parsing fails, skip edit distance
                pass

        results.append(
            {
                "key": sample["key"],
                "target": target,
                "predicted": predicted,
                "raw_predicted": raw_predicted if raw_predicted != predicted else None,
                "exact_match": is_match,
                "is_valid": is_valid,
                "edit_distance": edit_dist,
                "normalized_edit_distance": normalized_edit_dist,
                "attempts": attempts if log_attempts else None,
            }
        )

    accuracy = exact_matches / len(samples) if samples else 0
    validity_rate = valid_programs / len(samples) if samples else 0
    mean_edit_distance = sum(edit_distances) / len(edit_distances) if edit_distances else None

    return {
        "accuracy": accuracy,
        "validity_rate": validity_rate,
        "mean_normalized_edit_distance": mean_edit_distance,
        "exact_matches": exact_matches,
        "valid_programs": valid_programs,
        "total": len(samples),
        "samples": results,
    }


def log_samples_to_wandb(eval_results: dict, step: int = None):
    """Log sample predictions to wandb as a table."""
    if not WANDB_AVAILABLE:
        return

    # Create wandb table with validity and edit distance columns
    table = wandb.Table(columns=["key", "target", "predicted", "exact_match", "valid", "edit_dist"])

    for sample in eval_results["samples"]:
        edit_dist_str = f"{sample.get('normalized_edit_distance', 'N/A'):.3f}" if sample.get('normalized_edit_distance') is not None else "N/A"
        table.add_data(
            sample["key"],
            sample["target"],
            sample["predicted"],
            "✓" if sample["exact_match"] else "✗",
            "✓" if sample.get("is_valid", True) else "✗",
            edit_dist_str,
        )

    log_data = {
        "eval/sample_predictions": table,
        "eval/accuracy": eval_results["accuracy"],
        "eval/exact_matches": eval_results["exact_matches"],
        "eval/validity_rate": eval_results.get("validity_rate", 1.0),
        "eval/valid_programs": eval_results.get("valid_programs", eval_results["total"]),
    }
    
    if eval_results.get("mean_normalized_edit_distance") is not None:
        log_data["eval/mean_normalized_edit_distance"] = eval_results["mean_normalized_edit_distance"]

    if step is not None:
        wandb.log(log_data, step=step)
    else:
        wandb.log(log_data)


def print_eval_results(eval_results: dict):
    """Pretty print evaluation results."""
    logger.info("SAMPLE EVALUATION RESULTS")
    logger.info(
        f"Accuracy: {eval_results['exact_matches']}/{eval_results['total']} ({eval_results['accuracy']:.1%})"
    )
    logger.info(
        f"Validity: {eval_results.get('valid_programs', eval_results['total'])}/{eval_results['total']} ({eval_results.get('validity_rate', 1.0):.1%})"
    )
    if eval_results.get("mean_normalized_edit_distance") is not None:
        logger.info(
            f"Mean Normalized Edit Distance: {eval_results['mean_normalized_edit_distance']:.3f} (lower is better)"
        )

    for sample in eval_results["samples"]:
        match_status = "✓" if sample["exact_match"] else "✗"
        valid_status = "V" if sample.get("is_valid", True) else "X"
        edit_dist = sample.get("normalized_edit_distance")
        edit_str = f" ED={edit_dist:.2f}" if edit_dist is not None else ""
        logger.info(f"[{match_status}|{valid_status}]{edit_str} {sample['key']}")
        logger.info(f"  Target:    {sample['target']}")
        logger.info(f"  Predicted: {sample['predicted']}")
        if sample.get("raw_predicted"):
            logger.info(f"  Raw:       {sample['raw_predicted']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Motion-conditioned Parser")
    parser.add_argument(
        "--config", type=str, default="configs/parser.yaml", help="Config file"
    )
    parser.add_argument(
        "--eval-only", type=str, default=None, help="Checkpoint dir for eval only"
    )
    parser.add_argument(
        "--resume", type=str, default=None, 
        help="Resume training from checkpoint dir (e.g., results/parser/20260115_120430)"
    )
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config, args.overrides)

    # Setup device
    device = get_device(cfg.get("device", "auto"))
    model_dtype = torch.bfloat16  # Match model dtype

    # Setup
    set_seed(cfg.seed)

    # Handle resume vs new run
    resume_from_checkpoint = None
    if args.resume:
        # Resume from existing run directory
        output_dir = Path(args.resume)
        if not output_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {args.resume}")
        
        # Find the latest checkpoint in the directory
        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
        if checkpoints:
            resume_from_checkpoint = str(checkpoints[-1])
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        else:
            logger.warning(f"No checkpoints found in {output_dir}, starting fresh")
        
        # Load config from resumed run if it exists
        saved_config = output_dir / "config.yaml"
        if saved_config.exists():
            logger.info(f"Loading config from resumed run: {saved_config}")
            cfg = OmegaConf.load(saved_config)
            # Apply any CLI overrides on top
            if args.overrides:
                override_cfg = OmegaConf.from_dotlist(args.overrides)
                cfg = OmegaConf.merge(cfg, override_cfg)
    else:
        # Create new output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(cfg.get("output_dir", "results/parser")) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save config
        OmegaConf.save(cfg, output_dir / "config.yaml")

    # Get run name from output directory
    run_name = output_dir.name

    # Initialize wandb
    use_wandb = cfg.wandb_mode != "disabled" and WANDB_AVAILABLE
    if use_wandb:
        # Check if we're resuming and have a wandb run ID saved
        wandb_id_file = output_dir / "wandb_run_id.txt"
        wandb_resume = None
        wandb_id = None
        
        if args.resume and wandb_id_file.exists():
            wandb_id = wandb_id_file.read_text().strip()
            wandb_resume = "must"
            logger.info(f"Resuming wandb run: {wandb_id}")
        
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"parser_{run_name}",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb_mode,
            id=wandb_id,
            resume=wandb_resume,
        )
        
        # Save wandb run ID for future resume
        if not wandb_id_file.exists():
            wandb_id_file.write_text(wandb.run.id)

    logger.info("Motion-conditioned Parser")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    if args.resume:
        logger.info(f"Resuming from: {resume_from_checkpoint}")
    if use_wandb:
        logger.info(f"Wandb: {cfg.wandb_project}")

    # Eval-only mode
    if args.eval_only:
        logger.info(f"Eval-only mode: {args.eval_only}")
        model, tokenizer, _, model_dtype = load_checkpoint(args.eval_only, device)

        # Load eval samples
        eval_samples = load_eval_samples(
            cfg.eval_data, n_samples=cfg.get("eval_samples", 8)
        )

        eval_kwargs = {
            "max_new_tokens": cfg.get("generation_max_new_tokens", 256),
            "num_retries": cfg.get("generation_retries", 2),
            "retry_temperature": cfg.get("generation_retry_temperature", 0.8),
            "retry_top_p": cfg.get("generation_retry_top_p", 0.9),
            "use_constrained_decoding": cfg.get("use_constrained_decoding", True),
            "log_attempts": cfg.get("log_generation_attempts", False),
        }

        # Evaluate
        eval_results = evaluate_samples(
            model,
            tokenizer,
            eval_samples,
            device,
            model_dtype,
            **eval_kwargs,
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

    load_in_4bit = cfg.get("load_in_4bit", False)
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Loading base model with 4-bit quantization")

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=None if load_in_4bit else torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=not load_in_4bit,
        quantization_config=quant_config,
    )
    model_hidden_size = get_model_hidden_size(base_model)
    logger.info(f"Model: {cfg.model_name}, hidden_size: {model_hidden_size} (bf16)")

    # Apply LoRA
    logger.info("[2/4] Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.get(
            "target_modules",
            [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )),
        bias="none",
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Initialize ST-GCN trajectory encoder (keep in float32 for BatchNorm stability)
    logger.info("[3/4] Initializing ST-GCN trajectory encoder...")
    
    num_nodes = cfg.motion_dim // 3  # Assumes 3D coordinates (x,y,z)
    num_temporal_tokens = cfg.get("stgcn_num_temporal_tokens", 8)
    trajectory_encoder = STGCNEncoder(
        num_nodes=num_nodes,
        input_channels=3,
        hidden_channels=cfg.get("stgcn_hidden_channels", 64),
        output_dim=model_hidden_size,
        num_blocks=cfg.get("stgcn_num_blocks", 4),
        num_temporal_tokens=num_temporal_tokens,
        temporal_kernel_size=cfg.get("stgcn_temporal_kernel", 9),
        spatial_kernel_size=cfg.get("stgcn_spatial_kernel", 3),
        dropout=cfg.get("stgcn_dropout", 0.1),
        graph_strategy=cfg.get("graph_strategy", "spatial"),
    ).to(device="cuda")  # Keep float32 for BatchNorm, output will be cast to bf16
    
    logger.info(f"Using STGCNEncoder with {num_nodes} joints, {cfg.get('stgcn_num_blocks', 4)} blocks, {num_temporal_tokens} temporal tokens")

    # Count encoder parameters
    encoder_params = sum(p.numel() for p in trajectory_encoder.parameters())
    trainable_encoder_params = sum(p.numel() for p in trajectory_encoder.parameters() if p.requires_grad)
    logger.info(f"Encoder params: {trainable_encoder_params:,} trainable / {encoder_params:,} total")

    # Create motion-conditioned parser with cross-modal attention and alignment
    use_cross_attention = cfg.get("use_cross_attention", True)
    use_alignment_loss = cfg.get("use_alignment_loss", True)
    alignment_weight = cfg.get("alignment_weight", 0.1)
    alignment_latent_dim = cfg.get("alignment_latent_dim", 256)
    cross_attention_heads = cfg.get("cross_attention_heads", 8)
    
    model = MotionConditionedParser(
        model=base_model,
        trajectory_encoder=trajectory_encoder,
        tokenizer=tokenizer,
        motion_dim=cfg.motion_dim,
        use_cross_attention=use_cross_attention,
        use_alignment_loss=use_alignment_loss,
        alignment_weight=alignment_weight,
        alignment_latent_dim=alignment_latent_dim,
        cross_attention_heads=cross_attention_heads,
    )
    
    if use_cross_attention or use_alignment_loss:
        logger.info(f"Cross-modal attention: {use_cross_attention}, Alignment loss: {use_alignment_loss} (weight={alignment_weight})")
        # Count new module parameters
        if use_cross_attention:
            cross_attn_params = sum(p.numel() for p in model.cross_attention.parameters())
            logger.info(f"  Cross-attention params: {cross_attn_params:,}")
        if use_alignment_loss:
            proj_params = sum(p.numel() for p in model.motion_projection.parameters()) + \
                         sum(p.numel() for p in model.text_projection.parameters())
            logger.info(f"  Projection head params: {proj_params:,}")

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
        # Optionally limit eval dataset size during training for speed
        max_eval_samples = cfg.get("max_eval_samples_training", None)
        if max_eval_samples and len(eval_dataset) > max_eval_samples:
            import random
            indices = random.sample(range(len(eval_dataset)), max_eval_samples)
            eval_dataset = torch.utils.data.Subset(eval_dataset, indices)
            logger.info(f"Using {max_eval_samples} evaluation samples (subsampled for training speed)")
        else:
            logger.info(f"Loaded {len(eval_dataset)} evaluation samples")

    # Setup training arguments
    warmup_args = {}
    if cfg.get("warmup_ratio", 0) > 0:
        warmup_args["warmup_ratio"] = cfg.warmup_ratio
    elif cfg.get("warmup_steps", 0) > 0:
        warmup_args["warmup_steps"] = cfg.warmup_steps

    # Build save args
    save_args = {"save_strategy": cfg.save_strategy, "save_total_limit": cfg.save_total_limit}
    if cfg.save_strategy == "steps":
        save_args["save_steps"] = cfg.get("save_steps", 500)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=f"parser_{run_name}",
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        **warmup_args,
        **save_args,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy if eval_dataset else "no",
        eval_steps=cfg.get("eval_steps", 500) if eval_dataset else None,
        load_best_model_at_end=cfg.load_best_model_at_end if eval_dataset else False,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=False,  # Lower eval_loss is better
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=device.type == "cuda",
        dataloader_prefetch_factor=cfg.get("dataloader_prefetch_factor", 2),
        report_to="wandb" if use_wandb else "none",
        seed=cfg.seed,
        remove_unused_columns=False,
        save_safetensors=False,
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
    )

    # Setup callbacks
    callbacks = []
    early_stopping_patience = cfg.get("early_stopping_patience", 0)
    if early_stopping_patience > 0 and eval_dataset:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )
        logger.info(f"Early stopping enabled (patience={early_stopping_patience})")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=get_collate_fn(tokenizer),
        callbacks=callbacks if callbacks else None,
    )

    # Train
    if resume_from_checkpoint:
        logger.info(f"Resuming training from: {resume_from_checkpoint}")
    else:
        logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model components
    logger.info("Saving model...")
    trainer.save_model()

    # Save trajectory encoder
    torch.save(
        trajectory_encoder.state_dict(),
        output_dir / "trajectory_encoder.pt",
    )
    
    # Save cross-attention module if used
    if use_cross_attention:
        torch.save(
            model.cross_attention.state_dict(),
            output_dir / "cross_attention.pt",
        )
        logger.info("Saved cross-attention module")
    
    # Save projection heads if used
    if use_alignment_loss:
        torch.save(
            {
                "motion": model.motion_projection.state_dict(),
                "text": model.text_projection.state_dict(),
            },
            output_dir / "projections.pt",
        )
        logger.info("Saved projection heads")

    # Save LoRA adapter
    base_model.save_pretrained(output_dir / "lora_adapter")
    tokenizer.save_pretrained(output_dir / "lora_adapter")

    # Post-training evaluation on samples
    logger.info("Running sample evaluation...")
    eval_samples = load_eval_samples(
        cfg.eval_data, n_samples=cfg.get("eval_samples", 8)
    )

    eval_kwargs = {
        "max_new_tokens": cfg.get("generation_max_new_tokens", 256),
        "num_retries": cfg.get("generation_retries", 2),
        "retry_temperature": cfg.get("generation_retry_temperature", 0.8),
        "retry_top_p": cfg.get("generation_retry_top_p", 0.9),
        "use_constrained_decoding": cfg.get("use_constrained_decoding", True),
        "log_attempts": cfg.get("log_generation_attempts", False),
    }

    # Move model to device for evaluation
    model.to(device)
    eval_results = evaluate_samples(
        model,
        tokenizer,
        eval_samples,
        device,
        model_dtype,
        **eval_kwargs,
    )
    print_eval_results(eval_results)

    # Save results
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    if use_wandb:
        log_samples_to_wandb(eval_results)
        wandb.summary["final_accuracy"] = eval_results["accuracy"]
        wandb.summary["final_validity_rate"] = eval_results["validity_rate"]
        if eval_results.get("mean_normalized_edit_distance") is not None:
            wandb.summary["final_mean_edit_distance"] = eval_results["mean_normalized_edit_distance"]
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

    model_dtype = torch.bfloat16  # Match training dtype

    load_in_4bit = config.get("load_in_4bit", False)
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Loading base model with 4-bit quantization")

    # Load tokenizer and base model
    logger.info(f"Loading base model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in bfloat16 (same as training)
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=None if load_in_4bit else model_dtype,
        device_map="auto",
        low_cpu_mem_usage=not load_in_4bit,
        quantization_config=quant_config,
    )
    model_hidden_size = get_model_hidden_size(base_model)

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

    logger.info(f"Loading ST-GCN trajectory encoder from: {encoder_path}")

    # Build ST-GCN encoder (keep float32 for BatchNorm stability)
    num_nodes = config["motion_dim"] // 3
    trajectory_encoder = STGCNEncoder(
        num_nodes=num_nodes,
        input_channels=3,
        hidden_channels=config.get("stgcn_hidden_channels", 64),
        output_dim=model_hidden_size,
        num_blocks=config.get("stgcn_num_blocks", 4),
        num_temporal_tokens=config.get("stgcn_num_temporal_tokens", 8),
        temporal_kernel_size=config.get("stgcn_temporal_kernel", 9),
        spatial_kernel_size=config.get("stgcn_spatial_kernel", 3),
        dropout=config.get("stgcn_dropout", 0.1),
        graph_strategy=config.get("graph_strategy", "spatial"),
    ).to(device)  # Keep float32, output will be cast to model_dtype
    logger.info("Using STGCNEncoder")

    trajectory_encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    # Create motion-conditioned parser with optional cross-modal attention and alignment
    # Load cross-attention module if it was saved
    cross_attention_path = os.path.join(checkpoint_dir, "cross_attention.pt")
    projections_path = os.path.join(checkpoint_dir, "projections.pt")
    
    use_cross_attention = config.get("use_cross_attention", os.path.exists(cross_attention_path))
    use_alignment_loss = config.get("use_alignment_loss", os.path.exists(projections_path))
    
    model = MotionConditionedParser(
        model=base_model,
        trajectory_encoder=trajectory_encoder,
        tokenizer=tokenizer,
        use_cross_attention=use_cross_attention,
        use_alignment_loss=use_alignment_loss,
        alignment_weight=config.get("alignment_weight", 0.1),
        alignment_latent_dim=config.get("alignment_latent_dim", 256),
        cross_attention_heads=config.get("cross_attention_heads", 8),
    )
    
    # Load cross-attention weights if available
    if use_cross_attention and os.path.exists(cross_attention_path):
        logger.info(f"Loading cross-attention from: {cross_attention_path}")
        model.cross_attention.load_state_dict(torch.load(cross_attention_path, map_location=device))
    
    # Load projection heads if available
    if use_alignment_loss and os.path.exists(projections_path):
        logger.info(f"Loading projection heads from: {projections_path}")
        projections = torch.load(projections_path, map_location=device)
        model.motion_projection.load_state_dict(projections["motion"])
        model.text_projection.load_state_dict(projections["text"])
    
    model.eval()

    return model, tokenizer, config, model_dtype


if __name__ == "__main__":
    main()