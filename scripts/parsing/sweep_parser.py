#!/usr/bin/env python
"""W&B Sweep agent for Motion Parser hyperparameter optimization.

This script serves a dual purpose:
  1. As the sweep *agent* entry point (called by `wandb agent`).
     When invoked by the sweep controller, it reads hyperparameters from
     `wandb.config`, merges them with the base config, and runs training.

  2. As a convenient *launcher* that creates the sweep and spawns parallel
     agents on the same machine (useful for multi-GPU setups like H200).

Usage (uv)
──────────
# Option A — Create + launch in one shot (recommended for single-node):
  uv run scripts/parsing/sweep_parser.py --create --agents 4 --count 50

# Option B — Manual two-step (useful when agents run on different nodes):
  wandb sweep configs/sweep_parser.yaml
  uv run scripts/parsing/sweep_parser.py --sweep-id <SWEEP_ID> --agents 4 --count 50

# Option C — Single agent (for debugging):
  uv run scripts/parsing/sweep_parser.py --sweep-id <SWEEP_ID> --count 1
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Disable tokenizers parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# ── Resolve project root (exact/) ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import wandb

# Import everything we need from the training script
from parsing.train_parser import (
    WANDB_AVAILABLE,
    AlignmentLoggingCallback,
    EncoderCheckpointCallback,
    GenerationEvalCallback,
    evaluate_samples,
    get_collate_fn,
    get_device,
    get_model_hidden_size,
    load_eval_samples,
    log_samples_to_wandb,
    print_eval_results,
    set_seed,
)


# ── Sweep parameter keys ───────────────────────────────────────────────────
# These are the hyperparameters that the sweep controller can override.
SWEEP_PARAM_KEYS = [
    # ST-GCN architecture
    "stgcn_hidden_channels",
    "stgcn_num_blocks",
    "stgcn_num_temporal_tokens",
    "stgcn_temporal_kernel",
    "stgcn_dropout",
    "stgcn_joint_embedding",
    # Alignment / auxiliary losses
    "alignment_weight",
    "alignment_dim",
    "alignment_temperature",
    # LoRA
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    # Training
    "per_device_train_batch_size",
    "learning_rate",
    "encoder_lr",
    "warmup_ratio",
    "weight_decay",
    "gradient_accumulation_steps",
]


# ── Shared dataset cache (loaded once, reused across trials) ─────────────────
# Populated by _ensure_data_cached() before the first trial.
_CACHED_TRAIN_DATA = None   # (programs, obs) or None
_CACHED_EVAL_DATA = None    # (programs, obs) or None


def _ensure_data_cached(
    train_path: str,
    eval_path: str | None = None,
    train_fraction: float = 1.0,
):
    """Pre-load train/eval motion data into module-level cache.

    Uses the .pt cache on disk (created automatically on first run).
    Subsequent sweep trials skip all I/O.

    Args:
        train_path: Path to training HDF5 file.
        eval_path: Optional path to eval HDF5 file.
        train_fraction: Fraction of training data to keep (0.0–1.0).
            Useful for sweeps where loading the full dataset is too slow.
            Default 1.0 = use everything.
    """
    global _CACHED_TRAIN_DATA, _CACHED_EVAL_DATA
    from exact.data.dataset import load_motion_data, load_motion_data_partial

    if _CACHED_TRAIN_DATA is None:
        if 0 < train_fraction < 1.0:
            # Read only the fraction we need directly from HDF5.
            # This avoids building the full .pt cache (which reads ALL
            # 50K samples) when we only need 5K for a sweep.
            import h5py
            with h5py.File(train_path, "r") as f:
                n_total = len(f.keys())
            max_samples = max(1, int(n_total * train_fraction))
            logger.info(
                f"Sweep: loading {max_samples}/{n_total} "
                f"({train_fraction:.0%}) training samples …"
            )
            programs, obs = load_motion_data_partial(
                train_path, max_samples=max_samples, shuffle=True, seed=42
            )
        else:
            # Full dataset — use .pt cache for speed
            logger.info(f"Pre-loading full training data from {train_path} …")
            programs, obs = load_motion_data(train_path)

        _CACHED_TRAIN_DATA = (programs, obs)
        logger.info(f"Cached {len(programs)} training samples in memory.")

    if eval_path and os.path.exists(eval_path) and _CACHED_EVAL_DATA is None:
        logger.info(f"Pre-loading eval data from {eval_path} …")
        _CACHED_EVAL_DATA = load_motion_data(eval_path)
        logger.info(f"Cached {len(_CACHED_EVAL_DATA[0])} eval samples in memory.")


# ── Training function (called by each sweep agent) ─────────────────────────


def train_sweep():
    """Single sweep trial: read wandb.config, train, evaluate, report."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        get_scheduler,
    )

    from exact.data import TrajectoryGenerationDataset
    from exact.data.dataset import PROMPT_PREFIX
    from exact.encoder import STGCNEncoder
    from exact.parser import MotionPrefixParser

    # wandb.agent() requires the function to call wandb.init() itself
    wandb.init()

    # ── 1. Build merged config ──────────────────────────────────────────────
    base_config_path = PROJECT_ROOT / "configs" / "parser.yaml"
    cfg = OmegaConf.load(base_config_path)

    # Override with sweep parameters
    sweep_cfg = dict(wandb.config)
    for key in SWEEP_PARAM_KEYS:
        if key in sweep_cfg:
            cfg[key] = sweep_cfg[key]

    # Reduce epoch count for sweep trials (faster feedback)
    cfg.num_train_epochs = cfg.get("sweep_max_epochs", 10)

    # Disable generation eval during sweep (expensive and noisy)
    cfg.generation_eval_steps = 0

    # Reduce eval samples for speed
    cfg.eval_samples = 32

    logger.info(f"Sweep trial config: {json.dumps(OmegaConf.to_container(cfg), indent=2, default=str)}")

    # ── 2. Setup ────────────────────────────────────────────────────────────
    device = get_device(cfg.get("device", "auto"))
    model_dtype = torch.bfloat16
    set_seed(cfg.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.get("output_dir", "results/parser")) / f"sweep_{wandb.run.id}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    logger.info(f"Sweep trial output: {output_dir}")
    logger.info(f"Device: {device}")

    # ── 3. Load model and tokenizer ─────────────────────────────────────────
    logger.info("[1/4] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_in_4bit = cfg.get("load_in_4bit", False)
    quant_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=None if load_in_4bit else model_dtype,
        device_map="auto",
        low_cpu_mem_usage=not load_in_4bit,
        quantization_config=quant_config,
    )
    model_hidden_size = get_model_hidden_size(base_model)
    logger.info(f"Model: {cfg.model_name}, hidden_size: {model_hidden_size}")

    # ── 4. Apply LoRA ───────────────────────────────────────────────────────
    logger.info("[2/4] Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(
            cfg.get(
                "target_modules",
                ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
        bias="none",
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # ── 5. Build ST-GCN encoder ─────────────────────────────────────────────
    logger.info("[3/4] Initializing ST-GCN encoder...")
    num_nodes = cfg.motion_dim // 3
    num_temporal_tokens = cfg.get("stgcn_num_temporal_tokens", 32)

    trajectory_encoder = STGCNEncoder(
        num_nodes=num_nodes,
        input_channels=3,
        hidden_channels=cfg.get("stgcn_hidden_channels", 128),
        output_dim=model_hidden_size,
        num_blocks=cfg.get("stgcn_num_blocks", 6),
        num_temporal_tokens=num_temporal_tokens,
        temporal_kernel_size=cfg.get("stgcn_temporal_kernel", 9),
        spatial_kernel_size=cfg.get("stgcn_spatial_kernel", 3),
        dropout=cfg.get("stgcn_dropout", 0.1),
        graph_strategy=cfg.get("graph_strategy", "spatial"),
        joint_embedding=cfg.get("stgcn_joint_embedding", False),
    ).to(device="cuda")

    encoder_params = sum(p.numel() for p in trajectory_encoder.parameters())
    logger.info(
        f"ST-GCN: {num_nodes} joints, {cfg.get('stgcn_num_blocks', 6)} blocks, "
        f"{num_temporal_tokens} temporal tokens, {encoder_params:,} params"
    )

    # ── 6. Create MotionPrefixParser ────────────────────────────────────────
    model = MotionPrefixParser(
        model=base_model,
        trajectory_encoder=trajectory_encoder,
        tokenizer=tokenizer,
        encoder_dim=model_hidden_size,
        alignment_weight=cfg.get("alignment_weight", 0.1),
        alignment_dim=cfg.get("alignment_dim", 256),
        alignment_temperature=cfg.get("alignment_temperature", 0.07),
    )

    # Move heads to CUDA with bf16
    model.motion_projection = model.motion_projection.to(device="cuda", dtype=model_dtype)
    model.motion_align_head = model.motion_align_head.to(device="cuda", dtype=model_dtype)
    model.program_align_head = model.program_align_head.to(device="cuda", dtype=model_dtype)
    model.logit_scale.data = model.logit_scale.data.to(device="cuda")
    model._motion_scale.data = model._motion_scale.data.to(device="cuda")

    # ── 7. Load datasets (from in-memory cache when available) ───────────────
    logger.info("[4/4] Loading datasets...")
    if not os.path.exists(cfg.train_data):
        raise FileNotFoundError(f"Training data not found: {cfg.train_data}")

    # Ensure the module-level cache is populated (no-op after first trial)
    _ensure_data_cached(
        cfg.train_data,
        cfg.get("eval_data"),
        train_fraction=cfg.get("sweep_train_fraction", 0.1),
    )

    train_programs, train_obs = _CACHED_TRAIN_DATA
    train_dataset = TrajectoryGenerationDataset(
        path=cfg.train_data,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        programs=train_programs,
        obs=train_obs,
    )
    logger.info(f"Loaded {len(train_dataset)} training samples (cached)")

    eval_dataset = None
    if cfg.eval_data and os.path.exists(cfg.eval_data):
        eval_programs, eval_obs = _CACHED_EVAL_DATA
        eval_dataset = TrajectoryGenerationDataset(
            path=cfg.eval_data,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_seq_length,
            programs=eval_programs,
            obs=eval_obs,
        )
        # Subsample eval set for speed during sweeps
        max_eval = cfg.get("max_eval_samples_training", 100)
        if len(eval_dataset) > max_eval:
            eval_dataset = torch.utils.data.Subset(
                eval_dataset, list(range(max_eval))
            )
        logger.info(f"Eval dataset: {len(eval_dataset)} samples (cached)")

    # ── 8. Optimizer with differential learning rates ───────────────────────
    encoder_lr = cfg.get("encoder_lr", cfg.learning_rate * 0.2)
    projection_lr = cfg.get("projection_lr", cfg.learning_rate)
    scale_lr = cfg.get("scale_lr", cfg.learning_rate * 2)

    encoder_params_list = list(model.trajectory_encoder.parameters())
    projection_params_list = list(model.motion_projection.parameters())
    align_params_list = (
        list(model.motion_align_head.parameters())
        + list(model.program_align_head.parameters())
        + [model.logit_scale]
    )
    scale_params_list = [model._motion_scale]

    special_param_ids = set(
        id(p)
        for p in (
            encoder_params_list
            + projection_params_list
            + align_params_list
            + scale_params_list
        )
    )
    lora_params_list = [
        p for p in model.parameters() if p.requires_grad and id(p) not in special_param_ids
    ]

    optimizer_grouped_parameters = [
        {"params": encoder_params_list, "lr": encoder_lr, "weight_decay": cfg.weight_decay},
        {"params": projection_params_list, "lr": projection_lr, "weight_decay": cfg.weight_decay},
        {"params": align_params_list, "lr": projection_lr, "weight_decay": 0.0},
        {"params": scale_params_list, "lr": scale_lr, "weight_decay": 0.0},
        {"params": lora_params_list, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
    ]

    custom_optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-8
    )

    logger.info(
        f"Differential LR — encoder: {encoder_lr:.1e}, projection: {projection_lr:.1e}, "
        f"scale: {scale_lr:.1e}, LoRA: {cfg.learning_rate:.1e}"
    )

    # ── 9. Training arguments ───────────────────────────────────────────────
    warmup_args = {}
    if cfg.get("warmup_ratio", 0) > 0:
        warmup_args["warmup_ratio"] = cfg.warmup_ratio
    elif cfg.get("warmup_steps", 0) > 0:
        warmup_args["warmup_steps"] = cfg.warmup_steps

    save_args = {
        "save_strategy": "steps",
        "save_total_limit": 2,       # fewer checkpoints during sweep
    }
    save_args["save_steps"] = cfg.get("save_steps", 2000)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=f"sweep_{wandb.run.id}",
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        **warmup_args,
        **save_args,
        logging_steps=cfg.get("logging_steps", 100),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=cfg.get("eval_steps", 2000) if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 8),
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=cfg.get("dataloader_prefetch_factor", 2),
        report_to="wandb",
        seed=cfg.seed,
        remove_unused_columns=False,
        save_safetensors=False,
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
    )

    # ── 10. Callbacks ───────────────────────────────────────────────────────
    callbacks = []

    # Early stopping (more aggressive for sweep — e.g. patience=3)
    sweep_patience = cfg.get("sweep_early_stopping_patience", 3)
    if sweep_patience > 0 and eval_dataset:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=sweep_patience)
        )

    callbacks.append(EncoderCheckpointCallback(model))
    callbacks.append(AlignmentLoggingCallback(model))

    # ── 11. Scheduler ───────────────────────────────────────────────────────
    total_train_steps = (
        len(train_dataset)
        // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
        * cfg.num_train_epochs
    )
    warmup_steps = (
        int(total_train_steps * cfg.warmup_ratio)
        if cfg.get("warmup_ratio", 0) > 0
        else cfg.get("warmup_steps", 0)
    )
    custom_scheduler = get_scheduler(
        cfg.get("lr_scheduler_type", "cosine"),
        optimizer=custom_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    # ── 12. Train ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=get_collate_fn(
            tokenizer,
            max_frame=cfg.get("max_frame", 1024),
        ),
        callbacks=callbacks,
        optimizers=(custom_optimizer, custom_scheduler),
    )

    logger.info("Starting sweep trial training...")
    trainer.train()

    # ── 13. Extract best eval_loss ──────────────────────────────────────────
    best_eval_loss = None
    if trainer.state.best_metric is not None:
        best_eval_loss = trainer.state.best_metric
    elif eval_dataset:
        # Fallback: run a final eval pass
        eval_result = trainer.evaluate()
        best_eval_loss = eval_result.get("eval_loss")

    if best_eval_loss is not None:
        wandb.summary["eval_loss_best"] = best_eval_loss
        logger.info(f"Best eval_loss: {best_eval_loss:.4f}")

    # ── 14. Skip generation evaluation during sweeps ───────────────────────
    # Generation eval is expensive (constrained decoding per sample).
    # We rely on eval_loss as the sweep metric and keep generation eval
    # only for full training runs (train_parser.py).
    logger.info("Skipping post-training generation eval (sweep mode).")

    # ── 15. Cleanup to free GPU memory for next trial ───────────────────────
    wandb.finish()
    del model, trainer, base_model, trajectory_encoder
    torch.cuda.empty_cache()

    logger.success(f"Sweep trial complete. Output: {output_dir}")


# ── Launcher helpers ────────────────────────────────────────────────────────


def create_sweep(config_path: str, project: str, entity: str) -> str:
    """Create a W&B sweep and return the sweep ID."""
    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
        entity=entity,
    )
    logger.info(f"Created sweep: {sweep_id}")
    return sweep_id


def launch_agents_parallel(
    sweep_id: str,
    project: str,
    entity: str,
    num_agents: int,
    count_per_agent: int,
    gpus: list[int] | None = None,
):
    """Launch multiple sweep agents as subprocesses, each pinned to a GPU.

    For a single GPU (like an H200), multiple agents share the GPU and
    W&B's sweep controller ensures they don't duplicate work.  If you
    have multiple GPUs, each agent is pinned to its own GPU via
    CUDA_VISIBLE_DEVICES.

    Uses `uv run` to ensure the correct virtual environment is activated.
    """
    available_gpus = gpus or list(range(torch.cuda.device_count()))
    if not available_gpus:
        available_gpus = [0]

    # Detect whether to use `uv run` or fall back to sys.executable
    import shutil
    use_uv = shutil.which("uv") is not None
    script_path = str(Path(__file__).resolve())

    processes = []
    for i in range(num_agents):
        gpu_id = available_gpus[i % len(available_gpus)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        if use_uv:
            cmd = [
                "uv", "run",
                script_path,
                "--sweep-id", sweep_id,
                "--count", str(count_per_agent),
                "--project", project,
                "--entity", entity,
            ]
        else:
            cmd = [
                sys.executable,
                script_path,
                "--sweep-id", sweep_id,
                "--count", str(count_per_agent),
                "--project", project,
                "--entity", entity,
            ]
        logger.info(f"Launching agent {i+1}/{num_agents} on GPU {gpu_id}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, env=env, cwd=str(PROJECT_ROOT))
        processes.append(proc)
        time.sleep(5)  # stagger starts to avoid race conditions

    logger.info(f"All {num_agents} agents launched. Waiting for completion...")

    for proc in processes:
        proc.wait()

    logger.success("All sweep agents finished.")


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Motion Parser W&B Sweep — agent and launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sweep + launch 4 parallel agents, 50 trials each:
  uv run scripts/parsing/sweep_parser.py --create --agents 4 --count 50

  # Join an existing sweep (e.g. from another node):
  uv run scripts/parsing/sweep_parser.py --sweep-id abc123 --count 50

  # Single agent (the default when called by `wandb agent`):
  uv run scripts/parsing/sweep_parser.py --sweep-id abc123 --count 1
        """,
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a new sweep from configs/sweep_parser.yaml and launch agents",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Existing W&B sweep ID to join",
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "sweep_parser.yaml"),
        help="Path to sweep YAML config",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=1,
        help="Number of parallel sweep agents to launch (default: 1)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Max trials per agent (default: 50)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="exact",
        help="W&B project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="<CHOOSE_YOUR_ENTITY>",
        help="W&B entity/team",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g. '0,1,2,3'). Default: all available.",
    )

    args = parser.parse_args()

    gpus = None
    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(",")]

    # ── Mode 1: Create new sweep and launch parallel agents ─────────────
    if args.create:
        sweep_id = create_sweep(args.sweep_config, args.project, args.entity)
        if args.agents > 1:
            launch_agents_parallel(
                sweep_id=sweep_id,
                project=args.project,
                entity=args.entity,
                num_agents=args.agents,
                count_per_agent=args.count,
                gpus=gpus,
            )
        else:
            # Single agent — run in-process
            wandb.agent(
                sweep_id,
                function=train_sweep,
                count=args.count,
                project=args.project,
                entity=args.entity,
            )
        return

    # ── Mode 2: Join existing sweep (single agent in-process) ───────────
    if args.sweep_id:
        if args.agents > 1:
            launch_agents_parallel(
                sweep_id=args.sweep_id,
                project=args.project,
                entity=args.entity,
                num_agents=args.agents,
                count_per_agent=args.count,
                gpus=gpus,
            )
        else:
            wandb.agent(
                args.sweep_id,
                function=train_sweep,
                count=args.count,
                project=args.project,
                entity=args.entity,
            )
        return

    # ── Mode 3: Called directly (no args needed) ─────────────────────────
    # train_sweep() handles wandb.init() and wandb.finish() internally.
    logger.info("No --create or --sweep-id provided. Running single trial.")
    train_sweep()


if __name__ == "__main__":
    main()
