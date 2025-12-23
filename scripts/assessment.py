#!/usr/bin/env python
"""Activity Assessment using STG-NF normalizing flows."""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exact.anomaly import STG_NF, Graph, Trainer
from exact.anomaly.esk_dataset import get_esk_dataloaders, get_unique_activities

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
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device(device_str: str) -> torch.device:
    """Get torch device from config string."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


@torch.no_grad()
def evaluate_by_activity(
    model: STG_NF,
    test_loader,
    device: torch.device,
    target_activity: str,
    use_wandb: bool = False,
) -> dict:
    """
    Evaluate model and compute per-activity statistics.

    For a good anomaly detector:
    - Target activity should have HIGH probability (low NLL)
    - Other activities should have LOW probability (high NLL)

    Returns dict with:
    - Per-activity mean/std of log-probability
    - Separation metrics between target and others
    """
    model.eval()

    # Collect scores per activity
    activity_scores = defaultdict(list)

    pbar = tqdm(test_loader, desc="Evaluating")
    for batch in pbar:
        poses, labels, activities = batch
        poses = poses.to(device, non_blocking=True).float()

        # Forward pass - get negative log-likelihood
        _, nll = model(poses)

        # Convert NLL to log-probability (higher = more likely)
        log_probs = -nll.cpu().numpy()

        # Group by activity
        for i, activity in enumerate(activities):
            activity_scores[activity].append(log_probs[i])

    # Compute statistics per activity
    results = {
        "per_activity": {},
        "target_activity": target_activity,
    }

    all_target_scores = []
    all_other_scores = []

    for activity, scores in activity_scores.items():
        scores = np.array(scores)
        results["per_activity"][activity] = {
            "mean_log_prob": float(np.mean(scores)),
            "std_log_prob": float(np.std(scores)),
            "n_samples": len(scores),
        }

        if activity == target_activity:
            all_target_scores.extend(scores)
        else:
            all_other_scores.extend(scores)

    # Compute separation metrics
    if all_target_scores and all_other_scores:
        target_mean = np.mean(all_target_scores)
        other_mean = np.mean(all_other_scores)

        # Higher target mean = better (model assigns higher prob to target)
        results["target_mean_log_prob"] = float(target_mean)
        results["other_mean_log_prob"] = float(other_mean)
        results["separation"] = float(target_mean - other_mean)

        # Compute AUC: can we distinguish target from others?
        all_scores = np.array(all_target_scores + all_other_scores)
        all_labels = np.array(
            [1] * len(all_target_scores) + [0] * len(all_other_scores)
        )

        try:
            results["auc_target_vs_others"] = float(
                roc_auc_score(all_labels, all_scores)
            )
        except ValueError:
            results["auc_target_vs_others"] = 0.5

    # Log to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb_metrics = {
            "eval/target_mean_log_prob": results.get("target_mean_log_prob", 0),
            "eval/other_mean_log_prob": results.get("other_mean_log_prob", 0),
            "eval/separation": results.get("separation", 0),
            "eval/auc": results.get("auc_target_vs_others", 0.5),
        }
        for activity, stats in results["per_activity"].items():
            safe_name = activity.replace(" ", "_")
            wandb_metrics[f"eval/activity_{safe_name}_mean"] = stats["mean_log_prob"]
        wandb.log(wandb_metrics)

    return results


def print_results(results: dict):
    """Pretty print evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    target = results["target_activity"]
    print(f"\nTarget Activity (trained on): {target}")

    print("\n--- Per-Activity Log-Probability (higher = more likely) ---")
    print(f"{'Activity':<25} {'Mean':>12} {'Std':>12} {'N Samples':>12}")
    print("-" * 65)

    # Sort by mean log-prob (target should be highest)
    sorted_activities = sorted(
        results["per_activity"].items(),
        key=lambda x: x[1]["mean_log_prob"],
        reverse=True,
    )

    for activity, stats in sorted_activities:
        marker = " <-- TARGET" if activity == target else ""
        print(
            f"{activity:<25} {stats['mean_log_prob']:>12.4f} {stats['std_log_prob']:>12.4f} {stats['n_samples']:>12}{marker}"
        )

    print("\n--- Separation Metrics ---")
    if "separation" in results:
        print(f"Target mean log-prob:  {results['target_mean_log_prob']:.4f}")
        print(f"Others mean log-prob:  {results['other_mean_log_prob']:.4f}")
        print(f"Separation (target - others): {results['separation']:.4f}")
        print(f"AUC (target vs others): {results['auc_target_vs_others']:.4f}")

    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Activity Assessment")
    parser.add_argument(
        "--config", type=str, default="configs/assessment.yaml", help="Config file"
    )
    parser.add_argument(
        "--list-activities", action="store_true", help="List available activities"
    )
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config, args.overrides)

    # List activities mode
    if args.list_activities:
        print(f"\n=== Available Activities ({cfg.data.label_type}) ===")
        activities = get_unique_activities(cfg.data.esk_dir, cfg.data.label_type)
        for a in activities:
            print(f"  {a}")
        return

    # Set up
    set_seed(cfg.experiment.seed)
    device = get_device(cfg.experiment.device)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (
        cfg.experiment.name
        or f"{cfg.data.label_type}_{cfg.data.target_activity}_{timestamp}"
    )
    output_dir = Path(cfg.output.dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    OmegaConf.save(cfg, output_dir / "config.yaml")

    # Initialize wandb
    use_wandb = cfg.wandb_mode != "disabled" and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb_mode,
        )

    print(f"\n{'='*70}")
    print(f"Activity Assessment - STG-NF")
    print(f"{'='*70}")
    print(f"Label type: {cfg.data.label_type}")
    print(f"Target activity (normal): {cfg.data.target_activity}")
    print(f"Test activities: {cfg.data.test_activities or 'all'}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    if use_wandb:
        print(f"Wandb: {cfg.wandb_project}/{exp_name}")
    print(f"{'='*70}\n")

    # Load data
    print("Loading data...")
    train_loader, test_loader, info = get_esk_dataloaders(
        esk_dir=cfg.data.esk_dir,
        label_type=cfg.data.label_type,
        target_activity=cfg.data.target_activity,
        test_activities=(
            list(cfg.data.test_activities) if cfg.data.test_activities else None
        ),
        seg_len=cfg.data.seg_len,
        seg_stride=cfg.data.seg_stride,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    print(
        f"Training samples ({cfg.data.target_activity} only): {info['n_train_samples']}"
    )
    print(f"Test samples: {info['n_test_samples']}")
    print(f"Test activity distribution:")
    for act, count in sorted(info["test_activity_counts"].items(), key=lambda x: -x[1]):
        marker = " <-- TARGET" if act == cfg.data.target_activity else ""
        print(f"  {act}: {count}{marker}")

    if info["n_train_samples"] == 0:
        print(
            f"\nERROR: No training samples found for activity '{cfg.data.target_activity}'!"
        )
        print(f"Available activities: {info['label_names']}")
        if use_wandb:
            wandb.finish(exit_code=1)
        return

    # Build model
    print("\nBuilding model...")
    pose_shape = (3, cfg.data.seg_len, 24)  # (C, T, V) for SMPL

    graph = Graph(
        layout="smpl",
        strategy=cfg.model.adj_strategy,
        max_hop=1,
    )

    model = STG_NF(
        pose_shape=pose_shape,
        hidden_channels=cfg.model.hidden_channels,
        K=cfg.model.K,
        L=cfg.model.L,
        graph=graph,
        learn_prior=cfg.model.learn_prior,
        R=cfg.model.R,
        temporal_kernel_size=cfg.model.temporal_kernel,
        permutation=cfg.model.permutation,
        device=str(device),
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    if use_wandb:
        wandb.log({"model/n_params": n_params})

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = ExponentialLR(optimizer, gamma=cfg.training.lr_decay)

    # Trainer with wandb support
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        checkpoint_dir=(
            str(output_dir / "checkpoints") if cfg.output.save_checkpoints else None
        ),
        use_wandb=use_wandb,
    )

    # Load checkpoint if provided
    if cfg.checkpoint.path:
        print(f"\nLoading checkpoint: {cfg.checkpoint.path}")
        trainer.load_checkpoint(cfg.checkpoint.path)

    # Train (unless eval_only)
    if not cfg.checkpoint.eval_only:
        print("\nTraining...")
        history = trainer.train(
            epochs=cfg.training.epochs,
            grad_clip=cfg.training.grad_clip,
            log_interval=cfg.training.log_interval,
        )

        # Save training history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # Evaluate
    if cfg.evaluation.enabled:
        print("\nEvaluating per-activity...")
        results = evaluate_by_activity(
            model=model,
            test_loader=test_loader,
            device=device,
            target_activity=cfg.data.target_activity,
            use_wandb=use_wandb,
        )

        # Print results
        print_results(results)

        # Save results
        results_dict = {
            "config": OmegaConf.to_container(cfg, resolve=True),
            "info": info,
            "results": results,
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(results_dict, f, indent=2, default=str)

    # Finish wandb
    if use_wandb:
        # Log final results as summary
        if cfg.evaluation.enabled and "separation" in results:
            wandb.summary["final_auc"] = results["auc_target_vs_others"]
            wandb.summary["final_separation"] = results["separation"]
        wandb.finish()

    print(f"\nResults saved to: {output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
