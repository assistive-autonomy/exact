#!/usr/bin/env python
"""Hyperparameter Tuning for Activity Assessment using STG-NF.

This script is designed for sweep optimization. It trains N models (one per activity)
and evaluates each model on its respective activity test data, reporting the mean AUC
across all models to drive hyperparameter optimization.

Key difference from assessment.py:
- Each model is evaluated ONLY on its target activity test data
- Goal is to maximize the probability that each model assigns to its training activity
- Reports mean_auc as the primary metric for sweep optimization
- Faster execution per trial (no cross-activity evaluation matrix)
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exact.anomaly import STG_NF, Graph, Trainer
from exact.data.esk import get_esk_dataloaders, get_unique_activities

import wandb
import matplotlib

matplotlib.use("Agg")


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
def evaluate_target_activity(
    model: STG_NF,
    test_loader,
    device: torch.device,
    target_activity: str,
) -> dict:
    """
    Evaluate model specifically on its target activity.

    This is for hyperparameter tuning - we want to maximize the probability
    that the model assigns to samples from its training activity.

    Returns dict with:
    - auc: AUC for target vs non-target samples
    - target_mean_log_prob: Mean log probability for target activity
    - other_mean_log_prob: Mean log probability for other activities
    """
    model.eval()

    # Collect scores per activity
    target_scores = []
    other_scores = []

    for batch in test_loader:
        poses, labels, activities = batch
        poses = poses.to(device, non_blocking=True).float()

        # Forward pass - get negative log-likelihood
        _, nll = model(poses)

        # Convert NLL to log-probability (higher = more likely)
        log_probs = -nll.cpu().numpy()

        # Separate by target vs other
        for i, activity in enumerate(activities):
            if activity == target_activity:
                target_scores.append(log_probs[i])
            else:
                other_scores.append(log_probs[i])

    results = {
        "target_activity": target_activity,
        "n_target_samples": len(target_scores),
        "n_other_samples": len(other_scores),
    }

    if target_scores:
        results["target_mean_log_prob"] = float(np.mean(target_scores))
        results["target_std_log_prob"] = float(np.std(target_scores))

    if other_scores:
        results["other_mean_log_prob"] = float(np.mean(other_scores))
        results["other_std_log_prob"] = float(np.std(other_scores))

    # Compute AUC: can we distinguish target from others?
    if target_scores and other_scores:
        all_scores = np.array(target_scores + other_scores)
        all_labels = np.array([1] * len(target_scores) + [0] * len(other_scores))

        try:
            results["auc"] = float(roc_auc_score(all_labels, all_scores))
        except ValueError:
            results["auc"] = 0.5

        results["separation"] = results["target_mean_log_prob"] - results["other_mean_log_prob"]
    else:
        results["auc"] = 0.5
        results["separation"] = 0.0

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tune Activity Assessment Hyperparameters")
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
        logger.info(f"Available Activities ({cfg.data.label_type})")
        activities = get_unique_activities(cfg.data.esk_dir, cfg.data.label_type)
        for a in activities:
            logger.info(f"  {a}")
        return

    # Set up
    set_seed(cfg.experiment.seed)
    device = get_device(cfg.experiment.device)

    # Get target activities
    target_activities = list(cfg.data.target_activities)
    n_models = len(target_activities)

    # Initialize wandb
    use_wandb = cfg.wandb_mode != "disabled"
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb_mode,
        )

    logger.info("Hyperparameter Tuning - STG-NF Activity Assessment")
    logger.info(f"Label type: {cfg.data.label_type}")
    logger.info(f"Target activities: {target_activities}")
    logger.info(f"Number of models to train: {n_models}")
    logger.info(f"Device: {device}")
    if use_wandb:
        logger.info(f"Wandb: {cfg.wandb_project}")

    # Train one model per target activity and collect AUC scores
    all_aucs = []
    all_separations = []

    for i, target_activity in enumerate(target_activities):
        logger.info(f"[{i+1}/{n_models}] Training model for '{target_activity}'...")

        # Load data for this target activity
        train_loader, test_loader, info = get_esk_dataloaders(
            esk_dir=cfg.data.esk_dir,
            label_type=cfg.data.label_type,
            target_activity=target_activity,
            test_activities=target_activities,  # Test on all target activities
            seg_len=cfg.data.seg_len,
            seg_stride=cfg.data.seg_stride,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
        )

        logger.info(f"  Train samples: {info['n_train_samples']}, Test samples: {info['n_test_samples']}")

        if info["n_train_samples"] == 0:
            logger.warning(f"No training samples for '{target_activity}', skipping...")
            continue

        # Build model
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

        # Optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        scheduler = ExponentialLR(optimizer, gamma=cfg.training.lr_decay)

        # Trainer (no checkpointing for tuning)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=str(device),
            checkpoint_dir=None,  # No checkpoints during tuning
            use_wandb=False,
        )

        # Train
        trainer.train(
            epochs=cfg.training.epochs,
            grad_clip=cfg.training.grad_clip,
            log_interval=cfg.training.log_interval,
        )

        # Evaluate on target activity
        results = evaluate_target_activity(
            model=model,
            test_loader=test_loader,
            device=device,
            target_activity=target_activity,
        )

        auc = results["auc"]
        separation = results["separation"]
        all_aucs.append(auc)
        all_separations.append(separation)

        logger.info(f"  AUC: {auc:.4f}, Separation: {separation:.4f}")

        # Log per-model metrics
        if use_wandb:
            safe_name = target_activity.replace(" ", "_")
            wandb.log({
                f"model_{safe_name}/auc": auc,
                f"model_{safe_name}/separation": separation,
                f"model_{safe_name}/target_log_prob": results.get("target_mean_log_prob", 0),
                f"model_{safe_name}/other_log_prob": results.get("other_mean_log_prob", 0),
                "models_trained": i + 1,
            })

    # Compute and log final mean metrics
    if all_aucs:
        mean_auc = float(np.mean(all_aucs))
        std_auc = float(np.std(all_aucs))
        mean_separation = float(np.mean(all_separations))
        std_separation = float(np.std(all_separations))

        logger.success("TUNING RESULTS")
        logger.info(f"Models trained: {len(all_aucs)}")
        logger.info(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        logger.info(f"Mean Separation: {mean_separation:.4f} ± {std_separation:.4f}")

        if use_wandb:
            # Log final metrics - these are what the sweep optimizes on
            wandb.log({
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "mean_separation": mean_separation,
                "std_separation": std_separation,
                "n_models": len(all_aucs),
            })

            # Also set as summary for sweep access
            wandb.summary["mean_auc"] = mean_auc
            wandb.summary["std_auc"] = std_auc
            wandb.summary["mean_separation"] = mean_separation
            wandb.summary["std_separation"] = std_separation
            wandb.summary["n_models"] = len(all_aucs)

    else:
        logger.error("No models were trained successfully!")
        if use_wandb:
            wandb.summary["mean_auc"] = 0.5
            wandb.summary["n_models"] = 0

    # Finish wandb
    if use_wandb:
        wandb.finish()

    logger.success("Done!")


if __name__ == "__main__":
    main()
