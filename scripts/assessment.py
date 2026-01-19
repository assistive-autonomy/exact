#!/usr/bin/env python
"""Activity Assessment using STG-NF normalizing flows.

This script performs full evaluation of STG-NF models for activity recognition:
- Trains N models (one per target activity)
- Evaluates each model on ALL target activities (cross-activity evaluation)
- Generates separability matrix showing how well each model distinguishes activities
- Reports per-activity AUC and separation metrics

For hyperparameter tuning, use tune_assessment.py instead which is faster
and reports mean_auc for sweep optimization.
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
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
import matplotlib.pyplot as plt
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
    other_activity_means = []  # For balanced separation calculation

    for activity, scores in activity_scores.items():
        scores = np.array(scores)
        activity_mean = float(np.mean(scores))
        results["per_activity"][activity] = {
            "mean_log_prob": activity_mean,
            "std_log_prob": float(np.std(scores)),
            "n_samples": len(scores),
        }

        if activity == target_activity:
            all_target_scores.extend(scores)
        else:
            all_other_scores.extend(scores)
            other_activity_means.append(activity_mean)

    # Compute separation metrics (class-balanced: average of per-activity means)
    if all_target_scores and other_activity_means:
        target_mean = np.mean(all_target_scores)
        # Balanced: each activity contributes equally regardless of sample count
        other_mean_balanced = np.mean(other_activity_means)

        # Higher target mean = better (model assigns higher prob to target)
        results["target_mean_log_prob"] = float(target_mean)
        results["other_mean_log_prob"] = float(other_mean_balanced)
        results["separation"] = float(target_mean - other_mean_balanced)

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
    if use_wandb:
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
    target = results["target_activity"]
    logger.info("EVALUATION RESULTS")
    logger.info(f"Target Activity (trained on): {target}")

    logger.info("Per-Activity Log-Probability (higher = more likely)")
    header = f"{'Activity':<25} {'Mean':>12} {'Std':>12} {'N Samples':>12}"
    logger.info(header)

    # Sort by mean log-prob (target should be highest)
    sorted_activities = sorted(
        results["per_activity"].items(),
        key=lambda x: x[1]["mean_log_prob"],
        reverse=True,
    )

    for activity, stats in sorted_activities:
        marker = " <-- TARGET" if activity == target else ""
        logger.info(
            f"{activity:<25} {stats['mean_log_prob']:>12.4f} {stats['std_log_prob']:>12.4f} {stats['n_samples']:>12}{marker}"
        )

    if "separation" in results:
        logger.info("Separation Metrics")
        logger.info(f"Target mean log-prob:  {results['target_mean_log_prob']:.4f}")
        logger.info(f"Others mean log-prob:  {results['other_mean_log_prob']:.4f}")
        logger.info(f"Separation (target - others): {results['separation']:.4f}")
        logger.info(f"AUC (target vs others): {results['auc_target_vs_others']:.4f}")


def aggregate_results(results_dirs: list[Path]) -> dict:
    """
    Aggregate results from multiple model runs into separability matrices.

    Args:
        results_dirs: List of directories containing results.json files

    Returns:
        Dictionary with:
        - activities: List of activity names
        - score_matrix: NxN matrix of mean log-probabilities
          (row=model trained on, col=activity evaluated)
        - auc_matrix: NxN matrix of pairwise AUC scores
        - separation_vector: Per-model separation scores
    """
    # Load all results
    all_results = []
    for results_dir in results_dirs:
        results_file = results_dir / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                all_results.append(json.load(f))

    if not all_results:
        raise ValueError("No results.json files found")

    # Get all unique activities across all results
    all_activities = set()
    for r in all_results:
        all_activities.update(r["results"]["per_activity"].keys())
    activities = sorted(all_activities)
    n_activities = len(activities)
    activity_to_idx = {a: i for i, a in enumerate(activities)}

    # Initialize matrices
    score_matrix = np.full((n_activities, n_activities), np.nan)
    auc_matrix = np.full((n_activities, n_activities), np.nan)
    separation_vector = np.full(n_activities, np.nan)

    # Fill matrices from results
    for r in all_results:
        target = r["results"]["target_activity"]
        if target not in activity_to_idx:
            continue
        row_idx = activity_to_idx[target]

        # Fill score row
        for activity, stats in r["results"]["per_activity"].items():
            if activity in activity_to_idx:
                col_idx = activity_to_idx[activity]
                score_matrix[row_idx, col_idx] = stats["mean_log_prob"]

        # Fill separation
        if "separation" in r["results"]:
            separation_vector[row_idx] = r["results"]["separation"]

        # Fill AUC (target vs others is aggregate, compute pairwise from scores)
        if "auc_target_vs_others" in r["results"]:
            # Diagonal represents overall AUC
            auc_matrix[row_idx, row_idx] = r["results"]["auc_target_vs_others"]

    return {
        "activities": activities,
        "score_matrix": score_matrix,
        "auc_matrix": auc_matrix,
        "separation_vector": separation_vector,
        "n_models": len(all_results),
    }


def plot_separability_matrix(
    aggregated: dict,
    output_path: Path,
    title: str = "Activity Separability Matrix",
):
    """
    Plot separability matrix as a heatmap.

    Rows = model trained on activity
    Cols = activity being evaluated
    Values = mean log-probability (higher = model thinks it's more likely)

    Diagonal should be highest if models are working well.
    """
    activities = aggregated["activities"]
    score_matrix = aggregated["score_matrix"]
    n = len(activities)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize per row for better visualization (relative scores)
    normalized = score_matrix.copy()
    for i in range(n):
        row = normalized[i, :]
        valid = ~np.isnan(row)
        if valid.any():
            row_min, row_max = row[valid].min(), row[valid].max()
            if row_max > row_min:
                normalized[i, valid] = (row[valid] - row_min) / (row_max - row_min)

    im = ax.imshow(normalized, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized Score (per row)", rotation=-90, va="bottom")

    # Set ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(activities, rotation=45, ha="right")
    ax.set_yticklabels(activities)

    # Labels
    ax.set_xlabel("Evaluated Activity")
    ax.set_ylabel("Model Trained On")
    ax.set_title(title)

    # Add text annotations with actual values
    for i in range(n):
        for j in range(n):
            val = score_matrix[i, j]
            if not np.isnan(val):
                # Highlight diagonal
                weight = "bold" if i == j else "normal"
                color = "white" if normalized[i, j] < 0.5 else "black"
                ax.text(
                    j, i, f"{val:.1f}",
                    ha="center", va="center",
                    color=color, fontsize=8, fontweight=weight
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved separability matrix plot: {output_path}")


def plot_separation_summary(
    aggregated: dict,
    output_path: Path,
    title: str = "Per-Activity Separation & AUC",
):
    """
    Plot bar chart of separation scores and AUC per activity.
    """
    activities = aggregated["activities"]
    separation = aggregated["separation_vector"]
    auc_diag = np.diag(aggregated["auc_matrix"])
    n = len(activities)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Separation scores
    ax1 = axes[0]
    valid_sep = ~np.isnan(separation)
    colors = ["green" if s > 0 else "red" for s in separation]
    bars = ax1.bar(np.arange(n)[valid_sep], separation[valid_sep],
                   color=[c for c, v in zip(colors, valid_sep) if v])
    ax1.set_xticks(np.arange(n))
    ax1.set_xticklabels(activities, rotation=45, ha="right")
    ax1.set_ylabel("Separation (target - others)")
    ax1.set_title("Separation Score per Model")
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # AUC scores
    ax2 = axes[1]
    valid_auc = ~np.isnan(auc_diag)
    colors = ["green" if a > 0.5 else "red" for a in auc_diag]
    ax2.bar(np.arange(n)[valid_auc], auc_diag[valid_auc],
            color=[c for c, v in zip(colors, valid_auc) if v])
    ax2.set_xticks(np.arange(n))
    ax2.set_xticklabels(activities, rotation=45, ha="right")
    ax2.set_ylabel("AUC (target vs others)")
    ax2.set_title("AUC Score per Model")
    ax2.axhline(y=0.5, color="black", linestyle="--", linewidth=0.5, label="Random")
    ax2.set_ylim(0, 1)
    ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved separation summary plot: {output_path}")


def print_aggregated_results(aggregated: dict):
    """Print aggregated results summary."""
    activities = aggregated["activities"]
    separation = aggregated["separation_vector"]
    auc_diag = np.diag(aggregated["auc_matrix"])

    logger.info("AGGREGATED RESULTS SUMMARY")
    logger.info(f"Models evaluated: {aggregated['n_models']}")
    logger.info(f"Activities: {len(activities)}")

    logger.info("Per-Model Performance")
    header = f"{'Activity (trained on)':<25} {'Separation':>12} {'AUC':>12}"
    logger.info(header)

    for i, act in enumerate(activities):
        sep = separation[i]
        auc = auc_diag[i]
        sep_str = f"{sep:.4f}" if not np.isnan(sep) else "N/A"
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        logger.info(f"{act:<25} {sep_str:>12} {auc_str:>12}")

    # Summary stats
    valid_sep = separation[~np.isnan(separation)]
    valid_auc = auc_diag[~np.isnan(auc_diag)]

    if len(valid_sep) > 0:
        logger.info(f"Mean Separation: {np.mean(valid_sep):.4f} ± {np.std(valid_sep):.4f}")
    if len(valid_auc) > 0:
        logger.info(f"Mean AUC: {np.mean(valid_auc):.4f} ± {np.std(valid_auc):.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Activity Assessment")
    parser.add_argument(
        "--config", type=str, default="configs/assessment.yaml", help="Config file"
    )
    parser.add_argument(
        "--list-activities", action="store_true", help="List available activities"
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        nargs="+",
        metavar="DIR",
        help="Aggregate results from multiple runs and generate separability matrix",
    )
    parser.add_argument(
        "--aggregate-output",
        type=str,
        default="results/assessment/aggregated",
        help="Output directory for aggregated results",
    )
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")

    args = parser.parse_args()

    # Aggregate mode - combine results from multiple runs
    if args.aggregate:
        logger.info("Aggregating Results")
        results_dirs = [Path(d) for d in args.aggregate]
        logger.info(f"Input directories: {len(results_dirs)}")

        aggregated = aggregate_results(results_dirs)
        print_aggregated_results(aggregated)

        # Save and plot
        output_dir = Path(args.aggregate_output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save aggregated data
        with open(output_dir / "aggregated.json", "w") as f:
            json.dump(
                {
                    "activities": aggregated["activities"],
                    "score_matrix": aggregated["score_matrix"].tolist(),
                    "auc_matrix": aggregated["auc_matrix"].tolist(),
                    "separation_vector": aggregated["separation_vector"].tolist(),
                    "n_models": aggregated["n_models"],
                },
                f,
                indent=2,
            )

        # Generate plots
        plot_separability_matrix(
            aggregated,
            output_dir / "separability_matrix.png",
        )
        plot_separation_summary(
            aggregated,
            output_dir / "separation_summary.png",
        )

        logger.success(f"Aggregated results saved to: {output_dir}")
        return

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

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg.experiment.name or f"{cfg.data.label_type}_{timestamp}"
    output_dir = Path(cfg.output.dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    OmegaConf.save(cfg, output_dir / "config.yaml")

    # Get target activities
    target_activities = list(cfg.data.target_activities)
    n_models = len(target_activities)

    # Initialize wandb
    use_wandb = cfg.wandb_mode != "disabled"
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb_mode,
        )

    logger.info("Activity Assessment - STG-NF")
    logger.info(f"Label type: {cfg.data.label_type}")
    logger.info(f"Target activities: {target_activities}")
    logger.info(f"Number of models to train: {n_models}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    if use_wandb:
        logger.info(f"Wandb: {cfg.wandb_project}/{exp_name}")

    # Train one model per target activity
    all_results = []
    model_dirs = []

    for i, target_activity in enumerate(target_activities):
        logger.info(f"Model {i+1}/{n_models}: Training on '{target_activity}'")

        # Create model-specific output directory
        safe_name = target_activity.replace(" ", "_")
        model_dir = output_dir / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_dirs.append(model_dir)

        # Load data for this target activity
        logger.info("Loading data...")
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

        logger.info(f"Training samples ({target_activity} only): {info['n_train_samples']}")
        logger.info(f"Test samples: {info['n_test_samples']}")
        logger.info("Test activity distribution:")
        for act, count in sorted(info["test_activity_counts"].items(), key=lambda x: -x[1]):
            marker = " <-- TARGET" if act == target_activity else ""
            logger.info(f"  {act}: {count}{marker}")

        if info["n_train_samples"] == 0:
            logger.warning(f"No training samples for '{target_activity}', skipping...")
            continue

        # Build model
        logger.info("Building model...")
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

        if i == 0:
            n_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model parameters: {n_params:,}")
            if use_wandb:
                wandb.log({"model/n_params": n_params})

        # Optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        scheduler = ExponentialLR(optimizer, gamma=cfg.training.lr_decay)

        # Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=str(device),
            checkpoint_dir=(
                str(model_dir / "checkpoints") if cfg.output.save_checkpoints else None
            ),
            use_wandb=False,  # We'll log aggregated metrics
        )

        # Train
        if not cfg.checkpoint.eval_only:
            logger.info("Training...")
            history = trainer.train(
                epochs=cfg.training.epochs,
                grad_clip=cfg.training.grad_clip,
                log_interval=cfg.training.log_interval,
            )

            with open(model_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

        # Evaluate
        if cfg.evaluation.enabled:
            logger.info("Evaluating...")
            results = evaluate_by_activity(
                model=model,
                test_loader=test_loader,
                device=device,
                target_activity=target_activity,
                use_wandb=False,
            )

            print_results(results)

            # Save results
            results_dict = {
                "config": OmegaConf.to_container(cfg, resolve=True),
                "info": info,
                "results": results,
            }

            with open(model_dir / "results.json", "w") as f:
                json.dump(results_dict, f, indent=2, default=str)

            all_results.append(results)

    # Aggregate results and generate separability matrix
    if len(all_results) > 1:
        logger.info("AGGREGATING RESULTS")

        aggregated = aggregate_results(model_dirs)
        print_aggregated_results(aggregated)

        # Save aggregated data
        with open(output_dir / "aggregated.json", "w") as f:
            json.dump(
                {
                    "activities": aggregated["activities"],
                    "score_matrix": aggregated["score_matrix"].tolist(),
                    "auc_matrix": aggregated["auc_matrix"].tolist(),
                    "separation_vector": aggregated["separation_vector"].tolist(),
                    "n_models": aggregated["n_models"],
                },
                f,
                indent=2,
            )

        # Generate plots
        plot_separability_matrix(
            aggregated,
            output_dir / "separability_matrix.png",
            title=f"Separability Matrix ({cfg.data.label_type})",
        )
        plot_separation_summary(
            aggregated,
            output_dir / "separation_summary.png",
            title=f"Per-Activity Metrics ({cfg.data.label_type})",
        )

        # Log aggregated results to wandb as a summary table
        if use_wandb:
            activities = aggregated["activities"]
            separation = aggregated["separation_vector"]
            auc_diag = np.diag(aggregated["auc_matrix"])

            # Create results table (per-activity metrics)
            results_table = wandb.Table(
                columns=["activity", "auc", "separation", "target_mean_log_prob", "other_mean_log_prob"]
            )
            for i, act in enumerate(activities):
                auc_val = auc_diag[i] if not np.isnan(auc_diag[i]) else None
                sep_val = separation[i] if not np.isnan(separation[i]) else None
                # Get detailed metrics from all_results
                target_mean = None
                other_mean = None
                for r in all_results:
                    if r["target_activity"] == act:
                        target_mean = r.get("target_mean_log_prob")
                        other_mean = r.get("other_mean_log_prob")
                        break
                results_table.add_data(act, auc_val, sep_val, target_mean, other_mean)

            wandb.log({"results_table": results_table})

            # Create separability matrix table (NxN scores)
            score_matrix = aggregated["score_matrix"]
            matrix_table = wandb.Table(
                columns=["model_trained_on"] + activities
            )
            for i, act in enumerate(activities):
                row_data = [act] + [
                    score_matrix[i, j] if not np.isnan(score_matrix[i, j]) else None
                    for j in range(len(activities))
                ]
                matrix_table.add_data(*row_data)

            wandb.log({"separability_matrix": matrix_table})

            # Log plots as images
            wandb.log({
                "separability_matrix_plot": wandb.Image(str(output_dir / "separability_matrix.png")),
                "separation_summary_plot": wandb.Image(str(output_dir / "separation_summary.png")),
            })

            # Summary metrics
            valid_auc = auc_diag[~np.isnan(auc_diag)]
            valid_sep = separation[~np.isnan(separation)]

            mean_auc = float(np.mean(valid_auc)) if len(valid_auc) > 0 else 0.5
            mean_sep = float(np.mean(valid_sep)) if len(valid_sep) > 0 else 0
            std_auc = float(np.std(valid_auc)) if len(valid_auc) > 0 else 0
            std_sep = float(np.std(valid_sep)) if len(valid_sep) > 0 else 0

            wandb.summary["mean_auc"] = mean_auc
            wandb.summary["std_auc"] = std_auc
            wandb.summary["mean_separation"] = mean_sep
            wandb.summary["std_separation"] = std_sep
            wandb.summary["n_models"] = len(valid_auc)

    # Finish wandb
    if use_wandb:
        wandb.finish()

    logger.success(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
