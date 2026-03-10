#!/usr/bin/env python
"""Anomaly Detection — unified script for all methods.

Supports three anomaly-detection algorithms, selected via ``method`` in the
config file:

  nf           — Normalising-flow density estimator (STG-NF).
                 Trains one model per target activity; separability is measured
                 by each model's log-probability on every test activity.

  mean_sigmoid — Executable activity models + program edit distance.
                 Scores a test sequence by the *mean* sigmoid-transformed
                 distance from the test program to each activity's program set.

  min_sigmoid  — Same as above but uses the *minimum* distance rather than
                 the mean (picks the closest matching program in the model).

All three methods produce identical output artefacts:
  • results.json             — full numerical results
  • auc_matrix.png           — N × N heatmap (AUC[i,j] = model i vs activity j)
  • separation_summary.png   — per-activity AUROC bar-chart
  • wandb run                — tables + images

Usage:
    uv run scripts/tasks/anomaly_detection.py --config configs/anomaly_detection/nf_esk_verbs.yaml
    uv run scripts/tasks/anomaly_detection.py --config configs/anomaly_detection/exec_esk_verbs.yaml
    uv run scripts/tasks/anomaly_detection.py --config configs/anomaly_detection/exec_esk_verbs.yaml method=min_sigmoid
"""

import gc
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Shared helpers
# =============================================================================

def load_config(config_path: str, overrides: list[str] = None) -> DictConfig:
    base_cfg = OmegaConf.load(config_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(base_cfg, override_cfg)
    else:
        cfg = base_cfg
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def get_device(device_str: str):
    import torch
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# =============================================================================
# Shared plotting
# =============================================================================

def plot_auc_matrix(
    auc_matrix: np.ndarray,
    activity_names: list,
    output_path: str,
    title: str = "AUC Matrix",
):
    """N × N AUC heatmap.

    Convention: AUC[i, j] = how well model i separates its own activity i
    (positive class) from activity j (negative class).
    Diagonal = one-vs-rest AUROC for model i.
    """
    wrapped = [n.replace("_", "\n").replace(" ", "\n") for n in activity_names]
    sz = max(8, len(activity_names) * 1.2)
    fig, ax = plt.subplots(figsize=(sz, sz * 0.85))
    mask = np.isnan(auc_matrix)
    sns.heatmap(
        auc_matrix,
        annot=True, fmt=".2f",
        cmap="RdYlGn",
        square=True,
        linewidths=0.5, linecolor="white",
        vmin=0.0, vmax=1.0,
        mask=mask,
        cbar_kws={"label": "AUC", "shrink": 0.8},
        annot_kws={"size": 9},
        ax=ax,
        xticklabels=wrapped, yticklabels=wrapped,
    )
    ax.set_xlabel("Query Activity", fontsize=13, fontweight="bold")
    ax.set_ylabel("Target Activity (model)", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved AUC matrix → {output_path}")


def plot_separation_summary(
    activity_names: list,
    per_activity_auc: dict,
    output_path: str,
    title: str = "Per-Activity AUC (one-vs-rest)",
):
    """Bar chart of one-vs-rest AUC per activity."""
    aucs = [per_activity_auc.get(a, float("nan")) for a in activity_names]
    colors = ["#2ecc71" if (not np.isnan(v) and v >= 0.5) else "#e74c3c" for v in aucs]

    fig, ax = plt.subplots(figsize=(max(8, len(activity_names) * 1.0), 5))
    bars = ax.bar(range(len(activity_names)), aucs, color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(activity_names)))
    ax.set_xticklabels(activity_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="Random")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, val in zip(bars, aucs):
        if not np.isnan(val):
            ax.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width() / 2, val + 0.01),
                        ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved separation summary → {output_path}")


def _log_auc_to_wandb(wandb, activity_names, auc_matrix, per_activity_auc, mean_auc,
                      prefix: str = ""):
    """Log AUC matrix + per-activity scalars to wandb."""
    p = f"{prefix}/" if prefix else ""
    wandb.summary[f"{p}mean_auc"] = mean_auc
    for act, val in per_activity_auc.items():
        wandb.summary[f"{p}{act}/auc"] = val
    rows = []
    for i, row_act in enumerate(activity_names):
        row = [row_act] + [
            float(auc_matrix[i, j]) if not np.isnan(auc_matrix[i, j]) else None
            for j in range(len(activity_names))
        ]
        rows.append(row)
    wandb.log({f"{p}auc_matrix_table": wandb.Table(
        columns=["target"] + activity_names,
        data=rows,
    )})


# =============================================================================
# NF pipeline
# =============================================================================

def _nf_evaluate_by_activity(model, test_loader, device, target_activity: str) -> dict:
    """Forward-pass the NF model over the full test set; returns per-activity stats."""
    import torch
    model.eval()
    activity_scores = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval [{target_activity}]", leave=False):
            poses, labels, activities = batch
            poses = poses.to(device, non_blocking=True).float()
            _, nll = model(poses)
            log_probs = -nll.cpu().numpy()
            for i, act in enumerate(activities):
                activity_scores[act].append(log_probs[i])

    results = {"per_activity": {}, "target_activity": target_activity}
    all_target, all_other, other_means = [], [], []

    for act, scores in activity_scores.items():
        arr = np.array(scores)
        results["per_activity"][act] = {
            "mean_log_prob": float(np.mean(arr)),
            "std_log_prob":  float(np.std(arr)),
            "n_samples":     len(arr),
        }
        if act == target_activity:
            all_target.extend(arr)
        else:
            all_other.extend(arr)
            other_means.append(float(np.mean(arr)))

    if all_target and other_means:
        target_mean = float(np.mean(all_target))
        other_mean  = float(np.mean(other_means))  # class-balanced
        results["target_mean_log_prob"] = target_mean
        results["other_mean_log_prob"]  = other_mean
        results["separation"]           = target_mean - other_mean

        all_scores = np.array(all_target + all_other)
        all_labels = np.array([1] * len(all_target) + [0] * len(all_other))
        try:
            results["auc_target_vs_others"] = float(roc_auc_score(all_labels, all_scores))
        except ValueError:
            results["auc_target_vs_others"] = 0.5

        # Pairwise AUC: model-i vs each individual activity j
        target_arr  = np.array(all_target)
        auc_pairwise = {}
        for act, scores in activity_scores.items():
            if act == target_activity:
                continue
            other_arr = np.array(scores)
            combined  = np.concatenate([target_arr, other_arr])
            lbls      = np.array([1] * len(target_arr) + [0] * len(other_arr))
            try:
                auc_pairwise[act] = float(roc_auc_score(lbls, combined))
            except ValueError:
                auc_pairwise[act] = 0.5
        results["auc_pairwise"] = auc_pairwise

    return results


def _build_nf_auc_matrix(all_results: list, activity_names: list):
    n   = len(activity_names)
    idx = {a: i for i, a in enumerate(activity_names)}
    auc_matrix = np.full((n, n), np.nan)

    for r in all_results:
        target = r["target_activity"]
        if target not in idx:
            continue
        row = idx[target]
        if "auc_target_vs_others" in r:
            auc_matrix[row, row] = r["auc_target_vs_others"]
        for act, val in r.get("auc_pairwise", {}).items():
            if act in idx:
                auc_matrix[row, idx[act]] = val

    per_activity_auc = {
        activity_names[i]: float(auc_matrix[i, i])
        for i in range(n) if not np.isnan(auc_matrix[i, i])
    }
    valid    = [v for v in np.diag(auc_matrix) if not np.isnan(v)]
    mean_auc = float(np.mean(valid)) if valid else 0.5
    return auc_matrix, per_activity_auc, mean_auc


def run_nf_pipeline(cfg: DictConfig, output_dir: Path) -> dict:
    """Train one STG-NF model per target activity and evaluate cross-activity."""
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ExponentialLR
    from exact.anomaly import STG_NF, AnomalyTrainer as Trainer
    from exact.encoder.utils import Graph
    from exact.data.esk import get_esk_dataloaders

    device           = get_device(cfg.experiment.device)
    target_activities = list(cfg.data.target_activities)
    logger.info(f"NF pipeline — {len(target_activities)} models, device={device}")

    all_per_model_results = []

    for i, target in enumerate(target_activities):
        logger.info(f"[{i+1}/{len(target_activities)}] Training on '{target}'")
        model_dir = output_dir / target.replace(" ", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        train_loader, test_loader, info = get_esk_dataloaders(
            esk_dir      = cfg.data.esk_dir,
            label_type   = cfg.data.label_type,
            target_activity = target,
            test_activities = target_activities,
            seg_len      = cfg.data.seg_len,
            seg_stride   = cfg.data.seg_stride,
            batch_size   = cfg.training.batch_size,
            num_workers  = cfg.training.num_workers,
        )

        if info["n_train_samples"] == 0:
            logger.warning(f"No training samples for '{target}', skipping")
            continue

        graph = Graph(strategy=cfg.model.adj_strategy, max_hop=1)
        model = STG_NF(
            pose_shape      = (3, cfg.data.seg_len, 24),
            hidden_channels = cfg.model.hidden_channels,
            K               = cfg.model.K,
            L               = cfg.model.L,
            graph           = graph,
            learn_prior     = cfg.model.learn_prior,
            R               = cfg.model.R,
            temporal_kernel_size = cfg.model.temporal_kernel,
            permutation     = cfg.model.permutation,
            device          = str(device),
        )

        optimizer = AdamW(model.parameters(),
                          lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=cfg.training.lr_decay)

        trainer = Trainer(
            model        = model,
            train_loader = train_loader,
            test_loader  = test_loader,
            optimizer    = optimizer,
            scheduler    = scheduler,
            device       = str(device),
            checkpoint_dir = (
                str(model_dir / "checkpoints") if cfg.output.save_checkpoints else None
            ),
            use_wandb    = False,
        )

        if not cfg.checkpoint.eval_only:
            history = trainer.train(
                epochs       = cfg.training.epochs,
                grad_clip    = cfg.training.grad_clip,
                log_interval = cfg.training.log_interval,
            )
            with open(model_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

        eval_results = _nf_evaluate_by_activity(model, test_loader, device, target)
        with open(model_dir / "results.json", "w") as f:
            json.dump({"info": info, "results": eval_results}, f, indent=2, default=str)
        all_per_model_results.append(eval_results)

        # ── Explicit cleanup to prevent OOM between iterations ──
        del model, trainer, optimizer, scheduler, train_loader, test_loader
        gc.collect()
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    auc_matrix, per_activity_auc, mean_auc = _build_nf_auc_matrix(
        all_per_model_results, target_activities
    )
    logger.info(f"NF mean AUC (one-vs-rest): {mean_auc:.4f}")
    for act, val in per_activity_auc.items():
        logger.info(f"  {act}: {val:.4f}")

    return {
        "activity_names":      target_activities,
        "auc_matrix":          auc_matrix,
        "per_activity_auc":    per_activity_auc,
        "mean_auc":            mean_auc,
        "per_model_results":   all_per_model_results,
    }


# =============================================================================
# Exec pipeline  (mean_sigmoid / min_sigmoid)
# =============================================================================

def _load_programs(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {
        "activity_names":       data.get("activity_names", []),
        "programs_by_activity": data.get("programs_by_activity", {}),
        "metadata":             data.get("metadata", {}),
    }


def _filter_valid(programs: list, max_count: int = None) -> list:
    from exact.programs import parse_to_tree
    valid = []
    for p in programs:
        s = p.get("program", "") if isinstance(p, dict) else str(p)
        try:
            if parse_to_tree(s) is not None:
                valid.append(s)
        except Exception:
            continue
        if max_count and len(valid) >= max_count:
            break
    return valid


def _resolve_exec_paths(cfg: DictConfig):
    """Resolve models_path and test_programs_path from config or auto-detect."""
    # Infer a "data root" from esk_dir: .../benchmarks/<dataset> → .../
    esk_dir  = Path(cfg.data.get("esk_dir", "../exact_data/benchmarks/esk"))
    data_root = esk_dir.parent.parent  # exact_data/
    label    = cfg.data.label_type

    # Models path
    mp = cfg.data.get("models_path", None)
    if mp:
        models_path = Path(mp)
    else:
        name_map = {
            "verbs":    "models_verbs.json",
            "activity": "models_activity.json",
            "actions":  "models_humanact12.json",
        }
        models_path = data_root / "models" / name_map.get(label, "models.json")

    if not models_path.exists():
        models_path = None   # will fall back to train_programs_path

    # Test programs path
    tp = cfg.data.get("test_programs_path", None)
    if tp:
        test_path = Path(tp)
    else:
        name_map = {
            "verbs":    "programs_verbs_test.json",
            "activity": "programs_activity_test.json",
            "actions":  "programs_humanact12_test.json",
        }
        test_path = data_root / "programs" / "parsed" / name_map.get(label, "programs_test.json")

    return models_path, test_path


def _plot_edit_distance_matrix(matrix, activity_names, output_path, title):
    wrapped = [n.replace("_", "\n").replace(" ", "\n") for n in activity_names]
    sz  = max(8, len(activity_names) * 1.2)
    fig, ax = plt.subplots(figsize=(sz, sz * 0.85))
    sns.heatmap(
        matrix,
        annot=True, fmt=".1f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Min Edit Distance (lower = more similar)", "shrink": 0.8},
        annot_kws={"size": 10},
        ax=ax,
        xticklabels=wrapped, yticklabels=wrapped,
    )
    ax.set_xlabel("Model Activity", fontsize=13, fontweight="bold")
    ax.set_ylabel("Query Activity", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved edit distance matrix → {output_path}")


def run_exec_pipeline(cfg: DictConfig, method: str, output_dir: Path) -> dict:
    """Executable-model pipeline: edit distance → sigmoid scores → AUC matrix."""
    from exact.programs import ProgramDistanceMatrix, VALUE_TOLERANCE
    from exact.programs.selection import select_diverse_programs

    models_path, test_path = _resolve_exec_paths(cfg)

    if not test_path.exists():
        raise FileNotFoundError(f"Test programs not found: {test_path} — run 2_train_and_parse.sh first")

    train_budget = int(cfg.data.get("train_budget", 100))
    num_workers  = int(cfg.data.get("exec_num_workers", 1))

    # ── Load model programs ──────────────────────────────────────────────────
    if models_path is not None:
        logger.info(f"Loading executable models from {models_path}")
        with open(models_path) as f:
            models_data = json.load(f)
        all_activity_names = list(models_data["models"].keys())
        model_programs_by_activity = {
            act: info["original_programs"]
            for act, info in models_data["models"].items()
        }
        use_saved_models = True
    else:
        tpp = cfg.data.get("train_programs_path", None)
        if not tpp:
            raise ValueError(
                "No models_path found and no train_programs_path set — "
                "run scripts/2_train_and_parse.sh first."
            )
        logger.info(f"Loading train programs from {tpp}")
        train_data             = _load_programs(tpp)
        all_activity_names     = train_data["activity_names"]
        model_programs_by_activity = None
        use_saved_models       = False

    logger.info(f"Loading test programs from {test_path}")
    test_data = _load_programs(str(test_path))

    # ── Filter to requested activities ──────────────────────────────────────
    requested = list(cfg.data.get("target_activities", []))
    if requested:
        invalid = [a for a in requested if a not in all_activity_names]
        if invalid:
            raise ValueError(f"Unknown activities: {invalid}. Available: {all_activity_names}")
        activity_names = requested
    else:
        activity_names = all_activity_names

    logger.info(f"Activities ({len(activity_names)}): {activity_names}")

    # ── Selection strategy for sub-selecting from loaded models ────────────
    selection_method = str(cfg.data.get("selection_method", "greedy"))

    # ── Build distance matrix ────────────────────────────────────────────────
    dist_matrix  = ProgramDistanceMatrix(activity_names)
    train_counts = {}
    test_counts  = {}

    for activity in activity_names:
        if use_saved_models:
            raw  = model_programs_by_activity.get(activity, [])
            mprg = _filter_valid([{"program": p} if isinstance(p, str) else p for p in raw])
        else:
            raw  = train_data["programs_by_activity"].get(activity, [])
            random.shuffle(raw)
            mprg = _filter_valid(raw, max_count=train_budget)

        if not mprg:
            logger.warning(f"No valid model programs for {activity}")
            continue

        # Sub-select when we have more programs than train_budget
        if len(mprg) > train_budget:
            logger.info(
                f"  {activity}: sub-selecting {train_budget} from {len(mprg)} "
                f"using {selection_method}"
            )
            result = select_diverse_programs(
                mprg,
                budget=train_budget,
                method=selection_method,
                deduplicate=True,
                dedup_tolerance=0.0,
                show_progress=False,
            )
            mprg = result.selected_programs

        dist_matrix.set_model_programs(activity, mprg)
        train_counts[activity] = len(mprg)

        test_raw  = test_data["programs_by_activity"].get(activity, [])
        test_prgs = _filter_valid(test_raw)
        for prog in test_prgs:
            dist_matrix.add_test_program(activity, prog)
        test_counts[activity] = len(test_prgs)

        logger.info(f"  {activity}: {len(mprg)} model  |  {len(test_prgs)} test")

    logger.info(f"Computing edit distances ({num_workers} worker(s))…")
    raw_distance_matrix = dist_matrix.compute_matrix(verbose=True, num_workers=num_workers)
    distance_metrics    = dist_matrix.get_separability_metrics()
    logger.info(f"  diagonal mean (same):  {distance_metrics['diagonal_mean']:.2f}")
    logger.info(f"  off-diag mean (cross): {distance_metrics['off_diagonal_mean']:.2f}")
    logger.info(f"  separation:            {distance_metrics['separation']:.2f}")

    # ── Compute AUC for both sigmoid methods ─────────────────────────────────
    sigmoid_results = {}
    for sig_method in ("mean-sigmoid", "min-sigmoid"):
        logger.info(f"Computing {sig_method} AUC…")
        auc_mat, score_mat, auc_metrics = dist_matrix.compute_auc_matrix(
            method=sig_method, verbose=True
        )
        sigmoid_results[sig_method] = {
            "auc_matrix":   auc_mat,
            "score_matrix": score_mat,
            "metrics":      auc_metrics,
        }
        logger.info(f"  {sig_method} mean AUC: {auc_metrics['mean_auc']:.4f}")
        for act, val in auc_metrics["per_model_auc"].items():
            logger.info(f"    {act}: {val:.4f}")

    # Primary method drives the top-level reporting
    canonical = method.replace("_", "-")   # mean_sigmoid → mean-sigmoid
    primary   = sigmoid_results[canonical]

    return {
        "activity_names":      activity_names,
        "method":              method,
        "auc_matrix":          primary["auc_matrix"],
        "per_activity_auc":    primary["metrics"]["per_model_auc"],
        "mean_auc":            primary["metrics"]["mean_auc"],
        "raw_distance_matrix": raw_distance_matrix,
        "distance_metrics":    distance_metrics,
        "sigmoid_results":     sigmoid_results,
        "train_counts":        train_counts,
        "test_counts":         test_counts,
        "value_tolerance":     VALUE_TOLERANCE,
        "test_programs_path":  str(test_path),
        "models_path":         str(models_path) if models_path else None,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Anomaly Detection  (method: nf | mean_sigmoid | min_sigmoid)"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf dot-list overrides, e.g. method=min_sigmoid")
    args = parser.parse_args()

    cfg    = load_config(args.config, args.overrides)
    method = str(cfg.get("method", "nf")).lower()

    if method not in {"nf", "mean_sigmoid", "min_sigmoid"}:
        logger.error(f"Unknown method '{method}'. Choose: nf | mean_sigmoid | min_sigmoid")
        sys.exit(1)

    set_seed(cfg.experiment.seed)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_type = cfg.data.label_type
    exp_name   = cfg.experiment.get("name", None) or f"{method}_{label_type}_{timestamp}"
    output_dir = Path(cfg.output.dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    use_wandb = cfg.get("wandb_mode", "disabled") != "disabled"
    if use_wandb:
        import wandb as _wandb
        _wandb.init(
            project = cfg.get("wandb_project", "exact"),
            entity  = cfg.get("wandb_entity", None),
            name    = exp_name,
            config  = OmegaConf.to_container(cfg, resolve=True),
            mode    = cfg.get("wandb_mode", "online"),
        )

    logger.info("=" * 60)
    logger.info(f"Anomaly Detection  [method={method}]")
    logger.info(f"Label type : {label_type}")
    logger.info(f"Output     : {output_dir}")
    logger.info("=" * 60)

    # ── Run selected pipeline ────────────────────────────────────────────────
    if method == "nf":
        out = run_nf_pipeline(cfg, output_dir)
    else:
        out = run_exec_pipeline(cfg, method, output_dir)

    activity_names   = out["activity_names"]
    auc_matrix       = out["auc_matrix"]
    per_activity_auc = out["per_activity_auc"]
    mean_auc         = out["mean_auc"]

    logger.info(f"Mean AUC (one-vs-rest): {mean_auc:.4f}")

    # ── Shared plots ─────────────────────────────────────────────────────────
    auc_plot  = str(output_dir / "auc_matrix.png")
    sep_plot  = str(output_dir / "separation_summary.png")
    plot_auc_matrix(auc_matrix, activity_names, auc_plot,
                    title=f"AUC Matrix — {method} ({label_type})")
    plot_separation_summary(activity_names, per_activity_auc, sep_plot,
                            title=f"Per-Activity AUC — {method} ({label_type})")

    # ── Extra exec plots ─────────────────────────────────────────────────────
    extra_plots = {}
    if method in ("mean_sigmoid", "min_sigmoid"):
        sr = out["sigmoid_results"]

        dist_plot = str(output_dir / "edit_distance_matrix.png")
        _plot_edit_distance_matrix(
            out["raw_distance_matrix"], activity_names, dist_plot,
            title=f"Program Edit Distance ({label_type})",
        )
        extra_plots["edit_distance_matrix"] = dist_plot

        for sig_key in ("mean-sigmoid", "min-sigmoid"):
            sig_label = sig_key.replace("-", "_")
            p = str(output_dir / f"auc_matrix_{sig_label}.png")
            plot_auc_matrix(sr[sig_key]["auc_matrix"], activity_names, p,
                            title=f"AUC Matrix — {sig_key} ({label_type})")
            extra_plots[f"auc_matrix_{sig_label}"] = p

        # Side-by-side comparison bar chart
        ms_aucs  = sr["mean-sigmoid"]["metrics"]["per_model_auc"]
        min_aucs = sr["min-sigmoid"]["metrics"]["per_model_auc"]
        n, x, w = len(activity_names), np.arange(len(activity_names)), 0.35
        fig, ax = plt.subplots(figsize=(max(10, n * 1.0), 5))
        ax.bar(x - w / 2, [ms_aucs.get(a, 0)  for a in activity_names],
               w, label="mean-sigmoid", color="#3498db", alpha=0.8)
        ax.bar(x + w / 2, [min_aucs.get(a, 0) for a in activity_names],
               w, label="min-sigmoid",  color="#e67e22", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(activity_names, rotation=45, ha="right", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(f"AUC: mean-sigmoid vs min-sigmoid ({label_type})",
                     fontsize=14, fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=0.3, linestyle="--")
        comp_path = str(output_dir / "auc_comparison.png")
        plt.tight_layout()
        plt.savefig(comp_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        extra_plots["auc_comparison"] = comp_path

    # ── Save results JSON ─────────────────────────────────────────────────────
    payload: dict = {
        "method":          method,
        "label_type":      label_type,
        "activity_names":  activity_names,
        "mean_auc":        mean_auc,
        "per_activity_auc": per_activity_auc,
        "auc_matrix":      np.where(np.isnan(auc_matrix), None, auc_matrix).tolist(),
        "timestamp":       timestamp,
    }
    if method == "nf":
        payload["per_model_results"] = out.get("per_model_results", [])
    else:
        payload["raw_distance_matrix"] = out["raw_distance_matrix"].tolist()
        payload["distance_metrics"]    = out["distance_metrics"]
        payload["train_counts"]        = out["train_counts"]
        payload["test_counts"]         = out["test_counts"]
        for sig_key in ("mean-sigmoid", "min-sigmoid"):
            sl = sig_key.replace("-", "_")
            sr = out["sigmoid_results"][sig_key]
            payload[f"{sl}_auc_matrix"] = np.where(
                np.isnan(sr["auc_matrix"]), None, sr["auc_matrix"]
            ).tolist()
            payload[f"{sl}_metrics"] = sr["metrics"]

    with open(output_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Saved results.json → {output_dir}")

    # ── Wandb ─────────────────────────────────────────────────────────────────
    if use_wandb:
        import wandb as _wandb
        _wandb.summary["method"]       = method
        _wandb.summary["mean_auc"]     = mean_auc
        _wandb.summary["n_activities"] = len(activity_names)
        _log_auc_to_wandb(_wandb, activity_names, auc_matrix, per_activity_auc, mean_auc)

        if method in ("mean_sigmoid", "min_sigmoid"):
            sr = out["sigmoid_results"]
            for sig_key in ("mean-sigmoid", "min-sigmoid"):
                sl = sig_key.replace("-", "_")
                m  = sr[sig_key]["metrics"]
                _log_auc_to_wandb(_wandb, activity_names,
                                  sr[sig_key]["auc_matrix"],
                                  m["per_model_auc"], m["mean_auc"], prefix=sl)
            dm = out["distance_metrics"]
            _wandb.summary["distance/diagonal_mean"]    = dm["diagonal_mean"]
            _wandb.summary["distance/off_diagonal_mean"] = dm["off_diagonal_mean"]
            _wandb.summary["distance/separation"]       = dm["separation"]

        _wandb.log({
            "auc_matrix_plot":    _wandb.Image(auc_plot),
            "separation_summary": _wandb.Image(sep_plot),
            **{k: _wandb.Image(v) for k, v in extra_plots.items()},
        })
        _wandb.finish()

    logger.success(f"Done. Results → {output_dir}")


if __name__ == "__main__":
    main()
