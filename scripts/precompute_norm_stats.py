"""Pre-compute normalization statistics and cache them to disk.

This script computes normalization statistics once on a fast machine,
saves them to a pickle file, which can then be loaded in slower environments
(like Docker containers with network storage) to skip the expensive computation.

Usage:
    # Compute and save stats (run on fast machine):
    uv run scripts/precompute_norm_stats.py --save

    # Load and verify stats:
    uv run scripts/precompute_norm_stats.py --load
"""

import argparse
import pickle
from pathlib import Path

import hydra
import torch
from dlc2action.data.dataset import BehaviorDataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def get_cache_path(cfg: DictConfig) -> Path:
    """Generate cache path based on config hash."""
    # Create a unique identifier based on data config
    data_path = Path(cfg.project.data_path).name
    annotation_path = Path(cfg.project.annotation_path).name
    # Note: cfg.features.keys shadows dict method, use bracket notation
    features_str = "_".join(sorted(cfg.features["keys"]))
    
    cache_dir = Path("cache/norm_stats")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{data_path}_{annotation_path}_{features_str}_stats.pkl"
    return cache_file


def compute_and_save_stats(cfg: DictConfig) -> None:
    """Compute normalization statistics and save to cache."""
    logger.info("Computing normalization statistics...")
    
    # Build data parameters
    # Note: cfg.features.keys would shadow the dict method, use bracket notation
    feature_keys = list(cfg.features["keys"])
    data_params = {
        "data_path": cfg.project.data_path,
        "annotation_path": cfg.project.annotation_path,
        "data_suffix": cfg.data.data_suffix,
        "annotation_suffix": cfg.data.annotation_suffix,
        "canvas_shape": list(cfg.data.canvas_shape),
        "feature_extraction": cfg.data.get("feature_extraction"),
        "ignored_bodyparts": cfg.data.ignored_bodyparts,
        "likelihood_threshold": cfg.data.likelihood_threshold,
        "behaviors": cfg.data.behaviors,
        "filter_annotated": cfg.data.filter_annotated,
        "filter_background": cfg.data.filter_background,
        "visibility_min_score": cfg.data.visibility_min_score,
        "visibility_min_frac": cfg.data.visibility_min_frac,
        "len_segment": cfg.general.len_segment,
        "overlap": cfg.general.overlap,
        "feature_save_path": None,
        "ignored_clips": cfg.general.ignored_clips,
        "keys": feature_keys,
    }
    
    # Remove None values
    data_params = {k: v for k, v in data_params.items() if v is not None}
    
    logger.info(f"Creating dataset with params: {data_params}")
    
    # Create dataset
    dataset = BehaviorDataset(
        data_type=cfg.project.data_type,
        annotation_type=cfg.project.annotation_type,
        **data_params,
    )
    
    # Get skip keys from config if present
    skip_keys = cfg.training.get("skip_normalization_keys", [])
    
    logger.info(f"Computing stats (skip_keys={skip_keys})...")
    stats = dataset.get_normalization_stats(skip_keys=skip_keys)
    
    # Convert tensors to CPU for serialization
    stats_serializable = {}
    for key, value in stats.items():
        stats_serializable[key] = {
            "mean": value["mean"].cpu() if isinstance(value["mean"], torch.Tensor) else value["mean"],
            "std": value["std"].cpu() if isinstance(value["std"], torch.Tensor) else value["std"],
        }
    
    # Save to cache
    cache_path = get_cache_path(cfg)
    with open(cache_path, "wb") as f:
        pickle.dump(stats_serializable, f)
    
    logger.success(f"Saved normalization stats to {cache_path}")
    
    # Print summary
    logger.info("Stats summary:")
    for key, value in stats_serializable.items():
        mean_shape = value["mean"].shape if hasattr(value["mean"], "shape") else "scalar"
        std_shape = value["std"].shape if hasattr(value["std"], "shape") else "scalar"
        logger.info(f"  {key}: mean={mean_shape}, std={std_shape}")


def load_and_verify_stats(cfg: DictConfig) -> dict:
    """Load cached stats and verify they exist."""
    cache_path = get_cache_path(cfg)
    
    if not cache_path.exists():
        logger.error(f"Cache file not found: {cache_path}")
        logger.info("Run with --save first to compute stats")
        return None
    
    with open(cache_path, "rb") as f:
        stats = pickle.load(f)
    
    logger.success(f"Loaded normalization stats from {cache_path}")
    
    # Print summary
    logger.info("Stats summary:")
    for key, value in stats.items():
        mean_val = value["mean"].flatten()[:3].tolist()
        std_val = value["std"].flatten()[:3].tolist()
        logger.info(f"  {key}: mean[:3]={mean_val}, std[:3]={std_val}")
    
    return stats


@hydra.main(version_base=None, config_path="../configs", config_name="segmentation")
def main(cfg: DictConfig):
    """Main entry point."""
    # Check for mode in hydra overrides (use +save=true or +load=true)
    save_mode = cfg.get("save", False)
    load_mode = cfg.get("load", False)
    
    if save_mode:
        compute_and_save_stats(cfg)
    elif load_mode:
        load_and_verify_stats(cfg)
    else:
        logger.info("Usage:")
        logger.info("  uv run scripts/precompute_norm_stats.py +save=true   # Compute and save stats")
        logger.info("  uv run scripts/precompute_norm_stats.py +load=true   # Load and verify stats")
        logger.info("")
        logger.info("Cache path would be: {}", get_cache_path(cfg))


if __name__ == "__main__":
    main()
