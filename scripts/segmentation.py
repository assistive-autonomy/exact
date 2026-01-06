import pickle
from pathlib import Path

import hydra
from dlc2action.project import Project
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def get_cached_stats_path(cfg: DictConfig) -> Path:
    """Generate cache path for normalization stats based on config."""
    data_path = Path(cfg.project.data_path).name
    annotation_path = Path(cfg.project.annotation_path).name
    # Note: cfg.features.keys shadows dict method, use bracket notation
    features_str = "_".join(sorted(cfg.features["keys"]))
    
    cache_dir = Path("cache/norm_stats")
    cache_file = cache_dir / f"{data_path}_{annotation_path}_{features_str}_stats.pkl"
    return cache_file


def load_cached_stats(cfg: DictConfig) -> dict | None:
    """Load pre-computed normalization stats if available."""
    cache_path = get_cached_stats_path(cfg)
    
    if cache_path.exists():
        logger.info(f"Loading cached normalization stats from {cache_path}")
        with open(cache_path, "rb") as f:
            stats = pickle.load(f)
        logger.success(f"Loaded stats for {len(stats)} feature keys")
        return stats
    else:
        logger.info(f"No cached stats found at {cache_path}")
        return None


def build_params_dict(cfg: DictConfig) -> dict:
    """Build parameters dictionary for DLC2Action Project."""
    params = {
        "general": OmegaConf.to_container(cfg.general),
        "data": OmegaConf.to_container(cfg.data),
        "training": OmegaConf.to_container(cfg.training),
        "losses": OmegaConf.to_container(cfg.losses),
        "metrics": OmegaConf.to_container(cfg.metrics),
        "model": OmegaConf.to_container(cfg.model),
        "features": OmegaConf.to_container(cfg.features),
    }
    return params


@hydra.main(version_base=None, config_path="../configs", config_name="segmentation")
def main(cfg: DictConfig):
    """Run activity segmentation with DLC2Action framework."""
    logger.info("Activity Segmentation Configuration")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Extract project configuration
    project_cfg = cfg.project
    models = cfg.models
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build parameters dict for dlc2action
    params = build_params_dict(cfg)

    # Load cached normalization stats if available (HUGE speedup in slow I/O environments)
    cached_stats = load_cached_stats(cfg)
    if cached_stats is not None:
        params["training"]["stats"] = cached_stats
        logger.info("Using pre-computed normalization stats (skipping slow computation)")

    logger.info(
        f"Initializing project '{project_cfg.project_name}' with data type "
        f"'{project_cfg.data_type}' and annotation type '{project_cfg.annotation_type}'"
    )

    # Remove existing project if it exists
    Project.remove_project(
        project_cfg.project_name, projects_path=project_cfg.projects_path
    )

    # Initialize project
    project = Project(
        project_cfg.project_name,
        projects_path=project_cfg.projects_path,
        data_type=project_cfg.data_type,
        annotation_type=project_cfg.annotation_type,
        data_path=project_cfg.data_path,
        annotation_path=project_cfg.annotation_path,
    )

    logger.info("Updating project parameters...")
    project.update_parameters(params)

    logger.info("Starting hyperparameter search, training, and evaluation...")

    # Hyperparameter search
    logger.info(f"Running hyperparameter search for {len(models)} models...")
    for model in models:
        logger.info(f"  Searching for {model}...")
        project.run_default_hyperparameter_search(
            f"{model}_search",
            model_name=model,
            num_epochs=cfg.hyperparameter_search_epochs,
            n_trials=cfg.hyperparameter_search_trials,
            metric=cfg.hyperparameter_search_metric,
        )

    # Train best models
    logger.info(f"Training best models for {len(models)} architectures...")
    for model in models:
        logger.info(f"  Training best {model}...")
        project.run_episode(
            f"{model}_best",
            load_search=f"{model}_search",
            parameters_update={"general": {"model_name": model}},
            n_seeds=cfg.num_seeds,
            force=True,
        )

    # Plot training curves
    logger.info("Plotting training curves...")
    training_curves_path = output_dir / cfg.training_curves_path
    project.plot_episodes(
        [f"{model}_best" for model in models],
        metrics=["f1"],
        save_path=str(training_curves_path),
        title="Best model training curves",
    )
    logger.info(f"Training curves saved to {training_curves_path}")

    # Evaluate additional metrics
    logger.info("Evaluating additional metrics...")
    for model in models:
        logger.info(f"  Evaluating {model}...")
        project.evaluate(
            [f"{model}_best"],
            parameters_update={
                "general": {"metric_functions": ["segmental_f1", "pr-auc", "f1"]},
                "metrics": {"f1": {"average": "none"}},
            },
        )

    # Get and save results table
    logger.info("Generating results table...")
    results_df = project.get_results_table([f"{model}_best" for model in models])

    results_table_path = output_dir / cfg.results_table_path
    results_df.to_csv(str(results_table_path), index=False)
    logger.info(f"Results table saved to {results_table_path}")

    logger.success("Activity segmentation complete!")
    logger.info(f"Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
