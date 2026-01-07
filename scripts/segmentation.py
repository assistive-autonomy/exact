from pathlib import Path

import hydra
from dlc2action.project import Project
from loguru import logger
from omegaconf import DictConfig, OmegaConf


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


def build_search_space(cfg: DictConfig) -> dict:
    """Build search space dictionary for hyperparameter search.
    
    Converts the YAML config format to DLC2Action's expected format:
    {'param_name': ('type', arg1, arg2)}
    """
    if cfg.hyperparameter_search_space is None:
        return None
    
    search_space = {}
    for param_name, param_cfg in cfg.hyperparameter_search_space.items():
        param_type = param_cfg.type
        if param_type == "categorical":
            search_space[param_name] = ("categorical", list(param_cfg.choices))
        elif param_type in ["float", "float_log", "int", "int_log"]:
            search_space[param_name] = (param_type, param_cfg.low, param_cfg.high)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    return search_space


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

    # Build custom search space if provided
    search_space = build_search_space(cfg)

    # =========================================================================
    # PHASE 1: Hyperparameter Search
    # Use only training data from split file, further split into train/val
    # =========================================================================
    logger.info(f"Running hyperparameter search for {len(models)} models...")
    logger.info("  Using train portion split into train/val for hyperparameter tuning")
    if search_space:
        logger.info(f"  Custom search space with {len(search_space)} parameters:")
        for param_name, param_spec in search_space.items():
            logger.info(f"    - {param_name}: {param_spec}")
    
    # Parameters for hyperparameter search phase:
    # - Override partition_method to allow train/val split (file-based doesn't support val_frac)
    # - Use time-based splitting within training data for validation
    hp_search_params = {
        "general": {"model_name": None},  # Will be set per model
        "training": {
            "num_epochs": cfg.hyperparameter_search_epochs,
            "partition_method": "time",  # Override to allow val_frac to work
            "val_frac": cfg.get("hyperparameter_val_frac", 0.2),  # Split train into train/val
            "test_frac": 0,  # Don't use test data during HP search
        },
    }
    
    for model in models:
        logger.info(f"  Searching for {model}...")
        hp_search_params["general"]["model_name"] = model
        
        if search_space:
            # Use custom search space
            project.run_hyperparameter_search(
                f"{model}_search",
                search_space=search_space,
                metric=cfg.hyperparameter_search_metric,
                n_trials=cfg.hyperparameter_search_trials,
                parameters_update=hp_search_params,
                force=True,
            )
        else:
            # Use default search space for model
            project.run_default_hyperparameter_search(
                f"{model}_search",
                model_name=model,
                num_epochs=cfg.hyperparameter_search_epochs,
                n_trials=cfg.hyperparameter_search_trials,
                metric=cfg.hyperparameter_search_metric,
            )

    # =========================================================================
    # PHASE 2: Final Training
    # Train on full training data (no validation split) with best hyperparameters
    # =========================================================================
    logger.info(f"Training final models for {len(models)} architectures...")
    logger.info("  Using full train portion for final training")
    
    for model in models:
        logger.info(f"  Training {model} with best hyperparameters...")
        project.run_episode(
            f"{model}_best",
            load_search=f"{model}_search",
            parameters_update={
                "general": {"model_name": model},
                "training": {
                    "num_epochs": cfg.training.num_epochs,  # Full training epochs
                    "partition_method": "file",  # Use original file-based split
                    "split_path": cfg.training.split_path,  # Restore split file path
                    "val_frac": 0,  # No validation split - use full train data
                    "test_frac": 0,  # Test is determined by split file
                },
            },
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

    # =========================================================================
    # PHASE 3: Final Evaluation on Test Set
    # Evaluate trained models on held-out test data from split file
    # =========================================================================
    logger.info("Evaluating models on held-out test set...")
    logger.info("  Using test portion from split file for final evaluation")
    
    for model in models:
        logger.info(f"  Evaluating {model} on test set...")
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
