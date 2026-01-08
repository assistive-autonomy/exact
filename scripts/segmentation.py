import gc
from pathlib import Path

import hydra
import torch
from dlc2action.project import Project
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def cleanup_memory():
    """Force garbage collection and clear CUDA cache to free memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


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


# Model-specific parameter mappings
# Some models use different parameter names for similar concepts
MODEL_PARAM_SUPPORT = {
    # Models that support num_f_maps
    "ms_tcn3": {"num_f_maps": "model/num_f_maps"},
    "c2f_tcn": {"num_f_maps": "model/num_f_maps"},
    "c2f_transformer": {"num_f_maps": "model/num_f_maps"},
    # EDTCN uses mid_channels instead of num_f_maps
    "edtcn": {"num_f_maps": "model/mid_channels"},
}


def build_search_space(cfg: DictConfig, model_name: str = None) -> dict:
    """Build search space dictionary for hyperparameter search.
    
    Converts the YAML config format to DLC2Action's expected format:
    {'param_name': ('type', arg1, arg2)}
    
    Args:
        cfg: Configuration object
        model_name: Model name to adapt search space for (handles model-specific params)
    """
    if cfg.hyperparameter_search_space is None:
        return None
    
    search_space = {}
    for param_name, param_cfg in cfg.hyperparameter_search_space.items():
        # Handle model-specific parameter name mapping
        actual_param_name = param_name
        if model_name and param_name == "model/num_f_maps":
            # Map num_f_maps to model-specific parameter name
            if model_name in MODEL_PARAM_SUPPORT:
                actual_param_name = MODEL_PARAM_SUPPORT[model_name].get("num_f_maps", param_name)
        
        param_type = param_cfg.type
        if param_type == "categorical":
            search_space[actual_param_name] = ("categorical", list(param_cfg.choices))
        elif param_type in ["float", "float_log", "int", "int_log"]:
            search_space[actual_param_name] = (param_type, param_cfg.low, param_cfg.high)
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

    # Initialize project (will load existing project if it exists)
    project = Project(
        project_cfg.project_name,
        projects_path=project_cfg.projects_path,
        data_type=project_cfg.data_type,
        annotation_type=project_cfg.annotation_type,
        data_path=project_cfg.data_path,
        annotation_path=project_cfg.annotation_path,
    )

    # Check for existing searches and episodes (DataFrames - names are in index)
    searches_df = project.list_searches()
    episodes_df = project.list_episodes()
    existing_searches = set(searches_df.index.tolist()) if len(searches_df) > 0 else set()
    existing_episodes = set(episodes_df.index.tolist()) if len(episodes_df) > 0 else set()
    logger.info(f"Found {len(existing_searches)} existing searches: {existing_searches}")
    logger.info(f"Found {len(existing_episodes)} existing episodes: {existing_episodes}")

    logger.info("Updating project parameters...")
    project.update_parameters(params)

    logger.info("Starting hyperparameter search, training, and evaluation...")

    # =========================================================================
    # PHASE 1: Hyperparameter Search
    # Use only training data from split file, further split into train/val
    # =========================================================================
    logger.info(f"Running hyperparameter search for {len(models)} models...")
    logger.info("  Using train portion split into train/val for hyperparameter tuning")
    
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
        search_name = f"{model}_search"
        
        # Skip if search already exists
        if search_name in existing_searches:
            logger.info(f"  Skipping {model} - search '{search_name}' already exists")
            continue
        
        # Build model-specific search space
        search_space = build_search_space(cfg, model_name=model)
        if search_space:
            logger.info(f"  Custom search space for {model} with {len(search_space)} parameters:")
            for param_name, param_spec in search_space.items():
                logger.info(f"    - {param_name}: {param_spec}")
            
        logger.info(f"  Searching for {model}...")
        hp_search_params["general"]["model_name"] = model
        
        if search_space:
            # Use custom search space
            project.run_hyperparameter_search(
                search_name,
                search_space=search_space,
                metric=cfg.hyperparameter_search_metric,
                n_trials=cfg.hyperparameter_search_trials,
                parameters_update=hp_search_params,
                force=False,  # Don't overwrite existing searches
            )
        else:
            # Use default search space for model
            project.run_default_hyperparameter_search(
                search_name,
                model_name=model,
                num_epochs=cfg.hyperparameter_search_epochs,
                n_trials=cfg.hyperparameter_search_trials,
                metric=cfg.hyperparameter_search_metric,
            )
        
        # Clean up memory after each model's hyperparameter search
        logger.info(f"  Cleaning up memory after {model} search...")
        cleanup_memory()

    # =========================================================================
    # PHASE 2: Final Training
    # Train on full training data (no validation split) with best hyperparameters
    # =========================================================================
    logger.info(f"Training final models for {len(models)} architectures...")
    logger.info("  Using full train portion for final training")
    
    for model in models:
        episode_name = f"{model}_best"
        search_name = f"{model}_search"
        
        # Skip if episode already exists
        if episode_name in existing_episodes:
            logger.info(f"  Skipping {model} training - episode '{episode_name}' already exists")
            continue
        
        # Check if search exists (required for training)
        # Re-query searches in case new ones were added during this run
        current_searches = set(project.list_searches().index.tolist())
        if search_name not in existing_searches and search_name not in current_searches:
            logger.warning(f"  Skipping {model} training - search '{search_name}' not found")
            continue
            
        logger.info(f"  Training {model} with best hyperparameters...")
        project.run_episode(
            episode_name,
            load_search=search_name,
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
            force=False,  # Don't overwrite existing episodes
        )
        
        # Clean up memory after training
        cleanup_memory()

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
