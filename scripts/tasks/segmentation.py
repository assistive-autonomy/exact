import gc
from pathlib import Path

import hydra
import pandas as pd
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


def parse_split_file(split_path: str) -> dict:
    """Parse a train/val/test split file.
    
    Returns:
        dict with keys 'train', 'validation', 'test' containing lists of video names
    """
    splits = {"train": [], "validation": [], "test": []}
    current_section = None
    
    with open(split_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Train videos:"):
                current_section = "train"
            elif line.startswith("Validation videos:"):
                current_section = "validation"
            elif line.startswith("Test videos:"):
                current_section = "test"
            elif current_section:
                splits[current_section].append(line)
    
    return splits


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
    if "augmentations" in cfg and cfg.augmentations is not None:
        params["augmentations"] = OmegaConf.to_container(cfg.augmentations)
    return params


# Model-specific parameter mappings
# Some models use different parameter names for similar concepts
# None means the parameter should be skipped for that model
MODEL_PARAM_SUPPORT = {
    # Models that support num_f_maps
    "ms_tcn3": {"num_f_maps": "model/num_f_maps"},
    "c2f_tcn": {"num_f_maps": "model/num_f_maps"},
    # c2f_transformer requires num_f_maps to be divisible by heads (default 4)
    # Skip to avoid divisibility issues during HP search
    "c2f_transformer": {"num_f_maps": None},
    # EDTCN uses mid_channels which is a list, not comparable to num_f_maps
    # Skip num_f_maps search for EDTCN
    "edtcn": {"num_f_maps": None},
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
                mapped_name = MODEL_PARAM_SUPPORT[model_name].get("num_f_maps", param_name)
                if mapped_name is None:
                    # Skip this parameter for this model
                    logger.info(f"    Skipping {param_name} for {model_name} (not supported)")
                    continue
                actual_param_name = mapped_name
        
        param_type = param_cfg.type
        if param_type == "categorical":
            search_space[actual_param_name] = ("categorical", list(param_cfg.choices))
        elif param_type in ["float", "float_log", "int", "int_log"]:
            search_space[actual_param_name] = (param_type, param_cfg.low, param_cfg.high)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    return search_space


@hydra.main(version_base=None, config_path="../../configs", config_name="segmentation")
def main(cfg: DictConfig):
    """Run activity segmentation with DLC2Action framework."""
    logger.info("Activity Segmentation Configuration")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Unwrap nested config groups introduced by Hydra's directory-based naming
    # e.g. segmentation/augmented/esk_activities → cfg.segmentation.augmented.{...}
    while "project" not in cfg:
        keys = list(cfg.keys())
        if len(keys) == 1:
            cfg = cfg[keys[0]]
        else:
            break

    # Extract project configuration
    project_cfg = cfg.project
    models = cfg.models
    num_seeds = cfg.num_seeds
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_name = project_cfg.project_name
    logger.info(f"Project name: {project_name}")

    evaluate_only = cfg.get("evaluate_only", False)
    if evaluate_only:
        logger.info("Evaluate-only mode enabled: skipping hyperparameter search and training")

    # Parse the base split file
    base_split_path = cfg.training.split_path
    base_splits = parse_split_file(base_split_path)
    logger.info(f"Split: {len(base_splits['train'])} train, "
                f"{len(base_splits['validation'])} val, {len(base_splits['test'])} test videos")

    # Build parameters dict for dlc2action
    params = build_params_dict(cfg)

    logger.info(
        f"Initializing project '{project_name}' with data type "
        f"'{project_cfg.data_type}' and annotation type '{project_cfg.annotation_type}'"
    )

    # Initialize project (will load existing project if it exists)
    project = Project(
        project_name,
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

    if not evaluate_only:
        # =====================================================================
        # PHASE 1: Hyperparameter Search
        # =====================================================================
        logger.info(f"Running hyperparameter search for {len(models)} models...")

        # Parameters for hyperparameter search phase
        hp_search_params = {
            "general": {"model_name": None},  # Will be set per model
            "training": {
                "num_epochs": cfg.hyperparameter_search_epochs,
                "partition_method": "file",
                "split_path": base_split_path,
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

            try:
                if search_space:
                    project.run_hyperparameter_search(
                        search_name,
                        search_space=search_space,
                        metric=cfg.hyperparameter_search_metric,
                        n_trials=cfg.hyperparameter_search_trials,
                        parameters_update=hp_search_params,
                        force=False,
                    )
                else:
                    project.run_default_hyperparameter_search(
                        search_name,
                        model_name=model,
                        num_epochs=cfg.hyperparameter_search_epochs,
                        n_trials=cfg.hyperparameter_search_trials,
                        metric=cfg.hyperparameter_search_metric,
                    )
            except FileExistsError:
                # Stale search directory from a previous incomplete run;
                # remove it and retry once.
                import shutil as _shutil
                stale = Path(project_cfg.projects_path) / project_name / "results" / "searches" / search_name
                logger.warning(f"  Stale search dir found ({stale}), removing and retrying...")
                _shutil.rmtree(stale, ignore_errors=True)
                if search_space:
                    project.run_hyperparameter_search(
                        search_name,
                        search_space=search_space,
                        metric=cfg.hyperparameter_search_metric,
                        n_trials=cfg.hyperparameter_search_trials,
                        parameters_update=hp_search_params,
                        force=False,
                    )
                else:
                    project.run_default_hyperparameter_search(
                        search_name,
                        model_name=model,
                        num_epochs=cfg.hyperparameter_search_epochs,
                        n_trials=cfg.hyperparameter_search_trials,
                        metric=cfg.hyperparameter_search_metric,
                    )
            except RuntimeError as e:
                if "zero total variance" in str(e):
                    logger.warning(
                        f"  Optuna param-importance plot failed for {model} "
                        f"(zero variance across trials). Search results are still saved."
                    )
                else:
                    raise

            # Clean up memory after each model's hyperparameter search
            logger.info(f"  Cleaning up memory after {model} search...")
            cleanup_memory()

        # =====================================================================
        # PHASE 2: Final Training
        # Train with best hyperparameters across multiple seeds
        # =====================================================================
        logger.info(f"Training final models for {len(models)} architectures, {num_seeds} seeds each...")

        for model in models:
            search_name = f"{model}_search"

            # Check if search exists (required for training)
            current_searches = set(project.list_searches().index.tolist())
            if search_name not in existing_searches and search_name not in current_searches:
                logger.warning(f"  Skipping {model} training - search '{search_name}' not found")
                continue

            for seed_idx in range(num_seeds):
                episode_name = f"{model}_best_seed{seed_idx}"

                # Skip if episode already exists
                if episode_name in existing_episodes:
                    logger.info(f"  Skipping {model} seed {seed_idx} - episode '{episode_name}' already exists")
                    continue

                logger.info(f"  Training {model} seed {seed_idx}/{num_seeds-1}...")

                project.run_episode(
                    episode_name,
                    load_search=search_name,
                    parameters_update={
                        "general": {"model_name": model},
                        "training": {
                            "num_epochs": cfg.training.num_epochs,
                            "partition_method": "file",
                            "split_path": base_split_path,
                        },
                    },
                    n_seeds=1,  # We handle seeds manually
                    force=False,
                )

                # Clean up memory after each seed
                cleanup_memory()

    # =====================================================================
    # PHASE 3: Final Evaluation on Test Set
    # Evaluate all trained models on held-out test data
    # =====================================================================
    logger.info("Evaluating models on held-out test set...")
    logger.info("  Using test portion from split file for final evaluation")

    # Get current episodes for evaluation
    current_episodes = set(project.list_episodes().index.tolist())

    # Collect evaluation results
    all_results = []
    for model in models:
        for seed_idx in range(num_seeds):
            episode_name = f"{model}_best_seed{seed_idx}"

            if episode_name not in current_episodes:
                logger.warning(f"  Skipping {episode_name} - not found")
                continue

            logger.info(f"  Evaluating {episode_name} on test set...")
            try:
                metrics = project.evaluate(
                    [episode_name],
                    mode="test",
                    skip_updating_meta=False,
                    parameters_update={
                        "general": {"metric_functions": ["segmental_f1", "pr-auc", "f1"]},
                        "metrics": {"f1": {"average": "macro"}},
                        "training": {
                            "partition_method": "file",
                            "split_path": base_split_path,
                        },
                    },
                )
            except IndexError as e:
                # This can happen if model checkpoint files are missing
                logger.warning(f"  Skipping {episode_name} - model checkpoint not found: {e}")
                continue

            result_row = {
                "model": model,
                "seed": seed_idx,
                "episode": episode_name,
            }
            result_row.update(metrics)
            all_results.append(result_row)
            logger.info(f"    {episode_name} test metrics: {metrics}")

    # Create results DataFrame
    logger.info("Generating results table...")
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Compute summary statistics per model
        logger.info("\n=== Results Summary ===")
        for model in models:
            model_results = results_df[results_df["model"] == model]
            if len(model_results) > 0:
                for metric in ["f1", "segmental_f1", "pr-auc"]:
                    if metric in model_results.columns:
                        mean_val = model_results[metric].mean()
                        std_val = model_results[metric].std()
                        logger.info(f"  {model} {metric}: {mean_val:.4f} ± {std_val:.4f}")
    else:
        logger.warning("No results collected!")
        results_df = pd.DataFrame()

    # Save results
    results_table_path = output_dir / cfg.results_table_path
    results_df.to_csv(str(results_table_path), index=False)
    logger.info(f"Results table saved to {results_table_path}")
    logger.info(f"\nFull results:\n{results_df.to_string()}")

    logger.success("Activity segmentation complete!")
    logger.info(f"Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
