#!/usr/bin/env python
"""Data augmentation using Executable Activity Models.

This script generates augmented training data for activity segmentation
by using trained parsers to create executable activity models, then
generating new motion trajectories guided by these models.

Workflow:
1. Load training subset (e.g., 25% of data)
2. Parse activity segments into programs (mock for now, real parser later)
3. Aggregate programs by activity class into ExecutableActivityModels
4. Generate augmented samples using BehaviourModel with random init poses
5. Save augmented data in ESK format for DLC2Action pipeline

Usage:
    python scripts/parsing/augment_data.py \
        --load-models /pvc/esk/models.json \
        --num-samples 1000 \
        --output-dir /pvc/esk/augmented
"""

import argparse
import random
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

# Conditional imports for heavy dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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


def subsample_videos(videos: list[str], fraction: float, seed: int) -> list[str]:
    """Subsample videos to a given fraction."""
    if fraction >= 1.0:
        return videos
    
    n_keep = max(1, int(len(videos) * fraction))
    rng = random.Random(seed)
    return rng.sample(videos, n_keep)


def load_training_segments(
    data_path: str,
    label_path: str,
    video_names: list[str],
    data_suffix: str = "_pose3d_smpl.h5",
    label_suffix: str = "_labels.pickle",
) -> dict:
    """Load all activity segments from training videos.
    
    Args:
        data_path: Path to pose data directory
        label_path: Path to label directory
        video_names: List of video names to load
        data_suffix: Suffix for pose files
        label_suffix: Suffix for label files
        
    Returns:
        Dictionary mapping activity_name -> list of segment info dicts
    """
    from exact.data.utils import load_labels, extract_segment_poses
    from exact.data.esk import load_pose_file
    from exact.data.env import smpl_rotations_to_positions
    
    all_segments = {}
    activity_names = None
    
    for video_name in tqdm(video_names, desc="Loading training data"):
        pose_file = Path(data_path) / f"{video_name}{data_suffix}"
        label_file = Path(label_path) / f"{video_name}{label_suffix}"
        
        if not pose_file.exists() or not label_file.exists():
            logger.warning(f"Missing files for {video_name}, skipping")
            continue
        
        # Load SMPL axis-angle rotations (T, 24, 3) and convert to positions (T, 72)
        pose_data, _ = load_pose_file(pose_file)
        poses = smpl_rotations_to_positions(pose_data)  # (T, 72)
        labels = load_labels(str(label_file))
        
        if activity_names is None:
            activity_names = labels["activity_names"]
            for name in activity_names:
                all_segments[name] = []
        
        # Extract segments for each activity
        video_segments = labels["segments"][0]  # First (only) video
        for activity_idx, activity_name in enumerate(activity_names):
            segs = video_segments[activity_idx]
            for seg in segs:
                start, end, flag = seg
                if end > start:  # Valid segment
                    segment_poses = extract_segment_poses(poses, start, end)
                    all_segments[activity_name].append({
                        "video": video_name,
                        "start": start,
                        "end": end,
                        "duration": end - start,
                        "poses": segment_poses,
                    })
    
    return all_segments


def create_executable_models(
    segments_by_activity: dict,
    parser,
    eval_timesteps: int = 100,
    max_programs_per_activity: int = 50,
    program_budget: int | None = None,
    seed: int = 42,
) -> dict:
    """Create ExecutableActivityModels from parsed segments.
    
    Args:
        segments_by_activity: Dict mapping activity -> list of segment info
        parser: Parser to convert segments to programs (trained or mock)
        eval_timesteps: Common evaluation timesteps for all models
        max_programs_per_activity: Maximum programs to parse per activity
        program_budget: If set, select diverse subset using hierarchical clustering
        seed: Random seed for sampling
        
    Returns:
        Dictionary mapping activity_name -> ExecutableActivityModel
    """
    from exact.models import create_executable_model
    
    rng = random.Random(seed)
    models = {}
    
    for activity_name, segments in segments_by_activity.items():
        if not segments:
            logger.info(f"No segments for {activity_name}, skipping")
            continue
        
        # Sample segments if too many
        if len(segments) > max_programs_per_activity:
            segments = rng.sample(segments, max_programs_per_activity)
        
        # Parse segments into programs
        programs = []
        for seg in segments:
            poses = seg.get("poses")
            duration = seg["duration"]
            
            # Use parser (trained or mock)
            if poses is not None and hasattr(parser, 'parse') and not hasattr(parser, 'duration'):
                # Trained parser uses motion data
                program = parser.parse(poses)
            else:
                # Mock parser uses duration
                program = parser.parse(duration=duration)
            programs.append(program)
        
        # Apply program budget selection if specified
        if program_budget and len(programs) > program_budget:
            from exact.programs import select_diverse_programs
            logger.info(f"  Selecting {program_budget} diverse programs from {len(programs)} for {activity_name}...")
            result = select_diverse_programs(
                programs,
                budget=program_budget,
                method="hierarchical",
                show_progress=False,
            )
            programs = result.selected_programs
        
        # Create executable model
        model = create_executable_model(
            activity_name=activity_name,
            programs=programs,
            eval_timesteps=eval_timesteps,
            metadata={
                "num_source_segments": len(segments),
                "source_videos": list(set(s["video"] for s in segments)),
            },
        )
        models[activity_name] = model
        
        logger.info(f"Created model for {activity_name}: {len(programs)} programs")
    
    return models


def observation_to_pose96(obs: np.ndarray) -> np.ndarray:
    """Convert HumEnv observation (358-dim) to ESK-compatible pose format (96-dim).
    
    Extracts root-relative 3D joint positions from the HumEnv observation
    (24 joints × 3 xyz = 72 dims), then appends a likelihood column per
    joint to produce the 96-dim format expected by DLC2Action:
    24 joints × 4 (x, y, z, likelihood).
    
    Args:
        obs: HumEnv observation [358] or [N, 358]
        
    Returns:
        ESK-compatible pose [96] or [N, 96]
    """
    from exact.data.env import extract_joint_positions
    
    is_batched = obs.ndim > 1
    if not is_batched:
        obs = obs[np.newaxis, :]
    
    # Extract root-relative joint positions: (N, 72) = 24 joints × 3 xyz
    joint_positions = extract_joint_positions(obs)  # (N, 72)
    
    # Reshape to (N, 24, 3) for per-joint processing
    positions_3d = joint_positions.reshape(-1, 24, 3)
    
    # Add likelihood column (1.0 = confident) per joint
    likelihood = np.ones((*positions_3d.shape[:-1], 1), dtype=np.float32)
    
    # Concatenate: (x, y, z, likelihood) per joint → (N, 24, 4) → flatten to (N, 96)
    pose_with_lik = np.concatenate([positions_3d, likelihood], axis=-1)
    pose96 = pose_with_lik.reshape(-1, 96)
    
    if not is_batched:
        pose96 = pose96[0]
    
    return pose96.astype(np.float32)


def generate_augmented_data(
    executable_models: dict,
    num_samples: int,
    output_dir: str,
    behaviour_model: "BehaviourModel" = None,
    env: "HumEnv" = None,
    trajectory_length: int = 100,
    seed: int = 42,
    dry_run: bool = False,
    batch_envs: int = 1,
    z_variants: int = 1,
    video_name: str = "augmented_data",
) -> str:
    """Generate augmented data using executable activity models.

    When ``batch_envs > 1``, uses optimized batched generation:
    - Pre-computes z vectors per activity (shared across all trajectories)
    - Runs ``batch_envs`` environments simultaneously with batched GPU
      inference, drastically improving GPU utilisation.

    Args:
        executable_models: Dict mapping activity -> ExecutableActivityModel
        num_samples: Total number of samples to generate
        output_dir: Output directory for augmented data
        behaviour_model: BehaviourModel for trajectory generation (None for dry run)
        env: HumEnv environment (None for dry run)
        trajectory_length: Length of each generated trajectory
        seed: Random seed
        dry_run: If True, generate random placeholder data instead of real trajectories
        batch_envs: Number of environments to roll out simultaneously per GPU
        z_variants: Number of z vector variants per timestep for diversity

    Returns:
        Path to output directory with generated files
    """
    from exact.data import ESKDatasetWriter

    rng = random.Random(seed)

    # Get list of activities with models
    activities = list(executable_models.keys())
    num_activities = len(activities)

    if num_activities == 0:
        raise ValueError("No executable models provided")

    # Distribute samples equally across activities
    samples_per_activity = num_samples // num_activities
    remainder = num_samples % num_activities

    logger.info(f"Generating {num_samples} samples across {num_activities} activities")
    logger.info(f"  {samples_per_activity} per activity (+{remainder} extra)")
    if batch_envs > 1:
        logger.info(f"  Batched mode: {batch_envs} parallel envs, {z_variants} z variants")

    # Initialize dataset writer
    writer = ESKDatasetWriter(
        activity_names=activities,
        output_dir=output_dir,
        video_name=video_name,
    )

    # Generate trajectories for each activity
    for activity_idx, activity_name in enumerate(activities):
        model = executable_models[activity_name]

        # Add one extra sample to first 'remainder' activities
        n_samples = samples_per_activity + (1 if activity_idx < remainder else 0)

        logger.info(f"Generating {n_samples} samples for {activity_name}")

        if dry_run or behaviour_model is None or env is None:
            # ---- dry-run / placeholder path ----
            for _ in tqdm(range(n_samples), desc=activity_name):
                poses = np.random.randn(trajectory_length, 96).astype(np.float32) * 0.1
                writer.add_trajectory(poses, activity_name)
        else:
            # ---- optimised generation path ----
            from exact.data import HumEnv as _HumEnv

            effective_batch = min(batch_envs, n_samples)
            envs = [_HumEnv() for _ in range(effective_batch)]

            # Pre-compute z cache once for this activity (biggest speedup)
            logger.info(f"  Pre-computing z cache ({z_variants} variant(s), "
                        f"{trajectory_length} timesteps)...")
            z_cache = behaviour_model.precompute_z_cache(
                env, model, trajectory_length, z_variants,
            )
            logger.info(f"  Z cache ready ({len(z_cache)} unique timesteps)")

            z_offset = 0
            pbar = tqdm(total=n_samples, desc=activity_name)

            for batch_start in range(0, n_samples, effective_batch):
                batch_n = min(effective_batch, n_samples - batch_start)
                batch_results = behaviour_model.generate_trajectories_batched(
                    envs=envs[:batch_n],
                    executable_model=model,
                    num_steps=trajectory_length,
                    z_cache=z_cache,
                    z_offset=z_offset,
                )
                for r in batch_results:
                    poses = observation_to_pose96(r["observations"])
                    writer.add_trajectory(poses, activity_name)
                z_offset += batch_n
                pbar.update(batch_n)

            pbar.close()
            for e in envs:
                e.env.close()

    # Save to files
    pose_path, label_path = writer.save()

    logger.info(f"Saved augmented data:")
    logger.info(f"  Poses: {pose_path}")
    logger.info(f"  Labels: {label_path}")
    logger.info(f"  Summary: {writer.summary()}")

    return output_dir


# ---------------------------------------------------------------------------
# Multi-GPU parallel generation
# ---------------------------------------------------------------------------

def _gpu_worker(
    gpu_id: int,
    model_name: str,
    models_json: dict,
    activity_assignments: list[tuple[str, int]],
    trajectory_length: int,
    batch_envs: int,
    buffer_batch_size: int,
    relabel_workers: int,
    z_variants: int,
    seed: int,
    result_dir: str,
) -> None:
    """Worker process for one GPU.

    Spawned via ``torch.multiprocessing``.  Generates trajectories for its
    assigned activities and saves raw observations as ``.npy`` files in
    *result_dir* for the main process to collect.
    """
    import torch
    import numpy as np
    import random as _random
    from pathlib import Path
    from loguru import logger as _logger

    device = f"cuda:{gpu_id}"
    worker_seed = seed + gpu_id * 10_000
    _random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    _logger.info(f"[GPU {gpu_id}] Starting on {device}, "
                 f"{len(activity_assignments)} activities")

    from exact import BehaviourModel
    from exact.data import HumEnv
    from exact.models import ActivityModelCollection

    # Reconstruct models from JSON (avoids pickling closures)
    collection = ActivityModelCollection.from_dict(models_json)

    bm = BehaviourModel(
        model_name=model_name,
        batch_size=buffer_batch_size,
        device=device,
        relabel_workers=relabel_workers,
    )

    n_envs = batch_envs
    envs = [HumEnv() for _ in range(n_envs)]
    z_env = HumEnv()  # dedicated env for z pre-computation

    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    try:
        for activity_name, n_samples in activity_assignments:
            exec_model = collection.models[activity_name]

            _logger.info(f"[GPU {gpu_id}] {activity_name}: pre-computing z cache "
                         f"({z_variants} variants)...")
            z_cache = bm.precompute_z_cache(
                z_env, exec_model, trajectory_length, z_variants,
            )
            _logger.info(f"[GPU {gpu_id}] {activity_name}: z cache ready "
                         f"({len(z_cache)} unique timesteps)")

            all_obs: list[np.ndarray] = []
            z_offset = 0

            for batch_start in range(0, n_samples, n_envs):
                batch_n = min(n_envs, n_samples - batch_start)
                batch_results = bm.generate_trajectories_batched(
                    envs[:batch_n], exec_model, trajectory_length,
                    z_cache, z_offset,
                )
                for r in batch_results:
                    all_obs.append(r["observations"])
                z_offset += batch_n
                _logger.info(f"[GPU {gpu_id}] {activity_name}: "
                             f"{batch_start + batch_n}/{n_samples}")

            obs_array = np.stack(all_obs)  # (n_samples, T, obs_dim)
            np.save(result_path / f"{activity_name}.npy", obs_array)
            _logger.info(f"[GPU {gpu_id}] Saved {activity_name}: {obs_array.shape}")

    finally:
        for env in envs:
            env.env.close()
        z_env.env.close()
        _logger.info(f"[GPU {gpu_id}] Worker finished")


def generate_augmented_data_parallel(
    executable_models: dict,
    models_json: dict,
    num_samples: int,
    output_dir: str,
    model_name: str = "facebook/metamotivo-M-1",
    num_gpus: int | None = None,
    trajectory_length: int = 100,
    batch_envs: int = 32,
    buffer_batch_size: int = 1024,
    relabel_workers: int | None = None,
    z_variants: int = 4,
    seed: int = 42,
    video_name: str = "augmented_data",
) -> str:
    """Generate augmented data using multiple GPUs in parallel.

    Distributes activities across GPUs with load-balanced scheduling then:

    1.  Each GPU worker reconstructs executable models from *models_json*.
    2.  Pre-computes z vectors per activity (shared across all trajectories).
    3.  Generates trajectories in batches of *batch_envs*.
    4.  Saves raw observations to temporary ``.npy`` files.

    The main process collects the results and writes the final ESK dataset.

    Args:
        executable_models: Dict mapping activity -> ExecutableActivityModel
        models_json: JSON-serialisable dict (``ActivityModelCollection.to_dict()``)
        num_samples: Total samples to generate
        output_dir: Output directory for augmented data
        model_name: HuggingFace model for BehaviourModel
        num_gpus: Number of GPUs (auto-detected if *None*)
        trajectory_length: Steps per trajectory
        batch_envs: Parallel environments per GPU
        buffer_batch_size: Buffer sample size for z computation
        relabel_workers: CPU relabel threads per GPU (auto-scaled if *None*)
        z_variants: z vector variants per timestep for diversity
        seed: Random seed

    Returns:
        Path to output directory with generated files
    """
    import os
    import tempfile
    import shutil
    import torch.multiprocessing as mp
    from exact.data import ESKDatasetWriter

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    activities = list(executable_models.keys())
    num_activities = len(activities)

    if num_activities == 0:
        raise ValueError("No executable models provided")

    # Compute per-activity sample counts
    samples_per = num_samples // num_activities
    remainder = num_samples % num_activities
    samples_map: dict[str, int] = {}
    for i, name in enumerate(activities):
        samples_map[name] = samples_per + (1 if i < remainder else 0)

    # Auto-scale relabel workers to fill available CPU cores
    if relabel_workers is None:
        total_cpus = os.cpu_count() or 8
        relabel_workers = max(4, total_cpus // num_gpus)

    logger.info(
        f"Parallel generation: {num_gpus} GPUs, "
        f"{relabel_workers} relabel workers/GPU, "
        f"{batch_envs} envs/GPU, {z_variants} z variants, "
        f"buffer_batch_size={buffer_batch_size}"
    )

    # Load-balanced distribution across GPUs
    sorted_activities = sorted(samples_map.items(), key=lambda x: -x[1])
    gpu_assignments: dict[int, list[tuple[str, int]]] = {
        i: [] for i in range(num_gpus)
    }
    gpu_loads: dict[int, int] = {i: 0 for i in range(num_gpus)}

    for activity_name, n_samples in sorted_activities:
        min_gpu = min(gpu_loads, key=gpu_loads.get)
        gpu_assignments[min_gpu].append((activity_name, n_samples))
        gpu_loads[min_gpu] += n_samples

    for gpu_id, assignments in gpu_assignments.items():
        names = [a[0] for a in assignments]
        total = sum(a[1] for a in assignments)
        logger.info(f"  GPU {gpu_id}: {total} samples — {names}")

    # Temporary directory for inter-process results
    tmp_dir = tempfile.mkdtemp(prefix="augment_parallel_")

    try:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # already set

        processes: list[mp.Process] = []
        for gpu_id, assignments in gpu_assignments.items():
            if not assignments:
                continue
            p = mp.Process(
                target=_gpu_worker,
                args=(
                    gpu_id, model_name, models_json, assignments,
                    trajectory_length, batch_envs, buffer_batch_size,
                    relabel_workers, z_variants, seed,
                    os.path.join(tmp_dir, f"gpu_{gpu_id}"),
                ),
            )
            processes.append(p)
            p.start()
            logger.info(f"Launched GPU {gpu_id} worker (pid={p.pid})")

        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(
                    f"GPU worker (pid={p.pid}) exited with code {p.exitcode}"
                )

        logger.info("All GPU workers finished, collecting results...")

        # Collect results and build final dataset
        writer = ESKDatasetWriter(
            activity_names=activities,
            output_dir=output_dir,
            video_name=video_name,
        )

        for gpu_id, assignments in gpu_assignments.items():
            gpu_dir = Path(tmp_dir) / f"gpu_{gpu_id}"
            for activity_name, _ in assignments:
                obs_file = gpu_dir / f"{activity_name}.npy"
                if not obs_file.exists():
                    raise FileNotFoundError(
                        f"Missing results for '{activity_name}' from GPU {gpu_id}"
                    )
                obs_array = np.load(obs_file)  # (n_samples, T, obs_dim)
                for i in range(len(obs_array)):
                    poses = observation_to_pose96(obs_array[i])
                    writer.add_trajectory(poses, activity_name)

        pose_path, label_path = writer.save()
        logger.info(f"Saved augmented data:")
        logger.info(f"  Poses: {pose_path}")
        logger.info(f"  Labels: {label_path}")
        logger.info(f"  Summary: {writer.summary()}")

        return output_dir

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented data using Executable Activity Models"
    )
    
    # Model loading options (alternative to parsing from scratch)
    parser.add_argument(
        "--load-models",
        type=str,
        default=None,
        help="Load pre-built models from JSON (from build_models.py). "
             "If provided, skips parsing and uses these models directly.",
    )
    
    # Data paths (only needed if not using --load-models)
    parser.add_argument(
        "--data-path",
        type=str,
        default="esk/D2A_converted_pose_smpl",
        help="Path to pose data directory",
    )
    parser.add_argument(
        "--label-path",
        type=str,
        default="esk/D2A_converted_label_verbs",
        help="Path to label directory",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default="esk/trainvaltest_split.txt",
        help="Path to train/val/test split file",
    )
    
    # Augmentation parameters
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.25,
        help="Fraction of training data to use for parsing (default: 0.25)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Total number of augmented samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=100,
        help="Length of each generated trajectory (default: 100)",
    )
    parser.add_argument(
        "--eval-timesteps",
        type=int,
        default=100,
        help="Evaluation timesteps for executable models (default: 100)",
    )
    parser.add_argument(
        "--max-programs-per-activity",
        type=int,
        default=50,
        help="Maximum programs per activity model (default: 50)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/augmented",
        help="Output directory for augmented data (default: data/augmented)",
    )
    
    # Model options
    parser.add_argument(
        "--behaviour-model",
        type=str,
        default="facebook/metamotivo-M-1",
        help="HuggingFace model name for BehaviourModel",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for model inference (auto, cpu, cuda)",
    )
    
    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate random data instead of real trajectories (for testing)",
    )
    parser.add_argument(
        "--save-models",
        type=str,
        default=None,
        help="Path to save executable models JSON (optional)",
    )
    parser.add_argument(
        "--parser-checkpoint",
        type=str,
        default=None,
        help="Path to trained parser checkpoint (uses mock if not provided)",
    )
    parser.add_argument(
        "--program-budget",
        type=int,
        default=None,
        help="Select diverse subset of programs per activity using hierarchical clustering",
    )

    # --- Parallelism / performance options ---
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (auto-detected if omitted). "
             "Set to 1 to force single-GPU mode.",
    )
    parser.add_argument(
        "--batch-envs",
        type=int,
        default=32,
        help="Number of environments rolled out simultaneously per GPU (default: 32)",
    )
    parser.add_argument(
        "--buffer-batch-size",
        type=int,
        default=1024,
        help="Buffer sample size for z computation — larger = more accurate z "
             "(default: 1024)",
    )
    parser.add_argument(
        "--relabel-workers",
        type=int,
        default=None,
        help="CPU workers for MuJoCo relabeling per GPU (auto-scaled if omitted)",
    )
    parser.add_argument(
        "--z-variants",
        type=int,
        default=4,
        help="Number of z vector variants per timestep for diversity (default: 4)",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="augmented_data",
        help="Video name for the generated data files (default: augmented_data)",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger.info("=== Data Augmentation with Executable Activity Models ===")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(args.seed)

    # Detect number of GPUs
    num_gpus = args.num_gpus
    if num_gpus is None and TORCH_AVAILABLE and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    elif num_gpus is None:
        num_gpus = 0
    use_multi_gpu = (num_gpus > 1) and not args.dry_run

    logger.info(f"Detected {num_gpus} GPU(s), multi-GPU={use_multi_gpu}")

    # Keep a JSON-serialisable copy of the models (needed for multi-GPU)
    models_json = None

    # Option 1: Load pre-built models from JSON
    if args.load_models:
        logger.info(f"Loading pre-built models from: {args.load_models}")
        from exact.models import ActivityModelCollection
        import json
        
        with open(args.load_models, "r") as f:
            models_data = json.load(f)
        
        collection = ActivityModelCollection.from_dict(models_data)
        executable_models = collection.models
        models_json = models_data  # already JSON-serialisable
        logger.info(f"Loaded {len(executable_models)} activity models")
        for name, model in executable_models.items():
            logger.info(f"  {name}: {model.num_programs} programs")
    
    # Option 2: Parse from ESK data
    else:
        logger.info(f"Parsing from ESK data (train_fraction={args.train_fraction})")
        
        # Load split file and get training videos
        logger.info("Loading split file...")
        splits = parse_split_file(args.split_path)
        train_videos = splits["train"]
        logger.info(f"Found {len(train_videos)} training videos")
    
    # Subsample training videos
        # Subsample training videos
        train_subset = subsample_videos(train_videos, args.train_fraction, args.seed)
        logger.info(f"Using {len(train_subset)} videos ({args.train_fraction*100:.0f}%)")
        
        # Load training segments
        logger.info("Loading training segments...")
        segments = load_training_segments(
            data_path=args.data_path,
            label_path=args.label_path,
            video_names=train_subset,
        )
        
        # Log segment statistics
        total_segments = sum(len(segs) for segs in segments.values())
        logger.info(f"Loaded {total_segments} segments across {len(segments)} activities")
        for name, segs in sorted(segments.items(), key=lambda x: -len(x[1])):
            if segs:
                logger.info(f"  {name}: {len(segs)} segments")
        
        # Create parser (trained or mock)
        logger.info("Creating parser...")
        from exact.parser import load_parser
        parser_model = load_parser(checkpoint_path=args.parser_checkpoint)
        if args.parser_checkpoint:
            logger.info(f"Using trained parser from {args.parser_checkpoint}")
        else:
            logger.info("Using mock parser (no checkpoint provided)")
        
        logger.info("Creating executable activity models...")
        
        # Create executable models
        executable_models = create_executable_models(
            segments_by_activity=segments,
            parser=parser_model,
            eval_timesteps=args.eval_timesteps,
            max_programs_per_activity=args.max_programs_per_activity,
            program_budget=args.program_budget,
            seed=args.seed,
        )
        
        # Serialise models for multi-GPU / save
        from exact.models import ActivityModelCollection
        collection = ActivityModelCollection(eval_timesteps=args.eval_timesteps)
        for model in executable_models.values():
            collection.add_model(model)
        models_json = collection.to_dict()

        # Optionally save models
        if args.save_models:
            collection.save(args.save_models)
            logger.info(f"Saved models to {args.save_models}")

    # ------------------------------------------------------------------
    # Generate augmented data
    # ------------------------------------------------------------------
    logger.info("Generating augmented data...")

    if use_multi_gpu:
        # ----- Multi-GPU parallel path -----
        logger.info(f"Using multi-GPU parallel generation ({num_gpus} GPUs)")
        output_dir = generate_augmented_data_parallel(
            executable_models=executable_models,
            models_json=models_json,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            model_name=args.behaviour_model,
            num_gpus=num_gpus,
            trajectory_length=args.trajectory_length,
            batch_envs=args.batch_envs,
            buffer_batch_size=args.buffer_batch_size,
            relabel_workers=args.relabel_workers,
            z_variants=args.z_variants,
            seed=args.seed,
            video_name=args.video_name,
        )
    else:
        # ----- Single-GPU (or dry-run) path -----
        behaviour_model = None
        env = None

        if not args.dry_run:
            logger.info("Loading behaviour model and environment...")

            if args.device == "auto":
                device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
            else:
                device = args.device

            from exact import BehaviourModel, HumEnv

            relabel_workers = args.relabel_workers
            if relabel_workers is None:
                import os
                relabel_workers = max(8, (os.cpu_count() or 8) // 2)

            try:
                behaviour_model = BehaviourModel(
                    model_name=args.behaviour_model,
                    batch_size=args.buffer_batch_size,
                    device=device,
                    relabel_workers=relabel_workers,
                )
                env = HumEnv()
                logger.info(f"Loaded model on {device} "
                            f"(relabel_workers={relabel_workers}, "
                            f"buffer_batch_size={args.buffer_batch_size})")
            except Exception as e:
                logger.warning(f"Failed to load behaviour model: {e}")
                logger.warning("Falling back to dry run mode")
                args.dry_run = True

        output_dir = generate_augmented_data(
            executable_models=executable_models,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            behaviour_model=behaviour_model,
            env=env,
            trajectory_length=args.trajectory_length,
            seed=args.seed,
            dry_run=args.dry_run,
            batch_envs=args.batch_envs,
            z_variants=args.z_variants,
            video_name=args.video_name,
        )
    
    logger.success(f"Augmentation complete! Data saved to {output_dir}")
    
    # Print usage instructions
    print("\n" + "="*60)
    print("To use augmented data with segmentation:")
    print(f"  python scripts/tasks/segmentation.py \\")
    print(f"    project.data_path={output_dir} \\")
    print(f"    project.annotation_path={output_dir} \\")
    print(f"    training.split_path=<new_split_file>")
    print("="*60)


if __name__ == "__main__":
    main()