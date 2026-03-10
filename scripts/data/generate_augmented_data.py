#!/usr/bin/env python
"""Generate augmented motion data from Executable Activity Models (CPU-only).

Designed for CPU-only machines with many cores, following the same
multiprocessing pattern as generate_data.py.

Workflow:
  1. Load pre-built executable activity models from JSON
  2. Spawn one worker per activity (each with its own BehaviourModel on CPU)
  3. Each worker pre-computes z vectors, then rolls out trajectories
  4. Collect results and write ESK-format dataset

Usage:
    # Basic (auto-detect CPU count)
    uv run scripts/data/generate_augmented_data.py \
        --models ../esk/models_verbs.json \
        --num-samples 1000 \
        --output-dir ../esk/augmented_verbs

    # Control parallelism
    uv run scripts/data/generate_augmented_data.py \
        --models ../esk/models_verbs.json \
        --num-samples 5000 \
        --num-workers 16 \
        --relabel-workers 4 \
        --output-dir ../esk/augmented_verbs
"""

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


def observation_to_pose96(obs: np.ndarray) -> np.ndarray:
    """Convert HumEnv observation (358-dim) to ESK-compatible pose (96-dim).

    Extracts root-relative 3D joint positions (24 joints × 3 xyz = 72 dims),
    then appends a likelihood=1.0 column per joint → 24 × 4 = 96 dims.
    """
    from exact.data.env import extract_joint_positions

    is_batched = obs.ndim > 1
    if not is_batched:
        obs = obs[np.newaxis, :]

    joint_positions = extract_joint_positions(obs)  # (N, 72)
    positions_3d = joint_positions.reshape(-1, 24, 3)
    likelihood = np.ones((*positions_3d.shape[:-1], 1), dtype=np.float32)
    pose_with_lik = np.concatenate([positions_3d, likelihood], axis=-1)
    pose96 = pose_with_lik.reshape(-1, 96)

    if not is_batched:
        pose96 = pose96[0]
    return pose96.astype(np.float32)


# ---------------------------------------------------------------------------
# Worker logic
# ---------------------------------------------------------------------------

def _worker_generate_activity(
    activity_name: str,
    model_dict: dict,
    n_samples: int,
    trajectory_length: int,
    model_name: str,
    buffer_batch_size: int,
    relabel_workers: int,
    z_variants: int,
    seed: int,
) -> tuple[str, list[np.ndarray] | None, str | None]:
    """Worker: generate all samples for one activity on CPU.

    Each worker process loads its own BehaviourModel and HumEnv to avoid
    cross-process state sharing.  The z cache is computed once per activity,
    then reused across all trajectories.

    Returns:
        (activity_name, list_of_pose_arrays, error_string_or_None)
    """
    import warnings
    warnings.filterwarnings("ignore")

    # Limit per-worker threading to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    try:
        from exact.bm import BehaviourModel
        from exact.data import HumEnv
        from exact.models.executable import ExecutableActivityModel

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Reconstruct model from dict (avoids pickling closures)
        exec_model = ExecutableActivityModel.from_dict(model_dict)

        bm = BehaviourModel(
            model_name=model_name,
            batch_size=buffer_batch_size,
            device="cpu",
            relabel_workers=relabel_workers,
        )
        env = HumEnv()

        # Pre-compute z cache (one relabel call per unique normalised timestep)
        z_cache = bm.precompute_z_cache(
            env, exec_model, trajectory_length, n_variants=z_variants,
        )

        poses_list: list[np.ndarray] = []
        for i in range(n_samples):
            result = bm.generate_trajectories_batched(
                envs=[env],
                executable_model=exec_model,
                num_steps=trajectory_length,
                z_cache=z_cache,
                z_offset=i,
            )
            obs = result[0]["observations"]  # (T, obs_dim)
            poses = observation_to_pose96(obs)  # (T, 96)
            poses_list.append(poses)

        env.env.close()
        return activity_name, poses_list, None

    except Exception:
        return activity_name, None, traceback.format_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented data from executable activity models (CPU-only)",
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Path to models JSON (from build_models.py)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Total samples to generate (distributed evenly across activities)",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=100,
        help="Steps per trajectory (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for ESK-format augmented data",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="augmented_data",
        help="Video name for the generated data files (default: augmented_data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # ── Parallelism knobs ────────────────────────────────────────────
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Worker processes (default: min(num_activities, auto)). "
             "Each worker handles one activity and loads its own BehaviourModel.",
    )
    parser.add_argument(
        "--behaviour-model",
        type=str,
        default="facebook/metamotivo-M-1",
        help="HuggingFace model name for BehaviourModel (default: metamotivo-M-1)",
    )
    parser.add_argument(
        "--buffer-batch-size",
        type=int,
        default=256,
        help="Buffer sample size for z computation (default: 256)",
    )
    parser.add_argument(
        "--relabel-workers",
        type=int,
        default=4,
        help="MuJoCo relabel threads *per* worker process (default: 4). "
             "Total CPU usage ≈ num_workers × relabel_workers.",
    )
    parser.add_argument(
        "--z-variants",
        type=int,
        default=4,
        help="z-vector variants per timestep for trajectory diversity (default: 4)",
    )

    args = parser.parse_args()

    # ── Load models ──────────────────────────────────────────────────
    models_path = Path(args.models)
    if not models_path.exists():
        logger.error(f"Models file not found: {models_path}")
        sys.exit(1)

    logger.info(f"Loading models from {models_path}")
    with open(models_path, "r") as f:
        models_data = json.load(f)

    from exact.models import ActivityModelCollection

    collection = ActivityModelCollection.from_dict(models_data)
    activities = collection.activity_names
    num_activities = len(activities)

    logger.info(f"Loaded {num_activities} activity models")
    for name in activities:
        m = collection.models[name]
        logger.info(f"  {name}: {m.num_programs} programs")

    # ── Distribute samples across activities ─────────────────────────
    samples_per = args.num_samples // num_activities
    remainder = args.num_samples % num_activities
    activity_samples: list[tuple[str, int]] = []
    for i, name in enumerate(activities):
        n = samples_per + (1 if i < remainder else 0)
        activity_samples.append((name, n))

    # ── Worker pool sizing ───────────────────────────────────────────
    total_cpus = cpu_count() or 8
    # Each worker uses ~relabel_workers threads + 1 main thread
    max_by_cpu = max(1, total_cpus // (args.relabel_workers + 1))
    num_workers = args.num_workers or min(num_activities, max_by_cpu)
    num_workers = max(1, min(num_workers, num_activities))

    logger.info(
        f"Generating {args.num_samples} samples across {num_activities} activities"
    )
    logger.info(
        f"CPU cores: {total_cpus}, worker processes: {num_workers}, "
        f"relabel threads/worker: {args.relabel_workers}"
    )
    logger.info(
        f"Behaviour model: {args.behaviour_model}, "
        f"buffer_batch_size: {args.buffer_batch_size}, "
        f"z_variants: {args.z_variants}"
    )
    for name, n in activity_samples:
        logger.info(f"  {name}: {n} samples")

    # ── Pre-download model files to HF cache (avoids race in workers) ─
    logger.info("Pre-downloading behaviour model to local cache...")
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id=args.behaviour_model,
        filename="data/buffer_inference_500000.hdf5",
        repo_type="model",
    )
    logger.info("Model files cached; launching workers.")

    # ── Launch parallel workers ──────────────────────────────────────
    all_results: dict[str, list[np.ndarray]] = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for activity_name, n in activity_samples:
            model_dict = collection.models[activity_name].to_dict()
            fut = executor.submit(
                _worker_generate_activity,
                activity_name=activity_name,
                model_dict=model_dict,
                n_samples=n,
                trajectory_length=args.trajectory_length,
                model_name=args.behaviour_model,
                buffer_batch_size=args.buffer_batch_size,
                relabel_workers=args.relabel_workers,
                z_variants=args.z_variants,
                seed=args.seed + hash(activity_name) % 10_000,
            )
            futures[fut] = (activity_name, n)

        pbar = tqdm(total=len(futures), desc="Activities")
        for fut in as_completed(futures):
            act_name, n = futures[fut]
            name, poses_list, err = fut.result()
            if err:
                logger.error(f"Worker for '{act_name}' failed:\n{err}")
                sys.exit(1)
            logger.info(f"  ✓ {act_name}: {len(poses_list)} trajectories generated")
            all_results[name] = poses_list
            pbar.update(1)
        pbar.close()

    # ── Write ESK dataset ────────────────────────────────────────────
    from exact.data.utils import ESKDatasetWriter

    logger.info("Writing ESK dataset...")
    writer = ESKDatasetWriter(
        activity_names=activities,
        output_dir=args.output_dir,
        video_name=args.video_name,
    )

    for activity_name in activities:
        for poses in all_results[activity_name]:
            writer.add_trajectory(poses, activity_name)

    pose_path, label_path = writer.save()

    logger.info("Done!")
    logger.info(f"  Poses:  {pose_path}")
    logger.info(f"  Labels: {label_path}")
    logger.info(f"  Summary: {writer.summary()}")


if __name__ == "__main__":
    main()
