"""Generate synthetic motion-program training data."""

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import h5py
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from exact.bm import BehaviourModel
from exact.programs import generate_program, parse_program
from exact.data.utils import generate_trajectory

# Body parts available for program generation
BODY_PARTS = [
    "pelvis",
    "torso",
    "spine",
    "chest",
    "neck",
    "head",
    "lhip",
    "lknee",
    "lankle",
    "ltoe",
    "rhip",
    "rknee",
    "rankle",
    "rtoe",
    "lthorax",
    "lshoulder",
    "lelbow",
    "lwrist",
    "lhand",
    "rthorax",
    "rshoulder",
    "relbow",
    "rwrist",
    "rhand",
]


def generate_single_sample(program: str) -> tuple[str, np.ndarray]:
    """Generate a single motion sample from a program.
    
    Args:
        program: Motion program string
        
    Returns:
        Tuple of (program, motion_array)
    """
    try:
        # Create model per-process to avoid sharing state
        model = BehaviourModel(device="cpu")
        reward = parse_program(program)
        obs, _ = generate_trajectory(model, reward, device="cpu")
        return program, obs.cpu().numpy().astype(np.float32), None
    except Exception as e:
        # Return error as string to avoid pickling issues
        return program, None, traceback.format_exc()


def generate_batch(programs: list[str], worker_id: int = 0) -> list[tuple[str, np.ndarray]]:
    """Generate a batch of samples (for chunked processing).
    
    Args:
        programs: List of program strings
        worker_id: Worker identifier for logging
        
    Returns:
        List of (program, motion_array) tuples
    """
    model = BehaviourModel(device="cpu")
    results = []
    for program in programs:
        reward = parse_program(program)
        obs, _ = generate_trajectory(model, reward, device="cpu")
        results.append((program, obs.cpu().numpy().astype(np.float32)))
    return results


def init_worker():
    """Initialize worker process - suppress warnings and set up env."""
    import warnings
    warnings.filterwarnings("ignore")
    # Limit OpenMP threads per worker to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic motion-program data"
    )
    parser.add_argument(
        "--name", type=str, default="train", help="Dataset name (train/eval)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of samples"
    )
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: num CPUs)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Number of samples per worker task (larger = less overhead)",
    )
    # Program generation parameters
    parser.add_argument(
        "--min-preds",
        type=int,
        default=1,
        help="Minimum number of predicates per segment",
    )
    parser.add_argument(
        "--max-preds",
        type=int,
        default=5,
        help="Maximum number of predicates per segment",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=0.0,
        help="Minimum value multiplier for predicates",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=2.0,
        help="Maximum value multiplier for predicates",
    )
    parser.add_argument(
        "--value-step",
        type=float,
        default=0.1,
        help="Step size for value discretization",
    )
    parser.add_argument(
        "--num-intervals",
        type=int,
        default=10,
        help="Number of time intervals/segments in program",
    )
    parser.add_argument(
        "--min-interval-time",
        type=int,
        default=1,
        help="Minimum timesteps per interval",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=1024,
        help="Maximum total timesteps",
    )
    parser.add_argument(
        "--allowed-parts",
        type=str,
        default=None,
        help="Comma-separated list of allowed body parts (default: all)",
    )
    parser.add_argument(
        "--allowed-axes",
        type=str,
        default=None,
        help="Comma-separated list of allowed axes: x,y,z (default: all)",
    )
    args = parser.parse_args()

    num_workers = args.num_workers or max(1, cpu_count() - 1)
    logger.info(
        f"Generating {args.name} data with {args.num_samples} samples using {num_workers} workers"
    )
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Parse allowed parts and axes
    allowed_parts = BODY_PARTS
    if args.allowed_parts:
        allowed_parts = [p.strip() for p in args.allowed_parts.split(",")]
        invalid_parts = set(allowed_parts) - set(BODY_PARTS)
        if invalid_parts:
            logger.error(f"Invalid body parts: {invalid_parts}")
            logger.error(f"Valid parts: {BODY_PARTS}")
            sys.exit(1)
    
    allowed_axes = None
    if args.allowed_axes:
        allowed_axes = [a.strip() for a in args.allowed_axes.split(",")]
        invalid_axes = set(allowed_axes) - {"x", "y", "z"}
        if invalid_axes:
            logger.error(f"Invalid axes: {invalid_axes}. Must be x, y, or z")
            sys.exit(1)

    # Program generation parameters
    program_kwargs = dict(
        min_preds=args.min_preds,
        max_preds=args.max_preds,
        min_value=args.min_value,
        max_value=args.max_value,
        value_step=args.value_step,
        allowed_parts=allowed_parts,
        allowed_axes=allowed_axes,
        max_timesteps=args.max_timesteps,
        num_intervals=args.num_intervals,
        min_interval_time=args.min_interval_time,
    )
    
    logger.info(f"Program generation params: {program_kwargs}")

    logger.info("Generating programs...")
    programs = [generate_program(**program_kwargs) for _ in range(args.num_samples)]

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    hdf5_path = output_path / f"{args.name}.h5"

    logger.info(f"Generating trajectories in parallel and saving to {hdf5_path}")

    # Process in parallel using ProcessPoolExecutor (more robust with pickling)
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(generate_single_sample, prog) for prog in programs]
        for future in tqdm(futures, desc="Generating"):
            program, motion, error = future.result()
            if error:
                logger.error(f"Error processing program: {error}")
                continue
            results.append((program, motion))

    # Write results to HDF5 (sequential write is fast)
    logger.info("Writing to HDF5...")
    with h5py.File(str(hdf5_path), "w") as f:
        for idx, (program, motion) in enumerate(results):
            grp = f.create_group(f"motion_{idx}")
            grp.create_dataset("motion", data=motion)
            grp.attrs["program"] = program

    logger.info(f"Done! Saved {args.num_samples} samples to {hdf5_path}")


if __name__ == "__main__":
    main()
