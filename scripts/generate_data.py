"""Generate synthetic motion-program training data."""

import argparse
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
    args = parser.parse_args()

    logger.info(f"Generating {args.name} data with {args.num_samples} samples")
    torch.manual_seed(args.seed)

    # Program generation parameters (fixed, opinionated defaults)
    program_kwargs = dict(
        min_preds=1,
        max_preds=5,
        min_value=0.0,
        max_value=2.0,
        value_step=0.1,
        allowed_parts=BODY_PARTS,
        max_timesteps=600,
        num_intervals=30,
        min_interval_time=10,
    )

    logger.info("Generating programs...")
    programs = [generate_program(**program_kwargs) for _ in range(args.num_samples)]

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    hdf5_path = output_path / f"{args.name}.h5"

    logger.info(f"Generating trajectories and saving to {hdf5_path}")
    model = BehaviourModel(device="cpu")

    with h5py.File(str(hdf5_path), "w") as f:
        for idx, program in tqdm(
            enumerate(programs), total=len(programs), desc="Generating"
        ):
            reward = parse_program(program)
            obs, _ = generate_trajectory(model, reward, device="cpu")

            grp = f.create_group(f"motion_{idx}")
            grp.create_dataset("motion", data=obs.cpu().numpy().astype(np.float32))
            grp.attrs["program"] = program

    logger.info(f"Done! Saved {args.num_samples} samples to {hdf5_path}")


if __name__ == "__main__":
    main()
