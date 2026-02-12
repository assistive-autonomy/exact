"""Merge multiple HDF5 datasets into a single file."""

import argparse
from pathlib import Path
import h5py
import numpy as np
from loguru import logger
from tqdm import tqdm


def merge_hdf5_files(input_files: list[Path], output_path: Path, shuffle: bool = True):
    """Merge multiple HDF5 files into a single file.
    
    Args:
        input_files: List of input HDF5 file paths
        output_path: Output HDF5 file path
        shuffle: Whether to shuffle the samples after merging
    """
    logger.info(f"Merging {len(input_files)} files into {output_path}")
    
    # First pass: count total samples and collect all data
    all_samples = []
    
    for input_file in tqdm(input_files, desc="Loading files"):
        with h5py.File(str(input_file), "r") as f:
            # Get all motion groups
            motion_keys = [k for k in f.keys() if k.startswith("motion_")]
            logger.info(f"  {input_file.name}: {len(motion_keys)} samples")
            
            for key in motion_keys:
                grp = f[key]
                motion = grp["motion"][:]
                program = grp.attrs["program"]
                all_samples.append((program, motion))
    
    logger.info(f"Total samples: {len(all_samples)}")
    
    # Shuffle if requested
    if shuffle:
        logger.info("Shuffling samples...")
        np.random.shuffle(all_samples)
    
    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {output_path}...")
    
    with h5py.File(str(output_path), "w") as f:
        for idx, (program, motion) in enumerate(tqdm(all_samples, desc="Writing")):
            grp = f.create_group(f"motion_{idx}")
            grp.create_dataset("motion", data=motion)
            grp.attrs["program"] = program
    
    logger.info(f"Done! Merged {len(all_samples)} samples into {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple HDF5 datasets into a single file"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing HDF5 files to merge",
    )
    parser.add_argument(
        "--input-files",
        type=str,
        nargs="+",
        default=None,
        help="Specific HDF5 files to merge (alternative to --input-dir)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples after merging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Collect input files
    if args.input_files:
        input_files = [Path(f) for f in args.input_files]
    else:
        input_dir = Path(args.input_dir)
        input_files = sorted(input_dir.glob("*.h5"))
    
    if not input_files:
        logger.error("No HDF5 files found to merge")
        return
    
    logger.info(f"Found {len(input_files)} files to merge:")
    for f in input_files:
        logger.info(f"  - {f}")
    
    merge_hdf5_files(input_files, Path(args.output), shuffle=args.shuffle)


if __name__ == "__main__":
    main()
