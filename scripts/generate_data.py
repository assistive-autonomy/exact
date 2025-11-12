"""
Utility script for generating and caching training data.

This script generates motion-program pairs and saves them to disk,
allowing you to reuse the same dataset across multiple training runs.
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import pickle

from exact.bm import BehaviourModel
from exact.programs import generate_programs, RewardBuilder


def generate_and_save_dataset(
    output_path: str,
    num_samples: int = 1000,
    num_steps: int = 100,
    min_units: int = 1,
    max_units: int = 3,
    device: str = "cuda",
):
    """Generate and save a dataset of motion-program pairs.
    
    Args:
        output_path: Path to save the dataset
        num_samples: Number of samples to generate
        num_steps: Number of timesteps per motion
        min_units: Minimum body parts per program
        max_units: Maximum body parts per program
        device: Device for generation
    """
    print(f"Generating {num_samples} motion-program pairs...")
    print(f"Device: {device}")
    print(f"Steps per motion: {num_steps}")
    
    # Initialize behavior model
    print("\nLoading behavior model...")
    behaviour_model = BehaviourModel(
        model_name="facebook/metamotivo-M-1",
        batch_size=256,
        max_episode_steps=max(1000, num_steps + 100),
        device=device,
    )
    
    # Generate programs
    print("\nGenerating random programs...")
    programs = generate_programs(
        num_programs=num_samples * 2,  # Generate extra in case some fail
        min_units=min_units,
        max_units=max_units,
        min_value=0.1,
        max_value=1.0,
    )
    
    # Generate motions
    print("\nGenerating motions from programs...")
    motions = []
    valid_programs = []
    failed = []
    
    with tqdm(total=num_samples) as pbar:
        for program in programs:
            if len(motions) >= num_samples:
                break
            
            try:
                # Parse program
                reward_fn = RewardBuilder.reward_from_name(program)
                
                # Generate motion
                poses, actions = behaviour_model.generate(
                    reward_fn,
                    steps=num_steps,
                    render=False,
                )
                
                # Combine into full motion tensor [N, 256]
                motion = torch.cat([poses, actions], dim=-1)
                
                # Validate shape
                assert motion.shape == (num_steps, 256), f"Unexpected shape: {motion.shape}"
                
                motions.append(motion)
                valid_programs.append(program)
                pbar.update(1)
                
            except Exception as e:
                failed.append((program, str(e)))
                continue
    
    print(f"\n✓ Successfully generated {len(motions)} samples")
    if failed:
        print(f"✗ Failed to generate {len(failed)} samples")
        print("\nFirst 5 failures:")
        for prog, err in failed[:5]:
            print(f"  {prog}: {err}")
    
    # Save dataset
    print(f"\nSaving dataset to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = {
        "motions": motions,
        "programs": valid_programs,
        "metadata": {
            "num_samples": len(motions),
            "num_steps": num_steps,
            "min_units": min_units,
            "max_units": max_units,
            "failed": len(failed),
        }
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
    
    print(f"✓ Dataset saved")
    print(f"\nDataset info:")
    print(f"  Samples: {len(motions)}")
    print(f"  Steps per motion: {num_steps}")
    print(f"  Motion shape: [N, 256]")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def load_dataset(path: str):
    """Load a saved dataset.
    
    Args:
        path: Path to the dataset file
        
    Returns:
        Dictionary with 'motions', 'programs', and 'metadata'
    """
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    
    print(f"Loaded dataset from {path}")
    print(f"  Samples: {dataset['metadata']['num_samples']}")
    print(f"  Steps: {dataset['metadata']['num_steps']}")
    
    return dataset


def split_dataset(dataset, train_ratio=0.8):
    """Split dataset into train and validation sets.
    
    Args:
        dataset: Dictionary with 'motions' and 'programs'
        train_ratio: Ratio of data to use for training
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    num_samples = len(dataset["motions"])
    num_train = int(num_samples * train_ratio)
    
    indices = torch.randperm(num_samples).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_dataset = {
        "motions": [dataset["motions"][i] for i in train_indices],
        "programs": [dataset["programs"][i] for i in train_indices],
    }
    
    val_dataset = {
        "motions": [dataset["motions"][i] for i in val_indices],
        "programs": [dataset["programs"][i] for i in val_indices],
    }
    
    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate training data for inverse BM")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/motion_programs.pkl",
        help="Output path for dataset"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=1000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--num-steps", "-s",
        type=int,
        default=100,
        help="Number of timesteps per motion"
    )
    parser.add_argument(
        "--min-units",
        type=int,
        default=1,
        help="Minimum body parts per program"
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=3,
        help="Maximum body parts per program"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for generation"
    )
    
    args = parser.parse_args()
    
    generate_and_save_dataset(
        output_path=args.output,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        min_units=args.min_units,
        max_units=args.max_units,
        device=args.device,
    )


if __name__ == "__main__":
    main()
