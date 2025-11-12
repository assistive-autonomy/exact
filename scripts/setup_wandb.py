#!/usr/bin/env python3
"""
Setup script to help configure WandB for the project.

Usage:
    python scripts/setup_wandb.py
"""

import os
import sys

def main():
    """Guide user through WandB setup."""
    print("=" * 60)
    print("WandB Setup for EXACT SFT Training")
    print("=" * 60)
    print()
    
    # Check if wandb is installed
    try:
        import wandb
        print("✓ WandB is installed")
    except ImportError:
        print("✗ WandB is not installed")
        print("\nPlease install dependencies:")
        print("  uv sync")
        print("  # or")
        print("  pip install wandb")
        sys.exit(1)
    
    print()
    
    # Check if user is logged in
    try:
        api = wandb.Api()
        user = api.viewer
        print(f"✓ Logged in as: {user.username}")
        print(f"  Default entity: {user.entity or user.username}")
    except Exception:
        print("✗ Not logged in to WandB")
        print("\nTo login:")
        print("  wandb login")
        print("\nOr set the WANDB_API_KEY environment variable")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("Configuration Options")
    print("=" * 60)
    print()
    print("You can configure WandB in several ways:")
    print()
    print("1. Override in command line:")
    print("   python scripts/train_sft.py \\")
    print("       wandb.project=my-project \\")
    print("       wandb.entity=my-team")
    print()
    print("2. Edit configs/config.yaml:")
    print("   wandb:")
    print("     project: my-project")
    print("     entity: my-team")
    print()
    print("3. Disable WandB:")
    print("   python scripts/train_sft.py wandb.mode=disabled")
    print()
    print("4. Run offline (sync later):")
    print("   python scripts/train_sft.py wandb.mode=offline")
    print("   wandb sync outputs/YYYY-MM-DD/HH-MM-SS/wandb")
    print()
    print("=" * 60)
    print("Next Steps")
    print("=" * 60)
    print()
    print("1. Run a test training:")
    print("   python scripts/train_sft.py training=fast data=small")
    print()
    print("2. Check your WandB dashboard:")
    print(f"   https://wandb.ai/{user.username}/exact-sft")
    print()
    print("3. Start a full training run:")
    print("   python scripts/train_sft.py")
    print()
    print("✓ Setup complete!")
    

if __name__ == "__main__":
    main()
