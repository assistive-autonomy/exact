#!/usr/bin/env python
"""Full Pipeline: Parse ESK → Build Models → Evaluate (Segmentation & Assessment).

This script runs the complete ExAct pipeline:
1. Parse ESK dataset with trained parser → activity programs (JSON)
2. Build executable activity models from programs → models (JSON)
3. Run assessment using program edit distance → separability matrix
4. Optionally: Generate augmented data for segmentation

Usage:
    # Full pipeline with trained parser
    uv run scripts/run_pipeline.py \
        --parser-checkpoint results/parser/20260122_225017 \
        --esk-path /pvc/esk

    # Skip parsing if programs already exist
    uv run scripts/run_pipeline.py \
        --skip-parsing \
        --programs /pvc/esk/programs_train.json

    # Quick test with fewer programs
    uv run scripts/run_pipeline.py \
        --parser-checkpoint results/parser/20260122_225017 \
        --max-programs 20 \
        --max-test-programs 10
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run full ExAct pipeline")
    
    # Paths
    parser.add_argument(
        "--parser-checkpoint",
        type=str,
        default="results/parser/20260122_225017",
        help="Path to trained parser checkpoint",
    )
    parser.add_argument(
        "--esk-path",
        type=str,
        default="/pvc/esk",
        help="Path to ESK dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/pipeline/<timestamp>)",
    )
    
    # Existing data
    parser.add_argument(
        "--skip-parsing",
        action="store_true",
        help="Skip parsing step, use existing programs JSON",
    )
    parser.add_argument(
        "--programs",
        type=str,
        default=None,
        help="Existing programs JSON (required if --skip-parsing)",
    )
    parser.add_argument(
        "--skip-model-build",
        action="store_true",
        help="Skip model building, use existing models JSON",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Existing models JSON (required if --skip-model-build)",
    )
    
    # Pipeline options
    parser.add_argument(
        "--label-type",
        type=str,
        default="verbs",
        choices=["verbs", "nouns", "activity"],
        help="Label type for activities",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use (for quick experiments)",
    )
    parser.add_argument(
        "--max-programs",
        type=int,
        default=50,
        help="Maximum programs per activity for models",
    )
    parser.add_argument(
        "--max-train-programs",
        type=int,
        default=30,
        help="Maximum programs per activity for assessment training",
    )
    parser.add_argument(
        "--max-test-programs",
        type=int,
        default=20,
        help="Maximum test programs per activity for assessment",
    )
    
    # Steps to run
    parser.add_argument(
        "--run-segmentation",
        action="store_true",
        help="Also run segmentation evaluation",
    )
    parser.add_argument(
        "--run-augmentation",
        action="store_true",
        help="Generate augmented data for segmentation",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"Running: {description}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        logger.error(f"Failed: {description}")
        return False
    
    logger.success(f"Completed: {description}")
    return True


def main():
    args = parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"results/pipeline/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    esk_path = Path(args.esk_path)
    
    logger.info("=" * 70)
    logger.info("ExAct Pipeline: Parse → Build → Evaluate")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"ESK path: {esk_path}")
    logger.info(f"Parser checkpoint: {args.parser_checkpoint}")
    logger.info(f"Label type: {args.label_type}")
    
    # =========================================================================
    # Step 1: Parse ESK dataset
    # =========================================================================
    if args.skip_parsing:
        if args.programs:
            programs_path = Path(args.programs)
        else:
            programs_path = esk_path / "programs_train.json"
        logger.info(f"Skipping parsing, using: {programs_path}")
    else:
        programs_path = esk_path / f"programs_train_{args.label_type}.json"
        
        logger.info("-" * 70)
        logger.info("Step 1: Parsing ESK dataset")
        logger.info("-" * 70)
        
        cmd = [
            "uv", "run", "scripts/parse_esk.py",
            "--parser-checkpoint", args.parser_checkpoint,
            "--esk-path", str(esk_path),
            "--split", "train",
            "--label-type", args.label_type,
            "--output", str(programs_path),
            "--seed", str(args.seed),
        ]
        
        if args.train_fraction < 1.0:
            cmd.extend(["--train-fraction", str(args.train_fraction)])
        
        if not run_command(cmd, "Parse ESK → Programs"):
            return 1
    
    # Check programs file exists
    if not programs_path.exists():
        logger.error(f"Programs file not found: {programs_path}")
        return 1
    
    # =========================================================================
    # Step 2: Build executable models
    # =========================================================================
    if args.skip_model_build:
        if args.models:
            models_path = Path(args.models)
        else:
            models_path = esk_path / "models.json"
        logger.info(f"Skipping model building, using: {models_path}")
    else:
        models_path = esk_path / f"models_{args.label_type}.json"
        
        logger.info("-" * 70)
        logger.info("Step 2: Building executable activity models")
        logger.info("-" * 70)
        
        cmd = [
            "uv", "run", "scripts/build_models.py",
            "--programs", str(programs_path),
            "--output", str(models_path),
            "--max-programs", str(args.max_programs),
            "--validate",
            "--seed", str(args.seed),
        ]
        
        if not run_command(cmd, "Build executable models"):
            return 1
    
    # Check models file exists
    if not models_path.exists():
        logger.error(f"Models file not found: {models_path}")
        return 1
    
    # =========================================================================
    # Step 3: Run assessment (program edit distance)
    # =========================================================================
    logger.info("-" * 70)
    logger.info("Step 3: Running assessment (program edit distance)")
    logger.info("-" * 70)
    
    assessment_dir = output_dir / "assessment"
    
    cmd = [
        "uv", "run", "scripts/assessment_edit_dist.py",
        "--load-models", str(models_path),
        "--parser-checkpoint", args.parser_checkpoint,
        "--max-train-programs", str(args.max_train_programs),
        "--max-test-programs", str(args.max_test_programs),
        "--output-dir", str(assessment_dir),
        "--seed", str(args.seed),
    ]
    
    if not run_command(cmd, "Assessment (edit distance)"):
        logger.warning("Assessment failed, continuing...")
    
    # =========================================================================
    # Optional: Run augmentation
    # =========================================================================
    if args.run_augmentation:
        logger.info("-" * 70)
        logger.info("Step 4: Generating augmented data")
        logger.info("-" * 70)
        
        augmented_dir = esk_path / "augmented"
        
        cmd = [
            "uv", "run", "scripts/augment_data.py",
            "--load-models", str(models_path),
            "--output-dir", str(augmented_dir),
            "--num-samples", "1000",
            "--dry-run",  # Use dry-run for now (no BehaviourModel)
        ]
        
        run_command(cmd, "Generate augmented data")
    
    # =========================================================================
    # Optional: Run segmentation
    # =========================================================================
    if args.run_segmentation:
        logger.info("-" * 70)
        logger.info("Step 5: Running segmentation evaluation")
        logger.info("-" * 70)
        
        cmd = [
            "uv", "run", "scripts/segmentation.py",
        ]
        
        run_command(cmd, "Segmentation evaluation")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 70)
    logger.info("Pipeline Complete!")
    logger.info("=" * 70)
    logger.info(f"Programs: {programs_path}")
    logger.info(f"Models: {models_path}")
    logger.info(f"Assessment: {assessment_dir}")
    
    # Load and print assessment summary if available
    results_file = assessment_dir / "results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
        metrics = results.get("metrics", {})
        logger.info("")
        logger.info("Assessment Metrics:")
        logger.info(f"  Diagonal mean (same-activity): {metrics.get('diagonal_mean', 'N/A'):.2f}")
        logger.info(f"  Off-diagonal mean (cross-activity): {metrics.get('off_diagonal_mean', 'N/A'):.2f}")
        logger.info(f"  Separation (higher=better): {metrics.get('separation', 'N/A'):.2f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
