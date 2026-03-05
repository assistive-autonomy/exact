#!/usr/bin/env python
"""Build Executable Activity Models from parsed programs.

This script loads parsed programs (from parse_esk.py) and builds an
ActivityModelCollection that can be used for:
- Data augmentation (generate trajectories using BehaviourModel)
- Anomaly detection (compare test programs to activity models)

The collection is saved as a JSON file with the following structure:
    {
        "eval_timesteps": 100,
        "models": {
            "Cut": {
                "activity_name": "Cut",
                "eval_timesteps": 100,
                "original_programs": ["[0,50]rhand.x(0.3);...", ...],
                "num_programs": 25,
                ...
            },
            ...
        }
    }

Usage:
    # Build models from parsed programs
    uv run scripts/parsing/build_models.py --programs ../esk/programs_train.json
    
    # Build models with program budget (select diverse subset)
    uv run scripts/parsing/build_models.py --programs ../esk/programs_train.json --program-budget 50
    
    # Custom output path
    uv run scripts/parsing/build_models.py --programs ../esk/programs_train.json --output ../esk/models.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Build executable activity models")
    parser.add_argument(
        "--programs",
        type=str,
        required=True,
        help="Path to parsed programs JSON file (from parse_esk.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for models JSON (default: same dir as programs, named 'models.json')",
    )
    parser.add_argument(
        "--eval-timesteps",
        type=int,
        default=100,
        help="Common evaluation timesteps for all models (for temporal normalization)",
    )
    parser.add_argument(
        "--program-budget",
        type=int,
        default=None,
        help="If set, select N diverse programs per activity using the specified selection method",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default="tfidf",
        choices=["tfidf", "hierarchical", "greedy"],
        help="Selection method for --program-budget: 'tfidf' (fast, feature-based), "
             "'hierarchical' (edit distance clustering), 'greedy' (maximin). Default: tfidf",
    )
    parser.add_argument(
        "--max-programs",
        type=int,
        default=None,
        help="Maximum programs per activity (simple random sampling, faster than budget)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that all programs can be parsed before adding",
    )
    return parser.parse_args()


def select_diverse_programs(
    programs: list[str],
    budget: int,
    seed: int = 42,
) -> list[str]:
    """Select diverse subset of programs using TF-IDF based diversity selection.
    
    Args:
        programs: List of program strings
        budget: Number of programs to select
        seed: Random seed (unused for tfidf, kept for API compatibility)
        
    Returns:
        List of selected programs
    """
    from exact.programs import select_diverse_programs as _select
    
    if len(programs) <= budget:
        return programs
    
    result = _select(
        programs,
        budget=budget,
        method="tfidf",  # Use fast TF-IDF based diversity selection
        show_progress=True,
    )
    return result.selected_programs


def validate_program(program: str) -> bool:
    """Check if a program can be parsed into a Reward object."""
    try:
        from exact.programs import parse_program
        reward = parse_program(program)
        return reward is not None and len(reward.motions) > 0
    except Exception:
        return False


def main():
    args = parse_args()
    
    # Load programs
    programs_path = Path(args.programs)
    if not programs_path.exists():
        logger.error(f"Programs file not found: {programs_path}")
        return 1
    
    logger.info(f"Loading programs from: {programs_path}")
    with open(programs_path, "r") as f:
        programs_data = json.load(f)
    
    # Extract data
    metadata = programs_data.get("metadata", {})
    activity_names = programs_data.get("activity_names", [])
    programs_by_activity = programs_data.get("programs_by_activity", {})
    
    logger.info(f"Loaded {len(activity_names)} activities from {metadata.get('split', 'unknown')} split")
    
    # Build activity models
    from exact.models import ActivityModelCollection, ExecutableActivityModel
    
    collection = ActivityModelCollection(eval_timesteps=args.eval_timesteps)
    
    stats = {"total_programs": 0, "valid_programs": 0, "activities_with_programs": 0}
    
    for activity_name in activity_names:
        activity_programs = programs_by_activity.get(activity_name, [])
        
        if not activity_programs:
            logger.warning(f"No programs for activity: {activity_name}")
            continue
        
        # Extract just the program strings
        program_strings = [p["program"] for p in activity_programs]
        stats["total_programs"] += len(program_strings)
        
        # Validate if requested
        if args.validate:
            valid_programs = []
            for prog in program_strings:
                if validate_program(prog):
                    valid_programs.append(prog)
                else:
                    logger.debug(f"Skipping invalid program: {prog[:50]}...")
            program_strings = valid_programs
            stats["valid_programs"] += len(valid_programs)
        else:
            stats["valid_programs"] += len(program_strings)
        
        if not program_strings:
            logger.warning(f"No valid programs for activity: {activity_name}")
            continue
        
        # Apply program budget if specified
        if args.program_budget and len(program_strings) > args.program_budget:
            logger.info(f"  {activity_name}: selecting {args.program_budget} diverse from {len(program_strings)} using {args.selection_method}")
            from exact.programs import select_diverse_programs as full_select
            result = full_select(
                program_strings,
                budget=args.program_budget,
                method=args.selection_method,
                show_progress=False,
            )
            program_strings = result.selected_programs
        elif args.max_programs and len(program_strings) > args.max_programs:
            import random
            rng = random.Random(args.seed)
            program_strings = rng.sample(program_strings, args.max_programs)
            logger.info(f"  {activity_name}: randomly sampled {args.max_programs} programs")
        
        # Create and add model
        try:
            model = ExecutableActivityModel(
                activity_name=activity_name,
                eval_timesteps=args.eval_timesteps,
                metadata={
                    "source": str(programs_path),
                    "original_count": len(activity_programs),
                },
            )
            model.add_programs(program_strings)
            collection.add_model(model)
            stats["activities_with_programs"] += 1
            logger.info(f"  {activity_name}: {model.num_programs} programs")
        except Exception as e:
            logger.error(f"Failed to create model for {activity_name}: {e}")
            continue
    
    # Log summary
    logger.info(f"Built collection with {collection.num_activities} activities")
    logger.info(f"  Total programs: {stats['total_programs']}")
    logger.info(f"  Valid programs: {stats['valid_programs']}")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = programs_path.parent / "models.json"
    
    # Add build metadata
    collection_dict = collection.to_dict()
    collection_dict["build_metadata"] = {
        "source_programs": str(programs_path),
        "source_metadata": metadata,
        "program_budget": args.program_budget,
        "max_programs": args.max_programs,
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
    }
    
    # Save collection
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(collection_dict, f, indent=2)
    
    logger.info(f"Saved models to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
