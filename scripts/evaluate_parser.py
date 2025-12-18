"""Evaluate trained parser and generate predictions CSV."""
import argparse
import csv
import os

import h5py
import torch
from loguru import logger
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from exact.parser import MotionConditionedParser, TrajectoryEncoder


def load_model(checkpoint_dir: str, device: str = "cuda"):
    """Load trained model from checkpoint directory.

    Args:
        checkpoint_dir: Path to the training output directory
        device: Device to load model on

    Returns:
        model: MotionConditionedParser
        tokenizer: Tokenizer
        config: Loaded config dict
    """
    # Load the hydra config to get model parameters
    config_path = os.path.join(checkpoint_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Determine dtype
    use_bf16 = config.get("bf16", True) and torch.cuda.is_bf16_supported() and device == "cuda"
    model_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # Load tokenizer and base model
    logger.info(f"Loading base model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=model_dtype,
        device_map="auto",
    )
    model_hidden_size = base_model.config.hidden_size

    # Load LoRA adapter
    lora_path = os.path.join(checkpoint_dir, "lora_adapter")
    if os.path.exists(lora_path):
        logger.info(f"Loading LoRA adapter from: {lora_path}")
        base_model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        logger.warning(f"LoRA adapter not found at {lora_path}, using base model")

    # Load trajectory encoder
    encoder_path = os.path.join(checkpoint_dir, "trajectory_encoder.pt")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Trajectory encoder not found: {encoder_path}")

    logger.info(f"Loading trajectory encoder from: {encoder_path}")
    trajectory_encoder = TrajectoryEncoder(
        trajectory_dim=config["motion_dim"],
        hidden_dim=config["motion_hidden_dim"],
        output_dim=model_hidden_size,
        num_layers=config["motion_num_layers"],
        num_prefix_tokens=config["num_prefix_tokens"],
    ).to(device, dtype=model_dtype)
    trajectory_encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    # Create motion-conditioned parser
    model = MotionConditionedParser(
        model=base_model,
        trajectory_encoder=trajectory_encoder,
    )
    model.eval()

    return model, tokenizer, config, model_dtype


def load_data(h5_path: str):
    """Load motion and program data from HDF5 file.

    Args:
        h5_path: Path to HDF5 file

    Returns:
        List of (motion_tensor, program_str, key) tuples
    """
    data = []
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            motion = torch.tensor(f[key]["motion"][()], dtype=torch.float32)
            program = f[key].attrs["program"]
            data.append((motion, program, key))
    return data


def predict(
    model: MotionConditionedParser,
    tokenizer,
    motion: torch.Tensor,
    max_new_tokens: int = 256,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Generate program prediction from motion.

    Args:
        model: MotionConditionedParser
        tokenizer: Tokenizer
        motion: Motion tensor of shape (T, motion_dim)
        max_new_tokens: Maximum tokens to generate
        device: Device
        dtype: Model dtype

    Returns:
        Predicted program string
    """
    motion = motion.unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            motion=motion,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    predicted = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return predicted.strip()


def evaluate_dataset(
    model: MotionConditionedParser,
    tokenizer,
    h5_path: str,
    output_csv: str,
    max_new_tokens: int = 256,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Evaluate model on dataset and write results to CSV.

    Args:
        model: MotionConditionedParser
        tokenizer: Tokenizer
        h5_path: Path to HDF5 file
        output_csv: Path to output CSV file
        max_new_tokens: Maximum tokens to generate
        device: Device
        dtype: Model dtype
    """
    logger.info(f"Loading data from: {h5_path}")
    data = load_data(h5_path)
    logger.info(f"Loaded {len(data)} samples")

    results = []
    for motion, target_program, key in tqdm(data, desc="Predicting"):
        predicted_program = predict(
            model, tokenizer, motion, max_new_tokens, device, dtype
        )
        exact_match = predicted_program == target_program
        results.append({
            "key": key,
            "target": target_program,
            "predicted": predicted_program,
            "exact_match": exact_match,
        })

    # Write CSV
    logger.info(f"Writing results to: {output_csv}")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "target", "predicted", "exact_match"])
        writer.writeheader()
        writer.writerows(results)

    # Calculate and log metrics
    exact_matches = sum(1 for r in results if r["exact_match"])
    accuracy = exact_matches / len(results) if results else 0
    logger.info(f"Exact match accuracy: {exact_matches}/{len(results)} ({accuracy:.2%})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained parser")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to training output directory (e.g., outputs/2025-12-18/15-46-08)",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="train.h5",
        help="Path to training HDF5 file",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="eval.h5",
        help="Path to evaluation HDF5 file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSV files (defaults to checkpoint dir)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir or args.checkpoint
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    model, tokenizer, config, model_dtype = load_model(args.checkpoint, args.device)
    logger.info("Model loaded successfully")

    # Evaluate on training data
    if os.path.exists(args.train_data):
        train_csv = os.path.join(output_dir, "predictions_train.csv")
        logger.info(f"Evaluating on training data: {args.train_data}")
        evaluate_dataset(
            model, tokenizer, args.train_data, train_csv,
            args.max_new_tokens, args.device, model_dtype
        )
    else:
        logger.warning(f"Training data not found: {args.train_data}")

    # Evaluate on eval data
    if os.path.exists(args.eval_data):
        eval_csv = os.path.join(output_dir, "predictions_eval.csv")
        logger.info(f"Evaluating on evaluation data: {args.eval_data}")
        evaluate_dataset(
            model, tokenizer, args.eval_data, eval_csv,
            args.max_new_tokens, args.device, model_dtype
        )
    else:
        logger.warning(f"Evaluation data not found: {args.eval_data}")

    logger.success("Evaluation complete!")


if __name__ == "__main__":
    main()
