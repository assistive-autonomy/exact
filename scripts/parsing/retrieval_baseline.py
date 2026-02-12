#!/usr/bin/env python3
"""
Retrieval Baseline for Motion-to-Program Generation

This diagnostic script tests whether the ST-GCN encoder learns meaningful motion
representations by using a simple nearest-neighbor retrieval approach:

1. Encode all training motions with the ST-GCN encoder
2. For each test motion, find the nearest training motion by embedding distance
3. Return the training motion's program as the prediction
4. Compute accuracy/edit distance metrics

If this works well → encoder is learning, decoder is the problem
If this fails → encoder needs improvement
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from loguru import logger
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exact.encoder import STGCNEncoder
from exact.programs.edit_distance import program_edit_distance, parse_to_tree
from lark.exceptions import LarkError


def load_hdf5_dataset(path: str) -> tuple[list[torch.Tensor], list[str], list[str]]:
    """Load motions and programs from HDF5 file.
    
    Returns:
        motions: List of motion tensors [T, num_joints*3]
        programs: List of program strings
        names: List of sample names
    """
    motions = []
    programs = []
    names = []
    
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            motion = f[key]["motion"][()]
            program = f[key].attrs["program"]
            
            motions.append(torch.tensor(motion, dtype=torch.float32))
            programs.append(program)
            names.append(key)
    
    return motions, programs, names


def preprocess_motion(motion: torch.Tensor, num_frames: int, num_joints: int) -> torch.Tensor:
    """Preprocess motion for ST-GCN encoder.
    
    Args:
        motion: [T, num_joints*3] tensor (flattened joint positions)
        num_frames: Target number of frames
        num_joints: Number of joints
    
    Returns:
        Preprocessed motion [T', num_joints*3] where T'=num_frames
    """
    T, features = motion.shape
    
    # Interpolate to target number of frames if needed
    if T != num_frames:
        # [T, F] -> [1, F, T] for 1D interpolation
        motion = motion.t().unsqueeze(0)  # [1, F, T]
        motion = F.interpolate(motion, size=num_frames, mode='linear', align_corners=True)
        motion = motion.squeeze(0).t()  # [T', F]
    
    return motion


def load_encoder(checkpoint_dir: str, config: dict, device: str) -> tuple[STGCNEncoder, int]:
    """Load the ST-GCN encoder from a checkpoint.
    
    Returns:
        encoder: The loaded encoder
        num_nodes: Number of nodes (joints) the encoder was trained with
    """
    encoder_path = os.path.join(checkpoint_dir, "trajectory_encoder.pt")
    
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found at {encoder_path}")
    
    # Load state dict to infer architecture
    state_dict = torch.load(encoder_path, map_location=device)
    
    # Infer num_nodes from A matrix shape: [3, num_nodes, num_nodes]
    num_nodes = state_dict["A"].shape[1]
    
    # Infer hidden_channels from blocks.0.tcn.0.weight shape
    hidden_channels = state_dict["blocks.0.tcn.0.weight"].shape[0]
    
    # Infer output_dim from output_projection.3.weight shape
    output_dim = state_dict["output_projection.3.weight"].shape[0]
    
    logger.info(f"Inferred encoder config: num_nodes={num_nodes}, hidden={hidden_channels}, output_dim={output_dim}")
    
    encoder = STGCNEncoder(
        num_nodes=num_nodes,
        input_channels=config.get("stgcn_in_channels", 3),
        hidden_channels=hidden_channels,
        output_dim=output_dim,
        num_blocks=config.get("stgcn_num_blocks", 4),
        num_temporal_tokens=config.get("stgcn_num_temporal_tokens", 64),
        spatial_kernel_size=config.get("stgcn_spatial_kernel", 3),
        dropout=config.get("stgcn_dropout", 0.1),
        graph_strategy=config.get("graph_strategy", "spatial"),
    ).to(device)
    
    encoder.load_state_dict(state_dict)
    encoder.eval()
    
    logger.info(f"Loaded ST-GCN encoder from {encoder_path}")
    return encoder, num_nodes


def encode_dataset(
    encoder: STGCNEncoder,
    motions: list[torch.Tensor],
    num_frames: int,
    num_joints: int,
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode all motions and return embeddings."""
    
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(motions), batch_size), desc="Encoding"):
            batch_motions = []
            
            for j in range(i, min(i + batch_size, len(motions))):
                motion = preprocess_motion(motions[j], num_frames, num_joints)
                batch_motions.append(motion)
            
            # Stack and encode
            batch = torch.stack(batch_motions).to(device)  # [B, C, T, V]
            encoded = encoder(batch)  # [B, num_tokens, dim]
            
            # Pool to single vector (mean over temporal tokens)
            pooled = encoded.mean(dim=1)  # [B, dim]
            
            embeddings.append(pooled.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)
    logger.info(f"Encoded {len(motions)} samples, embedding shape: {embeddings.shape}")
    
    return embeddings


def find_nearest_neighbors(
    query_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    k: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find k nearest neighbors for each query in reference set."""
    
    # Normalize embeddings for cosine similarity
    query_norm = F.normalize(query_embeddings, p=2, dim=1)
    ref_norm = F.normalize(reference_embeddings, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.mm(query_norm, ref_norm.t())  # [num_query, num_ref]
    
    # Get top-k
    distances, indices = similarity.topk(k, dim=1, largest=True)
    
    return distances, indices


def compute_normalized_edit_distance(target: str, predicted: str) -> float | None:
    """Compute normalized tree edit distance between two programs."""
    try:
        target_tree = parse_to_tree(target)
        pred_tree = parse_to_tree(predicted)
        edit_dist = program_edit_distance(target_tree, pred_tree)
        max_size = max(len(target_tree), len(pred_tree))
        return edit_dist / max_size if max_size > 0 else 0.0
    except LarkError:
        return None


def evaluate_retrieval(
    test_programs: list[str],
    predicted_programs: list[str],
    test_names: list[str] = None,
) -> dict:
    """Evaluate retrieval results."""
    
    exact_matches = 0
    edit_distances = []
    
    results = []
    
    for i, (target, predicted) in enumerate(zip(test_programs, predicted_programs)):
        # Check exact match
        is_exact = target.strip() == predicted.strip()
        if is_exact:
            exact_matches += 1
        
        # Compute normalized edit distance
        ed = compute_normalized_edit_distance(predicted, target)
        if ed is not None:
            edit_distances.append(ed)
        
        name = test_names[i] if test_names else f"sample_{i}"
        results.append({
            "name": name,
            "target": target,
            "predicted": predicted,
            "exact_match": is_exact,
            "edit_distance": ed,
        })
    
    metrics = {
        "accuracy": exact_matches / len(test_programs),
        "exact_matches": exact_matches,
        "total": len(test_programs),
        "mean_edit_distance": np.mean(edit_distances) if edit_distances else None,
        "std_edit_distance": np.std(edit_distances) if edit_distances else None,
        "min_edit_distance": np.min(edit_distances) if edit_distances else None,
        "max_edit_distance": np.max(edit_distances) if edit_distances else None,
        "valid_comparisons": len(edit_distances),
    }
    
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Retrieval baseline for motion-to-program")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--config", type=str, default="configs/parser.yaml",
                        help="Path to config file")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of nearest neighbors to consider")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for encoding")
    parser.add_argument("--show-examples", type=int, default=10,
                        help="Number of example predictions to show")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        # Try checkpoint directory
        config_path = os.path.join(args.checkpoint, "config.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(args.checkpoint), "config.yaml")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    
    # Load encoder
    encoder, num_joints = load_encoder(args.checkpoint, config, device)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_motions, train_programs, train_names = load_hdf5_dataset(config["train_data"])
    # Use eval_data for test set (config uses eval_data, not test_data)
    test_data_key = "test_data" if "test_data" in config else "eval_data"
    test_motions, test_programs, test_names = load_hdf5_dataset(config[test_data_key])
    
    num_frames = config.get("stgcn_num_frames", 64)
    # Use inferred num_joints from encoder
    
    logger.info(f"Train samples: {len(train_motions)}, Test samples: {len(test_motions)}")
    
    # Encode all motions
    logger.info("Encoding training set...")
    train_embeddings = encode_dataset(
        encoder, train_motions, num_frames, num_joints, device, args.batch_size
    )
    
    logger.info("Encoding test set...")
    test_embeddings = encode_dataset(
        encoder, test_motions, num_frames, num_joints, device, args.batch_size
    )
    
    # Find nearest neighbors
    logger.info(f"Finding {args.k} nearest neighbors...")
    distances, indices = find_nearest_neighbors(
        test_embeddings, train_embeddings, k=args.k
    )
    
    # Get predicted programs (from nearest neighbor)
    predicted_programs = [train_programs[idx[0].item()] for idx in indices]
    
    # Evaluate
    logger.info("Evaluating retrieval results...")
    metrics, results = evaluate_retrieval(test_programs, predicted_programs, test_names)
    
    # Print results
    print("\n" + "="*80)
    print("RETRIEVAL BASELINE RESULTS")
    print("="*80)
    print(f"\nAccuracy (exact match): {metrics['accuracy']*100:.2f}% ({metrics['exact_matches']}/{metrics['total']})")
    print(f"Mean Edit Distance:     {metrics['mean_edit_distance']:.4f} (±{metrics['std_edit_distance']:.4f})")
    print(f"Min/Max Edit Distance:  {metrics['min_edit_distance']:.4f} / {metrics['max_edit_distance']:.4f}")
    
    # Compare with generative model
    print("\n" + "-"*80)
    print("COMPARISON WITH GENERATIVE MODEL (Adapter Parser)")
    print("-"*80)
    print(f"{'Metric':<30} {'Retrieval':<15} {'Generative':<15} {'Better?':<10}")
    print("-"*80)
    
    gen_accuracy = 0.0
    gen_edit_dist = 0.71363
    gen_validity = 0.85938
    
    ret_accuracy = metrics['accuracy']
    ret_edit_dist = metrics['mean_edit_distance']
    
    acc_better = "Retrieval" if ret_accuracy > gen_accuracy else ("Generative" if gen_accuracy > ret_accuracy else "Tie")
    ed_better = "Retrieval" if ret_edit_dist < gen_edit_dist else ("Generative" if gen_edit_dist < ret_edit_dist else "Tie")
    
    print(f"{'Accuracy':<30} {ret_accuracy*100:>12.2f}% {gen_accuracy*100:>12.2f}% {acc_better:<10}")
    print(f"{'Mean Edit Distance':<30} {ret_edit_dist:>12.4f}  {gen_edit_dist:>12.4f}  {ed_better:<10}")
    print(f"{'Validity Rate':<30} {'100.00%':>12}  {gen_validity*100:>11.2f}% {'Retrieval':<10}")
    
    # Diagnosis
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    if ret_edit_dist < gen_edit_dist:
        print("✅ Retrieval OUTPERFORMS generative model")
        print("   → The ST-GCN encoder IS learning meaningful representations")
        print("   → The decoder/generation process is the bottleneck")
        print("\n   Recommendations:")
        print("   1. Try beam search with motion-constrained reranking")
        print("   2. Consider segment-level generation instead of token-by-token")
        print("   3. Add auxiliary decoder supervision")
    elif ret_edit_dist > gen_edit_dist:
        print("❌ Retrieval UNDERPERFORMS generative model")
        print("   → The ST-GCN encoder may NOT be learning good representations")
        print("   → Motion embeddings don't capture program-relevant features")
        print("\n   Recommendations:")
        print("   1. Add auxiliary supervision to ST-GCN (predict joint importance)")
        print("   2. Generate more diverse synthetic training data")
        print("   3. Try different encoder architectures")
    else:
        print("⚠️ Retrieval and generative model perform similarly")
        print("   → Both encoder and decoder may need improvement")
    
    # Show examples
    if args.show_examples > 0:
        print("\n" + "="*80)
        print(f"EXAMPLE PREDICTIONS (showing {args.show_examples})")
        print("="*80)
        
        # Sort by edit distance to show range
        sorted_results = sorted(results, key=lambda x: x['edit_distance'])
        
        # Show best, worst, and some middle examples
        n = min(args.show_examples, len(sorted_results))
        example_indices = []
        if n >= 3:
            example_indices = [0, len(sorted_results)//2, -1]  # best, middle, worst
            remaining = n - 3
            step = len(sorted_results) // (remaining + 1) if remaining > 0 else 1
            for i in range(1, remaining + 1):
                example_indices.append(i * step)
            example_indices = sorted(set(example_indices))[:n]
        else:
            example_indices = list(range(n))
        
        for idx in example_indices:
            r = sorted_results[idx]
            match_icon = "✓" if r['exact_match'] else "✗"
            print(f"\n[{match_icon}] {r['name']} (ED={r['edit_distance']:.3f})")
            print(f"  Target:    {r['target'][:200]}{'...' if len(r['target']) > 200 else ''}")
            print(f"  Retrieved: {r['predicted'][:200]}{'...' if len(r['predicted']) > 200 else ''}")
    
    # Analyze embedding quality
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS")
    print("="*80)
    
    # Check embedding statistics
    train_mean = train_embeddings.mean(dim=0)
    train_std = train_embeddings.std(dim=0)
    test_mean = test_embeddings.mean(dim=0)
    
    # Distribution shift
    mean_diff = (train_mean - test_mean).abs().mean().item()
    print(f"Mean embedding difference (train vs test): {mean_diff:.4f}")
    
    # Embedding variance
    print(f"Train embedding std (mean across dims): {train_std.mean().item():.4f}")
    
    # Nearest neighbor distance distribution
    nn_distances = distances[:, 0]
    print(f"NN cosine similarity: mean={nn_distances.mean().item():.4f}, "
          f"min={nn_distances.min().item():.4f}, max={nn_distances.max().item():.4f}")
    
    # Check if embeddings are collapsed
    if train_std.mean().item() < 0.01:
        print("⚠️ WARNING: Embeddings appear collapsed (very low variance)")
        print("   This suggests the encoder is not learning discriminative features")


if __name__ == "__main__":
    main()
