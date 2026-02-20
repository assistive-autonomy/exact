#!/usr/bin/env python
"""Diagnostic script for the Motion Prefix Parser.

Investigates why the trained model produces 'Program:' (empty/invalid) for ~75%
of evaluation samples despite reasonable training loss (0.74 eval loss).

Diagnostics run:
1. Raw generation inspection (what tokens does the model actually produce?)
2. Temperature sweep (does higher temperature help?)
3. Greedy vs sampling decoding
4. With/without grammar-constrained decoding
5. Motion encoder output analysis (are embeddings informative?)
6. Attention to motion prefix tokens
7. Training data sanity check
"""

import os
import sys
import json
from pathlib import Path

import h5py
import torch
import numpy as np
from loguru import logger

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.parsing.train_parser import (
    load_checkpoint,
    load_eval_samples,
    get_device,
)
from exact.parser.utils import (
    validate_program,
    post_process_program,
    create_grammar_processor,
)
from exact.data.dataset import PROMPT_PREFIX


def strip_prompt(text: str) -> str:
    text = text.strip()
    if text.startswith(PROMPT_PREFIX):
        text = text[len(PROMPT_PREFIX):]
    return text.strip()


def diagnose_raw_generation(model, tokenizer, samples, device, n=5):
    """Test 1: Inspect raw generated tokens without any post-processing."""
    print("\n" + "=" * 80)
    print("TEST 1: RAW GENERATION INSPECTION (no grammar constraint)")
    print("=" * 80)

    model.eval()
    for i, sample in enumerate(samples[:n]):
        motion = sample["motion"].unsqueeze(0).to(device)
        target = sample["program"]

        # Generate WITHOUT grammar constraint
        generated_ids = model.generate(
            motion=motion,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            grammar_processor=None,
        )
        raw_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        clean_text = strip_prompt(
            tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        )

        # Also show token IDs
        gen_token_ids = generated_ids[0].tolist()

        print(f"\n--- Sample {i} ({sample['key']}) ---")
        print(f"  Target:     {target[:120]}...")
        print(f"  Raw output: {raw_text[:200]}")
        print(f"  Clean:      {clean_text[:200]}")
        print(f"  Token count: {len(gen_token_ids)}")
        print(f"  Last 10 tokens: {gen_token_ids[-10:]}")
        print(f"  Last 10 decoded: {[tokenizer.decode([t]) for t in gen_token_ids[-10:]]}")
        print(f"  Valid: {validate_program(clean_text)}")


def diagnose_grammar_constrained(model, tokenizer, samples, device, n=5):
    """Test 2: Compare with/without grammar constraint."""
    print("\n" + "=" * 80)
    print("TEST 2: WITH vs WITHOUT GRAMMAR-CONSTRAINED DECODING")
    print("=" * 80)

    model.eval()
    grammar_processor = create_grammar_processor(tokenizer)

    for i, sample in enumerate(samples[:n]):
        motion = sample["motion"].unsqueeze(0).to(device)
        target = sample["program"]

        # Without grammar
        gen_no_grammar = model.generate(
            motion=motion,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            grammar_processor=None,
        )
        text_no_grammar = strip_prompt(
            tokenizer.decode(gen_no_grammar[0], skip_special_tokens=True)
        )

        # With grammar
        gen_with_grammar = model.generate(
            motion=motion,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            grammar_processor=grammar_processor,
        )
        text_with_grammar = strip_prompt(
            tokenizer.decode(gen_with_grammar[0], skip_special_tokens=True)
        )

        print(f"\n--- Sample {i} ({sample['key']}) ---")
        print(f"  Target:          {target[:120]}...")
        print(f"  No grammar:      {text_no_grammar[:200]}")
        print(f"  With grammar:    {text_with_grammar[:200]}")
        print(
            f"  Valid (no gram):  {validate_program(text_no_grammar)} | "
            f"Valid (gram): {validate_program(text_with_grammar)}"
        )


def diagnose_temperature_sweep(model, tokenizer, samples, device, n=3):
    """Test 3: Try various temperatures and decoding strategies."""
    print("\n" + "=" * 80)
    print("TEST 3: TEMPERATURE SWEEP")
    print("=" * 80)

    model.eval()
    temperatures = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5]

    for i, sample in enumerate(samples[:n]):
        motion = sample["motion"].unsqueeze(0).to(device)
        target = sample["program"]

        print(f"\n--- Sample {i} ({sample['key']}) ---")
        print(f"  Target: {target[:100]}...")

        for temp in temperatures:
            gen = model.generate(
                motion=motion,
                max_new_tokens=256,
                temperature=temp if temp > 0 else 1.0,
                do_sample=temp > 0,
                grammar_processor=None,
            )
            text = strip_prompt(
                tokenizer.decode(gen[0], skip_special_tokens=True)
            )
            valid = validate_program(text)
            label = f"T={temp:.1f}"
            # Truncate for display
            display = text[:100] + ("..." if len(text) > 100 else "")
            print(f"  {label:6s} [{' V' if valid else ' X'}]: {display}")


def diagnose_encoder_outputs(model, samples, device, n=5):
    """Test 4: Analyze motion encoder outputs."""
    print("\n" + "=" * 80)
    print("TEST 4: MOTION ENCODER OUTPUT ANALYSIS")
    print("=" * 80)

    model.eval()
    all_features = []

    for i, sample in enumerate(samples[:n]):
        motion = sample["motion"].unsqueeze(0).to(device)
        
        # Get raw encoder output (before projection)
        raw_enc = model.trajectory_encoder(motion)
        # Get projected features
        proj_param = next(model.motion_projection.parameters())
        raw_enc_cast = raw_enc.to(dtype=proj_param.dtype, device=proj_param.device)
        projected = model.motion_projection(raw_enc_cast)

        raw_np = raw_enc.detach().cpu().float().numpy()[0]
        proj_np = projected.detach().cpu().float().numpy()[0]

        print(f"\n--- Sample {i} ({sample['key']}) ---")
        print(f"  Motion shape:    {sample['motion'].shape}")
        print(f"  Raw encoder out: shape={raw_np.shape}, "
              f"mean={raw_np.mean():.4f}, std={raw_np.std():.4f}, "
              f"min={raw_np.min():.4f}, max={raw_np.max():.4f}")
        print(f"  Projected out:   shape={proj_np.shape}, "
              f"mean={proj_np.mean():.4f}, std={proj_np.std():.4f}, "
              f"min={proj_np.min():.4f}, max={proj_np.max():.4f}")

        # Check if all temporal tokens look similar (collapsed representation)
        token_norms = np.linalg.norm(proj_np, axis=-1)
        print(f"  Token norms:     {token_norms}")
        
        # Cosine similarity between temporal tokens
        proj_normed = proj_np / (np.linalg.norm(proj_np, axis=-1, keepdims=True) + 1e-8)
        cos_sim = proj_normed @ proj_normed.T
        off_diag = cos_sim[np.triu_indices(cos_sim.shape[0], k=1)]
        print(f"  Inter-token cos sim: mean={off_diag.mean():.4f}, "
              f"std={off_diag.std():.4f}, min={off_diag.min():.4f}, max={off_diag.max():.4f}")

        all_features.append(proj_np)

    # Cross-sample similarity (are all samples encoding to similar features?)
    if len(all_features) >= 2:
        means = np.array([f.mean(axis=0) for f in all_features])
        means_normed = means / (np.linalg.norm(means, axis=-1, keepdims=True) + 1e-8)
        cross_sim = means_normed @ means_normed.T
        off_diag = cross_sim[np.triu_indices(cross_sim.shape[0], k=1)]
        print(f"\n  Cross-sample cos sim: mean={off_diag.mean():.4f}, "
              f"std={off_diag.std():.4f}")
        if off_diag.mean() > 0.95:
            print("  ⚠️ WARNING: Motion encoder outputs are highly similar across "
                  "different samples — encoder may have collapsed!")


def diagnose_embedding_scale(model, tokenizer, samples, device, n=3):
    """Test 5: Compare scale of motion embeddings vs text embeddings."""
    print("\n" + "=" * 80)
    print("TEST 5: MOTION vs TEXT EMBEDDING SCALE COMPARISON")
    print("=" * 80)

    model.eval()
    embed_layer = model.model.get_input_embeddings()

    for i, sample in enumerate(samples[:n]):
        motion = sample["motion"].unsqueeze(0).to(device)
        target = sample["program"]

        # Get motion embeddings
        motion_features = model._encode_motion(motion)
        
        # Get text embeddings for target program
        text = PROMPT_PREFIX + target
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        text_embeds = embed_layer(encoded["input_ids"])

        m_np = motion_features.detach().cpu().float().numpy()[0]
        t_np = text_embeds.detach().cpu().float().numpy()[0]

        print(f"\n--- Sample {i} ({sample['key']}) ---")
        print(f"  Motion embeds: shape={m_np.shape}, "
              f"mean={m_np.mean():.4f}, std={m_np.std():.4f}, "
              f"L2_norm={np.linalg.norm(m_np, axis=-1).mean():.4f}")
        print(f"  Text embeds:   shape={t_np.shape}, "
              f"mean={t_np.mean():.4f}, std={t_np.std():.4f}, "
              f"L2_norm={np.linalg.norm(t_np, axis=-1).mean():.4f}")
        
        ratio = np.linalg.norm(m_np, axis=-1).mean() / (np.linalg.norm(t_np, axis=-1).mean() + 1e-8)
        print(f"  Scale ratio (motion/text): {ratio:.2f}")
        if ratio > 5 or ratio < 0.2:
            print("  ⚠️ WARNING: Large scale mismatch between motion and text embeddings!")


def diagnose_training_data(cfg, n=5):
    """Test 6: Inspect training data distribution."""
    print("\n" + "=" * 80)
    print("TEST 6: TRAINING DATA INSPECTION")
    print("=" * 80)

    for split, path in [("train", cfg.get("train_data")), ("eval", cfg.get("eval_data"))]:
        if not path or not os.path.exists(path):
            print(f"  {split}: not found at {path}")
            continue

        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            print(f"\n  {split}: {len(keys)} samples from {path}")

            # Sample program lengths
            prog_lengths = []
            motion_lengths = []
            num_segments = []
            for key in keys[:200]:
                prog = f[key].attrs["program"]
                motion = f[key]["motion"][()]
                prog_lengths.append(len(prog))
                motion_lengths.append(motion.shape[0])
                num_segments.append(prog.count(";") + 1)

            print(f"    Program char lengths: mean={np.mean(prog_lengths):.0f}, "
                  f"std={np.std(prog_lengths):.0f}, "
                  f"min={np.min(prog_lengths)}, max={np.max(prog_lengths)}")
            print(f"    Motion lengths: mean={np.mean(motion_lengths):.0f}, "
                  f"std={np.std(motion_lengths):.0f}, "
                  f"min={np.min(motion_lengths)}, max={np.max(motion_lengths)}")
            print(f"    Segments per program: mean={np.mean(num_segments):.1f}, "
                  f"std={np.std(num_segments):.1f}, "
                  f"min={np.min(num_segments)}, max={np.max(num_segments)}")

            # Show a few examples
            for key in keys[:n]:
                prog = f[key].attrs["program"]
                motion = f[key]["motion"][()]
                print(f"    {key}: motion={motion.shape}, program='{prog[:80]}...'")


def diagnose_logit_analysis(model, tokenizer, samples, device, n=3):
    """Test 7: Analyze output logits to understand what the model 'wants' to generate."""
    print("\n" + "=" * 80)
    print("TEST 7: LOGIT ANALYSIS AT GENERATION START")
    print("=" * 80)

    model.eval()

    for i, sample in enumerate(samples[:n]):
        motion = sample["motion"].unsqueeze(0).to(device)
        
        # Encode motion and get prefix + prompt embeddings
        motion_features = model._encode_motion(motion)
        prefix_len = motion_features.shape[1]
        prompt_ids = model._get_prompt_ids(1, device)
        prompt_embeds = model._get_input_embeddings(prompt_ids)
        
        # Combined embeddings
        inputs_embeds = torch.cat([motion_features, prompt_embeds], dim=1)
        
        # Run forward to get logits
        outputs = model.model(inputs_embeds=inputs_embeds, return_dict=True)
        
        # Logits at the last position (what would be generated first)
        last_logits = outputs.logits[0, -1, :]  # [vocab_size]
        
        # Top-k predictions
        topk_vals, topk_ids = torch.topk(last_logits, k=20)
        
        print(f"\n--- Sample {i} ({sample['key']}) ---")
        print(f"  Target starts with: '{sample['program'][:40]}'")
        print(f"  Top-20 next token predictions:")
        for rank, (val, tid) in enumerate(zip(topk_vals, topk_ids)):
            token_str = tokenizer.decode([tid.item()])
            print(f"    #{rank+1:2d}: '{token_str}' (logit={val.item():.2f}, id={tid.item()})")

        # Check: What's the probability of '[' (the first char of valid programs)?
        bracket_tokens = tokenizer.encode("[", add_special_tokens=False)
        print(f"\n  '[' token IDs: {bracket_tokens}")
        for bt in bracket_tokens:
            logit_val = last_logits[bt].item()
            rank = (last_logits > logit_val).sum().item() + 1
            print(f"    Token {bt} ('{tokenizer.decode([bt])}'): logit={logit_val:.2f}, rank={rank}")
        
        # Check EOS token probability
        eos_id = tokenizer.eos_token_id
        eos_logit = last_logits[eos_id].item()
        eos_rank = (last_logits > eos_logit).sum().item() + 1
        print(f"\n  EOS token ({eos_id}): logit={eos_logit:.2f}, rank={eos_rank}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose Motion Parser")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/parser/20260216_204927",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Number of samples for diagnostics",
    )
    parser.add_argument(
        "--tests",
        type=str,
        default="all",
        help="Comma-separated test numbers to run (e.g., '1,2,5') or 'all'",
    )
    args = parser.parse_args()

    device = get_device("auto")
    
    # Load model
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    model, tokenizer, config, model_dtype = load_checkpoint(args.checkpoint, device)
    model.eval()

    # Load eval samples
    eval_data = config.get("eval_data", "../exact_data/eval_mid.h5")
    samples = load_eval_samples(eval_data, n_samples=args.n_samples)
    logger.info(f"Loaded {len(samples)} eval samples")

    # Determine which tests to run
    if args.tests == "all":
        tests_to_run = {1, 2, 3, 4, 5, 6, 7}
    else:
        tests_to_run = {int(t.strip()) for t in args.tests.split(",")}

    # Run diagnostics
    with torch.no_grad():
        if 1 in tests_to_run:
            diagnose_raw_generation(model, tokenizer, samples, device, n=min(5, args.n_samples))

        if 2 in tests_to_run:
            diagnose_grammar_constrained(model, tokenizer, samples, device, n=min(5, args.n_samples))

        if 3 in tests_to_run:
            diagnose_temperature_sweep(model, tokenizer, samples, device, n=min(3, args.n_samples))

        if 4 in tests_to_run:
            diagnose_encoder_outputs(model, samples, device, n=min(5, args.n_samples))

        if 5 in tests_to_run:
            diagnose_embedding_scale(model, tokenizer, samples, device, n=min(3, args.n_samples))

        if 6 in tests_to_run:
            from omegaconf import DictConfig
            diagnose_training_data(config, n=min(5, args.n_samples))

        if 7 in tests_to_run:
            diagnose_logit_analysis(model, tokenizer, samples, device, n=min(3, args.n_samples))

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
