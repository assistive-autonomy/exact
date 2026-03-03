#!/usr/bin/env python
"""Motion Parser: training and evaluation.

Architecture: ST-GCN motion prefix tokens + LoRA fine-tuned LLM.
Motion tokens are prepended as prefix embeddings, and the LLM's native
self-attention handles conditioning.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Disable tokenizers parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import h5py
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    EarlyStoppingCallback,
)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exact.data import TrajectoryGenerationDataset
from exact.data.dataset import PROMPT_PREFIX
from exact.parser import MotionPrefixParser
from exact.parser.utils import create_grammar_processor
from exact.encoder import STGCNEncoder

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─── Helpers ──────────────────────────────────────────────────────────────────


def load_config(config_path: str, overrides: list[str] = None) -> DictConfig:
    """Load config file and apply CLI overrides."""
    base_cfg = OmegaConf.load(config_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(base_cfg, override_cfg)
    else:
        cfg = base_cfg
    return cfg


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def set_tf32(enabled: bool = True):
    """Enable TF32 for float32 matmuls and convolutions (A100/H100/H200).
    
    ~2x throughput for the float32 ST-GCN encoder with negligible precision loss.
    """
    if enabled and torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Auto-tune conv algorithms
        logger.info("TF32 enabled for float32 matmuls and cuDNN (benchmark=True)")


def get_model_hidden_size(model) -> int:
    """Get hidden size from model config."""
    config = model.config
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    raise AttributeError(f"Cannot find hidden_size in model config: {type(config)}")


def get_collate_fn(tokenizer, prompt_prefix: str = PROMPT_PREFIX, max_frame: int = 1024):
    """Create a collate function with proper label masking.

    Labels are set to -100 for:
        - Padding tokens (attention_mask == 0)
        - Prompt prefix tokens ("Program: ")
    Only actual program tokens + EOS contribute to the LM loss.
    """
    # Pre-compute prompt prefix token count
    prompt_token_ids = tokenizer(
        prompt_prefix, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]
    prompt_len = prompt_token_ids.shape[1]

    def collate_fn(batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        obs = torch.stack([x["obs"] for x in batch])

        # Create labels with proper masking
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100     # mask padding
        labels[:, :prompt_len] = -100          # mask prompt prefix

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "motion": obs,
            "labels": labels,
        }

    return collate_fn


# ─── Callbacks ────────────────────────────────────────────────────────────────


class EncoderCheckpointCallback(TrainerCallback):
    """Save encoder and projection weights with each checkpoint."""

    def __init__(self, parser_model):
        self.parser_model = parser_model

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        if os.path.exists(checkpoint_dir):
            self.parser_model.save_pretrained(checkpoint_dir)
            logger.debug(f"Saved encoder + projections to {checkpoint_dir}")


class GenerationEvalCallback(TrainerCallback):
    """Run generation-based evaluation periodically during training.

    Unlike eval_loss (teacher-forced), this tests actual autoregressive
    generation — the metric that matters for deployment.  It catches the
    failure mode where eval_loss looks fine but generation is garbage.
    """

    def __init__(
        self,
        parser_model,
        tokenizer,
        eval_h5_path: str,
        device: torch.device,
        dtype: torch.dtype,
        cfg: DictConfig,
        every_n_steps: int = 4000,
        n_samples: int = 16,
    ):
        self.parser_model = parser_model
        self.tokenizer = tokenizer
        self.eval_h5_path = eval_h5_path
        self.device = device
        self.dtype = dtype
        self.cfg = cfg
        self.every_n_steps = every_n_steps
        self.n_samples = n_samples
        self._eval_samples = None

    def _get_eval_samples(self):
        if self._eval_samples is None:
            self._eval_samples = load_eval_samples(
                self.eval_h5_path, n_samples=self.n_samples
            )
        return self._eval_samples

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if (
            state.global_step > 0
            and state.global_step % self.every_n_steps == 0
        ):
            logger.info(
                f"[Step {state.global_step}] Running mid-training generation eval "
                f"({self.n_samples} samples)..."
            )
            samples = self._get_eval_samples()
            was_training = self.parser_model.training
            try:
                eval_results = evaluate_samples(
                    self.parser_model,
                    self.tokenizer,
                    samples,
                    self.device,
                    self.dtype,
                    max_new_tokens=self.cfg.get("generation_max_new_tokens", 256),
                    num_retries=0,  # skip retries for speed
                    generation_temperature=self.cfg.get("generation_temperature", 0.3),
                    use_constrained_decoding=self.cfg.get(
                        "use_constrained_decoding", True
                    ),
                    max_frame=self.cfg.get("max_frame", 1024),
                )
                logger.info(
                    f"[Step {state.global_step}] Generation eval: "
                    f"accuracy={eval_results['accuracy']:.1%}, "
                    f"validity={eval_results['validity_rate']:.1%}, "
                    f"edit_dist={eval_results.get('mean_normalized_edit_distance', 'N/A')}"
                )
                if WANDB_AVAILABLE and wandb.run is not None:
                    log_data = {
                        "gen_eval/accuracy": eval_results["accuracy"],
                        "gen_eval/validity_rate": eval_results["validity_rate"],
                    }
                    if eval_results.get("mean_normalized_edit_distance") is not None:
                        log_data["gen_eval/edit_distance"] = eval_results[
                            "mean_normalized_edit_distance"
                        ]
                    wandb.log(log_data, step=state.global_step)
            except Exception as e:
                logger.warning(f"Mid-training generation eval failed: {e}")
            finally:
                if was_training:
                    self.parser_model.train()


class AlignmentLoggingCallback(TrainerCallback):
    """Log lm_loss and alignment_loss as separate wandb/log metrics."""

    def __init__(self, parser_model):
        self.parser_model = parser_model
        self._last_lm_loss = None
        self._last_alignment_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # The model stores the most recent loss breakdown in forward()
        m = self.parser_model
        if hasattr(m, "_last_lm_loss") and m._last_lm_loss is not None:
            logs["train/lm_loss"] = m._last_lm_loss
        if hasattr(m, "_last_alignment_loss") and m._last_alignment_loss is not None:
            logs["train/alignment_loss"] = m._last_alignment_loss
            logs["train/logit_scale"] = m.logit_scale.exp().item()
        # Track learned motion embedding scale
        if hasattr(m, "_motion_scale"):
            logs["train/motion_scale"] = m._motion_scale.item()


# ─── Evaluation ───────────────────────────────────────────────────────────────


def load_eval_samples(h5_path: str, n_samples: int = 8):
    """Load a subset of samples for generation-based evaluation."""
    samples = []
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())[:n_samples]
        for key in keys:
            motion = f[key]["motion"][()]
            program = f[key].attrs["program"]
            samples.append(
                {
                    "key": key,
                    "motion": torch.tensor(motion, dtype=torch.float32),
                    "program": program,
                }
            )
    return samples


@torch.no_grad()
def evaluate_samples(
    model,
    tokenizer,
    samples: list,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int = 256,
    num_retries: int = 2,
    retry_temperature: float = 0.5,
    retry_top_p: float = 0.9,
    generation_temperature: float = 0.3,
    use_constrained_decoding: bool = True,
    log_attempts: bool = False,
    max_frame: int = 1024,
) -> dict:
    """Evaluate model on sample batch and return metrics."""
    from exact.parser.utils import post_process_program, validate_program
    from exact.programs.edit_distance import program_edit_distance, parse_to_tree
    from lark.exceptions import LarkError

    model.eval()

    grammar_processor = (
        create_grammar_processor(tokenizer) if use_constrained_decoding else None
    )

    results = []
    exact_matches = 0
    valid_programs = 0
    edit_distances = []

    def _strip_prompt(text: str) -> str:
        """Strip the prompt prefix from generated text."""
        text = text.strip()
        if text.startswith(PROMPT_PREFIX):
            text = text[len(PROMPT_PREFIX):]
        return text.strip()

    def _generate_with_retries(motion_batch: torch.Tensor):
        """Try generation with increasing temperature on grammar failures."""
        # First attempt: low temperature (near-greedy)
        generated_ids = model.generate(
            motion=motion_batch,
            max_new_tokens=max_new_tokens,
            temperature=generation_temperature,
            do_sample=generation_temperature > 0,
            grammar_processor=grammar_processor,
        )
        raw = _strip_prompt(
            tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        )
        program, is_valid = post_process_program(raw, repair=True, max_frame=max_frame)
        if is_valid:
            return program, raw, None

        # Retry with higher temperature
        all_attempts = [(raw, program, is_valid)]
        for retry in range(num_retries):
            temp = retry_temperature + retry * 0.2
            generated_ids = model.generate(
                motion=motion_batch,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_p=retry_top_p,
                do_sample=True,
                grammar_processor=grammar_processor,
            )
            raw = _strip_prompt(
                tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            )
            program, is_valid = post_process_program(
                raw, repair=True, max_frame=max_frame
            )
            all_attempts.append((raw, program, is_valid))
            if is_valid:
                return program, raw, all_attempts if log_attempts else None

        # Return best attempt (first valid, or last)
        for raw, program, is_valid in all_attempts:
            if is_valid:
                return program, raw, all_attempts if log_attempts else None
        return program, raw, all_attempts if log_attempts else None

    for sample in tqdm(samples, desc="Evaluating samples"):
        motion = sample["motion"].unsqueeze(0).to(device)
        target = sample["program"]

        try:
            predicted, raw_predicted, attempts = _generate_with_retries(motion)
        except Exception as e:
            logger.warning(f"Generation failed for {sample['key']}: {e}")
            predicted, raw_predicted, attempts = "", None, None

        # Check exact match
        is_exact = predicted.strip() == target.strip()
        if is_exact:
            exact_matches += 1

        # Check validity
        is_valid = validate_program(predicted) if predicted else False
        if is_valid:
            valid_programs += 1

        # Compute normalized edit distance
        norm_edit_dist = None
        if is_valid and predicted.strip():
            try:
                target_tree = parse_to_tree(target)
                pred_tree = parse_to_tree(predicted)
                raw_dist = program_edit_distance(target_tree, pred_tree)
                max_size = max(len(target_tree), len(pred_tree))
                norm_edit_dist = raw_dist / max_size if max_size > 0 else 0.0
                edit_distances.append(norm_edit_dist)
            except (LarkError, Exception) as e:
                logger.debug(f"Edit distance failed: {e}")

        results.append(
            {
                "key": sample["key"],
                "target": target,
                "predicted": predicted,
                "raw_predicted": raw_predicted,
                "exact_match": is_exact,
                "is_valid": is_valid,
                "normalized_edit_distance": norm_edit_dist,
                "attempts": attempts,
            }
        )

    accuracy = exact_matches / len(samples) if samples else 0
    validity_rate = valid_programs / len(samples) if samples else 0
    mean_edit_distance = (
        sum(edit_distances) / len(edit_distances) if edit_distances else None
    )

    return {
        "accuracy": accuracy,
        "validity_rate": validity_rate,
        "mean_normalized_edit_distance": mean_edit_distance,
        "exact_matches": exact_matches,
        "valid_programs": valid_programs,
        "total": len(samples),
        "samples": results,
    }


def log_samples_to_wandb(eval_results: dict, step: int = None):
    """Log sample predictions to wandb as a table."""
    if not WANDB_AVAILABLE:
        return

    table = wandb.Table(
        columns=["key", "target", "predicted", "exact_match", "valid", "edit_dist"]
    )

    for sample in eval_results["samples"]:
        edit_dist_str = (
            f"{sample.get('normalized_edit_distance', 'N/A'):.3f}"
            if sample.get("normalized_edit_distance") is not None
            else "N/A"
        )
        table.add_data(
            sample["key"],
            sample["target"],
            sample["predicted"],
            "✓" if sample["exact_match"] else "✗",
            "✓" if sample.get("is_valid", True) else "✗",
            edit_dist_str,
        )

    log_data = {
        "eval/sample_predictions": table,
        "eval/accuracy": eval_results["accuracy"],
        "eval/exact_matches": eval_results["exact_matches"],
        "eval/validity_rate": eval_results.get("validity_rate", 1.0),
        "eval/valid_programs": eval_results.get("valid_programs", eval_results["total"]),
    }

    if eval_results.get("mean_normalized_edit_distance") is not None:
        log_data["eval/mean_normalized_edit_distance"] = eval_results[
            "mean_normalized_edit_distance"
        ]

    if step is not None:
        wandb.log(log_data, step=step)
    else:
        wandb.log(log_data)


def print_eval_results(eval_results: dict):
    """Pretty print evaluation results."""
    logger.info("SAMPLE EVALUATION RESULTS")
    logger.info(
        f"Accuracy: {eval_results['exact_matches']}/{eval_results['total']} "
        f"({eval_results['accuracy']:.1%})"
    )
    logger.info(
        f"Validity: {eval_results.get('valid_programs', eval_results['total'])}/"
        f"{eval_results['total']} ({eval_results.get('validity_rate', 1.0):.1%})"
    )
    if eval_results.get("mean_normalized_edit_distance") is not None:
        logger.info(
            f"Mean Normalized Edit Distance: "
            f"{eval_results['mean_normalized_edit_distance']:.3f} (lower is better)"
        )

    for sample in eval_results["samples"]:
        match_status = "✓" if sample["exact_match"] else "✗"
        valid_status = "V" if sample.get("is_valid", True) else "X"
        edit_dist = sample.get("normalized_edit_distance")
        edit_str = f" ED={edit_dist:.2f}" if edit_dist is not None else ""
        logger.info(f"[{match_status}|{valid_status}]{edit_str} {sample['key']}")
        logger.info(f"  Target:    {sample['target']}")
        logger.info(f"  Predicted: {sample['predicted']}")
        if sample.get("raw_predicted") and sample["raw_predicted"] != sample["predicted"]:
            logger.info(f"  Raw:       {sample['raw_predicted']}")


# ─── Checkpoint Loading ──────────────────────────────────────────────────────


def load_checkpoint(checkpoint_dir: str, device: torch.device):
    """Load trained MotionPrefixParser from checkpoint directory."""
    import yaml

    # Find config.yaml (in checkpoint dir or parent)
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    if not os.path.exists(config_path):
        parent_dir = os.path.dirname(checkpoint_dir.rstrip("/"))
        config_path = os.path.join(parent_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found in: {checkpoint_dir} or parent directory"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_dtype = torch.bfloat16

    load_in_4bit = config.get("load_in_4bit", False)
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Load tokenizer + base model
    logger.info(f"Loading base model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=None if load_in_4bit else model_dtype,
        device_map="auto",
        low_cpu_mem_usage=not load_in_4bit,
        quantization_config=quant_config,
    )
    model_hidden_size = get_model_hidden_size(base_model)

    # Load LoRA adapter
    lora_path = os.path.join(checkpoint_dir, "lora_adapter")
    pytorch_model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.exists(lora_path):
        logger.info(f"Loading LoRA adapter from: {lora_path}")
        base_model = PeftModel.from_pretrained(base_model, lora_path)
    elif os.path.exists(pytorch_model_path):
        logger.info(f"Loading from Trainer checkpoint: {pytorch_model_path}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=0.0,
            target_modules=list(
                config.get(
                    "target_modules",
                    ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
            ),
            bias="none",
        )
        base_model = get_peft_model(base_model, lora_config)
        checkpoint = torch.load(pytorch_model_path, map_location=device)
        model_state = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint.items()
            if k.startswith("model.")
        }
        base_model.load_state_dict(model_state, strict=False)
        logger.info("Loaded LoRA weights from pytorch_model.bin")
    else:
        logger.warning(f"LoRA adapter not found at {lora_path}")

    # Load ST-GCN encoder
    encoder_path = os.path.join(checkpoint_dir, "trajectory_encoder.pt")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Trajectory encoder not found: {encoder_path}")

    num_nodes = config["motion_dim"] // 3
    trajectory_encoder = STGCNEncoder(
        num_nodes=num_nodes,
        input_channels=3,
        hidden_channels=config.get("stgcn_hidden_channels", 64),
        output_dim=model_hidden_size,
        num_blocks=config.get("stgcn_num_blocks", 4),
        num_temporal_tokens=config.get("stgcn_num_temporal_tokens", 16),
        temporal_kernel_size=config.get("stgcn_temporal_kernel", 9),
        spatial_kernel_size=config.get("stgcn_spatial_kernel", 3),
        dropout=config.get("stgcn_dropout", 0.1),
        graph_strategy=config.get("graph_strategy", "spatial"),
        joint_embedding=config.get("stgcn_joint_embedding", False),
    ).to(device)

    trajectory_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    logger.info("Loaded ST-GCN encoder")

    # Create MotionPrefixParser
    model = MotionPrefixParser(
        model=base_model,
        trajectory_encoder=trajectory_encoder,
        tokenizer=tokenizer,
        encoder_dim=model_hidden_size,
        alignment_weight=config.get("alignment_weight", 0.1),
        alignment_dim=config.get("alignment_dim", 256),
        alignment_temperature=config.get("alignment_temperature", 0.07),
    )

    # Load parser weights (projection + alignment heads)
    prefix_path = os.path.join(checkpoint_dir, "prefix_parser.pt")
    if os.path.exists(prefix_path):
        logger.info(f"Loading parser weights from: {prefix_path}")
        prefix_data = torch.load(prefix_path, map_location=device)
        model.motion_projection.load_state_dict(prefix_data["motion_projection"])
        if "motion_align_head" in prefix_data:
            model.motion_align_head.load_state_dict(prefix_data["motion_align_head"])
        if "program_align_head" in prefix_data:
            model.program_align_head.load_state_dict(prefix_data["program_align_head"])
        if "logit_scale" in prefix_data:
            model.logit_scale.data = prefix_data["logit_scale"]
        if "motion_scale" in prefix_data:
            model._motion_scale.data = prefix_data["motion_scale"]
    else:
        logger.warning(f"Parser weights not found at {prefix_path}")

    # Move to correct device/dtype
    model.motion_projection.to(device=device, dtype=model_dtype)
    model.motion_align_head.to(device=device, dtype=model_dtype)
    model.program_align_head.to(device=device, dtype=model_dtype)
    model.logit_scale.data = model.logit_scale.data.to(device=device)
    model._motion_scale.data = model._motion_scale.data.to(device=device)

    model.eval()
    logger.info("Loaded MotionPrefixParser")

    return model, tokenizer, config, model_dtype


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Motion Parser")
    parser.add_argument(
        "--config", type=str, default="configs/parser.yaml", help="Config file"
    )
    parser.add_argument(
        "--eval-only",
        type=str,
        default=None,
        help="Checkpoint dir for eval only",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint dir",
    )
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config, args.overrides)

    # Multi-GPU: use LOCAL_RANK for per-process device placement (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    is_main_process = local_rank == 0
    model_dtype = torch.bfloat16
    set_seed(cfg.seed)
    set_tf32(cfg.get("tf32", False))

    # Handle resume vs new run
    resume_from_checkpoint = None
    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {args.resume}")

        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[1]),
        )
        if checkpoints:
            resume_from_checkpoint = str(checkpoints[-1])
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        else:
            logger.warning(f"No checkpoints found in {output_dir}, starting fresh")

        saved_config = output_dir / "config.yaml"
        if saved_config.exists():
            logger.info(f"Loading config from resumed run: {saved_config}")
            cfg = OmegaConf.load(saved_config)
            if args.overrides:
                override_cfg = OmegaConf.from_dotlist(args.overrides)
                cfg = OmegaConf.merge(cfg, override_cfg)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(cfg.get("output_dir", "results/parser")) / timestamp
        if is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, output_dir / "config.yaml")
        # Ensure all ranks wait for rank 0 to create the directory
        if world_size > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
            else:
                # Not yet initialized — directory will exist by the time Trainer needs it
                pass
        output_dir.mkdir(parents=True, exist_ok=True)  # no-op on rank 0, creates on others if needed

    run_name = output_dir.name

    # Initialize wandb (rank 0 only — Trainer handles per-rank logging)
    use_wandb = cfg.get("wandb_mode", "disabled") != "disabled" and WANDB_AVAILABLE
    if use_wandb and is_main_process:
        wandb_id_file = output_dir / "wandb_run_id.txt"
        wandb_resume = None
        wandb_id = None

        if args.resume and wandb_id_file.exists():
            wandb_id = wandb_id_file.read_text().strip()
            wandb_resume = "must"
            logger.info(f"Resuming wandb run: {wandb_id}")

        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"parser_{run_name}",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb_mode,
            id=wandb_id,
            resume=wandb_resume,
        )

        if not wandb_id_file.exists():
            wandb_id_file.write_text(wandb.run.id)

    logger.info("Motion-conditioned Prefix Parser")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    if args.resume:
        logger.info(f"Resuming from: {resume_from_checkpoint}")

    # ── Eval-only mode ──────────────────────────────────────────────────────

    if args.eval_only:
        logger.info(f"Eval-only mode: {args.eval_only}")
        model, tokenizer, _, model_dtype = load_checkpoint(args.eval_only, device)

        eval_samples = load_eval_samples(
            cfg.eval_data, n_samples=cfg.get("eval_samples", 8)
        )

        eval_results = evaluate_samples(
            model,
            tokenizer,
            eval_samples,
            device,
            model_dtype,
            max_new_tokens=cfg.get("generation_max_new_tokens", 256),
            num_retries=cfg.get("generation_retries", 2),
            retry_temperature=cfg.get("generation_retry_temperature", 0.5),
            retry_top_p=cfg.get("generation_retry_top_p", 0.9),
            generation_temperature=cfg.get("generation_temperature", 0.3),
            use_constrained_decoding=cfg.get("use_constrained_decoding", True),
            log_attempts=cfg.get("log_generation_attempts", False),
            max_frame=cfg.get("max_frame", 1024),
        )
        print_eval_results(eval_results)

        if use_wandb:
            log_samples_to_wandb(eval_results)
            wandb.finish()
        return

    # ── Training mode ───────────────────────────────────────────────────────

    # [1/4] Load model and tokenizer
    logger.info("[1/4] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_in_4bit = cfg.get("load_in_4bit", False)
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Loading base model with 4-bit quantization")

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=None if load_in_4bit else torch.bfloat16,
        device_map={"":  local_rank} if load_in_4bit else None,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
    )
    if not load_in_4bit:
        base_model = base_model.to(device)
    model_hidden_size = get_model_hidden_size(base_model)
    logger.info(f"Model: {cfg.model_name}, hidden_size: {model_hidden_size}")

    # [2/4] Apply LoRA
    logger.info("[2/4] Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(
            cfg.get(
                "target_modules",
                ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
        bias="none",
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # [3/4] Initialize ST-GCN encoder
    logger.info("[3/4] Initializing ST-GCN trajectory encoder...")

    num_nodes = cfg.motion_dim // 3
    num_temporal_tokens = cfg.get("stgcn_num_temporal_tokens", 64)
    trajectory_encoder = STGCNEncoder(
        num_nodes=num_nodes,
        input_channels=3,
        hidden_channels=cfg.get("stgcn_hidden_channels", 64),
        output_dim=model_hidden_size,
        num_blocks=cfg.get("stgcn_num_blocks", 4),
        num_temporal_tokens=num_temporal_tokens,
        temporal_kernel_size=cfg.get("stgcn_temporal_kernel", 9),
        spatial_kernel_size=cfg.get("stgcn_spatial_kernel", 3),
        dropout=cfg.get("stgcn_dropout", 0.1),
        graph_strategy=cfg.get("graph_strategy", "spatial"),
        joint_embedding=cfg.get("stgcn_joint_embedding", False),
    ).to(device=device)  # Keep float32 for BatchNorm stability

    encoder_params = sum(p.numel() for p in trajectory_encoder.parameters())
    logger.info(
        f"ST-GCN: {num_nodes} joints, {cfg.get('stgcn_num_blocks', 4)} blocks, "
        f"{num_temporal_tokens} temporal tokens, {encoder_params:,} params"
    )

    # Create MotionPrefixParser
    model = MotionPrefixParser(
        model=base_model,
        trajectory_encoder=trajectory_encoder,
        tokenizer=tokenizer,
        encoder_dim=model_hidden_size,
        alignment_weight=cfg.get("alignment_weight", 0.1),
        alignment_dim=cfg.get("alignment_dim", 256),
        alignment_temperature=cfg.get("alignment_temperature", 0.07),
    )

    # Move projection and alignment modules to local GPU with bf16
    model.motion_projection = model.motion_projection.to(device=device, dtype=model_dtype)
    model.motion_align_head = model.motion_align_head.to(device=device, dtype=model_dtype)
    model.program_align_head = model.program_align_head.to(device=device, dtype=model_dtype)
    model.logit_scale.data = model.logit_scale.data.to(device=device)
    model._motion_scale.data = model._motion_scale.data.to(device=device)

    proj_params = sum(p.numel() for p in model.motion_projection.parameters())
    align_params = (
        sum(p.numel() for p in model.motion_align_head.parameters())
        + sum(p.numel() for p in model.program_align_head.parameters())
        + 1  # logit_scale
    )
    logger.info(
        f"MotionPrefixParser: {num_temporal_tokens} motion prefix tokens, "
        f"{proj_params:,} projection params, "
        f"alignment_weight={model.alignment_weight}, "
        f"{align_params:,} alignment params"
    )

    # [4/4] Load datasets
    logger.info("[4/4] Loading datasets...")
    if not os.path.exists(cfg.train_data):
        raise FileNotFoundError(f"Training data not found: {cfg.train_data}")

    train_dataset = TrajectoryGenerationDataset(
        path=cfg.train_data,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        augment_crop_prob=cfg.get("augment_crop_prob", 0.0),
        augment_min_segments=cfg.get("augment_min_segments", 1),
        augment_min_crop_frames=cfg.get("augment_min_crop_frames", 32),
    )
    logger.info(f"Loaded {len(train_dataset)} training samples")

    eval_dataset = None
    if cfg.eval_data and os.path.exists(cfg.eval_data):
        eval_dataset = TrajectoryGenerationDataset(
            path=cfg.eval_data,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_seq_length,
            augment_crop_prob=0.0,  # No augmentation for eval
        )
        eval_dataset.training_mode = False  # Ensure no augmentation at eval
        max_eval_samples = cfg.get("max_eval_samples_training", None)
        if max_eval_samples and len(eval_dataset) > max_eval_samples:
            import random

            indices = random.sample(range(len(eval_dataset)), max_eval_samples)
            eval_dataset = torch.utils.data.Subset(eval_dataset, indices)
            logger.info(
                f"Using {max_eval_samples} eval samples (subsampled for speed)"
            )
        else:
            logger.info(f"Loaded {len(eval_dataset)} evaluation samples")

    # ── Build optimizer with differential learning rates ────────────────────
    # The encoder trains from scratch and needs a lower LR to avoid
    # over-shooting.  LoRA / projection layers need the base LR.
    # _motion_scale gets a slightly higher LR so it can adapt quickly.
    encoder_lr = cfg.get("encoder_lr", cfg.learning_rate * 0.2)  # default: 1e-5
    projection_lr = cfg.get("projection_lr", cfg.learning_rate)    # default: 5e-5
    scale_lr = cfg.get("scale_lr", cfg.learning_rate * 2)          # default: 1e-4

    # Group parameters
    encoder_params_list = list(model.trajectory_encoder.parameters())
    projection_params_list = list(model.motion_projection.parameters())
    align_params_list = (
        list(model.motion_align_head.parameters())
        + list(model.program_align_head.parameters())
        + [model.logit_scale]
    )
    scale_params_list = [model._motion_scale]

    # All other params (LoRA weights) get the base LR
    special_param_ids = set(
        id(p)
        for p in (
            encoder_params_list
            + projection_params_list
            + align_params_list
            + scale_params_list
        )
    )
    lora_params_list = [
        p for p in model.parameters() if p.requires_grad and id(p) not in special_param_ids
    ]

    optimizer_grouped_parameters = [
        {"params": encoder_params_list, "lr": encoder_lr, "weight_decay": cfg.weight_decay},
        {"params": projection_params_list, "lr": projection_lr, "weight_decay": cfg.weight_decay},
        {"params": align_params_list, "lr": projection_lr, "weight_decay": 0.0},
        {"params": scale_params_list, "lr": scale_lr, "weight_decay": 0.0},
        {"params": lora_params_list, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
    ]

    custom_optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-8)

    logger.info(
        f"Differential LR — encoder: {encoder_lr:.1e}, projection: {projection_lr:.1e}, "
        f"scale: {scale_lr:.1e}, LoRA: {cfg.learning_rate:.1e}"
    )

    # ── Training arguments ──────────────────────────────────────────────────

    warmup_args = {}
    if cfg.get("warmup_ratio", 0) > 0:
        warmup_args["warmup_ratio"] = cfg.warmup_ratio
    elif cfg.get("warmup_steps", 0) > 0:
        warmup_args["warmup_steps"] = cfg.warmup_steps

    save_args = {
        "save_strategy": cfg.save_strategy,
        "save_total_limit": cfg.save_total_limit,
    }
    if cfg.save_strategy == "steps":
        save_args["save_steps"] = cfg.get("save_steps", 500)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=f"parser_{run_name}",
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        **warmup_args,
        **save_args,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy if eval_dataset else "no",
        eval_steps=cfg.get("eval_steps", 500) if eval_dataset else None,
        load_best_model_at_end=cfg.load_best_model_at_end if eval_dataset else False,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=device.type == "cuda",
        dataloader_persistent_workers=cfg.get("dataloader_persistent_workers", False) and cfg.dataloader_num_workers > 0,
        dataloader_prefetch_factor=cfg.get("dataloader_prefetch_factor", 2),
        report_to="wandb" if (use_wandb and is_main_process) else "none",
        seed=cfg.seed,
        remove_unused_columns=False,
        save_safetensors=False,
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        ddp_find_unused_parameters=False,
    )

    # ── Callbacks ───────────────────────────────────────────────────────────

    callbacks = []
    early_stopping_patience = cfg.get("early_stopping_patience", 0)
    if early_stopping_patience > 0 and eval_dataset:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )
        logger.info(f"Early stopping enabled (patience={early_stopping_patience})")

    callbacks.append(EncoderCheckpointCallback(model))
    callbacks.append(AlignmentLoggingCallback(model))

    # Mid-training generation-based evaluation
    gen_eval_steps = cfg.get("generation_eval_steps", 0)
    gen_eval_samples = cfg.get("generation_eval_samples", 16)
    if gen_eval_steps > 0 and cfg.eval_data and os.path.exists(cfg.eval_data):
        callbacks.append(
            GenerationEvalCallback(
                parser_model=model,
                tokenizer=tokenizer,
                eval_h5_path=cfg.eval_data,
                device=device,
                dtype=model_dtype,
                cfg=cfg,
                every_n_steps=gen_eval_steps,
                n_samples=gen_eval_samples,
            )
        )
        logger.info(
            f"Generation eval enabled every {gen_eval_steps} steps "
            f"({gen_eval_samples} samples)"
        )

    # ── Train ───────────────────────────────────────────────────────────────

    # ── Scheduler (cosine with warmup, matched to custom optimizer) ──────
    from transformers import get_scheduler

    total_train_steps = (
        len(train_dataset)
        // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * world_size)
        * cfg.num_train_epochs
    )
    warmup_steps = (
        int(total_train_steps * cfg.warmup_ratio)
        if cfg.get("warmup_ratio", 0) > 0
        else cfg.get("warmup_steps", 0)
    )
    custom_scheduler = get_scheduler(
        cfg.get("lr_scheduler_type", "cosine"),
        optimizer=custom_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )
    logger.info(f"Scheduler: {cfg.get('lr_scheduler_type', 'cosine')}, "
                f"warmup={warmup_steps}, total={total_train_steps}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=get_collate_fn(
            tokenizer,
            max_frame=cfg.get("max_frame", 1024),
        ),
        callbacks=callbacks,
        optimizers=(custom_optimizer, custom_scheduler),
    )

    if resume_from_checkpoint:
        logger.info(f"Resuming training from: {resume_from_checkpoint}")
    else:
        logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ── Save (rank 0 only, while DDP is still active) ─────────────────────

    if is_main_process:
        logger.info("Saving model...")
        trainer.save_model()

        # Save encoder + projection weights
        model.save_pretrained(str(output_dir))

        # Save LoRA adapter
        base_model.save_pretrained(output_dir / "lora_adapter")
        tokenizer.save_pretrained(output_dir / "lora_adapter")

    # ── Tear down DDP before long-running single-process eval ─────────────
    # Without this, non-main ranks wait at the barrier while rank 0 runs
    # generation evaluation, causing NCCL watchdog timeouts (SIGABRT).
    if world_size > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()           # sync so all ranks finish saving
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed.")

    # ── Post-training evaluation (rank 0 only, single-process) ────────────

    if is_main_process:
        if cfg.get("skip_post_training_eval", False):
            logger.info("Skipping post-training generation evaluation (skip_post_training_eval=true)")
        else:
            logger.info("Running sample evaluation...")
            eval_samples = load_eval_samples(
                cfg.eval_data, n_samples=cfg.get("eval_samples", 8)
            )

            # Unwrap DDP wrapper if present so generation works single-process
            if hasattr(model, "module"):
                model = model.module

            model.to(device)
            eval_results = evaluate_samples(
                model,
                tokenizer,
                eval_samples,
                device,
                model_dtype,
                max_new_tokens=cfg.get("generation_max_new_tokens", 256),
                num_retries=cfg.get("generation_retries", 2),
                retry_temperature=cfg.get("generation_retry_temperature", 0.5),
                retry_top_p=cfg.get("generation_retry_top_p", 0.9),
                generation_temperature=cfg.get("generation_temperature", 0.3),
                use_constrained_decoding=cfg.get("use_constrained_decoding", True),
                log_attempts=cfg.get("log_generation_attempts", False),
                max_frame=cfg.get("max_frame", 1024),
            )
            print_eval_results(eval_results)

            # Save results
            with open(output_dir / "eval_results.json", "w") as f:
                json.dump(eval_results, f, indent=2)

            if use_wandb:
                log_samples_to_wandb(eval_results)
                wandb.summary["final_accuracy"] = eval_results["accuracy"]
                wandb.summary["final_validity_rate"] = eval_results["validity_rate"]
                if eval_results.get("mean_normalized_edit_distance") is not None:
                    wandb.summary["final_mean_edit_distance"] = eval_results[
                        "mean_normalized_edit_distance"
                    ]

        if use_wandb:
            wandb.finish()

        logger.success(f"Training complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()