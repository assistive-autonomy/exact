"""
Train inverse behavior model using GRPO (Group Relative Policy Optimization).

This script uses Hydra for configuration management. Example usage:

    # Default training
    python scripts/train_grpo.py

    # Override specific parameters
    python scripts/train_grpo.py training.learning_rate=1e-5 data.num_train_samples=1000

    # Use different config presets
    python scripts/train_grpo.py training=fast data=small

    # Multi-run with hyperparameter search
    python scripts/train_grpo.py -m training.learning_rate=1e-6,5e-6,1e-5

    # Disable WandB
    python scripts/train_grpo.py wandb.mode=disabled

    # Multi-GPU with accelerate
    accelerate launch scripts/train_grpo.py
"""

import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from exact.bm import BehaviourModel
from exact.programs import generate_programs, RewardBuilder
from exact.trainer_grpo import train_motion_to_program_grpo
from exact.utils import generate_program

def generate_training_data(
    num_samples: int,
    num_steps: int,
    program_config: DictConfig,
    behaviour_model: BehaviourModel,
    device: str,
):
    """Generate training data by running the behavior model on random programs.
    
    Args:
        num_samples: Number of program-motion pairs to generate
        num_steps: Number of timesteps for each motion sequence
        program_config: Configuration for program generation
        behaviour_model: Pre-trained behaviour model
        device: Device to use for generation
        
    Returns:
        Tuple of (motions, programs) where motions is a list of [N, 256] tensors
    """
    # Generate random programs
    programs = generate_programs(
        num_programs=num_samples,
        min_units=program_config.min_units,
        max_units=program_config.max_units,
        min_value=program_config.min_value,
        max_value=program_config.max_value,
    )
    
    motions = []
    valid_programs = []
    
    print(f"Generating {num_samples} motion sequences...")
    for i, program in enumerate(programs):
        if i % 100 == 0:
            print(f"Progress: {i}/{num_samples}")
        
        try:
            # Generate motion from program
            reward_fn = RewardBuilder.reward_from_name(program)
            poses, actions = behaviour_model.generate(
                reward_fn,
                steps=num_steps,
                render=False,
            )
            
            # Combine poses and actions into full motion tensor [N, 256]
            # poses: [N, 214], actions: [N, 42] -> motion: [N, 256]
            motion = torch.cat([poses, actions], dim=-1)
            
            motions.append(motion)
            valid_programs.append(program)
            
        except Exception as e:
            print(f"Failed to generate motion for program '{program}': {e}")
            continue
    
    print(f"Successfully generated {len(motions)} motion sequences")
    return motions, valid_programs


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training script."""
    
    # Print configuration
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    # Device setup
    if cfg.device:
        DEVICE = cfg.device
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nDevice: {DEVICE}")
    
    # Initialize wandb
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=cfg.wandb.dir,
            mode=cfg.wandb.mode,
        )
        print(f"WandB run: {wandb.run.name}")
        print(f"WandB URL: {wandb.run.url}")
    
    # Initialize behaviour model
    print("\n1. Loading behaviour model...")
    behaviour_model = BehaviourModel(
        model_name=cfg.behavior_model.name,
        batch_size=cfg.behavior_model.batch_size,
        max_episode_steps=cfg.behavior_model.max_episode_steps,
        device=DEVICE,
    )
    
    # Generate training data
    print("\n2. Generating training data...")
    train_motions, train_programs = generate_training_data(
        num_samples=cfg.data.num_train_samples,
        num_steps=cfg.data.num_steps,
        program_config=cfg.data.program_generation,
        behaviour_model=behaviour_model,
        device=DEVICE,
    )
    
    if cfg.wandb.mode != "disabled":
        wandb.log({"data/num_train_samples": len(train_motions)})
    
    print("\n3. Generating validation data...")
    val_motions, val_programs = generate_training_data(
        num_samples=cfg.data.num_val_samples,
        num_steps=cfg.data.num_steps,
        program_config=cfg.data.program_generation,
        behaviour_model=behaviour_model,
        device=DEVICE,
    )
    
    if cfg.wandb.mode != "disabled":
        wandb.log({"data/num_val_samples": len(val_motions)})
    
    # Train the model with GRPO
    print("\n4. Training inverse model with GRPO...")
    trainer = train_motion_to_program_grpo(
        train_motions=train_motions,
        train_programs=train_programs,
        val_motions=val_motions,
        val_programs=val_programs,
        model_name=cfg.model.name,
        behaviour_model=behaviour_model,
        output_dir=cfg.output_dir,
        motion_encoder_config=cfg.model.motion_encoder,
        training_config=cfg.training,
        wandb_enabled=(cfg.wandb.mode != "disabled"),
    )
    
    # Test inference
    print("\n5. Testing inference...")
    test_motion = val_motions[0].unsqueeze(0).to(DEVICE)  # [1, N, 256]
    generated_programs = generate_program(
        model=trainer.model,
        motion_encoder=trainer.motion_encoder,
        tokenizer=trainer.tokenizer,
        motion=test_motion,
        num_beams=5,
    )
    
    print(f"\nTest example:")
    print(f"Original program:  {val_programs[0]}")
    print(f"Generated program: {generated_programs[0]}")
    
    # Log example to wandb
    if cfg.wandb.mode != "disabled":
        wandb.log({
            "examples/original_program": val_programs[0],
            "examples/generated_program": generated_programs[0],
        })
    
    print("\n✓ Training complete!")
    print(f"Model saved to: {cfg.output_dir}")
    
    # Finish wandb run
    if cfg.wandb.mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()
