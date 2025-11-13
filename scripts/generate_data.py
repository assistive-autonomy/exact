import torch
import hydra
from omegaconf import OmegaConf
from exact.bm import BehaviourModel
from exact.programs import generate_program, RewardBuilder
from exact.config import DataConfig 

@hydra.main(version_base=None, config_path="../configs/data", config_name="default")
def main(cfg: DataConfig):
    """Generate and save data based on the provided configuration."""
    print("Generating data with the following configuration:")
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)

    bm = BehaviourModel()
    for i in range(cfg.num_samples):
        print(f"Generating sample {i+1}/{cfg.num_samples}...")
        program = generate_program(
            min_units=cfg.min_units,
            max_units=cfg.max_units,
            min_value=cfg.min_value,
            max_value=cfg.max_value,
            value_step=cfg.value_step,
            allowed_parts=cfg.allowed_parts,
        )
        print(f"Generated program: {program}")
        # reward_fn = RewardBuilder.reward_from_name(program)
        # poses, _ = bm.generate(reward_fn,steps=cfg.num_steps,render=False)

if __name__ == "__main__":
    main()