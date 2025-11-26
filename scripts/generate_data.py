import torch
import hydra
from tqdm import tqdm
from loguru import logger
from omegaconf import OmegaConf
import h5py
import numpy as np

from exact.bm import BehaviourModel
from exact.programs import generate_program, parse_program
from exact.trajectories import trajectory_generation
from exact.config import DataConfig


@hydra.main(version_base=None, config_path="../configs", config_name="data")
def main(cfg: DataConfig):
    logger.info(f"Generating {cfg.name} data")
    logger.info(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    model = BehaviourModel()

    with h5py.File(f"{cfg.name}.hdf5", "w") as f:
        for i in tqdm(range(cfg.num_samples), desc="Generating samples"):
            program = generate_program(
                min_preds=cfg.min_preds,
                max_preds=cfg.max_preds,
                min_value=cfg.min_value,
                max_value=cfg.max_value,
                value_step=cfg.value_step,
                allowed_parts=cfg.allowed_parts,
                max_timesteps=cfg.num_motion_steps,
                num_intervals=cfg.num_intervals,
                min_interval_time=cfg.min_interval_time,
            )

            reward = parse_program(program)
            obs, actions = trajectory_generation(model, reward)
            
            poses = obs[:, :214]
            motion_data = torch.cat([poses, actions], dim=-1)

            grp = f.create_group(f"motion_{i}")
            grp.create_dataset("motion", data=motion_data.cpu().numpy().astype(np.float32))
            grp.attrs["program"] = program

    logger.info(f"Data saved to {cfg.name}.hdf5")


if __name__ == "__main__":
    main()