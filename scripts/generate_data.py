import torch
import hydra
from loguru import logger
from omegaconf import OmegaConf
import h5py
import numpy as np

from exact.bm import BehaviourModel
from exact.programs import generate_program, parse_program
from exact.generation import generate_trajectory
from exact.config import DataConfig


@hydra.main(version_base=None, config_path="../configs", config_name="data")
def main(cfg: DataConfig):
    logger.info(f"Generating {cfg.name} data")
    logger.info(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BehaviourModel(device=device)

    # Pre-generate all programs
    logger.info("Generating programs...")
    programs = []
    rewards = []
    for _ in range(cfg.num_samples):
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
        programs.append(program)
        rewards.append(parse_program(program))

    # Generate trajectories and save to HDF5
    logger.info("Generating trajectories...")
    with h5py.File(f"{cfg.name}.hdf5", "w") as f:
        from tqdm import tqdm

        for i, (program, reward) in tqdm(
            enumerate(zip(programs, rewards)),
            total=len(programs),
            desc="Generating samples",
        ):
            obs, actions = generate_trajectory(
                model,
                reward,
                device=device,
                render=cfg.render,
                render_path=f"media/{cfg.name}_sample_{i}.mp4" if cfg.render else None,
            )

            motion_data = torch.cat([obs, actions], dim=-1)

            grp = f.create_group(f"motion_{i}")
            grp.create_dataset(
                "motion", data=motion_data.cpu().numpy().astype(np.float32)
            )
            grp.attrs["program"] = program

    logger.info(f"Data saved to {cfg.name}.hdf5")


if __name__ == "__main__":
    main()
