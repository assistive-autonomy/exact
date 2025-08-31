import argparse

from packaging.version import Version

import torch
import h5py
import mediapy
from huggingface_hub import hf_hub_download
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.buffers.buffers import DictBuffer
from humenv.env import make_from_name
from metamotivo.wrappers.humenvbench import relabel
from humenv import make_humenv
import gymnasium
from gymnasium.wrappers import FlattenObservation, TransformObservation


def main(args):
    
    model = FBcprModel.from_pretrained(args.model)

    if Version("0.26") <= Version(gymnasium.__version__) < Version("1.0"):
        transform_obs_wrapper = lambda env: TransformObservation(
                env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=args.device)
            )
    else:
        transform_obs_wrapper = lambda env: TransformObservation(
                env, lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=args.device), env.observation_space
            )

    env, _ = make_humenv(
        num_envs=1,
        wrappers=[
            FlattenObservation,
            transform_obs_wrapper,
        ],
        state_init="Default",
    )

    buffer_path = hf_hub_download(
        repo_id=args.model,
        filename="data/buffer_inference_500000.hdf5",
    )
    with h5py.File(buffer_path, "r") as buffer_file:
        data = {k:v[:] for k, v in buffer_file.items()}
    buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
    buffer.extend(data)
    del data

    reward_fn = make_from_name(args.reward)
    rewards = relabel(env,
        qpos=buffer["next_qpos"],
        qvel=buffer["next_qvel"],
        action=buffer["action"],
        reward_fn=reward_fn, 
        max_workers=8
    )

    model.to(args.device)

    z = model.reward_inference(
        next_obs=buffer["next_observation"],
        reward=torch.tensor(rewards, device=model.cfg.device, dtype=torch.float32)
    )    
    
    observation, _ = env.reset()
    frames = [env.render()]
    for i in range(30):
        action = model.act(observation, z, mean=True)
        observation, reward, terminated, truncated, info = env.step(action.cpu().numpy().ravel())
        frames.append(env.render())

    mediapy.write_video(f'raward_{args.reward}_policy.mp4', frames, fps=30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run random policy.')
    parser.add_argument('--model', type=str, default='facebook/metamotivo-S-1', help='Model to use')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on')
    parser.add_argument('--reward', type=str, default="move-ego-0-2", help='Reward function to use')
    args = parser.parse_args()
    main(args)