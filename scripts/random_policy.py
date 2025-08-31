import argparse

from packaging.version import Version

import torch
import mediapy
from metamotivo.fb_cpr.huggingface import FBcprModel
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

    model.to(args.device)

    z = model.sample_z(args.seed)

    observation, _ = env.reset()
    frames = [env.render()]
    for i in range(30):
        action = model.act(observation, z, mean=True)
        observation, _, _, _, _ = env.step(action.cpu().numpy().ravel())
        frames.append(env.render())

    mediapy.write_video(f'random_policy.mp4', frames, fps=30)

if __name__ == "__main__":
    """
    Random policy with metamotivo model
    """
    parser = argparse.ArgumentParser(description='Run random policy.')
    parser.add_argument('--model', type=str, default='facebook/metamotivo-S-1', help='Model to use')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()
    main(args)