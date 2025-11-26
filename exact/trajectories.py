import numpy as np

from exact.env import HumEnv
from exact.bm import BehaviourModel
from exact.programs.rewards import SensorReward, Reward
import torch
import mediapy as media

def motion_generation(
    env: HumEnv,
    model: BehaviourModel,
    obs: torch.Tensor,
    steps: int,
    reward_fn: SensorReward,
    render: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Generate the motion in the env for <steps> using following the reward function."""
    z = model.z_from_reward(env, reward_fn)
    obss, actions = [], []
    frames = [env.render()] if render else None

    for _ in range(steps):
        action = model.act(obs, z)
        obs, _, done, _, _ = env.step(action.cpu().numpy().ravel())
        obss.append(obs)
        actions.append(action)
        if render:
            frames.append(env.render())
        if done:
            obs = env.reset()

    obbs = torch.stack(obss).squeeze(1)
    actions = torch.stack(actions).squeeze(1)

    return obbs, actions, frames

def trajectory_generation(
    model: BehaviourModel,
    reward_fn: Reward,
    device: str = "cuda",
    render: bool = False,
    render_path: str = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a trajectory from """

    env = HumEnv(device=device)
    obs, _ = env.reset()
    obss, actions = [], []
    frames = [env.render()] if render else None

    for motion in reward_fn.motions:
        steps = motion.end - motion.start
        reward_fn = motion.sensor_reward
        motion_obbs, motion_actions, motion_frames = motion_generation(
            env, model, obs, steps, reward_fn, render
        )
        obs = motion_obbs[-1:]  # Keep last observation for next motion [1, 358]
        obss.append(motion_obbs)
        actions.append(motion_actions)
        if motion_frames is not None:
            frames.append(motion_frames)

    obbs = torch.cat(obss, dim=0)
    actions = torch.cat(actions, dim=0)

    if render and render_path is not None:
        media.write_video(render_path, np.array(frames), fps=30)

    return obbs, actions