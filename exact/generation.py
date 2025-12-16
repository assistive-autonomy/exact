import numpy as np
import torch
import mediapy as media

from exact.env import HumEnv
from exact.bm import BehaviourModel
from exact.programs.rewards import SensorReward, Reward


@torch.inference_mode()
def generate_motion(
    env: HumEnv,
    model: BehaviourModel,
    obs: np.ndarray,
    steps: int,
    reward_fn: SensorReward,
    device: str = "cuda",
    render: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list | None]:
    """Generate motion in the env for <steps> following <reward_fn>.

    Args:
        env: HumEnv environment instance
        model: BehaviourModel for action generation
        obs: Initial observation numpy array [1, obs_dim]
        steps: Number of steps to generate
        reward_fn: Reward function to follow
        device: Device for tensor operations
        render: Whether to collect rendered frames

    Returns:
        Tuple of (observations, actions, frames)
        - observations: [steps, obs_dim] torch.Tensor
        - actions: [steps, action_dim] torch.Tensor
        - frames: list of frames if render=True, else None
    """
    # Convert numpy obs to tensor for model (ensure 2D: [1, obs_dim])
    obs_tensor = torch.from_numpy(obs).to(device)
    if obs_tensor.dim() == 1:
        obs_tensor = obs_tensor.unsqueeze(0)
    z = model.z_from_reward(env, reward_fn)

    # Pre-allocate tensors for efficiency
    obs_dim = obs.shape[-1]
    action_dim = 69

    obss = torch.empty((steps, obs_dim), dtype=torch.float32, device=device)
    actions_tensor = torch.empty(
        (steps, action_dim), dtype=torch.float32, device=device
    )

    frames = [] if render else None
    if render:
        frames.append(env.render())

    for i in range(steps):
        action = model.act(obs_tensor, z)
        actions_tensor[i] = action.squeeze(0)

        # Step environment with numpy action
        obs, _, done, _, _ = env.step(action.squeeze(0).cpu().numpy())
        obs_tensor = torch.from_numpy(obs).to(device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        obss[i] = obs_tensor.squeeze(0)

        if render:
            frames.append(env.render())

        if done:
            obs, _ = env.reset()
            obs_tensor = torch.from_numpy(obs).to(device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

    return obss, actions_tensor, frames


@torch.inference_mode()
def generate_trajectory(
    model: BehaviourModel,
    reward_fn: Reward,
    device: str = "cuda",
    render: bool = False,
    render_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a full trajectory from a Reward program.

    Args:
        model: BehaviourModel for motion generation
        reward_fn: Reward containing motion intervals
        device: Device for tensor operations
        render: Whether to render and collect frames
        render_path: Path to save video (if render=True)

    Returns:
        Tuple of (observations, actions)
        - observations: [total_steps, obs_dim] torch.Tensor
        - actions: [total_steps, action_dim] torch.Tensor
    """
    env = HumEnv()
    obs, _ = env.reset()

    # Calculate total steps for pre-allocation
    total_steps = sum(m.end - m.start for m in reward_fn.motions)
    obs_dim = obs.shape[-1]
    action_dim = 69

    # Pre-allocate output tensors
    all_obss = torch.empty((total_steps, obs_dim), dtype=torch.float32, device=device)
    all_actions = torch.empty(
        (total_steps, action_dim), dtype=torch.float32, device=device
    )

    all_frames = [] if render else None
    if render:
        all_frames.append(env.render())

    idx = 0

    for motion in reward_fn.motions:
        steps = motion.end - motion.start
        sensor_reward = motion.sensor_reward

        motion_obss, motion_actions, motion_frames = generate_motion(
            env, model, obs, steps, sensor_reward, device, render
        )

        # Copy to pre-allocated tensors
        all_obss[idx : idx + steps] = motion_obss
        all_actions[idx : idx + steps] = motion_actions
        idx += steps

        # Keep last observation for next motion (as numpy array for env)
        obs = motion_obss[-1:].cpu().numpy()

        if motion_frames is not None:
            all_frames.extend(
                motion_frames[1:]
            )  # Skip first frame (already in previous)

    if render and render_path is not None:
        media.write_video(render_path, np.array(all_frames), fps=30)

    return all_obss, all_actions
