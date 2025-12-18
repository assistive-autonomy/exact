import numpy as np
import torch

from exact.data.env import HumEnv, extract_joint_positions, JOINT_POS_DIM
from exact.bm import BehaviourModel
from exact.programs.rewards import SensorReward, Reward


@torch.inference_mode()
def generate_motion(
    env: HumEnv,
    model: BehaviourModel,
    obs: np.ndarray,
    steps: int,
    reward_fn: SensorReward,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate motion in environment for N steps following reward function.
    
    Args:
        env: HumEnv environment instance
        model: BehaviourModel for action generation
        obs: Initial observation [obs_dim]
        steps: Number of steps to generate
        reward_fn: Reward function to follow
        device: Device for tensor operations
        
    Returns:
        Tuple of (observations, actions)
        - observations: [steps, obs_dim]
        - actions: [steps, action_dim]
    """
    obs_tensor = torch.from_numpy(obs).to(device).unsqueeze(0) if obs.ndim == 1 else torch.from_numpy(obs).to(device)
    z = model.z_from_reward(env, reward_fn)

    obs_dim = obs.shape[-1]
    action_dim = 69
    
    obss = torch.empty((steps, obs_dim), dtype=torch.float32, device=device)
    actions_tensor = torch.empty((steps, action_dim), dtype=torch.float32, device=device)

    for i in range(steps):
        action = model.act(obs_tensor, z)
        actions_tensor[i] = action.squeeze(0)
        obs, _, done, _, _ = env.step(action.squeeze(0).cpu().numpy())
        obs_tensor = torch.from_numpy(obs).to(device).unsqueeze(0) if obs.ndim == 1 else torch.from_numpy(obs).to(device)
        obss[i] = obs_tensor.squeeze(0)
        
        if done:
            obs, _ = env.reset()
            obs_tensor = torch.from_numpy(obs).to(device).unsqueeze(0) if obs.ndim == 1 else torch.from_numpy(obs).to(device)

    return obss, actions_tensor


@torch.inference_mode()
def generate_trajectory(
    model: BehaviourModel,
    reward_fn: Reward,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate full trajectory from reward program.
    
    Args:
        model: BehaviourModel for motion generation
        reward_fn: Reward containing motion intervals
        device: Device for tensor operations
        
    Returns:
        Tuple of (observations, actions)
        - observations: [total_steps, 72] (24 joints * 3 xyz)
        - actions: [total_steps, action_dim]
    """
    env = HumEnv()
    obs, _ = env.reset()
    
    total_steps = sum(m.end - m.start for m in reward_fn.motions)
    action_dim = 69
    
    all_obss = torch.empty((total_steps, JOINT_POS_DIM), dtype=torch.float32, device=device)
    all_actions = torch.empty((total_steps, action_dim), dtype=torch.float32, device=device)
    
    idx = 0
    for motion in reward_fn.motions:
        steps = motion.end - motion.start
        motion_obss, motion_actions = generate_motion(
            env, model, obs, steps, motion.sensor_reward, device
        )
        
        # Extract joint positions from full observations
        joint_positions = extract_joint_positions(motion_obss.cpu().numpy())
        joint_positions = torch.from_numpy(joint_positions).to(device, dtype=torch.float32)
        
        all_obss[idx : idx + steps] = joint_positions
        all_actions[idx : idx + steps] = motion_actions
        idx += steps
        obs = motion_obss[-1:].cpu().numpy()

    return all_obss, all_actions
