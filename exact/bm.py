from typing import TYPE_CHECKING, Optional
import os

import h5py
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from metamotivo.fb_cpr.huggingface import FBcprModel
from metamotivo.buffers.buffers import DictBuffer
from metamotivo.wrappers.humenvbench import relabel

from exact.data import HumEnv
from exact.programs import SensorReward

if TYPE_CHECKING:
    from exact.models import ExecutableActivityModel


class BehaviourModel:
    """Generates motions from reward programs using pre-trained MetaMotivo model."""

    def __init__(
        self,
        model_name: str = "facebook/metamotivo-M-1",
        batch_size: int = 256,
        device: str = "cpu",
        relabel_workers: int | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.relabel_workers = relabel_workers if relabel_workers is not None else 8

        self.model = FBcprModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        buffer_path = hf_hub_download(
            repo_id=model_name,
            filename="data/buffer_inference_500000.hdf5",
            repo_type="model",
        )
        with h5py.File(buffer_path, "r") as f:
            data = {k: v[:] for k, v in f.items()}
            self.buffer = DictBuffer(capacity=data["qpos"].shape[0], device=device)
            self.buffer.extend(data)

    def z_from_reward(self, env: HumEnv, reward_fn: SensorReward) -> torch.Tensor:
        """Compute latent z from reward function using buffer relabeling."""
        batch = self.buffer.sample(batch_size=self.batch_size)
        rewards = relabel(
            env.env,
            qpos=batch["qpos"].cpu(),
            qvel=batch["qvel"].cpu(),
            action=batch["action"].cpu(),
            reward_fn=reward_fn,
            max_workers=self.relabel_workers,
        )
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        z = self.model.reward_wr_inference(batch["next_observation"], rewards)
        return z.squeeze()  # ensure 1D [z_dim]

    def act(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Get action from observation and latent z."""
        return self.model.act(obs, z)

    def z_from_executable_model(
        self,
        env: HumEnv,
        executable_model: "ExecutableActivityModel",
        timestep: int,
    ) -> torch.Tensor:
        """Compute latent z from ExecutableActivityModel at a specific timestep.
        
        The ExecutableActivityModel provides a reward function at each timestep
        that combines multiple programs via logical disjunction.
        
        Args:
            env: HumEnv environment
            executable_model: ExecutableActivityModel with combined programs
            timestep: Current timestep in normalized time [0, eval_timesteps]
            
        Returns:
            Latent z vector for the behaviour model
        """
        # Get reward function for this timestep from executable model
        reward_fn = executable_model.get_reward_fn(timestep)
        
        if reward_fn is None:
            # No active reward at this timestep, return zero z
            return torch.zeros(
                self.model.cfg.z_dim,
                device=self.device,
                dtype=torch.float32,
            )
        
        # Create a wrapper that matches SensorReward interface
        class ExecutableRewardWrapper:
            def __init__(self, reward_fn):
                self._reward_fn = reward_fn
            
            def compute(self, model, data):
                return self._reward_fn(model, data)
            
            def __call__(self, model, qpos, qvel, ctrl):
                import mujoco
                data = mujoco.MjData(model)
                data.qpos[:] = qpos.detach().cpu().numpy()
                data.qvel[:] = qvel.detach().cpu().numpy()
                data.ctrl[:] = ctrl.detach().cpu().numpy()
                mujoco.mj_forward(model, data)
                result = self.compute(model, data)
                return torch.as_tensor(result, device=qpos.device, dtype=qpos.dtype)
        
        wrapper = ExecutableRewardWrapper(reward_fn)
        return self.z_from_reward(env, wrapper)

    def precompute_z_cache(
        self,
        env: HumEnv,
        executable_model: "ExecutableActivityModel",
        num_steps: int,
        n_variants: int = 1,
    ) -> dict[int, list[torch.Tensor]]:
        """Pre-compute z vectors for all unique timesteps of an activity model.

        Instead of computing z on-the-fly for every (trajectory, timestep) pair,
        this pre-computes z once per unique normalized timestep. This reduces
        expensive relabel() calls from N_trajectories * N_unique_timesteps to
        just N_unique_timesteps * n_variants.

        Args:
            env: HumEnv environment for reward relabeling
            executable_model: The activity model to compute z for
            num_steps: Number of trajectory steps (determines unique timesteps)
            n_variants: Number of z variants per timestep for diversity

        Returns:
            Dict mapping normalized_t -> list of z tensors (one per variant)
        """
        z_cache: dict[int, list[torch.Tensor]] = {}
        for step in range(num_steps):
            normalized_t = int(step * executable_model.eval_timesteps / num_steps)
            if normalized_t not in z_cache:
                variants = []
                for _ in range(n_variants):
                    z = self.z_from_executable_model(
                        env, executable_model, normalized_t
                    )
                    variants.append(z)
                z_cache[normalized_t] = variants
        return z_cache

    def generate_trajectories_batched(
        self,
        envs: list[HumEnv],
        executable_model: "ExecutableActivityModel",
        num_steps: int,
        z_cache: dict[int, list[torch.Tensor]],
        z_offset: int = 0,
    ) -> list[dict]:
        """Generate K trajectories simultaneously with batched GPU inference.

        Runs K environments in parallel, batching their observations for a
        single GPU forward pass per timestep.  z vectors are taken from a
        pre-computed cache (see :meth:`precompute_z_cache`) and cycled across
        environments for diversity.

        Args:
            envs: List of K HumEnv environments
            executable_model: Activity model (only used for eval_timesteps)
            num_steps: Number of steps per trajectory
            z_cache: Pre-computed z vectors from precompute_z_cache()
            z_offset: Offset into z variants for diversity across batches

        Returns:
            List of K result dicts, each with ``observations`` (num_steps, obs_dim)
            and ``actions`` (num_steps, act_dim).
        """
        K = len(envs)
        if K == 0:
            return []

        # Reset all envs and collect initial observations
        obs_np = np.stack([env.reset()[0] for env in envs])  # (K, obs_dim)
        obs_batch = torch.tensor(obs_np, device=self.device, dtype=torch.float32)

        obs_frames: list[np.ndarray] = [obs_np.copy()]
        act_frames: list[np.ndarray] = []

        for step in range(num_steps):
            normalized_t = int(step * executable_model.eval_timesteps / num_steps)
            variants = z_cache[normalized_t]
            n_variants = len(variants)

            # Build z batch – cycle through variants for diversity
            z_tensors = [
                variants[(z_offset + i) % n_variants] for i in range(K)
            ]
            z_batch = torch.stack(z_tensors)  # (K, z_dim)

            # Batched action inference on GPU
            with torch.no_grad():
                actions_batch = self.act(obs_batch, z_batch)  # (K, act_dim)
            actions_np = actions_batch.cpu().numpy()
            act_frames.append(actions_np)

            # Step all environments
            next_obs_list: list[np.ndarray] = []
            for i in range(K):
                next_obs, _, done, truncated, _ = envs[i].step(actions_np[i])
                if done or truncated:
                    next_obs, _ = envs[i].reset()
                next_obs_list.append(next_obs)

            obs_np = np.stack(next_obs_list)
            obs_batch = torch.tensor(
                obs_np, device=self.device, dtype=torch.float32
            )
            obs_frames.append(obs_np)

        # Stack: (num_steps+1, K, obs_dim) and (num_steps, K, act_dim)
        obs_array = np.stack(obs_frames[:-1])  # exclude final obs
        act_array = np.stack(act_frames)

        return [
            {
                "observations": obs_array[:, i],  # (num_steps, obs_dim)
                "actions": act_array[:, i],        # (num_steps, act_dim)
            }
            for i in range(K)
        ]

    def generate_trajectory(
        self,
        env: HumEnv,
        executable_model: "ExecutableActivityModel",
        num_steps: Optional[int] = None,
        return_rewards: bool = False,
    ) -> dict:
        """Generate a motion trajectory using an ExecutableActivityModel.
        
        Rolls out the environment using the executable model's reward function
        at each timestep. The reward function evolves over time according to
        the normalized temporal structure of the combined programs.
        
        Args:
            env: HumEnv environment
            executable_model: ExecutableActivityModel to guide generation
            num_steps: Number of steps to generate (defaults to eval_timesteps)
            return_rewards: Whether to return per-timestep rewards
            
        Returns:
            Dictionary with:
                - observations: [num_steps, obs_dim] observation trajectory
                - actions: [num_steps, action_dim] action trajectory
                - rewards: [num_steps] rewards (if return_rewards=True)
        """
        if num_steps is None:
            num_steps = executable_model.eval_timesteps
        
        # Reset environment to random initial state
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)
        
        observations = [obs]
        actions = []
        rewards = [] if return_rewards else None
        
        # Pre-compute z for each unique timestep interval
        # For efficiency, we compute z at the start of each motion interval
        z_cache = {}
        
        for step in range(num_steps):
            # Map step to normalized timestep
            normalized_t = int(step * executable_model.eval_timesteps / num_steps)
            
            # Get or compute z for this timestep
            if normalized_t not in z_cache:
                z_cache[normalized_t] = self.z_from_executable_model(
                    env, executable_model, normalized_t
                )
            z = z_cache[normalized_t]
            
            # Get action from policy
            with torch.no_grad():
                action = self.act(obs_tensor.unsqueeze(0), z.unsqueeze(0))
                action = action.squeeze(0).cpu().numpy()
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            observations.append(next_obs)
            actions.append(action)
            
            if return_rewards:
                # Compute reward from executable model
                model_reward = executable_model.compute_reward(
                    normalized_t, env.model, env.data
                )
                rewards.append(model_reward)
            
            obs = next_obs
            obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)
            
            if done or truncated:
                break
        
        result = {
            "observations": np.stack(observations[:-1]),  # Exclude final obs
            "actions": np.stack(actions),
        }
        
        if return_rewards:
            result["rewards"] = np.array(rewards)
        
        return result
