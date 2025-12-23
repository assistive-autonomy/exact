import numpy as np

from humenv import make_humenv
from gymnasium.wrappers import FlattenObservation

# Observation slices for SMPL humanoid (total 358 dims)
OBS_SLICES = {
    "root_h_obs": (0, 1),
    "local_body_pos": (1, 70),  # 23 joints * 3 (x,y,z)
    "local_body_rot_obs": (70, 214),
    "local_body_vel": (214, 286),
    "local_body_ang_vel": (286, 358),
}

# Number of SMPL body joints (23 body + 1 root = 24 total)
NUM_JOINTS = 24
JOINT_POS_DIM = NUM_JOINTS * 3  # 72


def extract_joint_positions(obs: np.ndarray) -> np.ndarray:
    """Extract ground-relative joint positions from observation.

    Takes local_body_pos (23 joints) which are relative to root, and rebases
    so root joint is at (0, 0, 0). All z-coordinates remain as-is since
    local_body_pos already contains positions relative to root.

    Args:
        obs: Full observation array [..., 358]

    Returns:
        Joint positions [..., 72] (24 joints * 3 xyz)
        Root joint is at (0, 0, 0), other joints relative to root.
    """
    is_batched = obs.ndim > 1
    if not is_batched:
        obs = obs[np.newaxis, :]

    local_pos = obs[..., 1:70]  # [..., 69] = 23 joints * 3

    # Reshape to [..., 23, 3] for easier manipulation
    local_pos = local_pos.reshape(obs.shape[:-1] + (23, 3))

    # Create root joint at (0, 0, 0)
    root_joint = np.zeros(obs.shape[:-1] + (1, 3), dtype=obs.dtype)

    # Concatenate: root + 23 body joints = 24 joints
    joint_pos = np.concatenate([root_joint, local_pos], axis=-2)

    # Flatten back to [..., 72]
    joint_pos = joint_pos.reshape(obs.shape[:-1] + (JOINT_POS_DIM,))

    if not is_batched:
        joint_pos = joint_pos.squeeze(0)

    return joint_pos


def unpack_obs(obs: np.ndarray) -> dict[str, np.ndarray]:
    """Unpack flattened observation into named components."""
    if obs.ndim == 1:
        return {name: obs[start:end] for name, (start, end) in OBS_SLICES.items()}
    return {name: obs[..., start:end] for name, (start, end) in OBS_SLICES.items()}


def pack_obs(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Pack observation dictionary back into flattened array."""
    return np.concatenate(
        [
            obs_dict["root_h_obs"],
            obs_dict["local_body_pos"],
            obs_dict["local_body_rot_obs"],
            obs_dict["local_body_vel"],
            obs_dict["local_body_ang_vel"],
        ],
        axis=-1,
    )


class HumEnv:
    """Wrapper for humanoid environment with flattened numpy observations."""

    def __init__(self, state_init: str = "Default"):
        self.env, _ = make_humenv(
            wrappers=[FlattenObservation],
            state_init=state_init,
        )

    def reset(self) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset()
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, done, truncated, info = self.env.step(action)
        return obs.astype(np.float32), reward, done, truncated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def model(self):
        return self.env.unwrapped.model

    @property
    def data(self):
        return self.env.unwrapped.data
