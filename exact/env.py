import numpy as np

from humenv import make_humenv
from gymnasium.wrappers import FlattenObservation

# Observation slices for SMPL humanoid (total 358 dims)
OBS_SLICES = {
    "root_h_obs": (0, 1),           # 1 dim
    "local_body_pos": (1, 70),      # 69 dims (23 bodies * 3)
    "local_body_rot_obs": (70, 214), # 144 dims (24 bodies * 6)
    "local_body_vel": (214, 286),   # 72 dims (24 bodies * 3)
    "local_body_ang_vel": (286, 358), # 72 dims (24 bodies * 3)
}


def unpack_obs(obs: np.ndarray) -> dict[str, np.ndarray]:
    """Unpack flattened observation into named components.
    
    Args:
        obs: Flattened observation array of shape (358,) or (batch, 358)
        
    Returns:
        Dictionary with named observation components:
            - root_h_obs: (1,) or (batch, 1)
            - local_body_pos: (69,) or (batch, 69)
            - local_body_rot_obs: (144,) or (batch, 144)
            - local_body_vel: (72,) or (batch, 72)
            - local_body_ang_vel: (72,) or (batch, 72)
    """
    if obs.ndim == 1:
        return {name: obs[start:end] for name, (start, end) in OBS_SLICES.items()}
    else:
        return {name: obs[..., start:end] for name, (start, end) in OBS_SLICES.items()}


def pack_obs(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Pack observation dictionary back into flattened array.
    
    Args:
        obs_dict: Dictionary with named observation components
        
    Returns:
        Flattened observation array of shape (358,) or (batch, 358)
    """
    return np.concatenate([
        obs_dict["root_h_obs"],
        obs_dict["local_body_pos"],
        obs_dict["local_body_rot_obs"],
        obs_dict["local_body_vel"],
        obs_dict["local_body_ang_vel"],
    ], axis=-1)


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

    def render(self) -> np.ndarray:
        return self.env.render()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def model(self):
        return self.env.unwrapped.model

    @property
    def data(self):
        return self.env.unwrapped.data