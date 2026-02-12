import numpy as np
import mujoco

from humenv import make_humenv
from gymnasium.wrappers import FlattenObservation
from scipy.spatial.transform import Rotation

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


# --- SMPL axis-angle to MuJoCo qpos conversion constants ---
# 90° X-rotation quaternion (wxyz) to convert SMPL Y-up to MuJoCo Z-up
_BASE_QUAT_WXYZ = np.array([0.7071068, 0.7071068, 0.0, 0.0])
# Same quaternion in scipy (xyzw) convention
_BASE_QUAT_XYZW = np.array([_BASE_QUAT_WXYZ[1], _BASE_QUAT_WXYZ[2],
                             _BASE_QUAT_WXYZ[3], _BASE_QUAT_WXYZ[0]])
# Default standing height for the SMPL humanoid
DEFAULT_STANDING_HEIGHT = 0.94


def _axisangle_to_euler_xyz(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotation vector to XYZ Euler angles.

    Args:
        rotvec: Axis-angle rotation vector (3,)

    Returns:
        XYZ Euler angles (3,)
    """
    if np.linalg.norm(rotvec) < 1e-8:
        return np.zeros(3)
    return Rotation.from_rotvec(rotvec).as_euler("XYZ")


def smpl_to_qpos(
    rotations: np.ndarray,
    standing_height: float = DEFAULT_STANDING_HEIGHT,
) -> np.ndarray:
    """Convert a single frame of SMPL axis-angle rotations to MuJoCo qpos.

    Maps 24 SMPL joints (axis-angle) to the 76-dim HumEnv qpos:
      [0:3]  = root position (x, y, z)
      [3:7]  = root quaternion (w, x, y, z)
      [7:76] = 23 body joints × 3 hinge angles (XYZ Euler)

    Args:
        rotations: SMPL axis-angle rotations (24, 3)
        standing_height: Root z-position in metres

    Returns:
        MuJoCo qpos array (76,)
    """
    qpos = np.zeros(76)

    # Root position
    qpos[0:3] = [0.0, 0.0, standing_height]

    # Root orientation: base rotation (Y-up→Z-up) composed with pelvis rotation
    pelvis_rotvec = rotations[0]
    if np.linalg.norm(pelvis_rotvec) > 1e-8:
        pelvis_rot = Rotation.from_rotvec(pelvis_rotvec)
        combined = Rotation.from_quat(_BASE_QUAT_XYZW) * pelvis_rot
        q_xyzw = combined.as_quat()
        qpos[3:7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # wxyz
    else:
        qpos[3:7] = _BASE_QUAT_WXYZ

    # Body joints 1–23: axis-angle → XYZ Euler angles
    for j in range(1, 24):
        qpos[7 + (j - 1) * 3 : 7 + j * 3] = _axisangle_to_euler_xyz(rotations[j])

    return qpos


def smpl_rotations_to_positions(
    rotations: np.ndarray,
    standing_height: float = DEFAULT_STANDING_HEIGHT,
) -> np.ndarray:
    """Convert SMPL axis-angle rotations to root-relative 3D joint positions.

    Performs MuJoCo forward kinematics on the HumEnv SMPL humanoid to obtain
    world-space body positions, then root-centres them to match the format
    produced by :func:`extract_joint_positions`.

    Args:
        rotations: Axis-angle rotations, shape ``(T, 24, 3)`` or ``(24, 3)``.
        standing_height: Root z-position in metres (default 0.94 for standing).

    Returns:
        Root-relative joint positions, shape ``(T, 72)`` or ``(72,)``
        (24 joints × 3 xyz, root at origin).
    """
    single_frame = rotations.ndim == 2
    if single_frame:
        rotations = rotations[np.newaxis, :]

    T = rotations.shape[0]

    # Create a lightweight MuJoCo model (no rendering, no env step needed)
    env = HumEnv()
    model = env.model
    data = mujoco.MjData(model)

    positions = np.zeros((T, JOINT_POS_DIM), dtype=np.float32)

    for t in range(T):
        # Convert rotations to qpos
        data.qpos[:] = smpl_to_qpos(rotations[t], standing_height)
        mujoco.mj_forward(model, data)

        # Extract 24 SMPL body positions (bodies 1–24, skipping world body 0)
        body_pos = data.xpos[1:25].copy()  # (24, 3)

        # Root-relative
        root = body_pos[0:1]
        relative = body_pos - root  # (24, 3)
        positions[t] = relative.flatten()

    env.env.close()

    if single_frame:
        return positions[0]
    return positions
