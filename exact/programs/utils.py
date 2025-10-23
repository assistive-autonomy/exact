import torch
import numpy as np
import mujoco


def rot2eul(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to Euler angles.
    
    Args:
        R: Rotation matrix of shape (3, 3)
        
    Returns:
        torch.Tensor: Euler angles (alpha, beta, gamma) in radians
    """
    beta = -torch.arcsin(R[2, 0])
    cos_beta = torch.cos(beta)
    alpha = torch.atan2(R[2, 1] / cos_beta, R[2, 2] / cos_beta)
    gamma = torch.atan2(R[1, 0] / cos_beta, R[0, 0] / cos_beta)
    return torch.stack((alpha, beta, gamma))


def get_xpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> torch.Tensor:
    """Get the position of a body as a PyTorch tensor.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        name: Name of the body
        
    Returns:
        torch.Tensor: Position vector (x, y, z)
    """
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert index > -1, f"Body {name} not found in model"
    return torch.tensor(data.xpos[index].copy(), dtype=torch.float32)


def get_xmat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> torch.Tensor:
    """Get the rotation matrix of a body as a PyTorch tensor.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        name: Name of the body
        
    Returns:
        torch.Tensor: Rotation matrix of shape (3, 3)
    """
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert index > -1, f"Body {name} not found in model"
    return torch.tensor(data.xmat[index].reshape((3, 3)).copy(), dtype=torch.float32)


def get_chest_upright(model: mujoco.MjModel, data: mujoco.MjData) -> torch.Tensor:
    """Get the upright orientation of the chest.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        
    Returns:
        torch.Tensor: Upright orientation value
    """
    chest_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Chest")
    assert chest_index > -1, "Chest body not found in model"
    return torch.tensor(data.xmat[chest_index][-2], dtype=torch.float32)


def get_sensor_data(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> torch.Tensor:
    """Get sensor data as a PyTorch tensor.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        name: Name of the sensor
        
    Returns:
        torch.Tensor: Sensor data
    """
    sensor_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    assert sensor_index > -1, f"Sensor {name} not found in model"
    start = model.sensor_adr[sensor_index]
    end = start + model.sensor_dim[sensor_index]
    return torch.tensor(data.sensordata[start:end].copy(), dtype=torch.float32)


def get_center_of_mass_linvel(model: mujoco.MjModel, data: mujoco.MjData) -> torch.Tensor:
    """Get the linear velocity of the center of mass.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        
    Returns:
        torch.Tensor: Linear velocity vector (vx, vy, vz)
    """
    sensor_name = "Chest_subtreelinvel"
    sensor_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    assert sensor_index > -1, f"Sensor {sensor_name} not found in model"
    start = model.sensor_adr[sensor_index]
    end = start + model.sensor_dim[sensor_index]
    return torch.tensor(data.sensordata[start:end].copy(), dtype=torch.float32)