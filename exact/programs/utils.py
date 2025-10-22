import numpy as np
from mujoco import mujoco


def rot2eul(R: np.ndarray):
    beta = -np.arcsin(R[2, 0])
    alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
    gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
    return np.array((alpha, beta, gamma))


def get_xpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert index > -1
    xpos = data.xpos[index].copy()
    return xpos


def get_xmat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert index > -1
    xmat = data.xmat[index].reshape((3, 3)).copy()
    return xmat


def get_chest_upright(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    chest_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Chest")
    assert chest_index > -1
    chest_upright = data.xmat[chest_index][-2]
    return chest_upright


def get_sensor_data(model: mujoco.MjModel, data: mujoco.MjData, name: str):
    chest_gyro_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)  # in global coordinate
    assert chest_gyro_index > -1
    start = model.sensor_adr[chest_gyro_index]
    end = start + model.sensor_dim[chest_gyro_index]
    sensord = data.sensordata[start:end].copy()
    return sensord


def get_center_of_mass_linvel(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    chest_subtree_linvel_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "Chest_subtreelinvel")  # in global coordinate
    start = model.sensor_adr[chest_subtree_linvel_index]
    end = start + model.sensor_dim[chest_subtree_linvel_index]
    center_of_mass_velocity = data.sensordata[start:end].copy()
    return center_of_mass_velocity