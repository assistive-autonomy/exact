import abc
from typing import Optional
import numpy as np
import mujoco

class RewardFunction(abc.ABC):
    @abc.abstractmethod
    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float: ...

    @staticmethod
    @abc.abstractmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]: ...

    def __call__(
        self,
        model: mujoco.MjModel,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
    ):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl
        mujoco.mj_forward(model, data)
        return self.compute(model, data)