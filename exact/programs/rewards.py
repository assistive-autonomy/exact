import abc
import re
import sys
import dataclasses
from functools import reduce
import inspect
from typing import Optional

import torch
import mujoco
from dm_control.utils import rewards

from exact.programs.utils import get_xpos


class Reward(abc.ABC):
    @abc.abstractmethod
    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float: ...

    @staticmethod
    @abc.abstractmethod
    def reward_from_name(name: str) -> Optional["Reward"]: ...

    def __call__(
        self,
        model: mujoco.MjModel,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        ctrl: torch.Tensor,
    ) -> torch.Tensor:
        data = mujoco.MjData(model)
        data.qpos[:] = qpos.detach().cpu().numpy()
        data.qvel[:] = qvel.detach().cpu().numpy()
        data.ctrl[:] = ctrl.detach().cpu().numpy()
        mujoco.mj_forward(model, data)
        return torch.tensor(self.compute(model, data), device=qpos.device, dtype=qpos.dtype)
    

@dataclasses.dataclass
class PoseReward(Reward):
    obj_body: str = "Head"
    target_pose: float = 1.4

    @staticmethod
    def get_pose_from_pattern(name: str, pattern: str) -> Optional[float]:
        match = re.search(pattern, name)
        if match:
            target_pos = float(match.group(1))
            return target_pos
        return None

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        current_pos = get_xpos(model, data, name=self.obj_body)[-1]
        return float(rewards.tolerance(
            current_pos,
            bounds=(self.target_pose - 0.1, self.target_pose + 0.1),
            margin=0.1,
        ))


@dataclasses.dataclass
class LHip(PoseReward):
    obj_body: str = "L_Hip"

    @staticmethod
    def reward_from_name(name: str) -> Optional["LHip"]:
        pattern = r"^lhip\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LHip(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LKnee(PoseReward):
    obj_body: str = "L_Knee"

    @staticmethod
    def reward_from_name(name: str) -> Optional["LKnee"]:
        pattern = r"^lknee\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LKnee(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LAnkle(PoseReward):
    obj_body: str = "L_Ankle"

    @staticmethod
    def reward_from_name(name: str) -> Optional["LAnkle"]:
        pattern = r"^lankle\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LAnkle(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LToe(PoseReward):
    obj_body: str = "L_Toe"

    @staticmethod
    def reward_from_name(name: str) -> Optional["LToe"]:
        pattern = r"^ltoe\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LToe(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RHip(PoseReward):
    obj_body: str = "R_Hip"

    @staticmethod
    def reward_from_name(name: str) -> Optional["RHip"]:
        pattern = r"^rhip\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RHip(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RKnee(PoseReward):
    obj_body: str = "R_Knee"

    @staticmethod
    def reward_from_name(name: str) -> Optional["RKnee"]:
        pattern = r"^rknee\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RKnee(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RAnkle(PoseReward):
    obj_body: str = "R_Ankle"
    target_pose: float = 0.5

    @staticmethod
    def reward_from_name(name: str) -> Optional["RAnkle"]:
        pattern = r"^rankle\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RAnkle(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RToe(PoseReward):
    obj_body: str = "R_Toe"

    @staticmethod
    def reward_from_name(name: str) -> Optional["RToe"]:
        pattern = r"^rtoe\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RToe(target_pose=target_pos)
        return None


@dataclasses.dataclass
class Torso(PoseReward):
    obj_body: str = "Torso"

    @staticmethod
    def reward_from_name(name: str) -> Optional["Torso"]:
        pattern = r"^torso\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return Torso(target_pose=target_pos)
        return None


@dataclasses.dataclass
class Spine(PoseReward):
    obj_body: str = "Spine"

    @staticmethod
    def reward_from_name(name: str) -> Optional["Spine"]:
        pattern = r"^spine\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return Spine(target_pose=target_pos)
        return None


@dataclasses.dataclass
class Chest(PoseReward):
    obj_body: str = "Chest"

    @staticmethod
    def reward_from_name(name: str) -> Optional["Chest"]:
        pattern = r"^chest\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return Chest(target_pose=target_pos)
        return None


@dataclasses.dataclass
class Neck(PoseReward):
    obj_body: str = "Neck"

    @staticmethod
    def reward_from_name(name: str) -> Optional["Neck"]:
        pattern = r"^neck\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return Neck(target_pose=target_pos)
        return None


@dataclasses.dataclass
class Head(PoseReward):
    obj_body: str = "Head"

    @staticmethod
    def reward_from_name(name: str) -> Optional["Head"]:
        pattern = r"^head\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return Head(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LThorax(PoseReward):
    obj_body: str = "L_Thorax"

    @staticmethod
    def reward_from_name(name: str) -> Optional["LThorax"]:
        pattern = r"^lthorax\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LThorax(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LShoulder(PoseReward):
    obj_body: str = "L_Shoulder"

    @staticmethod
    def reward_from_name(name: str) -> Optional["LShoulder"]:
        pattern = r"^lshoulder\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LShoulder(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LElbow(PoseReward):
    obj_body: str = "L_Elbow"

    @staticmethod
    def reward_from_name(name: str) -> Optional["LElbow"]:
        pattern = r"^lelbow\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LElbow(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LWrist(PoseReward):
    obj_body: str = "L_Wrist"

    @staticmethod
    def reward_from_name(name: str) -> Optional["LWrist"]:
        pattern = r"^lwrist\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LWrist(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LHand(PoseReward):
    obj_body: str = "L_Hand"

    @staticmethod
    def reward_from_name(name: str) -> Optional["LHand"]:
        pattern = r"^lhand\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LHand(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RThorax(PoseReward):
    obj_body: str = "R_Thorax"

    @staticmethod
    def reward_from_name(name: str) -> Optional["RThorax"]:
        pattern = r"^rthorax\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RThorax(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RShoulder(PoseReward):
    obj_body: str = "R_Shoulder"

    @staticmethod
    def reward_from_name(name: str) -> Optional["RShoulder"]:
        pattern = r"^rshoulder\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RShoulder(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RElbow(PoseReward):
    obj_body: str = "R_Elbow"

    @staticmethod
    def reward_from_name(name: str) -> Optional["RElbow"]:
        pattern = r"^relbow\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RElbow(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RWrist(PoseReward):
    obj_body: str = "R_Wrist"

    @staticmethod
    def reward_from_name(name: str) -> Optional["RWrist"]:
        pattern = r"^rwrist\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RWrist(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RHand(PoseReward):
    obj_body: str = "R_Hand"

    @staticmethod
    def reward_from_name(name: str) -> Optional["RHand"]:
        pattern = r"^rhand\((\d+\.?\d*)\)$"
        target_pos = PoseReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RHand(target_pose=target_pos)
        return None


def make_from_name(
    name: str | None = None,
):
    all_rewards = inspect.getmembers(sys.modules["exact.programs.rewards"], inspect.isclass)
    for _, reward_cls in all_rewards:
        if not inspect.isabstract(reward_cls) and not isinstance(reward_cls, RewardBuilder):
            reward_obj = reward_cls.reward_from_name(name)
            if reward_obj is not None:
                return reward_obj
    raise ValueError(f"Unknown reward name: {name}")


@dataclasses.dataclass
class RewardBuilder(Reward):
    """Combines multiple reward functions by multiplying their outputs."""
    rewards: list[Reward]

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        return reduce(lambda acc, reward_fn: acc * reward_fn.compute(model, data), self.rewards, 1.0)

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardBuilder"]:
        reward_parts = name.split("*")
        rewards = [make_from_name(part) for part in reward_parts]
        return RewardBuilder(rewards=rewards)