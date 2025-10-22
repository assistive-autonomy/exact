import dataclasses
import sys
import inspect
from typing import Optional
import mujoco
import re
from dm_control.utils import rewards

from exact.programs.reward_base import RewardFunction
from exact.programs.utils import get_xpos


@dataclasses.dataclass
class SimpleReward(RewardFunction):
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
        return rewards.tolerance(
            current_pos,
            bounds=(self.target_pose - 0.1, self.target_pose + 0.1),
            margin=0.1,
        )


@dataclasses.dataclass
class LHip(SimpleReward):
    obj_body: str = "L_Hip"
    target_pose: float = 0.5

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^lhip-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LHip(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LKnee(SimpleReward):
    obj_body: str = "L_Knee"
    target_pose: float = 0.5

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^lknee-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LKnee(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LAnkle(SimpleReward):
    obj_body: str = "L_Ankle"
    target_pose: float = 0.5

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^lankle-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LAnkle(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LToe(SimpleReward):
    obj_body: str = "L_Toe"
    target_pose: float = 0.5

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^ltoe-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LToe(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RHip(SimpleReward):
    obj_body: str = "R_Hip"
    target_pose: float = 0.5

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^rhip-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RHip(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RKnee(SimpleReward):
    obj_body: str = "R_Knee"
    target_pose: float = 0.5

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^rknee-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RKnee(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RAnkle(SimpleReward):
    obj_body: str = "R_Ankle"
    target_pose: float = 0.5

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^rankle-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RAnkle(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RToe(SimpleReward):
    obj_body: str = "R_Toe"
    target_pose: float = 0.5

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^rtoe-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RToe(target_pose=target_pos)
        return None


@dataclasses.dataclass
class Torso(SimpleReward):
    obj_body: str = "Torso"
    target_pose: float = 1.4

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^torso-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return Torso(target_pose=target_pos)
        return None


@dataclasses.dataclass
class Spine(SimpleReward):
    obj_body: str = "Spine"
    target_pose: float = 1.2

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^spine-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return Spine(target_pose=target_pos)
        return None


@dataclasses.dataclass
class Chest(SimpleReward):
    obj_body: str = "Chest"
    target_pose: float = 1.3

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^chest-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return Chest(target_pose=target_pos)
        return None


@dataclasses.dataclass
class Neck(SimpleReward):
    obj_body: str = "Neck"
    target_pose: float = 1.4

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^neck-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return Neck(target_pose=target_pos)
        return None


@dataclasses.dataclass
class Head(SimpleReward):
    obj_body: str = "Head"
    target_pose: float = 1.4

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^head-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return Head(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LThorax(SimpleReward):
    obj_body: str = "L_Thorax"
    target_pose: float = 1.4

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^lthorax-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LThorax(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LShoulder(SimpleReward):
    obj_body: str = "L_Shoulder"
    target_pose: float = 1.4

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^lshoulder-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LShoulder(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LElbow(SimpleReward):
    obj_body: str = "L_Elbow"
    target_pose: float = 1.4

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^lelbow-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LElbow(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LWrist(SimpleReward):
    obj_body: str = "L_Wrist"
    target_pose: float = 1.4

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^lwrist-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LWrist(target_pose=target_pos)
        return None


@dataclasses.dataclass
class LHand(SimpleReward):
    obj_body: str = "L_Hand"
    target_pose: float = 1.4

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^lhand-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return LHand(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RThorax(SimpleReward):
    obj_body: str = "R_Thorax"
    target_pose: float = 1.4

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^rthorax-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RThorax(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RShoulder(SimpleReward):
    obj_body: str = "R_Shoulder"
    target_pose: float = 1.4

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^rshoulder-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RShoulder(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RElbow(SimpleReward):
    obj_body: str = "R_Elbow"
    target_pose: float = 1.4

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^relbow-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RElbow(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RWrist(SimpleReward):
    obj_body: str = "R_Wrist"
    target_pose: float = 1.4

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^rwrist-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RWrist(target_pose=target_pos)
        return None


@dataclasses.dataclass
class RHand(SimpleReward):
    obj_body: str = "R_Hand"
    target_pose: float = 1.4

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^rhand-(-?\d+\.*\d*)$"
        target_pos = SimpleReward.get_pose_from_pattern(name, pattern)
        if target_pos is not None:
            return RHand(target_pose=target_pos)
        return None


def make_from_name(
    name: str | None = None,
):
    all_rewards = inspect.getmembers(sys.modules["exact.programs.reward_simple"], inspect.isclass)
    for reward_class_name, reward_cls in all_rewards:
        if not inspect.isabstract(reward_cls):
            reward_obj = reward_cls.reward_from_name(name)
            if reward_obj is not None:
                return reward_obj
    raise ValueError(f"Unknown reward name: {name}")
