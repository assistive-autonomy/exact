import os
from functools import reduce
from typing import Callable
from dataclasses import dataclass

import torch
import mujoco
from lark import Lark, Transformer

# Body name mapping: program names -> MuJoCo body names
BODY_NAMES = {
    "pelvis": "Pelvis",
    "torso": "Torso",
    "spine": "Spine",
    "chest": "Chest",
    "neck": "Neck",
    "head": "Head",
    "lhip": "L_Hip",
    "lknee": "L_Knee",
    "lankle": "L_Ankle",
    "ltoe": "L_Toe",
    "rhip": "R_Hip",
    "rknee": "R_Knee",
    "rankle": "R_Ankle",
    "rtoe": "R_Toe",
    "lthorax": "L_Thorax",
    "lshoulder": "L_Shoulder",
    "lelbow": "L_Elbow",
    "lwrist": "L_Wrist",
    "lhand": "L_Hand",
    "rthorax": "R_Thorax",
    "rshoulder": "R_Shoulder",
    "relbow": "R_Elbow",
    "rwrist": "R_Wrist",
    "rhand": "R_Hand",
}

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def get_joint_position(model: mujoco.MjModel, data: mujoco.MjData, joint: str) -> tuple:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, joint)
    assert index > -1, f"Joint {joint} not found in model"
    return torch.tensor(data.xpos[index].copy(), dtype=torch.float32)


def make_base_reward(body: str, axis: str, target: float) -> Callable:
    """Create a reward function for a single sensor using sigmoid activation.

    Reward is computed as sigmoid of the negated distance between current value and target.
    This provides smooth activation based on proximity to the target position.
    """
    mujoco_body = BODY_NAMES[body]
    axis_idx = AXIS_INDEX[axis]

    def reward_fn(model: mujoco.MjModel, data: mujoco.MjData) -> float:
        value = get_joint_position(model, data, mujoco_body)[axis_idx]
        distance = torch.abs(value - target)
        return torch.sigmoid(-distance)

    return reward_fn


class SensorReward:
    """Callable reward that can evaluate motion states."""

    def __init__(self, reward_fns: list[Callable]):
        self.reward_fns = reward_fns

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        """Compute combined reward (product of all predicates)."""
        return reduce(lambda acc, fn: acc * fn(model, data), self.reward_fns, 1.0)

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
        result = self.compute(model, data)
        return torch.as_tensor(result, device=qpos.device, dtype=qpos.dtype)


@dataclass
class MotionReward:
    """Reward for a motion interval."""

    start: int
    end: int
    sensor_reward: SensorReward


@dataclass
class Reward:
    """Reward for ExAct program."""

    motions: list[MotionReward]

    def get_reward_fn(self, timestep: int) -> SensorReward | None:
        for motion in self.motions:
            if motion.start <= timestep < motion.end:
                return motion.sensor_reward
        return None


class ProgramTransformer(Transformer):
    """Transform parsed tree into reward objects."""

    def sensor(self, children):
        # children: [JOINT token, AXIS token, VALUE token]
        joint, axis, value = children
        return (str(joint), str(axis), float(value))

    def motion(self, children):
        # children: [FRAME, FRAME, sensor tuples...]
        start, end, *sensors = children
        reward_fns = [make_base_reward(j, a, v) for j, a, v in sensors]
        return MotionReward(int(start), int(end), SensorReward(reward_fns))

    def start(self, children):
        return Reward(list(children))


def parse_program(program: str, grm: str = "grammar.lark") -> Reward:
    """Parse a program string into a Reward object."""
    with open(os.path.join(os.path.dirname(__file__), grm), "r") as file:
        grammar = file.read()
    parser = Lark(grammar, start="start", parser="earley")
    tree = parser.parse(program)

    return ProgramTransformer().transform(tree)
