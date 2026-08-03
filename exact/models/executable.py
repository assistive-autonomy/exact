"""Executable Behaviour Representations (EBR).

This module implements the core logic for aggregating multiple ExAct motion
programs, sharing an action label, into a single executable behaviour
representation using logical disjunction. (The class name
``ExecutableActivityModel`` is retained for backwards compatibility.)

Given N programs p_1, p_2, ..., p_N for the same action, we combine them
into a single reward function using the formula:

    r = 1 - prod_{i=1}^{N} (1 - p_i(state))

This is a "soft OR" - if any program gives high reward, the combined reward
is high. This enables:
- Data augmentation for activity segmentation
- Compositional generalization for anomaly detection
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import mujoco

from exact.programs import Reward, SensorReward, MotionReward, parse_program


@dataclass
class NormalizedProgram:
    """A program with normalized temporal intervals.
    
    Stores both the original program and a version normalized to a
    common evaluation window [0, eval_timesteps].
    """
    original_program: str
    original_reward: Reward
    normalized_reward: Reward
    original_duration: int
    scale_factor: float  # eval_timesteps / original_duration
    
    @classmethod
    def from_program(
        cls,
        program: str,
        eval_timesteps: int,
    ) -> "NormalizedProgram":
        """Create a normalized program from a program string.
        
        Args:
            program: The program string to normalize
            eval_timesteps: The target evaluation window duration
            
        Returns:
            NormalizedProgram with scaled temporal intervals
        """
        original_reward = parse_program(program)
        original_duration = cls._get_program_duration(original_reward)
        
        if original_duration == 0:
            raise ValueError(f"Program has zero duration: {program}")
        
        scale_factor = eval_timesteps / original_duration
        normalized_reward = cls._scale_reward(original_reward, scale_factor)
        
        return cls(
            original_program=program,
            original_reward=original_reward,
            normalized_reward=normalized_reward,
            original_duration=original_duration,
            scale_factor=scale_factor,
        )
    
    @staticmethod
    def _get_program_duration(reward: Reward) -> int:
        """Get the total duration (max end timestep) of a program."""
        if not reward.motions:
            return 0
        return max(motion.end for motion in reward.motions)
    
    @staticmethod
    def _scale_reward(reward: Reward, scale_factor: float) -> Reward:
        """Scale all temporal intervals in a reward by the given factor.
        
        Args:
            reward: The original Reward object
            scale_factor: Factor to multiply all timesteps by
            
        Returns:
            New Reward with scaled intervals
        """
        scaled_motions = []
        for motion in reward.motions:
            scaled_start = int(round(motion.start * scale_factor))
            scaled_end = int(round(motion.end * scale_factor))
            scaled_motions.append(MotionReward(
                start=scaled_start,
                end=scaled_end,
                sensor_reward=motion.sensor_reward,
            ))
        
        return Reward(motions=scaled_motions)
    
    def get_reward_at_timestep(
        self,
        timestep: int,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        """Get the reward for this program at a given timestep.
        
        Uses the normalized temporal intervals for evaluation.
        Returns 0.0 if no motion interval is active at this timestep.
        
        Args:
            timestep: The timestep to evaluate (in normalized time)
            model: MuJoCo model
            data: MuJoCo data with current state
            
        Returns:
            Reward value in [0, 1]
        """
        sensor_reward = self.normalized_reward.get_reward_fn(timestep)
        if sensor_reward is None:
            return 0.0
        return float(sensor_reward.compute(model, data))


@dataclass
class ExecutableActivityModel:
    """Executable model combining multiple programs for a single activity.
    
    This class implements the disjunctive combination of activity programs:
    
        r = 1 - prod_{i=1}^{N} (1 - p_i(state))
    
    Each program represents a different execution of the same activity,
    and the combination captures the variability in how the activity
    can be performed.
    
    Attributes:
        activity_name: Name/label of the activity this model represents
        programs: List of normalized programs
        eval_timesteps: Common evaluation window duration
        original_programs: List of original program strings (for reference)
        metadata: Optional additional metadata
    """
    activity_name: str
    programs: list[NormalizedProgram] = field(default_factory=list)
    eval_timesteps: int = 100
    original_programs: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the model after initialization."""
        if self.programs and not self.original_programs:
            # Extract original programs if not provided
            self.original_programs = [p.original_program for p in self.programs]
    
    @property
    def num_programs(self) -> int:
        """Number of programs in this model."""
        return len(self.programs)
    
    def add_program(self, program: str) -> None:
        """Add a program to this executable model.
        
        The program will be parsed and normalized to the common
        evaluation window.
        
        Args:
            program: Program string to add
        """
        normalized = NormalizedProgram.from_program(program, self.eval_timesteps)
        self.programs.append(normalized)
        self.original_programs.append(program)
    
    def add_programs(self, programs: list[str]) -> None:
        """Add multiple programs to this executable model.
        
        Args:
            programs: List of program strings to add
        """
        for program in programs:
            self.add_program(program)
    
    def compute_reward(
        self,
        timestep: int,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        """Compute the disjunctive reward at a given timestep.
        
        Implements: r = 1 - prod_{i=1}^{N} (1 - p_i(state))
        
        Args:
            timestep: The timestep to evaluate (in normalized time [0, eval_timesteps])
            model: MuJoCo model
            data: MuJoCo data with current state
            
        Returns:
            Combined reward value in [0, 1]
        """
        if not self.programs:
            return 0.0
        
        product = 1.0
        for program in self.programs:
            r_i = program.get_reward_at_timestep(timestep, model, data)
            product *= (1.0 - r_i)
        
        return 1.0 - product
    
    def compute_reward_breakdown(
        self,
        timestep: int,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> dict:
        """Compute reward with per-program breakdown for analysis.
        
        Args:
            timestep: The timestep to evaluate
            model: MuJoCo model
            data: MuJoCo data with current state
            
        Returns:
            Dictionary with combined reward and individual program rewards
        """
        individual_rewards = []
        product = 1.0
        
        for i, program in enumerate(self.programs):
            r_i = program.get_reward_at_timestep(timestep, model, data)
            individual_rewards.append({
                "program_idx": i,
                "reward": r_i,
                "original_program": program.original_program[:50] + "...",
            })
            product *= (1.0 - r_i)
        
        return {
            "combined_reward": 1.0 - product,
            "individual_rewards": individual_rewards,
            "timestep": timestep,
            "activity": self.activity_name,
        }
    
    def get_reward_fn(self, timestep: int) -> Optional[Callable]:
        """Get a reward function for a specific timestep.
        
        This provides compatibility with the Reward class interface.
        
        Args:
            timestep: The timestep to get the reward function for
            
        Returns:
            A callable that takes (model, data) and returns the reward,
            or None if no programs are loaded.
        """
        if not self.programs:
            return None
        
        def reward_fn(model: mujoco.MjModel, data: mujoco.MjData) -> float:
            return self.compute_reward(timestep, model, data)
        
        return reward_fn
    
    def __call__(
        self,
        model: mujoco.MjModel,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        ctrl: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """Evaluate the model on a state (compatible with SensorReward interface).
        
        Args:
            model: MuJoCo model
            qpos: Position state tensor
            qvel: Velocity state tensor
            ctrl: Control tensor
            timestep: Current timestep (in normalized time)
            
        Returns:
            Reward tensor
        """
        data = mujoco.MjData(model)
        data.qpos[:] = qpos.detach().cpu().numpy()
        data.qvel[:] = qvel.detach().cpu().numpy()
        data.ctrl[:] = ctrl.detach().cpu().numpy()
        mujoco.mj_forward(model, data)
        
        result = self.compute_reward(timestep, model, data)
        return torch.as_tensor(result, device=qpos.device, dtype=qpos.dtype)
    
    def to_dict(self) -> dict:
        """Serialize the model to a dictionary.
        
        Returns:
            Dictionary representation of the model
        """
        return {
            "activity_name": self.activity_name,
            "eval_timesteps": self.eval_timesteps,
            "original_programs": self.original_programs,
            "metadata": self.metadata,
            "num_programs": self.num_programs,
            "program_details": [
                {
                    "original_program": p.original_program,
                    "original_duration": p.original_duration,
                    "scale_factor": p.scale_factor,
                }
                for p in self.programs
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExecutableActivityModel":
        """Deserialize a model from a dictionary.
        
        Args:
            data: Dictionary representation of the model
            
        Returns:
            Reconstructed ExecutableActivityModel
        """
        model = cls(
            activity_name=data["activity_name"],
            eval_timesteps=data["eval_timesteps"],
            metadata=data.get("metadata", {}),
        )
        model.add_programs(data["original_programs"])
        return model
    
    def save(self, path: str) -> None:
        """Save the model to a JSON file.
        
        Args:
            path: Path to save the model to
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ExecutableActivityModel":
        """Load a model from a JSON file.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded ExecutableActivityModel
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        return (
            f"ExecutableActivityModel("
            f"activity='{self.activity_name}', "
            f"num_programs={self.num_programs}, "
            f"eval_timesteps={self.eval_timesteps})"
        )
    
    @classmethod
    def from_programs(
        cls,
        programs: list[str],
        activity_name: str,
        eval_timesteps: int = 100,
        metadata: Optional[dict] = None,
    ) -> "ExecutableActivityModel":
        """Create an ExecutableActivityModel from a list of programs.
        
        This is a convenience class method that creates a model and adds
        all programs in one call.
        
        Args:
            programs: List of program strings for this activity
            activity_name: Name of the activity
            eval_timesteps: Common evaluation window duration
            metadata: Optional metadata dictionary
            
        Returns:
            ExecutableActivityModel with all programs added
            
        Example:
            >>> model = ExecutableActivityModel.from_programs(
            ...     programs=["[0,50]head.y(1.6)", "[0,30]lhand.x(0.3)"],
            ...     activity_name="Grab",
            ... )
        """
        model = cls(
            activity_name=activity_name,
            eval_timesteps=eval_timesteps,
            metadata=metadata or {},
        )
        model.add_programs(programs)
        return model


def create_executable_model(
    activity_name: str,
    programs: list[str],
    eval_timesteps: int = 100,
    metadata: Optional[dict] = None,
) -> ExecutableActivityModel:
    """Factory function to create an ExecutableActivityModel.
    
    This is a convenience function for creating models with programs
    already added.
    
    Args:
        activity_name: Name of the activity
        programs: List of program strings for this activity
        eval_timesteps: Common evaluation window duration
        metadata: Optional metadata dictionary
        
    Returns:
        ExecutableActivityModel with all programs added
        
    Example:
        >>> model = create_executable_model(
        ...     activity_name="walking",
        ...     programs=[
        ...         "[0,20]head.y(1.6)*pelvis.y(0.9);[20,40]lknee.z(0.5)",
        ...         "[0,30]head.y(1.65);[30,50]rknee.z(0.45)",
        ...     ],
        ...     eval_timesteps=100,
        ... )
        >>> model.num_programs
        2
    """
    model = ExecutableActivityModel(
        activity_name=activity_name,
        eval_timesteps=eval_timesteps,
        metadata=metadata or {},
    )
    model.add_programs(programs)
    return model


@dataclass
class ActivityModelCollection:
    """Collection of ExecutableActivityModels for multiple activities.
    
    This class manages a set of activity models, one per activity class.
    Useful for segmentation and anomaly detection tasks where you need to
    evaluate against multiple activity types.
    
    Attributes:
        models: Dictionary mapping activity names to their models
        eval_timesteps: Common evaluation window for all models
    """
    models: dict[str, ExecutableActivityModel] = field(default_factory=dict)
    eval_timesteps: int = 100
    
    @property
    def activity_names(self) -> list[str]:
        """List of activity names in this collection."""
        return list(self.models.keys())
    
    @property
    def num_activities(self) -> int:
        """Number of activities in this collection."""
        return len(self.models)
    
    def add_model(self, model: ExecutableActivityModel) -> None:
        """Add an activity model to the collection.
        
        Args:
            model: ExecutableActivityModel to add
        """
        if model.eval_timesteps != self.eval_timesteps:
            raise ValueError(
                f"Model eval_timesteps ({model.eval_timesteps}) does not match "
                f"collection eval_timesteps ({self.eval_timesteps})"
            )
        self.models[model.activity_name] = model
    
    def create_and_add_model(
        self,
        activity_name: str,
        programs: list[str],
        metadata: Optional[dict] = None,
    ) -> ExecutableActivityModel:
        """Create a new model and add it to the collection.
        
        Args:
            activity_name: Name of the activity
            programs: List of program strings for this activity
            metadata: Optional metadata dictionary
            
        Returns:
            The created ExecutableActivityModel
        """
        model = create_executable_model(
            activity_name=activity_name,
            programs=programs,
            eval_timesteps=self.eval_timesteps,
            metadata=metadata,
        )
        self.add_model(model)
        return model
    
    def get_model(self, activity_name: str) -> Optional[ExecutableActivityModel]:
        """Get the model for an activity.
        
        Args:
            activity_name: Name of the activity
            
        Returns:
            ExecutableActivityModel or None if not found
        """
        return self.models.get(activity_name)
    
    def compute_all_rewards(
        self,
        timestep: int,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> dict[str, float]:
        """Compute rewards for all activities at a given timestep.
        
        Args:
            timestep: The timestep to evaluate
            model: MuJoCo model
            data: MuJoCo data with current state
            
        Returns:
            Dictionary mapping activity names to rewards
        """
        return {
            name: activity_model.compute_reward(timestep, model, data)
            for name, activity_model in self.models.items()
        }
    
    def classify_activity(
        self,
        timestep: int,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> tuple[str, float]:
        """Classify the current state to the most likely activity.
        
        Args:
            timestep: The timestep to evaluate
            model: MuJoCo model
            data: MuJoCo data with current state
            
        Returns:
            Tuple of (activity_name, reward) for the best matching activity
        """
        rewards = self.compute_all_rewards(timestep, model, data)
        if not rewards:
            return ("unknown", 0.0)
        best_activity = max(rewards, key=rewards.get)
        return (best_activity, rewards[best_activity])
    
    def to_dict(self) -> dict:
        """Serialize the collection to a dictionary.
        
        Returns:
            Dictionary representation of the collection
        """
        return {
            "eval_timesteps": self.eval_timesteps,
            "models": {
                name: model.to_dict()
                for name, model in self.models.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ActivityModelCollection":
        """Deserialize a collection from a dictionary.
        
        Args:
            data: Dictionary representation of the collection
            
        Returns:
            Reconstructed ActivityModelCollection
        """
        collection = cls(eval_timesteps=data["eval_timesteps"])
        for name, model_data in data["models"].items():
            model = ExecutableActivityModel.from_dict(model_data)
            collection.add_model(model)
        return collection
    
    def save(self, path: str) -> None:
        """Save the collection to a JSON file.
        
        Args:
            path: Path to save the collection to
        """
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ActivityModelCollection":
        """Load a collection from a JSON file.
        
        Args:
            path: Path to load the collection from
            
        Returns:
            Loaded ActivityModelCollection
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        return (
            f"ActivityModelCollection("
            f"activities={self.activity_names}, "
            f"eval_timesteps={self.eval_timesteps})"
        )
