"""Mock program generator for testing the augmentation pipeline.

This module provides a mock parser that generates random valid programs
conforming to the ExAct grammar. It will be replaced with the trained
neural parser once available.
"""

import random
from typing import Optional

# Body parts available in the grammar
BODY_PARTS = [
    "pelvis", "torso", "spine", "chest", "neck", "head",
    "lhip", "lknee", "lankle", "ltoe",
    "rhip", "rknee", "rankle", "rtoe",
    "lthorax", "lshoulder", "lelbow", "lwrist", "lhand",
    "rthorax", "rshoulder", "relbow", "rwrist", "rhand",
]

AXES = ["x", "y", "z"]

# Typical value ranges for different body parts (approximate)
# These are rough estimates for realistic-looking programs
BODY_PART_RANGES = {
    "head": {"y": (1.5, 1.8), "x": (-0.3, 0.3), "z": (-0.3, 0.3)},
    "pelvis": {"y": (0.7, 1.0), "x": (-0.2, 0.2), "z": (-0.2, 0.2)},
    "chest": {"y": (1.2, 1.5), "x": (-0.2, 0.2), "z": (-0.2, 0.2)},
    "lhand": {"y": (0.3, 1.5), "x": (-0.5, 0.5), "z": (-0.5, 0.5)},
    "rhand": {"y": (0.3, 1.5), "x": (-0.5, 0.5), "z": (-0.5, 0.5)},
    "lknee": {"y": (0.3, 0.6), "x": (-0.3, 0.3), "z": (-0.3, 0.3)},
    "rknee": {"y": (0.3, 0.6), "x": (-0.3, 0.3), "z": (-0.3, 0.3)},
    # Default range for other parts
    "_default": {"x": (-0.5, 0.5), "y": (0.2, 1.5), "z": (-0.5, 0.5)},
}


def get_value_range(body_part: str, axis: str) -> tuple[float, float]:
    """Get appropriate value range for a body part and axis."""
    if body_part in BODY_PART_RANGES:
        ranges = BODY_PART_RANGES[body_part]
        if axis in ranges:
            return ranges[axis]
    return BODY_PART_RANGES["_default"][axis]


def generate_sensor(
    body_part: Optional[str] = None,
    axis: Optional[str] = None,
    value: Optional[float] = None,
) -> str:
    """Generate a single sensor predicate.
    
    Format: body.axis(value)
    Example: head.y(1.65)
    """
    if body_part is None:
        body_part = random.choice(BODY_PARTS)
    if axis is None:
        axis = random.choice(AXES)
    if value is None:
        low, high = get_value_range(body_part, axis)
        value = round(random.uniform(low, high), 2)
    
    return f"{body_part}.{axis}({value:.2f})"


def generate_motion(
    num_sensors: Optional[int] = None,
    min_sensors: int = 1,
    max_sensors: int = 4,
) -> str:
    """Generate a motion string with multiple sensor predicates.
    
    Format: sensor*sensor*...
    Example: head.y(1.65)*rhand.z(0.45)
    """
    if num_sensors is None:
        num_sensors = random.randint(min_sensors, max_sensors)
    
    sensors = [generate_sensor() for _ in range(num_sensors)]
    return "*".join(sensors)


def generate_mock_program(
    duration: int = 100,
    num_intervals: Optional[int] = None,
    min_intervals: int = 1,
    max_intervals: int = 4,
    min_sensors: int = 1,
    max_sensors: int = 4,
) -> str:
    """Generate a random valid program conforming to ExAct grammar.
    
    Format: [start,end]motion;[start,end]motion;...
    Example: [0,50]head.y(1.65)*rhand.z(0.45);[50,100]pelvis.y(0.85)
    
    Args:
        duration: Total duration of the program in timesteps
        num_intervals: Number of motion intervals (random if None)
        min_intervals: Minimum number of intervals
        max_intervals: Maximum number of intervals
        min_sensors: Minimum sensors per motion
        max_sensors: Maximum sensors per motion
        
    Returns:
        Program string conforming to ExAct grammar
    """
    if num_intervals is None:
        num_intervals = random.randint(min_intervals, max_intervals)
    
    # Generate interval boundaries
    if num_intervals == 1:
        boundaries = [0, duration]
    else:
        # Generate random split points
        inner_points = sorted(random.sample(range(1, duration), num_intervals - 1))
        boundaries = [0] + inner_points + [duration]
    
    # Generate motion for each interval
    intervals = []
    for i in range(num_intervals):
        start = boundaries[i]
        end = boundaries[i + 1]
        motion = generate_motion(min_sensors=min_sensors, max_sensors=max_sensors)
        intervals.append(f"[{start},{end}]{motion}")
    
    return ";".join(intervals)


def generate_programs_for_activity(
    activity_name: str,
    num_programs: int,
    duration_range: tuple[int, int] = (50, 150),
    seed: Optional[int] = None,
) -> list[str]:
    """Generate multiple mock programs for an activity.
    
    In the real implementation, these would come from the neural parser
    applied to motion segments of this activity class.
    
    Args:
        activity_name: Name of the activity (used for seeding consistency)
        num_programs: Number of programs to generate
        duration_range: Range of durations (min, max)
        seed: Random seed for reproducibility
        
    Returns:
        List of program strings
    """
    if seed is not None:
        # Create activity-specific seed for reproducibility
        activity_seed = hash(activity_name) + seed
        random.seed(activity_seed)
    
    programs = []
    for _ in range(num_programs):
        duration = random.randint(*duration_range)
        program = generate_mock_program(duration=duration)
        programs.append(program)
    
    return programs


class MockParser:
    """Mock parser that generates random programs from motion segments.
    
    This class simulates the neural parser interface. Given a motion segment,
    it generates a random valid program. This allows testing the augmentation
    pipeline before the real parser is trained.
    
    In the real implementation, this would be replaced by:
        parser = CrossAttentionParser(...)
        program = parser.generate(motion_segment)
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        min_intervals: int = 1,
        max_intervals: int = 4,
        min_sensors: int = 1,
        max_sensors: int = 4,
    ):
        """Initialize mock parser.
        
        Args:
            seed: Random seed for reproducibility
            min_intervals: Minimum motion intervals per program
            max_intervals: Maximum motion intervals per program
            min_sensors: Minimum sensors per motion
            max_sensors: Maximum sensors per motion
        """
        self.seed = seed
        self.min_intervals = min_intervals
        self.max_intervals = max_intervals
        self.min_sensors = min_sensors
        self.max_sensors = max_sensors
        
        if seed is not None:
            random.seed(seed)
    
    def parse(
        self,
        motion: "torch.Tensor | None" = None,
        duration: Optional[int] = None,
    ) -> str:
        """Parse a motion segment into a program.
        
        Args:
            motion: Motion tensor (ignored in mock, used for duration)
            duration: Program duration (uses motion length if not provided)
            
        Returns:
            Program string
        """
        if duration is None:
            if motion is not None:
                duration = motion.shape[0] if motion.ndim > 1 else 100
            else:
                duration = random.randint(50, 150)
        
        return generate_mock_program(
            duration=duration,
            min_intervals=self.min_intervals,
            max_intervals=self.max_intervals,
            min_sensors=self.min_sensors,
            max_sensors=self.max_sensors,
        )
    
    def parse_batch(
        self,
        motions: list,
        durations: Optional[list[int]] = None,
    ) -> list[str]:
        """Parse multiple motion segments.
        
        Args:
            motions: List of motion tensors
            durations: List of durations (optional)
            
        Returns:
            List of program strings
        """
        if durations is None:
            durations = [None] * len(motions)
        
        return [
            self.parse(motion=m, duration=d)
            for m, d in zip(motions, durations)
        ]
