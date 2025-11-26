import random
from typing import Optional

BODY_PARTS = [
    "pelvis", "torso", "spine", "chest", "neck", "head",
    "lhip", "lknee", "lankle", "ltoe",
    "rhip", "rknee", "rankle", "rtoe",
    "lthorax", "lshoulder", "lelbow", "lwrist", "lhand",
    "rthorax", "rshoulder", "relbow", "rwrist", "rhand",
]

AXES = ["x", "y", "z"]

def generate_motion(
    min_preds: int = 1,
    max_preds: int = 5,
    min_value: float = 0.0,
    max_value: float = 2.0,
    value_step: float = 0.1,
    allowed_parts: Optional[list[str]] = None,
    allowed_axes: Optional[list[str]] = None,
) -> str:
    """Generate a random motion string.
    
    Format: body.axis(value)*body.axis(value)*...
    Example: head.z(1.4)*lhand.x(0.5)
    """
    parts = allowed_parts or BODY_PARTS
    axes = allowed_axes or AXES
    
    num_preds = random.randint(min_preds, max_preds)
    predicates = []
    
    for _ in range(num_preds):
        body = random.choice(parts)
        axis = random.choice(axes)
        value = round(random.uniform(min_value, max_value) // value_step * value_step, 1)
        predicates.append(f"{body}.{axis}({value:.1f})")
    
    return "*".join(predicates)


def generate_program(
    min_preds: int = 1,
    max_preds: int = 5,
    min_value: float = 0.0,
    max_value: float = 2.0,
    value_step: float = 0.1,
    allowed_parts: Optional[list[str]] = None,
    allowed_axes: Optional[list[str]] = None,
    max_timesteps: int = 1000,
    num_intervals: int = 2,
    min_interval_time: int = 100,
) -> str:
    """Generate a random program string with N timestep intervals.
    
    Format: [start,end]motion,[start,end]motion,...
    Example: [0,10]head.z(1.4)*lhand.x(0.5),[10,20]pelvis.y(0.8)
    
    The intervals are non-overlapping and cover the full time from 0 to max_timesteps.
    Interval sizes vary randomly but are at least min_interval_time.
    """
    if num_intervals * min_interval_time > max_timesteps:
        raise ValueError(f"Cannot fit {num_intervals} intervals of minimum size {min_interval_time} into {max_timesteps} timesteps")
    
    intervals = []
    
    # Generate random interval sizes that sum to max_timesteps
    remaining_time = max_timesteps - (num_intervals * min_interval_time)
    sizes = [min_interval_time] * num_intervals
    
    # Distribute remaining time randomly
    for _ in range(remaining_time):
        sizes[random.randint(0, num_intervals - 1)] += 1
    
    current_time = 0
    for i in range(num_intervals):
        motion_str = generate_motion(
            min_preds=min_preds,
            max_preds=max_preds,
            min_value=min_value,
            max_value=max_value,
            value_step=value_step,
            allowed_parts=allowed_parts,
            allowed_axes=allowed_axes,
        )
        start = current_time
        end = current_time + sizes[i]
        intervals.append(f"[{start},{end}]{motion_str}")
        current_time = end
    
    return ";".join(intervals)