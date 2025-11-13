import random
from typing import List, Optional

BODY_PARTS = [
    "lhip",
    "lknee",
    "lankle",
    "ltoe",
    "rhip",
    "rknee",
    "rankle",
    "rtoe",
    "torso",
    "spine",
    "chest",
    "neck",
    "head",
    "lthorax",
    "lshoulder",
    "lelbow",
    "lwrist",
    "lhand",
    "rthorax",
    "rshoulder",
    "relbow",
    "rwrist",
    "rhand",
]


def generate_program(
    min_units: int = 1,
    max_units: int = 5,
    min_value: float = 0.1,
    max_value: float = 1.0,
    value_step: float = 0.1,
    allowed_parts: Optional[List[str]] = None,
) -> str:
    """
    Generate a single reward program string.

    Args:
        min_units: Minimum number of units in the program
        max_units: Maximum number of units in the program
        min_value: Minimum value for reward weights
        max_value: Maximum value for reward weights
        value_step: Step size for reward weights
        allowed_parts: List of allowed body parts to use. If None, all parts are allowed.

    Returns:
        Generated program string in format "part1(value1)*part2(value2)*..."
    """
    parts = allowed_parts if allowed_parts is not None else BODY_PARTS
    
    # Validate allowed parts
    if allowed_parts is not None:
        invalid_parts = [p for p in allowed_parts if p not in BODY_PARTS]
        if invalid_parts:
            raise ValueError(f"Invalid body parts specified: {invalid_parts}")
    
    # Generate random number of units
    num_units = random.randint(min_units, max_units)
    
    # Generate each unit
    program_parts = []
    for _ in range(num_units):
        # Select random body part
        part = random.choice(parts)
        
        # Generate random value within range
        min_val = int(min_value / value_step)
        max_val = int(max_value / value_step)
        value = random.randint(min_val, max_val) * value_step
        
        # Format as "part(value)"
        program_parts.append(f"{part}({value:.1f})")
    
    # Join with "*" operator
    return "*".join(program_parts)


