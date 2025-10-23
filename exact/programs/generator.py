import random
from typing import List, Optional
from dataclasses import dataclass, field

# Define the available body parts and their corresponding reward classes
BODY_PARTS = [
    "lhip", "lknee", "lankle", "ltoe",
    "rhip", "rknee", "rankle", "rtoe",
    "torso", "spine", "chest", "neck", "head",
    "lthorax", "lshoulder", "lelbow", "lwrist", "lhand",
    "rthorax", "rshoulder", "relbow", "rwrist", "rhand"
]

@dataclass
class GenerationConfig:
    """Configuration for reward program generation.
    
    Attributes:
        min_units: Minimum number of units in the program
        max_units: Maximum number of units in the program
        min_value: Minimum value for the reward (can be negative)
        max_value: Maximum value for the reward
        value_step: Step size for generating values
        allowed_parts: List of allowed body parts. If None, all parts are allowed.
    """
    min_units: int = 1
    max_units: int = 5
    min_value: float = -1.0  # Allow negative values by default
    max_value: float = 1.0
    value_step: float = 0.1
    allowed_parts: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.allowed_parts is None:
            self.allowed_parts = BODY_PARTS
        else:
            # Validate that all specified parts are valid
            invalid_parts = [p for p in self.allowed_parts if p not in BODY_PARTS]
            if invalid_parts:
                raise ValueError(f"Invalid body parts specified: {invalid_parts}")

class RewardProgramGenerator:
    """Generates reward programs based on the specified grammar and configuration."""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config if config is not None else GenerationConfig()
    
    def generate_program(self) -> str:
        """Generate a single reward program string."""
        num_units = random.randint(self.config.min_units, self.config.max_units)
        program_parts = []
        
        for _ in range(num_units):
            # Select a random body part
            part = random.choice(self.config.allowed_parts)
            
            # Generate a random value within the specified range and step
            value = self._generate_random_value()
            
            # Format the unit as "part(value)" and handle negative numbers
            sign = '' if value >= 0 else ''  # No need for explicit + sign
            program_parts.append(f"{part}({sign}{value:.1f})")
        
        # Join units with "*" as per the grammar
        return "*".join(program_parts)
    
    def _generate_random_value(self) -> float:
        """Generate a random value within the configured range and step."""
        # Allow negative values by extending the range if min_value is negative
        min_val = int(self.config.min_value / self.config.value_step)
        max_val = int(self.config.max_value / self.config.value_step)
        return random.randint(min_val, max_val) * self.config.value_step

def generate_programs(
    num_programs: int = 10,
    min_units: int = 1,
    max_units: int = 5,
    min_value: float = 0.1,
    max_value: float = 1.0,
    value_step: float = 0.1,
    allowed_parts: Optional[List[str]] = None
) -> List[str]:
    """
    Generate multiple reward programs with the specified parameters.
    
    Args:
        num_programs: Number of programs to generate
        min_units: Minimum number of units in each program
        max_units: Maximum number of units in each program
        min_value: Minimum value for reward weights
        max_value: Maximum value for reward weights
        value_step: Step size for reward weights
        allowed_parts: List of allowed body parts to use in programs
        
    Returns:
        List of generated program strings
    """
    config = GenerationConfig(
        min_units=min_units,
        max_units=max_units,
        min_value=min_value,
        max_value=max_value,
        value_step=value_step,
        allowed_parts=allowed_parts
    )
    
    generator = RewardProgramGenerator(config)
    return [generator.generate_program() for _ in range(num_programs)]

def generate_programs_with_config(
    config: GenerationConfig,
    num_programs: int = 10
) -> List[str]:
    """Generate multiple reward programs using a pre-configured GenerationConfig."""
    generator = RewardProgramGenerator(config)
    return [generator.generate_program() for _ in range(num_programs)]
