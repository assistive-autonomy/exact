from .rewards import Reward, RewardBuilder
from .generator import generate_program

def generate_programs(num_programs: int, **kwargs) -> list[str]:
    """Generate multiple programs using the functional generator."""
    return [generate_program(**kwargs) for _ in range(num_programs)]

__all__ = [
    "Reward",
    "RewardBuilder",
    "generate_program",
    "generate_programs",
]
