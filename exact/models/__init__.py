"""Executable Activity Models module.

This module provides classes for combining multiple activity programs
into unified executable models using logical disjunction.
"""

from .executable import (
    ExecutableActivityModel,
    ActivityModelCollection,
    NormalizedProgram,
    create_executable_model,
)
from .mock_parser import MockParser, generate_mock_program

__all__ = [
    "ExecutableActivityModel",
    "ActivityModelCollection",
    "NormalizedProgram",
    "create_executable_model",
    "MockParser",
    "generate_mock_program",
]
