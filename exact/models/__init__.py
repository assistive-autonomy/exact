"""Executable Behaviour Representations (EBR) module.

This module provides classes for aggregating multiple ExAct programs, by action
label, into executable behaviour representations using logical disjunction. (The
class name ``ExecutableActivityModel`` is retained for backwards compatibility.)
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
