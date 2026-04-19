"""Differential-testing utilities for comparing runtime behavior."""

from rddl2puffer.testing.differential import (
    Mismatch,
    MismatchReport,
    ReferenceEnv,
    ResetResult,
    RolloutTrace,
    TraceStep,
    Transition,
)
from rddl2puffer.testing.rollout_compare import compare_rollouts, format_mismatch_report, rollout

__all__ = [
    "Mismatch",
    "MismatchReport",
    "ReferenceEnv",
    "ResetResult",
    "RolloutTrace",
    "TraceStep",
    "Transition",
    "compare_rollouts",
    "format_mismatch_report",
    "rollout",
]

