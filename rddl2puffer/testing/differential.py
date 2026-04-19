"""Shared types for differential rollout comparison."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence

from rddl2puffer.frontend.schema import Scalar


@dataclass(frozen=True, slots=True)
class ResetResult:
    """Environment output produced by `reset(seed)`."""

    observation: tuple[Scalar, ...]
    hidden_state: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class Transition:
    """Environment output produced by `step(action)`."""

    observation: tuple[Scalar, ...]
    reward: float
    done: bool
    hidden_state: Mapping[str, object] | None = None


class ReferenceEnv(ABC):
    """Abstract runtime used for differential testing."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> ResetResult:
        """Reset the environment and return the initial observation."""

    @abstractmethod
    def step(self, action: Sequence[Scalar]) -> Transition:
        """Advance the environment by one action."""


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One action/transition pair in a rollout trace."""

    step_index: int
    action: tuple[Scalar, ...]
    transition: Transition


@dataclass(frozen=True, slots=True)
class RolloutTrace:
    """Recorded reset and step outputs for one seeded rollout."""

    reset: ResetResult
    steps: tuple[TraceStep, ...]


@dataclass(frozen=True, slots=True)
class Mismatch:
    """Single field-level mismatch between two traces."""

    location: str
    field: str
    left: object
    right: object


@dataclass(frozen=True, slots=True)
class MismatchReport:
    """Human-readable summary of rollout parity."""

    left_name: str
    right_name: str
    mismatches: tuple[Mismatch, ...]
    left_trace: RolloutTrace
    right_trace: RolloutTrace

    @property
    def matches(self) -> bool:
        return not self.mismatches

    def to_text(self) -> str:
        if self.matches:
            return f"{self.left_name} and {self.right_name} matched exactly."

        lines = [f"Found {len(self.mismatches)} mismatch(es):"]
        for mismatch in self.mismatches:
            lines.append(
                f"- {mismatch.location} {mismatch.field}: "
                f"{self.left_name}={mismatch.left!r}, {self.right_name}={mismatch.right!r}"
            )
        return "\n".join(lines)

