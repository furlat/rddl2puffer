"""Schema objects for grounded fluents and flat layouts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from math import prod
from typing import Iterable, Mapping, TypeAlias

Scalar: TypeAlias = bool | int | float


class FluentRole(StrEnum):
    """Logical role for a grounded fluent."""

    STATE = "state"
    NEXT_STATE = "next_state"
    ACTION = "action"
    OBSERVATION = "observation"
    INTERMEDIATE = "intermediate"
    REWARD = "reward"
    TERMINAL = "terminal"


class FluentDType(StrEnum):
    """Coarse data type for a grounded fluent."""

    BOOL = "bool"
    INT = "int"
    REAL = "real"


ROLE_PRIORITY = {
    FluentRole.STATE: 0,
    FluentRole.NEXT_STATE: 1,
    FluentRole.ACTION: 2,
    FluentRole.OBSERVATION: 3,
    FluentRole.INTERMEDIATE: 4,
    FluentRole.REWARD: 5,
    FluentRole.TERMINAL: 6,
}


@dataclass(frozen=True, slots=True)
class FluentSpec:
    """Metadata for a single grounded fluent."""

    name: str
    role: FluentRole
    dtype: FluentDType
    parameters: tuple[str, ...] = ()
    shape: tuple[int, ...] = ()
    bounds: tuple[Scalar | None, Scalar | None] = (None, None)
    default: Scalar = 0
    flat_index: int | None = None

    @property
    def qualified_name(self) -> str:
        if not self.parameters:
            return self.name
        params = ",".join(self.parameters)
        return f"{self.name}[{params}]"

    @property
    def size(self) -> int:
        return prod(self.shape) if self.shape else 1

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "role": self.role.value,
            "dtype": self.dtype.value,
            "parameters": list(self.parameters),
            "shape": list(self.shape),
            "bounds": list(self.bounds),
            "default": self.default,
            "flat_index": self.flat_index,
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class LayoutSpec:
    """Fixed contiguous layout for a family of grounded fluents."""

    name: str
    fluents: tuple[FluentSpec, ...]

    @property
    def total_size(self) -> int:
        return sum(fluent.size for fluent in self.fluents)

    @property
    def offsets(self) -> dict[str, int]:
        return {
            fluent.qualified_name: fluent.flat_index if fluent.flat_index is not None else -1
            for fluent in self.fluents
        }

    def index_of(self, qualified_name: str) -> int:
        try:
            return self.offsets[qualified_name]
        except KeyError as exc:
            raise KeyError(f"Unknown fluent in layout {self.name!r}: {qualified_name}") from exc

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "total_size": self.total_size,
            "fluents": [fluent.to_debug_dict() for fluent in self.fluents],
            "offsets": self.offsets,
        }


@dataclass(frozen=True, slots=True)
class EnvSchema:
    """Canonical flat layouts for a grounded single-agent environment."""

    state_layout: LayoutSpec
    action_layout: LayoutSpec
    observation_layout: LayoutSpec
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "state_layout": self.state_layout.to_debug_dict(),
            "action_layout": self.action_layout.to_debug_dict(),
            "observation_layout": self.observation_layout.to_debug_dict(),
            "metadata": dict(self.metadata),
        }


def canonical_fluent_key(spec: FluentSpec) -> tuple[object, ...]:
    """Stable sort key for grounded fluents.

    The rule is intentionally simple and deterministic:
    role priority, then base name, then grounding parameters, then dtype, then shape.
    """

    return (
        ROLE_PRIORITY[spec.role],
        spec.name,
        spec.parameters,
        spec.dtype.value,
        spec.shape,
    )


def canonicalize_fluents(fluents: Iterable[FluentSpec]) -> tuple[FluentSpec, ...]:
    """Return a stable, deterministic canonical ordering for grounded fluents."""

    return tuple(sorted(fluents, key=canonical_fluent_key))


def build_layout(name: str, fluents: Iterable[FluentSpec]) -> LayoutSpec:
    """Build a flat layout that preserves source order and contiguous offsets."""

    ordered = tuple(fluents)
    next_offset = 0
    indexed: list[FluentSpec] = []
    for fluent in ordered:
        indexed.append(replace(fluent, flat_index=next_offset))
        next_offset += fluent.size
    return LayoutSpec(name=name, fluents=tuple(indexed))
