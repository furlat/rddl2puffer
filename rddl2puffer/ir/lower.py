"""Helpers for assembling IR programs from future grounded frontends."""

from __future__ import annotations

from typing import Iterable, Mapping

from rddl2puffer.frontend.schema import EnvSchema
from rddl2puffer.ir.nodes import CPFNode, IRProgram
from rddl2puffer.ir.validate import validate_program


def assemble_program(
    name: str,
    schema: EnvSchema,
    nodes: Iterable[CPFNode],
    metadata: Mapping[str, object] | None = None,
) -> IRProgram:
    """Build and validate an ordered IR program from already-lowered nodes."""

    program = IRProgram(
        name=name,
        state_layout=schema.state_layout,
        action_layout=schema.action_layout,
        observation_layout=schema.observation_layout,
        nodes=tuple(nodes),
        metadata=dict(metadata or schema.metadata),
    )
    validate_program(program, allow_sample_nodes=True)
    return program


def lower_grounded_cpfs(*_args: object, **_kwargs: object) -> IRProgram:
    """Placeholder for the real RDDL-to-IR lowering pass."""

    raise NotImplementedError(
        "Grounded CPF lowering has not been wired in yet. "
        "Use assemble_program() with structured mock grounded data for now."
    )

