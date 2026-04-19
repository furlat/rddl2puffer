"""Validation for the executable IR."""

from __future__ import annotations

from rddl2puffer.ir.nodes import IRProgram, NodeOp


class IRValidationError(ValueError):
    """Raised when the ordered IR is structurally invalid."""


_ARITY = {
    NodeOp.CONST: 0,
    NodeOp.LOAD_STATE: 0,
    NodeOp.LOAD_ACTION: 0,
    NodeOp.NEG: 1,
    NodeOp.NOT: 1,
    NodeOp.SIN: 1,
    NodeOp.COS: 1,
    NodeOp.ADD: 2,
    NodeOp.SUB: 2,
    NodeOp.MUL: 2,
    NodeOp.DIV: 2,
    NodeOp.MIN: 2,
    NodeOp.MAX: 2,
    NodeOp.LT: 2,
    NodeOp.LE: 2,
    NodeOp.GT: 2,
    NodeOp.GE: 2,
    NodeOp.EQ: 2,
    NodeOp.NE: 2,
    NodeOp.AND: 2,
    NodeOp.OR: 2,
    NodeOp.SELECT: 3,
    NodeOp.SAMPLE: 2,
    NodeOp.STORE_NEXT_STATE: 1,
    NodeOp.STORE_OBS: 1,
    NodeOp.STORE_REWARD: 1,
    NodeOp.STORE_DONE: 1,
}


def validate_program(program: IRProgram, *, allow_sample_nodes: bool = False) -> None:
    """Validate node references, slots, and the basic instruction schema."""

    seen: set[str] = set()

    for node in program.nodes:
        if node.node_id in seen:
            raise IRValidationError(f"Duplicate node id: {node.node_id}")
        seen.add(node.node_id)

        expected_arity = _ARITY[node.op]
        if len(node.args) != expected_arity:
            raise IRValidationError(
                f"Node {node.node_id} expected {expected_arity} args for {node.op.value}, "
                f"got {len(node.args)}"
            )

        if node.op is NodeOp.CONST and node.value is None:
            raise IRValidationError(f"Const node {node.node_id} is missing a value")

        if node.op in {NodeOp.LOAD_STATE, NodeOp.STORE_NEXT_STATE}:
            if node.slot is None or not 0 <= node.slot < program.state_layout.total_size:
                raise IRValidationError(f"Node {node.node_id} has invalid state slot {node.slot}")

        if node.op in {NodeOp.LOAD_ACTION}:
            if node.slot is None or not 0 <= node.slot < program.action_layout.total_size:
                raise IRValidationError(f"Node {node.node_id} has invalid action slot {node.slot}")

        if node.op in {NodeOp.STORE_OBS}:
            if node.slot is None or not 0 <= node.slot < program.observation_layout.total_size:
                raise IRValidationError(f"Node {node.node_id} has invalid observation slot {node.slot}")

        if node.op is NodeOp.SAMPLE and not allow_sample_nodes:
            raise IRValidationError(
                f"Node {node.node_id} uses SAMPLE, but RNG nodes are not enabled yet"
            )

        for ref in node.args:
            if ref not in seen:
                raise IRValidationError(
                    f"Node {node.node_id} references {ref!r} before it is defined"
                )
