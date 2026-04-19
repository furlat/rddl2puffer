"""Typed instruction nodes for grounded CPF execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping

from rddl2puffer.frontend.schema import LayoutSpec, Scalar


class NodeOp(StrEnum):
    """Small deterministic instruction set for v1 IR execution."""

    CONST = "const"
    LOAD_STATE = "load_state"
    LOAD_ACTION = "load_action"
    NEG = "neg"
    NOT = "not"
    SIN = "sin"
    COS = "cos"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    MIN = "min"
    MAX = "max"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    EQ = "eq"
    NE = "ne"
    AND = "and"
    OR = "or"
    SELECT = "select"
    SAMPLE = "sample"
    STORE_NEXT_STATE = "store_next_state"
    STORE_OBS = "store_obs"
    STORE_REWARD = "store_reward"
    STORE_DONE = "store_done"


@dataclass(frozen=True, slots=True)
class CPFNode:
    """Single IR instruction or store operation.

    `args` references prior node ids, which makes the program easy to validate
    and straightforward to interpret in order.
    """

    node_id: str
    op: NodeOp
    args: tuple[str, ...] = ()
    value: Scalar | None = None
    slot: int | None = None
    comment: str | None = None

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "op": self.op.value,
            "args": list(self.args),
            "value": self.value,
            "slot": self.slot,
            "comment": self.comment,
        }


@dataclass(frozen=True, slots=True)
class IRProgram:
    """Fully ordered executable IR for one grounded environment instance."""

    name: str
    state_layout: LayoutSpec
    action_layout: LayoutSpec
    observation_layout: LayoutSpec
    nodes: tuple[CPFNode, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "state_layout": self.state_layout.to_debug_dict(),
            "action_layout": self.action_layout.to_debug_dict(),
            "observation_layout": self.observation_layout.to_debug_dict(),
            "metadata": dict(self.metadata),
            "nodes": [node.to_debug_dict() for node in self.nodes],
        }
