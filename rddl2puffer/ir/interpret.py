"""Deterministic Python interpreter for the RDDL execution IR."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Mapping, Sequence

from rddl2puffer.frontend.schema import Scalar
from rddl2puffer.ir.nodes import IRProgram, NodeOp
from rddl2puffer.ir.validate import validate_program


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Interpreter output for one environment step."""

    next_state: tuple[Scalar, ...]
    observation: tuple[Scalar, ...]
    reward: float
    done: bool
    values: Mapping[str, Scalar]

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "next_state": list(self.next_state),
            "observation": list(self.observation),
            "reward": self.reward,
            "done": self.done,
            "values": dict(self.values),
        }


def step_ir(
    program: IRProgram,
    state: Sequence[Scalar],
    action: Sequence[Scalar],
    rng: object | None = None,
) -> ExecutionResult:
    """Execute a single deterministic IR step.

    RNG is accepted as a placeholder for future stochastic nodes. The current
    interpreter raises if the program contains `sample` instructions.
    """

    validate_program(program, allow_sample_nodes=True)

    if len(state) != program.state_layout.total_size:
        raise ValueError(
            f"State length {len(state)} does not match layout size {program.state_layout.total_size}"
        )
    if len(action) != program.action_layout.total_size:
        raise ValueError(
            f"Action length {len(action)} does not match layout size {program.action_layout.total_size}"
        )

    values: dict[str, Scalar] = {}
    next_state = list(state)
    observation: list[Scalar] = [0] * program.observation_layout.total_size
    reward = 0.0
    done = False

    for node in program.nodes:
        if node.op is NodeOp.CONST:
            values[node.node_id] = node.value  # type: ignore[assignment]
        elif node.op is NodeOp.LOAD_STATE:
            values[node.node_id] = state[node.slot]  # type: ignore[index]
        elif node.op is NodeOp.LOAD_ACTION:
            values[node.node_id] = action[node.slot]  # type: ignore[index]
        elif node.op is NodeOp.NEG:
            values[node.node_id] = -_as_number(values[node.args[0]])
        elif node.op is NodeOp.NOT:
            values[node.node_id] = not bool(values[node.args[0]])
        elif node.op is NodeOp.SIN:
            values[node.node_id] = sin(_as_number(values[node.args[0]]))
        elif node.op is NodeOp.COS:
            values[node.node_id] = cos(_as_number(values[node.args[0]]))
        elif node.op is NodeOp.ADD:
            values[node.node_id] = _as_number(values[node.args[0]]) + _as_number(values[node.args[1]])
        elif node.op is NodeOp.SUB:
            values[node.node_id] = _as_number(values[node.args[0]]) - _as_number(values[node.args[1]])
        elif node.op is NodeOp.MUL:
            values[node.node_id] = _as_number(values[node.args[0]]) * _as_number(values[node.args[1]])
        elif node.op is NodeOp.DIV:
            values[node.node_id] = _as_number(values[node.args[0]]) / _as_number(values[node.args[1]])
        elif node.op is NodeOp.MIN:
            values[node.node_id] = min(_as_number(values[node.args[0]]), _as_number(values[node.args[1]]))
        elif node.op is NodeOp.MAX:
            values[node.node_id] = max(_as_number(values[node.args[0]]), _as_number(values[node.args[1]]))
        elif node.op is NodeOp.LT:
            values[node.node_id] = _as_number(values[node.args[0]]) < _as_number(values[node.args[1]])
        elif node.op is NodeOp.LE:
            values[node.node_id] = _as_number(values[node.args[0]]) <= _as_number(values[node.args[1]])
        elif node.op is NodeOp.GT:
            values[node.node_id] = _as_number(values[node.args[0]]) > _as_number(values[node.args[1]])
        elif node.op is NodeOp.GE:
            values[node.node_id] = _as_number(values[node.args[0]]) >= _as_number(values[node.args[1]])
        elif node.op is NodeOp.EQ:
            values[node.node_id] = values[node.args[0]] == values[node.args[1]]
        elif node.op is NodeOp.NE:
            values[node.node_id] = values[node.args[0]] != values[node.args[1]]
        elif node.op is NodeOp.AND:
            values[node.node_id] = bool(values[node.args[0]]) and bool(values[node.args[1]])
        elif node.op is NodeOp.OR:
            values[node.node_id] = bool(values[node.args[0]]) or bool(values[node.args[1]])
        elif node.op is NodeOp.SELECT:
            predicate = bool(values[node.args[0]])
            values[node.node_id] = values[node.args[1]] if predicate else values[node.args[2]]
        elif node.op is NodeOp.SAMPLE:
            raise NotImplementedError(
                "RNG-backed sample nodes are reserved for the next milestone. "
                f"Received rng={rng!r}."
            )
        elif node.op is NodeOp.STORE_NEXT_STATE:
            next_state[node.slot] = values[node.args[0]]  # type: ignore[index]
        elif node.op is NodeOp.STORE_OBS:
            observation[node.slot] = values[node.args[0]]  # type: ignore[index]
        elif node.op is NodeOp.STORE_REWARD:
            reward = float(_as_number(values[node.args[0]]))
        elif node.op is NodeOp.STORE_DONE:
            done = bool(values[node.args[0]])
        else:
            raise NotImplementedError(f"Unsupported IR op: {node.op}")

    return ExecutionResult(
        next_state=tuple(next_state),
        observation=tuple(observation),
        reward=reward,
        done=done,
        values=values,
    )


def _as_number(value: Scalar) -> int | float:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    raise TypeError(f"Expected a scalar number, got {type(value)!r}")
