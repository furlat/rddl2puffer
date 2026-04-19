"""Shared helpers for Puffer C code emission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rddl2puffer.ir.nodes import CPFNode, IRProgram, NodeOp


@dataclass(frozen=True, slots=True)
class CNodeValue:
    """C expression and storage type for one IR node result."""

    ctype: str


def infer_c_node_values(program: IRProgram) -> dict[str, CNodeValue]:
    """Infer C scalar types and expressions for each non-store node."""

    inferred: dict[str, CNodeValue] = {}
    for node in program.nodes:
        if node.op is NodeOp.CONST:
            inferred[node.node_id] = _infer_const(node.value)
        elif node.op is NodeOp.LOAD_STATE:
            inferred[node.node_id] = CNodeValue("float")
        elif node.op is NodeOp.LOAD_ACTION:
            inferred[node.node_id] = CNodeValue("float")
        elif node.op is NodeOp.NEG:
            inferred[node.node_id] = CNodeValue("float")
        elif node.op is NodeOp.NOT:
            inferred[node.node_id] = CNodeValue("bool")
        elif node.op is NodeOp.SIN:
            inferred[node.node_id] = CNodeValue("float")
        elif node.op is NodeOp.COS:
            inferred[node.node_id] = CNodeValue("float")
        elif node.op in {NodeOp.ADD, NodeOp.SUB, NodeOp.MUL, NodeOp.DIV, NodeOp.MIN, NodeOp.MAX}:
            inferred[node.node_id] = CNodeValue("float")
        elif node.op in {NodeOp.LT, NodeOp.LE, NodeOp.GT, NodeOp.GE, NodeOp.EQ, NodeOp.NE}:
            inferred[node.node_id] = CNodeValue("bool")
        elif node.op in {NodeOp.AND, NodeOp.OR}:
            inferred[node.node_id] = CNodeValue("bool")
        elif node.op is NodeOp.SELECT:
            inferred[node.node_id] = _infer_select_type(node, inferred)
        elif node.op in {
            NodeOp.STORE_NEXT_STATE,
            NodeOp.STORE_OBS,
            NodeOp.STORE_REWARD,
            NodeOp.STORE_DONE,
        }:
            continue
        elif node.op is NodeOp.SAMPLE:
            raise NotImplementedError("Puffer C codegen for SAMPLE nodes is not implemented yet.")
        else:
            raise NotImplementedError(f"Unsupported IR op for C codegen: {node.op}")
    return inferred


def sanitize_c_identifier(value: str) -> str:
    """Turn an arbitrary identifier into a legal C identifier fragment."""

    chars: list[str] = []
    for char in value:
        chars.append(char if char.isalnum() else "_")
    text = "".join(chars).strip("_")
    if not text:
        text = "value"
    if text[0].isdigit():
        text = f"n_{text}"
    return text


def emit_c_value_declarations(
    program: IRProgram,
    inferred: dict[str, CNodeValue],
    *,
    state_targets: Mapping[int, str] | None = None,
) -> str:
    """Emit C local variable declarations for each non-store node."""

    lines: list[str] = []
    for index, node in enumerate(program.nodes):
        if node.node_id not in inferred:
            continue
        info = inferred[node.node_id]
        name = node_var_name(index, node.node_id)
        expr = _emit_node_expr(program, node, state_targets=state_targets)
        comment = f" // {node.comment}" if node.comment else ""
        lines.append(f"{info.ctype} {name} = {expr};{comment}")
    return "\n".join(lines)


def emit_c_store_statements(
    program: IRProgram,
    *,
    next_state_targets: Mapping[int, str] | None = None,
    observation_targets: Mapping[int, str] | None = None,
) -> str:
    """Emit C assignments for next-state, observation, reward, and done outputs."""

    lines: list[str] = []
    for index, node in enumerate(program.nodes):
        if node.op is NodeOp.STORE_NEXT_STATE:
            source = lookup_var_name(program, node.args[0])
            if node.slot is None:
                raise ValueError("STORE_NEXT_STATE node is missing a slot.")
            target = (
                next_state_targets[node.slot]
                if next_state_targets is not None
                else f"next_state[{node.slot}]"
            )
            lines.append(f"{target} = {source};")
        elif node.op is NodeOp.STORE_OBS:
            source = lookup_var_name(program, node.args[0])
            if node.slot is None:
                raise ValueError("STORE_OBS node is missing a slot.")
            target = (
                observation_targets[node.slot]
                if observation_targets is not None
                else f"obs[{node.slot}]"
            )
            lines.append(f"{target} = {source};")
        elif node.op is NodeOp.STORE_REWARD:
            source = lookup_var_name(program, node.args[0])
            lines.append(f"reward = (float)({source});")
        elif node.op is NodeOp.STORE_DONE:
            source = lookup_var_name(program, node.args[0])
            lines.append(f"terminated = {source};")
    return "\n".join(lines)


def lookup_var_name(program: IRProgram, node_id: str) -> str:
    """Return the emitted C variable name for a referenced IR node id."""

    for index, node in enumerate(program.nodes):
        if node.node_id == node_id:
            return node_var_name(index, node.node_id)
    raise KeyError(f"Unknown IR node id: {node_id}")


def node_var_name(index: int, node_id: str) -> str:
    """Stable C local variable name for one IR node."""

    return f"v_{index}_{sanitize_c_identifier(node_id)}"


def action_var_name(slot: int) -> str:
    """Stable C local variable name for one sanitized action slot."""

    return f"action_{slot}"


def _infer_const(value: object) -> CNodeValue:
    if isinstance(value, bool):
        return CNodeValue("bool")
    if isinstance(value, int):
        return CNodeValue("float")
    if isinstance(value, float):
        return CNodeValue("float")
    if value is None:
        raise ValueError("CONST node is missing a value.")
    raise TypeError(f"Unsupported const literal type: {type(value)!r}")


def _emit_numeric_expr(node: CPFNode, lhs: str, rhs: str) -> str:
    if node.op is NodeOp.ADD:
        return f"({lhs} + {rhs})"
    if node.op is NodeOp.SUB:
        return f"({lhs} - {rhs})"
    if node.op is NodeOp.MUL:
        return f"({lhs} * {rhs})"
    if node.op is NodeOp.DIV:
        return f"({lhs} / {rhs})"
    if node.op is NodeOp.MIN:
        return f"fminf({lhs}, {rhs})"
    if node.op is NodeOp.MAX:
        return f"fmaxf({lhs}, {rhs})"
    raise NotImplementedError(f"Unsupported numeric node: {node.op}")


def _emit_compare_expr(node: CPFNode, lhs: str, rhs: str) -> str:
    operator = {
        NodeOp.LT: "<",
        NodeOp.LE: "<=",
        NodeOp.GT: ">",
        NodeOp.GE: ">=",
        NodeOp.EQ: "==",
        NodeOp.NE: "!=",
    }[node.op]
    return f"({lhs} {operator} {rhs})"


def _emit_logical_expr(node: CPFNode, lhs: str, rhs: str) -> str:
    operator = "&&" if node.op is NodeOp.AND else "||"
    return f"({lhs} {operator} {rhs})"


def _infer_select_type(node: CPFNode, inferred: dict[str, CNodeValue]) -> CNodeValue:
    when_true = inferred[node.args[1]]
    when_false = inferred[node.args[2]]
    ctype = "bool" if when_true.ctype == when_false.ctype == "bool" else "float"
    return CNodeValue(ctype=ctype)


def _emit_node_expr(
    program: IRProgram,
    node: CPFNode,
    *,
    state_targets: Mapping[int, str] | None = None,
) -> str:
    refs = [lookup_var_name(program, arg) for arg in node.args]
    if node.op is NodeOp.CONST:
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        if isinstance(node.value, int):
            return _float_literal(float(node.value))
        if isinstance(node.value, float):
            return _float_literal(node.value)
        raise TypeError(f"Unsupported const literal type: {type(node.value)!r}")
    if node.op is NodeOp.LOAD_STATE:
        if node.slot is None:
            raise ValueError("LOAD_STATE node is missing a slot.")
        if state_targets is not None:
            return state_targets[node.slot]
        return f"env->state[{node.slot}]"
    if node.op is NodeOp.LOAD_ACTION:
        return action_var_name(int(node.slot))
    if node.op is NodeOp.NEG:
        return f"(-{refs[0]})"
    if node.op is NodeOp.NOT:
        return f"(!{refs[0]})"
    if node.op is NodeOp.SIN:
        return f"sinf({refs[0]})"
    if node.op is NodeOp.COS:
        return f"cosf({refs[0]})"
    if node.op in {NodeOp.ADD, NodeOp.SUB, NodeOp.MUL, NodeOp.DIV, NodeOp.MIN, NodeOp.MAX}:
        return _emit_numeric_expr(node, refs[0], refs[1])
    if node.op in {NodeOp.LT, NodeOp.LE, NodeOp.GT, NodeOp.GE, NodeOp.EQ, NodeOp.NE}:
        return _emit_compare_expr(node, refs[0], refs[1])
    if node.op in {NodeOp.AND, NodeOp.OR}:
        return _emit_logical_expr(node, refs[0], refs[1])
    if node.op is NodeOp.SELECT:
        return f"({refs[0]} ? {refs[1]} : {refs[2]})"
    raise NotImplementedError(f"Unsupported node expression emission for {node.op}")


def _float_literal(value: float) -> str:
    text = repr(float(value))
    if "e" not in text and "." not in text:
        text += ".0"
    return f"{text}f"
