"""Real grounding helpers for the supported scalar deterministic RDDL subset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rddl2puffer.frontend.schema import EnvSchema, FluentDType, FluentRole, FluentSpec, Scalar, build_layout
from rddl2puffer.rddl_plus.parser.cpf import CPF
from rddl2puffer.rddl_plus.parser.expr import Expression
from rddl2puffer.rddl_plus.parser.rddl import RDDL


@dataclass(frozen=True, slots=True)
class GroundedRDDLModel:
    """Grounded scalar deterministic RDDL model ready for lowering to IR."""

    schema: EnvSchema
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    non_fluents: Mapping[str, Scalar]
    intermediate_cpfs: tuple[CPF, ...]
    state_cpfs: tuple[CPF, ...]
    observation_cpfs: tuple[CPF, ...]
    reward: Expression
    terminals: tuple[Expression, ...]
    metadata: Mapping[str, object]


def ground_rddl_ast(ast: RDDL) -> GroundedRDDLModel:
    """Ground a parsed local AST into the supported scalar deterministic subset."""

    ast.build()
    _validate_scalar_unparameterized_subset(ast)

    state_defaults = _state_defaults(ast)
    non_fluent_values = _non_fluent_values(ast)

    state_order = _source_ordered_fluent_names(ast, "state")
    action_order = _source_ordered_fluent_names(ast, "action")
    observation_order = (
        _source_ordered_fluent_names(ast, "observation")
        if getattr(ast.domain, "observation_fluents", {})
        else state_order
    )

    schema = EnvSchema(
        state_layout=build_layout(
            "state",
            tuple(
                FluentSpec(
                    name=ast.domain.state_fluents[name].name,
                    role=FluentRole.STATE,
                    dtype=_dtype_for_range(ast.domain.state_fluents[name].range),
                    default=state_defaults[name],
                )
                for name in state_order
            ),
        ),
        action_layout=build_layout(
            "action",
            tuple(
                FluentSpec(
                    name=ast.domain.action_fluents[name].name,
                    role=FluentRole.ACTION,
                    dtype=_dtype_for_range(ast.domain.action_fluents[name].range),
                    default=ast.domain.action_fluents[name].default or 0,
                    bounds=_action_bounds(ast, name),
                )
                for name in action_order
            ),
        ),
        observation_layout=build_layout(
            "observation",
            tuple(
                FluentSpec(
                    name=_observation_name(ast, name),
                    role=FluentRole.OBSERVATION,
                    dtype=_observation_dtype(ast, name),
                    default=_observation_default(ast, name, state_defaults),
                )
                for name in observation_order
            ),
        ),
        metadata={
            "domain": ast.domain.name,
            "instance": ast.instance.name,
            "horizon": int(ast.instance.horizon),
            "discount": float(ast.instance.discount),
        },
    )

    return GroundedRDDLModel(
        schema=schema,
        state_names=state_order,
        action_names=action_order,
        observation_names=observation_order,
        non_fluents=non_fluent_values,
        intermediate_cpfs=tuple(ast.domain.intermediate_cpfs),
        state_cpfs=tuple(ast.domain.state_cpfs),
        observation_cpfs=tuple(ast.domain.observation_cpfs),
        reward=ast.domain.reward,
        terminals=tuple(ast.domain.terminals),
        metadata=schema.metadata,
    )


def _validate_scalar_unparameterized_subset(ast: RDDL) -> None:
    for fluent_map in (
        ast.domain.non_fluents,
        ast.domain.state_fluents,
        ast.domain.action_fluents,
        ast.domain.intermediate_fluents,
        ast.domain.observation_fluents,
    ):
        for pvar in fluent_map.values():
            if pvar.arity != 0:
                raise NotImplementedError(
                    "The current direct RDDL compiler supports only scalar unparameterized fluents. "
                    f"Received {pvar.name}/{pvar.arity}."
                )


def _source_ordered_fluent_names(ast: RDDL, role: str) -> tuple[str, ...]:
    ordered: list[str] = []
    for pvar in ast.domain.pvariables:
        if role == "state" and pvar.is_state_fluent():
            ordered.append(str(pvar))
        elif role == "action" and pvar.is_action_fluent():
            ordered.append(str(pvar))
        elif role == "observation" and pvar.is_observ_fluent():
            ordered.append(str(pvar))
    return tuple(ordered)


def _state_defaults(ast: RDDL) -> dict[str, Scalar]:
    defaults = {
        name: _normalize_scalar(ast.domain.state_fluents[name].default or 0)
        for name in ast.domain.state_fluent_ordering
    }
    for (name, _params), value in ast.instance.init_state:
        canonical = f"{name}/0"
        defaults[canonical] = _normalize_scalar(value)
    return defaults


def _non_fluent_values(ast: RDDL) -> dict[str, Scalar]:
    values = {
        name: _normalize_scalar(ast.domain.non_fluents[name].default or 0)
        for name in ast.domain.non_fluent_ordering
    }
    init_values = getattr(ast.non_fluents, "init_non_fluent", []) or []
    for (name, _params), value in init_values:
        canonical = f"{name}/0"
        values[canonical] = _normalize_scalar(value)
    return values


def _dtype_for_range(range_type: str) -> FluentDType:
    normalized = range_type.lower()
    if normalized == "real":
        return FluentDType.REAL
    if normalized == "int":
        return FluentDType.INT
    if normalized == "bool":
        return FluentDType.BOOL
    raise NotImplementedError(f"Unsupported RDDL range type: {range_type}")


def _action_bounds(ast: RDDL, canonical_name: str) -> tuple[Scalar | None, Scalar | None]:
    lower_expr = ast.domain.action_lower_bound_constraints.get(canonical_name)
    upper_expr = ast.domain.action_upper_bound_constraints.get(canonical_name)
    return (
        _expr_constant_value(lower_expr) if lower_expr is not None else None,
        _expr_constant_value(upper_expr) if upper_expr is not None else None,
    )


def _observation_name(ast: RDDL, canonical_name: str) -> str:
    if canonical_name in ast.domain.state_fluents:
        return ast.domain.state_fluents[canonical_name].name
    return ast.domain.observation_fluents[canonical_name].name


def _observation_dtype(ast: RDDL, canonical_name: str) -> FluentDType:
    if canonical_name in ast.domain.state_fluents:
        return _dtype_for_range(ast.domain.state_fluents[canonical_name].range)
    return _dtype_for_range(ast.domain.observation_fluents[canonical_name].range)


def _observation_default(
    ast: RDDL,
    canonical_name: str,
    state_defaults: Mapping[str, Scalar],
) -> Scalar:
    if canonical_name in state_defaults:
        return state_defaults[canonical_name]
    return _normalize_scalar(ast.domain.observation_fluents[canonical_name].default or 0)


def _expr_constant_value(expr: Expression) -> Scalar:
    if expr.is_constant_expression():
        return _normalize_scalar(expr.value)

    if expr.etype == ("arithmetic", "-") and len(expr.args) == 1:
        inner = _expr_constant_value(expr.args[0])
        if isinstance(inner, bool):
            raise NotImplementedError("Boolean values are not valid numeric action bounds.")
        return -inner

    raise NotImplementedError(
        "The current direct RDDL compiler requires constant action bounds. "
        f"Received {expr.etype}."
    )


def _normalize_scalar(value: Scalar) -> Scalar:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    return float(value)
