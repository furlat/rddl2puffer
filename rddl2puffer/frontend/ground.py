"""Grounding helpers for the currently supported RDDL subset."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

from rddl2puffer.frontend.schema import EnvSchema, FluentDType, FluentRole, FluentSpec, Scalar, build_layout
from rddl2puffer.rddl_plus.parser.expr import Expression
from rddl2puffer.rddl_plus.parser.pvariable import PVariable
from rddl2puffer.rddl_plus.parser.rddl import RDDL
from rddl2puffer.rddl_plus.parser import utils as parser_utils


@dataclass(frozen=True, slots=True)
class GroundedRule:
    """One grounded scalar rule ready for lowering."""

    name: str
    expr: Expression


@dataclass(frozen=True, slots=True)
class GroundedRDDLModel:
    """Grounded scalar RDDL model ready for lowering to native code."""

    schema: EnvSchema
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    non_fluents: Mapping[str, Scalar]
    intermediate_cpfs: tuple[GroundedRule, ...]
    state_cpfs: tuple[GroundedRule, ...]
    observation_cpfs: tuple[GroundedRule, ...]
    reward: Expression
    terminals: tuple[Expression, ...]
    metadata: Mapping[str, object]


def ground_rddl_ast(ast: RDDL) -> GroundedRDDLModel:
    """Ground a parsed AST into the supported scalar subset."""

    ast.build()

    state_specs = _build_fluent_specs(ast, role="state")
    action_specs = _build_fluent_specs(ast, role="action")
    observation_specs = (
        _build_fluent_specs(ast, role="observation")
        if getattr(ast.domain, "observation_fluents", {})
        else tuple(
            FluentSpec(
                name=spec.name,
                role=FluentRole.OBSERVATION,
                dtype=spec.dtype,
                parameters=spec.parameters,
                bounds=spec.bounds,
                default=spec.default,
            )
            for spec in state_specs
        )
    )

    state_defaults = {spec.qualified_name: spec.default for spec in state_specs}
    non_fluent_values = _non_fluent_values(ast)
    metadata = {
        "domain": ast.domain.name,
        "instance": ast.instance.name,
        "horizon": int(ast.instance.horizon),
        "discount": float(ast.instance.discount),
    }

    schema = EnvSchema(
        state_layout=build_layout("state", state_specs),
        action_layout=build_layout("action", action_specs),
        observation_layout=build_layout("observation", observation_specs),
        metadata=metadata,
    )

    state_names = tuple(spec.qualified_name for spec in schema.state_layout.fluents)
    action_names = tuple(spec.qualified_name for spec in schema.action_layout.fluents)
    observation_names = tuple(spec.qualified_name for spec in schema.observation_layout.fluents)

    return GroundedRDDLModel(
        schema=schema,
        state_names=state_names,
        action_names=action_names,
        observation_names=observation_names,
        non_fluents=non_fluent_values,
        intermediate_cpfs=_ground_cpfs(ast, _ordered_intermediate_cpfs(ast)),
        state_cpfs=_ground_state_cpfs(ast),
        observation_cpfs=_ground_cpfs(ast, ast.domain.observation_cpfs),
        reward=_ground_expression(ast, ast.domain.reward, {}),
        terminals=tuple(_ground_expression(ast, expr, {}) for expr in ast.domain.terminals),
        metadata=metadata,
    )


def _build_fluent_specs(ast: RDDL, *, role: str) -> tuple[FluentSpec, ...]:
    pvars = tuple(_iter_role_pvariables(ast, role))
    specs: list[FluentSpec] = []
    init_overrides = (
        _initializer_values(getattr(ast.instance, "init_state", []))
        if role == "state"
        else {}
    )
    if role == "state":
        fluent_map = ast.domain.state_fluents
    elif role == "action":
        fluent_map = ast.domain.action_fluents
    elif role == "observation":
        fluent_map = ast.domain.observation_fluents
    else:
        raise ValueError(f"Unsupported role {role!r}")

    for pvar in pvars:
        param_types = tuple(pvar.param_types or ())
        for params in _groundings_for_types(ast, param_types):
            qualified_name = _qualified_name(pvar.name, params)
            default = _normalize_scalar(pvar.default or 0)
            if role == "state":
                default = init_overrides.get(qualified_name, default)
            specs.append(
                FluentSpec(
                    name=pvar.name,
                    role={
                        "state": FluentRole.STATE,
                        "action": FluentRole.ACTION,
                        "observation": FluentRole.OBSERVATION,
                    }[role],
                    dtype=_dtype_for_range(pvar.range),
                    parameters=params,
                    bounds=_action_bounds(ast, pvar, params) if role == "action" else (None, None),
                    default=default,
                )
            )
    return tuple(specs)


def _iter_role_pvariables(ast: RDDL, role: str) -> Iterable[PVariable]:
    for pvar in ast.domain.pvariables:
        if role == "state" and pvar.is_state_fluent():
            yield pvar
        elif role == "action" and pvar.is_action_fluent():
            yield pvar
        elif role == "observation" and pvar.is_observ_fluent():
            yield pvar


def _ground_cpfs(ast: RDDL, cpfs: Sequence[object]) -> tuple[GroundedRule, ...]:
    grounded: list[GroundedRule] = []
    for cpf in cpfs:
        pvar_name, params = cpf.pvar[1]
        param_types = _pvar_param_types(ast, cpf.name)
        for grounded_params in _groundings_for_types(ast, param_types):
            bindings = _bindings_for_params(params or (), grounded_params)
            grounded.append(
                GroundedRule(
                    name=_qualified_name(
                        pvar_name[:-1] if pvar_name.endswith("'") else pvar_name,
                        grounded_params,
                    ),
                    expr=_ground_expression(ast, cpf.expr, bindings),
                )
            )
    return tuple(grounded)


def _ordered_intermediate_cpfs(ast: RDDL) -> tuple[object, ...]:
    _, cpfs = ast.domain.cpfs
    indexed = [
        (index, cpf)
        for index, cpf in enumerate(cpfs)
        if cpf.name in ast.domain.intermediate_fluents
    ]
    indexed.sort(key=lambda item: (ast.domain.intermediate_fluents[item[1].name].level, item[0]))
    return tuple(cpf for _, cpf in indexed)


def _ground_state_cpfs(ast: RDDL) -> tuple[GroundedRule, ...]:
    grounded: list[GroundedRule] = []
    for cpf in ast.domain.state_cpfs:
        pvar_name, params = cpf.pvar[1]
        base_name = parser_utils.rename_next_state_fluent(f"{pvar_name}/{len(params or ())}")
        pvar = ast.domain.state_fluents[base_name]
        for grounded_params in _groundings_for_types(ast, tuple(pvar.param_types or ())):
            bindings = _bindings_for_params(params or (), grounded_params)
            grounded.append(
                GroundedRule(
                    name=_qualified_name(pvar.name, grounded_params),
                    expr=_ground_expression(ast, cpf.expr, bindings),
                )
            )
    return tuple(grounded)


def _ground_expression(ast: RDDL, expr: Expression, bindings: Mapping[str, str]) -> Expression:
    etype, operator = expr.etype
    if expr.is_constant_expression():
        return expr

    if expr.is_pvariable_expression():
        functor, params = expr.args
        grounded_params = None
        if params is not None:
            grounded_params = [_resolve_param(param, bindings) for param in params]
        return Expression(("pvar_expr", (functor, grounded_params)))

    if etype in {"arithmetic", "boolean", "relational"}:
        return Expression((operator, tuple(_ground_expression(ast, arg, bindings) for arg in expr.args)))

    if etype == "func":
        return Expression(("func", (operator, tuple(_ground_expression(ast, arg, bindings) for arg in expr.args))))

    if etype == "randomvar":
        return Expression(
            ("randomvar", (operator, tuple(_ground_expression(ast, arg, bindings) for arg in expr.args)))
        )

    if etype == "control" and operator == "if":
        return Expression(("if", tuple(_ground_expression(ast, arg, bindings) for arg in expr.args)))

    if etype == "aggregation":
        typed_vars = expr.args[:-1]
        body = expr.args[-1]
        return _expand_aggregation(ast, operator, typed_vars, body, bindings)

    raise NotImplementedError(f"Unsupported expression in grounding: {expr.etype}")


def _expand_aggregation(
    ast: RDDL,
    operator: str,
    typed_vars: Sequence[object],
    body: Expression,
    bindings: Mapping[str, str],
) -> Expression:
    grounded_bodies = list(_expand_typed_var_bindings(ast, typed_vars, body, bindings))
    if operator == "sum":
        return _fold_nary("+", grounded_bodies, _const_expr(0.0))
    if operator == "prod":
        return _fold_nary("*", grounded_bodies, _const_expr(1.0))
    if operator == "forall":
        return _fold_nary("^", grounded_bodies, _const_expr(True))
    if operator == "exists":
        return _fold_nary("|", grounded_bodies, _const_expr(False))
    if operator == "maximum":
        return _fold_function("max", grounded_bodies)
    if operator == "minimum":
        return _fold_function("min", grounded_bodies)
    if operator == "avg":
        if not grounded_bodies:
            return _const_expr(0.0)
        total = _fold_nary("+", grounded_bodies, _const_expr(0.0))
        return Expression(("/", (total, _const_expr(float(len(grounded_bodies))))))
    raise NotImplementedError(f"Unsupported aggregation in grounding: {operator}")


def _expand_typed_var_bindings(
    ast: RDDL,
    typed_vars: Sequence[object],
    body: Expression,
    bindings: Mapping[str, str],
) -> Iterable[Expression]:
    if not typed_vars:
        yield _ground_expression(ast, body, bindings)
        return

    raw_typed_var = typed_vars[0]
    if not isinstance(raw_typed_var, tuple) or raw_typed_var[0] != "typed_var":
        raise NotImplementedError(f"Unsupported typed var form: {raw_typed_var!r}")
    var_name, type_name = raw_typed_var[1]
    objects = ast.object_table[type_name]["objects"]
    for obj_name in objects:
        nested = dict(bindings)
        nested[var_name] = obj_name
        yield from _expand_typed_var_bindings(ast, typed_vars[1:], body, nested)


def _fold_nary(operator: str, values: Sequence[Expression], identity: Expression) -> Expression:
    if not values:
        return identity
    result = values[0]
    for value in values[1:]:
        result = Expression((operator, (result, value)))
    return result


def _fold_function(name: str, values: Sequence[Expression]) -> Expression:
    if not values:
        raise NotImplementedError(f"Cannot ground empty {name} aggregation.")
    result = values[0]
    for value in values[1:]:
        result = Expression(("func", (name, (result, value))))
    return result


def _const_expr(value: Scalar) -> Expression:
    if isinstance(value, bool):
        return Expression(("boolean", value))
    return Expression(("number", value))


def _resolve_param(param: object, bindings: Mapping[str, str]) -> str:
    if isinstance(param, str):
        return bindings.get(param, param)
    raise NotImplementedError(f"Unsupported pvariable parameter expression: {param!r}")


def _bindings_for_params(params: Sequence[object], grounded_params: Sequence[str]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for raw_param, grounded_param in zip(params, grounded_params, strict=True):
        if isinstance(raw_param, str):
            bindings[raw_param] = grounded_param
    return bindings


def _groundings_for_types(ast: RDDL, param_types: Sequence[str]) -> Iterable[tuple[str, ...]]:
    if not param_types:
        yield ()
        return
    object_lists = [tuple(ast.object_table[param_type]["objects"]) for param_type in param_types]
    yield from product(*object_lists)


def _qualified_name(name: str, params: Sequence[str]) -> str:
    if not params:
        return name
    return f"{name}[{','.join(params)}]"


def _initializer_values(initializers: Sequence[tuple[tuple[object, object], Scalar]]) -> dict[str, Scalar]:
    values: dict[str, Scalar] = {}
    for (name, params), value in initializers:
        param_values = tuple(params or ())
        values[_qualified_name(str(name), tuple(str(param) for param in param_values))] = _normalize_scalar(value)
    return values


def _non_fluent_values(ast: RDDL) -> dict[str, Scalar]:
    values: dict[str, Scalar] = {}
    for pvar in ast.domain.pvariables:
        if not pvar.is_non_fluent():
            continue
        for params in _groundings_for_types(ast, tuple(pvar.param_types or ())):
            values[_qualified_name(pvar.name, params)] = _normalize_scalar(pvar.default or 0)

    for (name, params), value in getattr(ast.non_fluents, "init_non_fluent", []) or []:
        param_values = tuple(params or ())
        values[_qualified_name(str(name), tuple(str(param) for param in param_values))] = _normalize_scalar(value)
    return values


def _pvar_param_types(ast: RDDL, canonical_name: str) -> tuple[str, ...]:
    if canonical_name in ast.domain.non_fluents:
        return tuple(ast.domain.non_fluents[canonical_name].param_types or ())
    if canonical_name in ast.domain.state_fluents:
        return tuple(ast.domain.state_fluents[canonical_name].param_types or ())
    if canonical_name in ast.domain.action_fluents:
        return tuple(ast.domain.action_fluents[canonical_name].param_types or ())
    if canonical_name in ast.domain.intermediate_fluents:
        return tuple(ast.domain.intermediate_fluents[canonical_name].param_types or ())
    if canonical_name in ast.domain.observation_fluents:
        return tuple(ast.domain.observation_fluents[canonical_name].param_types or ())
    raise KeyError(f"Unknown pvariable {canonical_name!r}")


def _dtype_for_range(range_type: str) -> FluentDType:
    normalized = range_type.lower()
    if normalized == "real":
        return FluentDType.REAL
    if normalized == "int":
        return FluentDType.INT
    if normalized == "bool":
        return FluentDType.BOOL
    raise NotImplementedError(f"Unsupported RDDL range type: {range_type}")


def _action_bounds(ast: RDDL, pvar: PVariable, params: Sequence[str]) -> tuple[Scalar | None, Scalar | None]:
    canonical_name = str(pvar)
    lower_expr = ast.domain.action_lower_bound_constraints.get(canonical_name)
    upper_expr = ast.domain.action_upper_bound_constraints.get(canonical_name)
    if lower_expr is None or upper_expr is None:
        derived_lower, derived_upper = _derive_action_bounds_from_preconditions(ast, pvar, params)
        lower_expr = lower_expr or derived_lower
        upper_expr = upper_expr or derived_upper
    return (
        _expr_constant_value(lower_expr) if lower_expr is not None else None,
        _expr_constant_value(upper_expr) if upper_expr is not None else None,
    )


def _derive_action_bounds_from_preconditions(
    ast: RDDL,
    pvar: PVariable,
    params: Sequence[str],
) -> tuple[Expression | None, Expression | None]:
    lower_expr: Expression | None = None
    upper_expr: Expression | None = None
    for precond in ast.domain.preconds:
        for relational in _iter_relational_bounds(precond):
            derived_lower = _extract_lower_bound_expr(relational, pvar, params)
            derived_upper = _extract_upper_bound_expr(relational, pvar, params)
            lower_expr = lower_expr or derived_lower
            upper_expr = upper_expr or derived_upper
    return lower_expr, upper_expr


def _iter_relational_bounds(expr: Expression) -> Iterable[Expression]:
    if expr.etype[0] == "relational":
        yield expr
        return
    if expr.etype in {("boolean", "^"), ("boolean", "&")}:
        for arg in expr.args:
            yield from _iter_relational_bounds(arg)
        return
    if expr.etype == ("aggregation", "forall"):
        yield from _iter_relational_bounds(expr.args[-1])


def _extract_lower_bound_expr(expr: Expression, pvar: PVariable, params: Sequence[str]) -> Expression | None:
    lhs, rhs = expr.args
    if expr.etype[1] in {"<=", "<"} and _matches_action_pvar(rhs, pvar, params):
        return lhs
    if expr.etype[1] in {">=", ">"} and _matches_action_pvar(lhs, pvar, params):
        return rhs
    return None


def _extract_upper_bound_expr(expr: Expression, pvar: PVariable, params: Sequence[str]) -> Expression | None:
    lhs, rhs = expr.args
    if expr.etype[1] in {"<=", "<"} and _matches_action_pvar(lhs, pvar, params):
        return rhs
    if expr.etype[1] in {">=", ">"} and _matches_action_pvar(rhs, pvar, params):
        return lhs
    return None


def _matches_action_pvar(expr: Expression, pvar: PVariable, params: Sequence[str]) -> bool:
    if not expr.is_pvariable_expression():
        return False
    functor, expr_params = expr.args
    if str(functor) != pvar.name:
        return False
    normalized = tuple(str(param) for param in (expr_params or ()))
    return normalized == tuple(params)


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
