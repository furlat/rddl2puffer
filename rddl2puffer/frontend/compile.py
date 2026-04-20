"""Direct source-driven compiler from supported RDDL subsets to IR."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Mapping

from rddl2puffer.frontend.ground import GroundedRDDLModel, ground_rddl_ast
from rddl2puffer.frontend.parse import ParsedRDDLProgram, parse_rddl_files, parse_rddl_sources
from rddl2puffer.ir.lower import assemble_program
from rddl2puffer.ir.nodes import CPFNode, IRProgram, NodeOp
from rddl2puffer.rddl_plus.parser.expr import Expression

_DIRECTIVE_PATTERN = re.compile(r"^\s*//%\s*([A-Za-z0-9_.-]+)\s*=\s*(.+?)\s*$")


def compile_parsed_rddl_program(
    parsed: ParsedRDDLProgram,
    *,
    env_name: str | None = None,
    metadata_overrides: Mapping[str, object] | None = None,
) -> IRProgram:
    """Compile a parsed RDDL AST into the supported deterministic IR."""

    grounded = ground_rddl_ast(parsed.ast)
    lowerer = _SubsetLowerer(grounded)
    program = lowerer.lower_program(env_name=env_name or grounded.metadata["domain"])  # type: ignore[index]
    merged_overrides = _merge_metadata_overrides(
        _extract_rddl_plus_metadata(parsed),
        metadata_overrides,
    )
    if merged_overrides:
        program = assemble_program(
            program.name,
            grounded.schema,
            program.nodes,
            metadata={**program.metadata, **merged_overrides},
        )
    return program


def compile_rddl_files(
    domain_path: str | Path,
    instance_path: str | Path,
    *,
    nonfluents_path: str | Path | None = None,
    env_name: str | None = None,
    metadata_overrides: Mapping[str, object] | None = None,
) -> IRProgram:
    """Parse and compile RDDL source files directly to IR."""

    parsed = parse_rddl_files(domain_path, instance_path, nonfluents_path=nonfluents_path)
    return compile_parsed_rddl_program(
        parsed,
        env_name=env_name,
        metadata_overrides=metadata_overrides,
    )


def compile_rddl_sources(
    domain_text: str,
    instance_text: str,
    *,
    nonfluents_text: str | None = None,
    env_name: str | None = None,
    metadata_overrides: Mapping[str, object] | None = None,
) -> IRProgram:
    """Parse and compile in-memory RDDL source directly to IR."""

    parsed = parse_rddl_sources(domain_text, instance_text, nonfluents_text=nonfluents_text)
    return compile_parsed_rddl_program(
        parsed,
        env_name=env_name,
        metadata_overrides=metadata_overrides,
    )


@dataclass(slots=True)
class _SubsetLowerer:
    grounded: GroundedRDDLModel
    _nodes: list[CPFNode] = field(init=False)
    _counter: int = field(init=False)
    _value_cache: dict[tuple[object, ...], str] = field(init=False)
    _const_values: dict[str, object] = field(init=False)
    _current_symbols: dict[str, str] = field(init=False)
    _next_state_symbols: dict[str, str] = field(init=False)
    _observation_symbols: dict[str, str] = field(init=False)
    _state_slots: dict[str, int | None] = field(init=False)
    _action_slots: dict[str, int | None] = field(init=False)
    _observation_slots: dict[str, int | None] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_nodes", [])
        object.__setattr__(self, "_counter", 0)
        object.__setattr__(self, "_value_cache", {})
        object.__setattr__(self, "_const_values", {})
        object.__setattr__(self, "_current_symbols", {})
        object.__setattr__(self, "_next_state_symbols", {})
        object.__setattr__(self, "_observation_symbols", {})
        object.__setattr__(
            self,
            "_state_slots",
            {
            name: fluent.flat_index
            for name, fluent in zip(
                self.grounded.state_names,
                self.grounded.schema.state_layout.fluents,
                strict=True,
            )
            },
        )
        object.__setattr__(
            self,
            "_action_slots",
            {
            name: fluent.flat_index
            for name, fluent in zip(
                self.grounded.action_names,
                self.grounded.schema.action_layout.fluents,
                strict=True,
            )
            },
        )
        object.__setattr__(
            self,
            "_observation_slots",
            {
            name: fluent.flat_index
            for name, fluent in zip(
                self.grounded.observation_names,
                self.grounded.schema.observation_layout.fluents,
                strict=True,
            )
            },
        )
        self._seed_base_symbols()

    def lower_program(self, *, env_name: str) -> IRProgram:
        for cpf in self.grounded.intermediate_cpfs:
            self._current_symbols[cpf.name] = self._lower_expression(cpf.expr, self._current_symbols)

        for cpf in self.grounded.state_cpfs:
            self._next_state_symbols[cpf.name] = self._lower_expression(cpf.expr, self._current_symbols)

        reward_node = self._lower_expression(self.grounded.reward, self._current_symbols)
        done_node = self._lower_terminals()

        for canonical_name, node_id in self._next_state_symbols.items():
            self._emit(
                NodeOp.STORE_NEXT_STATE,
                node_id=f"store_next_{_sanitize_name(canonical_name)}",
                args=(node_id,),
                slot=self._require_slot(self._state_slots, canonical_name, "state"),
            )

        if self.grounded.observation_cpfs:
            observation_symbols = dict(self._current_symbols)
            observation_symbols.update(self._next_state_symbols)
            for cpf in self.grounded.observation_cpfs:
                self._observation_symbols[cpf.name] = self._lower_expression(cpf.expr, observation_symbols)
            for canonical_name, node_id in self._observation_symbols.items():
                self._emit(
                    NodeOp.STORE_OBS,
                    node_id=f"store_obs_{_sanitize_name(canonical_name)}",
                    args=(node_id,),
                    slot=self._require_slot(self._observation_slots, canonical_name, "observation"),
                )
        else:
            for canonical_name, slot in self._observation_slots.items():
                source_name = canonical_name if canonical_name in self._next_state_symbols else None
                if source_name is None:
                    raise NotImplementedError(
                        "Observation lowering without observation CPFs currently expects MDP-style state observations."
                    )
                self._emit(
                    NodeOp.STORE_OBS,
                    node_id=f"store_obs_{_sanitize_name(canonical_name)}",
                    args=(self._next_state_symbols[source_name],),
                    slot=slot,
                )

        self._emit(NodeOp.STORE_REWARD, node_id="store_reward", args=(reward_node,))
        self._emit(NodeOp.STORE_DONE, node_id="store_done", args=(done_node,))
        return assemble_program(env_name, self.grounded.schema, tuple(self._nodes), metadata=self.grounded.metadata)

    def _seed_base_symbols(self) -> None:
        for canonical_name, slot in self._state_slots.items():
            self._current_symbols[canonical_name] = self._emit(
                NodeOp.LOAD_STATE,
                node_id=f"load_state_{_sanitize_name(canonical_name)}",
                slot=self._require_slot(self._state_slots, canonical_name, "state"),
            )

        for canonical_name, slot in self._action_slots.items():
            self._current_symbols[canonical_name] = self._emit(
                NodeOp.LOAD_ACTION,
                node_id=f"load_action_{_sanitize_name(canonical_name)}",
                slot=self._require_slot(self._action_slots, canonical_name, "action"),
            )

        for canonical_name, value in self.grounded.non_fluents.items():
            self._current_symbols[canonical_name] = self._emit_const(
                node_id=f"const_{_sanitize_name(canonical_name)}",
                value=value,
            )

    def _lower_terminals(self) -> str:
        if not self.grounded.terminals:
            return self._emit_const(node_id="const_terminal_false", value=False)

        terminal_symbols = dict(self._current_symbols)
        terminal_symbols.update(self._next_state_symbols)
        lowered = [self._lower_expression(expr, terminal_symbols) for expr in self.grounded.terminals]
        result = lowered[0]
        for index, node_id in enumerate(lowered[1:], start=1):
            result = self._emit_cached(
                NodeOp.OR,
                cache_key=(NodeOp.OR.value, result, node_id),
                node_id=f"terminal_or_{index}",
                args=(result, node_id),
            )
        return result

    def _lower_expression(self, expr: Expression, symbols: Mapping[str, str]) -> str:
        if expr.is_constant_expression():
            return self._emit_const(value=expr.value)
        if expr.is_pvariable_expression():
            name = _grounded_expr_name(expr)
            try:
                return symbols[name]
            except KeyError as exc:
                raise NotImplementedError(f"Unsupported unresolved pvariable reference: {name}") from exc

        expr_type = expr.etype
        etype, operator = expr_type
        args = expr.args

        if expr_type == ("control", "if"):
            predicate = self._lower_expression(args[0], symbols)
            when_true = self._lower_expression(args[1], symbols)
            when_false = self._lower_expression(args[2], symbols)
            return self._emit_cached(
                NodeOp.SELECT,
                cache_key=(NodeOp.SELECT.value, predicate, when_true, when_false),
                node_id=f"select_{self._counter}",
                args=(predicate, when_true, when_false),
            )

        if expr_type == ("arithmetic", "-") and len(args) == 1:
            inner = self._lower_expression(args[0], symbols)
            return self._emit_cached(
                NodeOp.NEG,
                cache_key=(NodeOp.NEG.value, inner),
                node_id=f"neg_{self._counter}",
                args=(inner,),
            )

        if expr_type == ("boolean", "~"):
            inner = self._lower_expression(args[0], symbols)
            return self._emit_cached(
                NodeOp.NOT,
                cache_key=(NodeOp.NOT.value, inner),
                node_id=f"not_{self._counter}",
                args=(inner,),
            )

        if expr_type in {("arithmetic", "+"), ("arithmetic", "-"), ("arithmetic", "*"), ("arithmetic", "/")}:
            lhs = self._lower_expression(args[0], symbols)
            rhs = self._lower_expression(args[1], symbols)
            op = {
                "+": NodeOp.ADD,
                "-": NodeOp.SUB,
                "*": NodeOp.MUL,
                "/": NodeOp.DIV,
            }[operator]
            return self._emit_cached(
                op,
                cache_key=(op.value, lhs, rhs),
                node_id=f"{op.value}_{self._counter}",
                args=(lhs, rhs),
            )

        if expr_type == ("boolean", "|"):
            lhs = self._lower_expression(args[0], symbols)
            rhs = self._lower_expression(args[1], symbols)
            return self._emit_cached(
                NodeOp.OR,
                cache_key=(NodeOp.OR.value, lhs, rhs),
                node_id=f"or_{self._counter}",
                args=(lhs, rhs),
            )

        if expr_type in {("boolean", "^"), ("boolean", "&")}:
            lhs = self._lower_expression(args[0], symbols)
            rhs = self._lower_expression(args[1], symbols)
            return self._emit_cached(
                NodeOp.AND,
                cache_key=(NodeOp.AND.value, lhs, rhs),
                node_id=f"and_{self._counter}",
                args=(lhs, rhs),
            )

        if etype == "relational":
            lhs = self._lower_expression(args[0], symbols)
            rhs = self._lower_expression(args[1], symbols)
            op = {
                "<": NodeOp.LT,
                "<=": NodeOp.LE,
                ">": NodeOp.GT,
                ">=": NodeOp.GE,
                "==": NodeOp.EQ,
                "~=": NodeOp.NE,
            }[operator]
            return self._emit_cached(
                op,
                cache_key=(op.value, lhs, rhs),
                node_id=f"{op.value}_{self._counter}",
                args=(lhs, rhs),
            )

        if expr_type == ("func", "sin"):
            inner = self._lower_expression(args[0], symbols)
            return self._emit_cached(
                NodeOp.SIN,
                cache_key=(NodeOp.SIN.value, inner),
                node_id=f"sin_{self._counter}",
                args=(inner,),
            )

        if expr_type == ("func", "cos"):
            inner = self._lower_expression(args[0], symbols)
            return self._emit_cached(
                NodeOp.COS,
                cache_key=(NodeOp.COS.value, inner),
                node_id=f"cos_{self._counter}",
                args=(inner,),
            )

        if expr_type == ("func", "min"):
            lhs = self._lower_expression(args[0], symbols)
            rhs = self._lower_expression(args[1], symbols)
            return self._emit_cached(
                NodeOp.MIN,
                cache_key=(NodeOp.MIN.value, lhs, rhs),
                node_id=f"min_{self._counter}",
                args=(lhs, rhs),
            )

        if expr_type == ("func", "max"):
            lhs = self._lower_expression(args[0], symbols)
            rhs = self._lower_expression(args[1], symbols)
            return self._emit_cached(
                NodeOp.MAX,
                cache_key=(NodeOp.MAX.value, lhs, rhs),
                node_id=f"max_{self._counter}",
                args=(lhs, rhs),
            )

        if expr_type == ("func", "abs"):
            inner = self._lower_expression(args[0], symbols)
            zero = self._emit_const(value=0.0)
            is_negative = self._emit_cached(
                NodeOp.LT,
                cache_key=(NodeOp.LT.value, inner, zero),
                node_id=f"abs_lt_{self._counter}",
                args=(inner, zero),
            )
            neg_inner = self._emit_cached(
                NodeOp.NEG,
                cache_key=(NodeOp.NEG.value, inner),
                node_id=f"abs_neg_{self._counter}",
                args=(inner,),
            )
            return self._emit_cached(
                NodeOp.SELECT,
                cache_key=(NodeOp.SELECT.value, is_negative, neg_inner, inner),
                node_id=f"abs_{self._counter}",
                args=(is_negative, neg_inner, inner),
            )

        if expr_type == ("func", "sgn"):
            inner = self._lower_expression(args[0], symbols)
            zero = self._emit_const(value=0.0)
            one = self._emit_const(value=1.0)
            neg_one = self._emit_const(value=-1.0)
            is_positive = self._emit_cached(
                NodeOp.GT,
                cache_key=(NodeOp.GT.value, inner, zero),
                node_id=f"sgn_gt_{self._counter}",
                args=(inner, zero),
            )
            is_negative = self._emit_cached(
                NodeOp.LT,
                cache_key=(NodeOp.LT.value, inner, zero),
                node_id=f"sgn_lt_{self._counter}",
                args=(inner, zero),
            )
            negative_or_zero = self._emit_cached(
                NodeOp.SELECT,
                cache_key=(NodeOp.SELECT.value, is_negative, neg_one, zero),
                node_id=f"sgn_neg_or_zero_{self._counter}",
                args=(is_negative, neg_one, zero),
            )
            return self._emit_cached(
                NodeOp.SELECT,
                cache_key=(NodeOp.SELECT.value, is_positive, one, negative_or_zero),
                node_id=f"sgn_{self._counter}",
                args=(is_positive, one, negative_or_zero),
            )

        if expr_type == ("func", "pow"):
            base = self._lower_expression(args[0], symbols)
            return self._lower_pow(base, args[1])

        if expr_type == ("randomvar", "Uniform"):
            low = self._lower_expression(args[0], symbols)
            high = self._lower_expression(args[1], symbols)
            return self._emit(
                NodeOp.SAMPLE,
                node_id=f"uniform_{self._counter}",
                args=(low, high),
                value="uniform",
            )

        raise NotImplementedError(f"Unsupported RDDL expression in direct compiler: {expr.etype}")

    def _lower_pow(self, base_node: str, exponent_expr: Expression) -> str:
        if not exponent_expr.is_constant_expression():
            raise NotImplementedError("Direct RDDL compiler currently supports only constant exponents in pow.")

        exponent = exponent_expr.value
        if not isinstance(exponent, int):
            if isinstance(exponent, float) and exponent.is_integer():
                exponent = int(exponent)
            else:
                raise NotImplementedError("Direct RDDL compiler currently supports only integer exponents in pow.")

        if exponent < 0:
            raise NotImplementedError("Direct RDDL compiler currently supports only non-negative integer exponents.")
        if exponent == 0:
            return self._emit_const(value=1.0)
        if exponent == 1:
            return base_node

        result = base_node
        for index in range(exponent - 1):
            result = self._emit_cached(
                NodeOp.MUL,
                cache_key=(NodeOp.MUL.value, result, base_node),
                node_id=f"pow_mul_{self._counter}_{index}",
                args=(result, base_node),
            )
        return result

    def _emit_const(self, value: object, node_id: str | None = None) -> str:
        cache_key = (NodeOp.CONST.value, type(value).__name__, value)
        return self._emit_cached(
            NodeOp.CONST,
            cache_key=cache_key,
            node_id=node_id or f"const_{self._counter}",
            value=value,
        )

    def _emit(
        self,
        op: NodeOp,
        *,
        node_id: str,
        args: tuple[str, ...] = (),
        value: object | None = None,
        slot: int | None = None,
    ) -> str:
        self._nodes.append(CPFNode(node_id=node_id, op=op, args=args, value=value, slot=slot))
        if op is NodeOp.CONST:
            self._const_values[node_id] = value
        self._counter += 1
        return node_id

    def _emit_cached(
        self,
        op: NodeOp,
        *,
        cache_key: tuple[object, ...],
        node_id: str,
        args: tuple[str, ...] = (),
        value: object | None = None,
        slot: int | None = None,
    ) -> str:
        folded = self._try_fold(op, args=args, value=value)
        if folded is not None:
            return folded

        existing = self._value_cache.get(cache_key)
        if existing is not None:
            return existing

        emitted = self._emit(op, node_id=node_id, args=args, value=value, slot=slot)
        self._value_cache[cache_key] = emitted
        return emitted

    def _try_fold(
        self,
        op: NodeOp,
        *,
        args: tuple[str, ...],
        value: object | None = None,
    ) -> str | None:
        const_args = [self._const_values.get(arg) for arg in args]

        if op is NodeOp.SELECT:
            predicate = const_args[0]
            if isinstance(predicate, bool):
                return args[1] if predicate else args[2]
            if len(args) == 3 and args[1] == args[2]:
                return args[1]
            return None

        if op is NodeOp.NOT and isinstance(const_args[0], bool):
            return self._emit_const(value=not const_args[0])

        if op is NodeOp.NEG and isinstance(const_args[0], int | float):
            return self._emit_const(value=-float(const_args[0]))

        if op in {NodeOp.SIN, NodeOp.COS} and isinstance(const_args[0], int | float):
            folded_value = math.sin(float(const_args[0])) if op is NodeOp.SIN else math.cos(float(const_args[0]))
            return self._emit_const(value=folded_value)

        if op in {NodeOp.ADD, NodeOp.SUB, NodeOp.MUL, NodeOp.DIV, NodeOp.MIN, NodeOp.MAX}:
            lhs, rhs = const_args
            if isinstance(lhs, int | float) and isinstance(rhs, int | float):
                left = float(lhs)
                right = float(rhs)
                folded_value = {
                    NodeOp.ADD: left + right,
                    NodeOp.SUB: left - right,
                    NodeOp.MUL: left * right,
                    NodeOp.DIV: left / right,
                    NodeOp.MIN: min(left, right),
                    NodeOp.MAX: max(left, right),
                }[op]
                return self._emit_const(value=folded_value)
            return self._fold_numeric_identity(op, args, lhs, rhs)

        if op in {NodeOp.LT, NodeOp.LE, NodeOp.GT, NodeOp.GE, NodeOp.EQ, NodeOp.NE}:
            lhs, rhs = const_args
            if isinstance(lhs, bool) and isinstance(rhs, bool):
                left = lhs
                right = rhs
            elif isinstance(lhs, int | float) and isinstance(rhs, int | float):
                left = float(lhs)
                right = float(rhs)
            else:
                return None
            folded_value = {
                NodeOp.LT: left < right,
                NodeOp.LE: left <= right,
                NodeOp.GT: left > right,
                NodeOp.GE: left >= right,
                NodeOp.EQ: left == right,
                NodeOp.NE: left != right,
            }[op]
            return self._emit_const(value=folded_value)

        if op in {NodeOp.AND, NodeOp.OR}:
            lhs, rhs = const_args
            if isinstance(lhs, bool) and isinstance(rhs, bool):
                folded_value = (lhs and rhs) if op is NodeOp.AND else (lhs or rhs)
                return self._emit_const(value=folded_value)
            if isinstance(lhs, bool):
                if op is NodeOp.AND:
                    return args[1] if lhs else self._emit_const(value=False)
                return self._emit_const(value=True) if lhs else args[1]
            if isinstance(rhs, bool):
                if op is NodeOp.AND:
                    return args[0] if rhs else self._emit_const(value=False)
                return self._emit_const(value=True) if rhs else args[0]

        return None

    def _fold_numeric_identity(
        self,
        op: NodeOp,
        args: tuple[str, ...],
        lhs: object | None,
        rhs: object | None,
    ) -> str | None:
        if isinstance(lhs, int | float):
            left = float(lhs)
            if op is NodeOp.ADD and left == 0.0:
                return args[1]
            if op is NodeOp.MUL:
                if left == 0.0:
                    return self._emit_const(value=0.0)
                if left == 1.0:
                    return args[1]
            if op is NodeOp.MIN and math.isinf(left) and left > 0:
                return args[1]
            if op is NodeOp.MAX and math.isinf(left) and left < 0:
                return args[1]

        if isinstance(rhs, int | float):
            right = float(rhs)
            if op in {NodeOp.ADD, NodeOp.SUB} and right == 0.0:
                return args[0]
            if op in {NodeOp.MUL, NodeOp.DIV} and right == 1.0:
                return args[0]
            if op is NodeOp.MUL and right == 0.0:
                return self._emit_const(value=0.0)
            if op is NodeOp.MIN and math.isinf(right) and right > 0:
                return args[0]
            if op is NodeOp.MAX and math.isinf(right) and right < 0:
                return args[0]

        return None

    @staticmethod
    def _require_slot(slots: Mapping[str, int | None], canonical_name: str, label: str) -> int:
        slot = slots.get(canonical_name)
        if slot is None:
            raise KeyError(f"Missing {label} slot for {canonical_name}")
        return slot


def _sanitize_name(name: str) -> str:
    return (
        name.replace("'", "_next")
        .replace("/", "_")
        .replace("-", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace(",", "_")
    )


def _grounded_expr_name(expr: Expression) -> str:
    functor, params = expr.args
    if params is None:
        return str(functor)
    resolved = []
    for param in params:
        if not isinstance(param, str):
            raise NotImplementedError(f"Unsupported grounded pvariable parameter: {param!r}")
        if param.startswith("?"):
            raise NotImplementedError(f"Encountered ungrounded parameter {param!r} during lowering.")
        resolved.append(param)
    return f"{functor}[{','.join(resolved)}]"


def _merge_metadata_overrides(*values: Mapping[str, object] | None) -> dict[str, object]:
    merged: dict[str, object] = {}
    for value in values:
        if not value:
            continue
        merged = _deep_merge(merged, dict(value))
    return merged


def _deep_merge(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    result = dict(left)
    for key, value in right.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            result[key] = _deep_merge(existing, dict(value))
        else:
            result[key] = value
    return result


def _extract_rddl_plus_metadata(parsed: ParsedRDDLProgram) -> dict[str, object]:
    merged: dict[str, object] = {}
    for text in (
        parsed.source.domain_text,
        parsed.source.nonfluents_text or "",
        parsed.source.instance_text,
    ):
        merged = _deep_merge(merged, _extract_directives(text))
    return merged


def _extract_directives(text: str) -> dict[str, object]:
    payload: dict[str, object] = {}
    for line in text.splitlines():
        match = _DIRECTIVE_PATTERN.match(line)
        if not match:
            continue
        dotted_key, raw_value = match.groups()
        _assign_directive(payload, dotted_key.split("."), _parse_directive_value(raw_value))
    return payload


def _assign_directive(target: dict[str, object], path: list[str], value: object) -> None:
    current = target
    for key in path[:-1]:
        existing = current.get(key)
        if not isinstance(existing, dict):
            existing = {}
            current[key] = existing
        current = existing
    current[path[-1]] = value


def _parse_directive_value(raw_value: str) -> object:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        lowered = raw_value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            if any(ch in raw_value for ch in (".", "e", "E")):
                return float(raw_value)
            return int(raw_value)
        except ValueError:
            return raw_value
