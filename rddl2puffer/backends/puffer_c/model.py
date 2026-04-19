"""Backend-specific metadata derived from the generic IR program."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rddl2puffer.frontend.schema import FluentDType, FluentSpec
from rddl2puffer.ir.nodes import IRProgram


@dataclass(frozen=True, slots=True)
class ActionHeadSpec:
    """One Puffer action head emitted from a scalar action fluent."""

    name: str
    size: int
    dtype: FluentDType
    flat_index: int
    default_value: float
    lower_bound: float | None
    upper_bound: float | None


@dataclass(frozen=True, slots=True)
class LogCounterSpec:
    """One metadata-driven episode counter exported through Puffer logs."""

    name: str
    source: str


@dataclass(frozen=True, slots=True)
class ResetObservationSpec:
    """How one observation slot should be initialized on reset."""

    observation_name: str
    observation_slot: int
    state_slot: int | None
    default_value: float


@dataclass(frozen=True, slots=True)
class GeneratedEnvSpec:
    """Concrete codegen contract for one generated Ocean environment."""

    env_name: str
    struct_name: str
    prefix: str
    action_heads: tuple[ActionHeadSpec, ...]
    initial_state: tuple[float, ...]
    reset_observations: tuple[ResetObservationSpec, ...]
    horizon: int
    discount: float
    zero_reward_on_done: bool
    score_mode: str
    perf_mode: str
    log_counters: tuple[LogCounterSpec, ...]
    observation_tensor_type: str = "FloatTensor"

    @property
    def num_actions(self) -> int:
        return len(self.action_heads)

    @property
    def act_sizes_literal(self) -> str:
        values = ", ".join(str(head.size) for head in self.action_heads)
        return "{" + values + "}"


def build_env_spec(program: IRProgram, env_name: str) -> GeneratedEnvSpec:
    """Derive the backend-specific codegen contract from a generic program."""

    _validate_scalar_layouts(program)
    state_by_name = {
        fluent.qualified_name: fluent.flat_index for fluent in program.state_layout.fluents
    }
    reset_observations: list[ResetObservationSpec] = []
    for fluent in program.observation_layout.fluents:
        if fluent.flat_index is None:
            raise ValueError(f"Observation fluent {fluent.qualified_name} is missing a flat index.")
        state_slot = state_by_name.get(fluent.qualified_name)
        reset_observations.append(
            ResetObservationSpec(
                observation_name=fluent.qualified_name,
                observation_slot=fluent.flat_index,
                state_slot=state_slot,
                default_value=float(fluent.default),
            )
        )

    action_heads = tuple(_build_action_head(fluent) for fluent in program.action_layout.fluents)
    initial_state = tuple(float(fluent.default) for fluent in program.state_layout.fluents)

    horizon = int(program.metadata.get("horizon", 0))
    discount = float(program.metadata.get("discount", 1.0))
    runtime = _mapping_metadata(program.metadata.get("runtime_semantics"))
    logging = _mapping_metadata(program.metadata.get("puffer_logging"))

    return GeneratedEnvSpec(
        env_name=env_name,
        struct_name=_camel_case(env_name),
        prefix=env_name.upper(),
        action_heads=action_heads,
        initial_state=initial_state,
        reset_observations=tuple(reset_observations),
        horizon=horizon,
        discount=discount,
        zero_reward_on_done=bool(runtime.get("zero_reward_on_done", False)),
        score_mode=str(logging.get("score_mode", "episode_return")),
        perf_mode=str(logging.get("perf_mode", "episode_return_div_horizon")),
        log_counters=_parse_log_counters(logging),
    )


def _validate_scalar_layouts(program: IRProgram) -> None:
    for layout_name, layout in (
        ("state", program.state_layout),
        ("action", program.action_layout),
        ("observation", program.observation_layout),
    ):
        for fluent in layout.fluents:
            if fluent.size != 1:
                raise NotImplementedError(
                    f"Puffer C codegen currently supports only scalar {layout_name} fluents. "
                    f"Received {fluent.qualified_name} with size {fluent.size}."
                )


def _build_action_head(fluent: FluentSpec) -> ActionHeadSpec:
    if fluent.dtype is FluentDType.REAL:
        size = 1
    elif fluent.dtype is FluentDType.BOOL:
        size = 2
    elif fluent.dtype is FluentDType.INT:
        lower, upper = fluent.bounds
        if lower is None or upper is None:
            raise NotImplementedError(
                f"Integer action fluent {fluent.qualified_name} requires explicit bounds for codegen."
            )
        size = int(upper) - int(lower) + 1
    else:
        raise NotImplementedError(f"Unsupported action dtype: {fluent.dtype}")

    if fluent.flat_index is None:
        raise ValueError(f"Action fluent {fluent.qualified_name} is missing a flat index.")

    return ActionHeadSpec(
        name=fluent.qualified_name,
        size=size,
        dtype=fluent.dtype,
        flat_index=fluent.flat_index,
        default_value=float(fluent.default),
        lower_bound=None if fluent.bounds[0] is None else float(fluent.bounds[0]),
        upper_bound=None if fluent.bounds[1] is None else float(fluent.bounds[1]),
    )


def _mapping_metadata(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _parse_log_counters(logging: Mapping[str, object]) -> tuple[LogCounterSpec, ...]:
    raw = logging.get("log_counters", ())
    if not isinstance(raw, list | tuple):
        return ()

    counters: list[LogCounterSpec] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        source = item.get("source")
        if not isinstance(name, str) or not isinstance(source, str):
            continue
        counters.append(LogCounterSpec(name=name, source=source))
    return tuple(counters)


def _camel_case(name: str) -> str:
    parts = [part for part in name.replace("-", "_").split("_") if part]
    if not parts:
        return "GeneratedEnv"
    return "".join(part[:1].upper() + part[1:] for part in parts)
