"""Reference runtime adapters backed by pyRDDLGym."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pyRDDLGym
from pyRDDLGym.core.compiler.model import RDDLPlanningModel

from rddl2puffer.frontend.schema import EnvSchema, FluentDType, FluentSpec, Scalar
from rddl2puffer.testing.differential import ReferenceEnv, ResetResult, Transition


@dataclass(frozen=True, slots=True)
class FlatValue:
    """One flattened fluent value captured from pyRDDLGym."""

    spec: FluentSpec
    value: Scalar


class PyRDDLGymReferenceEnv(ReferenceEnv):
    """Adapt a pyRDDLGym environment to the project differential-testing API."""

    def __init__(
        self,
        *,
        domain: str,
        instance: str,
        schema: EnvSchema,
        auto_reset_on_done: bool = False,
    ) -> None:
        self._domain = domain
        self._instance = instance
        self._schema = schema
        self._auto_reset_on_done = auto_reset_on_done
        self._env = pyRDDLGym.make(domain, instance)

    def reset(self, seed: int | None = None) -> ResetResult:
        observation, _info = self._env.reset(seed=seed)
        hidden_state = self._build_hidden_state(
            terminated=False,
            truncated=False,
        )
        return ResetResult(
            observation=self._flatten_values(self._schema.observation_layout.fluents, observation),
            hidden_state=hidden_state,
        )

    def step(self, action: Sequence[Scalar]) -> Transition:
        action_dict = self._unflatten_action(action)
        observation, reward, terminated, truncated, _info = self._env.step(action_dict)
        done = bool(terminated or truncated)
        if self._auto_reset_on_done and done:
            observation, _info = self._env.reset()
            hidden_state = self._build_hidden_state(terminated=terminated, truncated=truncated)
        else:
            hidden_state = self._build_hidden_state(terminated=terminated, truncated=truncated)
        return Transition(
            observation=self._flatten_values(self._schema.observation_layout.fluents, observation),
            reward=float(reward),
            done=done,
            hidden_state=hidden_state,
        )

    def close(self) -> None:
        self._env.close()

    def _unflatten_action(self, action: Sequence[Scalar]) -> dict[str, Scalar]:
        if len(action) != self._schema.action_layout.total_size:
            raise ValueError(
                f"Action length {len(action)} does not match layout size "
                f"{self._schema.action_layout.total_size}."
            )

        action_dict: dict[str, Scalar] = {}
        for fluent in self._schema.action_layout.fluents:
            if fluent.size != 1:
                raise NotImplementedError(
                    "The pyRDDLGym adapter currently supports only scalar action fluents."
                )
            slot = fluent.flat_index
            if slot is None:
                raise ValueError(f"Action fluent {fluent.qualified_name} is missing a flat index.")
            action_dict[_reference_key(fluent)] = _coerce_scalar(action[slot], fluent.dtype)
        return action_dict

    def _build_hidden_state(self, *, terminated: bool, truncated: bool) -> Mapping[str, object]:
        state = getattr(self._env, "state", None)
        if not isinstance(state, Mapping):
            state = {}
        flat_state = {
            item.spec.qualified_name: item.value
            for item in self._iter_flat_values(self._schema.state_layout.fluents, state)
        }
        flat_state["terminated"] = bool(terminated)
        flat_state["truncated"] = bool(truncated)
        return flat_state

    def _flatten_values(
        self,
        fluents: Sequence[FluentSpec],
        values: Mapping[str, Any],
    ) -> tuple[Scalar, ...]:
        return tuple(item.value for item in self._iter_flat_values(fluents, values))

    def _iter_flat_values(
        self,
        fluents: Sequence[FluentSpec],
        values: Mapping[str, Any],
    ) -> tuple[FlatValue, ...]:
        flattened: list[FlatValue] = []
        for fluent in fluents:
            if fluent.size != 1:
                raise NotImplementedError(
                    "The pyRDDLGym adapter currently supports only scalar state and observation fluents."
                )
            raw_value = values[_reference_key(fluent)]
            flattened.append(
                FlatValue(
                    spec=fluent,
                    value=_coerce_scalar(raw_value, fluent.dtype),
                )
            )
        return tuple(flattened)


def _reference_key(fluent: FluentSpec) -> str:
    if fluent.parameters:
        return RDDLPlanningModel.ground_var(fluent.name, fluent.parameters)
    return fluent.name


def _coerce_scalar(value: Any, dtype: FluentDType) -> Scalar:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if dtype is FluentDType.BOOL:
        return bool(value)
    if dtype is FluentDType.INT:
        return int(value)
    if dtype is FluentDType.REAL:
        return float(value)
    raise TypeError(f"Unsupported fluent dtype: {dtype}")
