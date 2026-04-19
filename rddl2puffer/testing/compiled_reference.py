"""Reference runtime for compiled RDDL programs."""

from __future__ import annotations

import random
from typing import Mapping, Sequence

from rddl2puffer.frontend.schema import FluentSpec, LayoutSpec, Scalar
from rddl2puffer.ir.interpret import step_ir
from rddl2puffer.ir.nodes import IRProgram
from rddl2puffer.testing.differential import ReferenceEnv, ResetResult, Transition


class CompiledProgramReferenceEnv(ReferenceEnv):
    """Execute one compiled IR program with Puffer-like auto-reset semantics."""

    def __init__(self, program: IRProgram) -> None:
        self._program = program
        self._horizon = int(program.metadata.get("horizon", 0))
        self._state = _initial_state(program.state_layout)
        self._step_count = 0
        self._rng = random.Random(0)

    def reset(self, seed: int | None = None) -> ResetResult:
        self._rng = random.Random(0 if seed is None else seed)
        self._state = _initial_state(self._program.state_layout)
        self._step_count = 0
        return ResetResult(
            observation=_reset_observation(self._program, self._state),
            hidden_state=_hidden_state(
                self._program,
                self._state,
                terminated=False,
                truncated=False,
            ),
        )

    def step(self, action: Sequence[Scalar]) -> Transition:
        result = step_ir(self._program, state=self._state, action=action, rng=self._rng)
        proposed_state = result.next_state
        self._step_count += 1

        terminated = bool(result.done)
        truncated = self._horizon > 0 and self._step_count >= self._horizon
        done = terminated or truncated

        if done:
            reset_state = _initial_state(self._program.state_layout)
            reset_observation = _reset_observation(self._program, reset_state)
            self._state = reset_state
            self._step_count = 0
            return Transition(
                observation=reset_observation,
                reward=result.reward,
                done=True,
                hidden_state=_hidden_state(
                    self._program,
                    reset_state,
                    terminated=terminated,
                    truncated=truncated,
                ),
            )

        self._state = proposed_state
        return Transition(
            observation=tuple(result.observation),
            reward=result.reward,
            done=False,
            hidden_state=_hidden_state(
                self._program,
                proposed_state,
                terminated=False,
                truncated=False,
            ),
        )


def _initial_state(layout: LayoutSpec) -> tuple[Scalar, ...]:
    return tuple(fluent.default for fluent in layout.fluents)


def _reset_observation(program: IRProgram, state: Sequence[Scalar]) -> tuple[Scalar, ...]:
    state_slots = {
        fluent.qualified_name: fluent.flat_index
        for fluent in program.state_layout.fluents
        if fluent.flat_index is not None
    }
    values: list[Scalar] = [0] * program.observation_layout.total_size
    for fluent in program.observation_layout.fluents:
        slot = fluent.flat_index
        if slot is None:
            continue
        state_slot = state_slots.get(fluent.qualified_name)
        values[slot] = state[state_slot] if state_slot is not None else fluent.default
    return tuple(values)


def _hidden_state(
    program: IRProgram,
    state: Sequence[Scalar],
    *,
    terminated: bool,
    truncated: bool,
) -> Mapping[str, object]:
    payload = {
        fluent.qualified_name: state[fluent.flat_index]
        for fluent in program.state_layout.fluents
        if fluent.flat_index is not None
    }
    payload["terminated"] = terminated
    payload["truncated"] = truncated
    return payload
