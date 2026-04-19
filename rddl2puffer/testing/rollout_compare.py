"""Rollout recording and exact differential comparison utilities."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from rddl2puffer.frontend.schema import Scalar
from rddl2puffer.testing.differential import (
    Mismatch,
    MismatchReport,
    ReferenceEnv,
    RolloutTrace,
    TraceStep,
)


def rollout(
    env: ReferenceEnv,
    actions: Iterable[Sequence[Scalar]],
    seed: int | None = None,
) -> RolloutTrace:
    """Record a seeded exact rollout trace."""

    reset_result = env.reset(seed)
    steps: list[TraceStep] = []

    for step_index, action in enumerate(actions):
        action_tuple = tuple(action)
        transition = env.step(action_tuple)
        steps.append(TraceStep(step_index=step_index, action=action_tuple, transition=transition))
        if transition.done:
            break

    return RolloutTrace(reset=reset_result, steps=tuple(steps))


def compare_rollouts(
    left_env: ReferenceEnv,
    right_env: ReferenceEnv,
    actions: Iterable[Sequence[Scalar]],
    *,
    seed: int | None = None,
    left_name: str = "left",
    right_name: str = "right",
    include_hidden_state: bool = True,
) -> MismatchReport:
    """Run two seeded rollouts and compare them field by field."""

    cached_actions = [tuple(action) for action in actions]
    left_trace = rollout(left_env, cached_actions, seed=seed)
    right_trace = rollout(right_env, cached_actions, seed=seed)

    mismatches: list[Mismatch] = []
    _compare_value(
        mismatches,
        location="reset",
        field="observation",
        left=left_trace.reset.observation,
        right=right_trace.reset.observation,
    )
    if include_hidden_state:
        _compare_value(
            mismatches,
            location="reset",
            field="hidden_state",
            left=left_trace.reset.hidden_state,
            right=right_trace.reset.hidden_state,
        )

    max_steps = max(len(left_trace.steps), len(right_trace.steps))
    for index in range(max_steps):
        if index >= len(left_trace.steps) or index >= len(right_trace.steps):
            left_value = len(left_trace.steps)
            right_value = len(right_trace.steps)
            mismatches.append(
                Mismatch(location="rollout", field="length", left=left_value, right=right_value)
            )
            break

        left_step = left_trace.steps[index]
        right_step = right_trace.steps[index]
        location = f"step {index}"

        _compare_value(
            mismatches,
            location=location,
            field="action",
            left=left_step.action,
            right=right_step.action,
        )
        _compare_value(
            mismatches,
            location=location,
            field="observation",
            left=left_step.transition.observation,
            right=right_step.transition.observation,
        )
        _compare_value(
            mismatches,
            location=location,
            field="reward",
            left=left_step.transition.reward,
            right=right_step.transition.reward,
        )
        _compare_value(
            mismatches,
            location=location,
            field="done",
            left=left_step.transition.done,
            right=right_step.transition.done,
        )
        if include_hidden_state:
            _compare_value(
                mismatches,
                location=location,
                field="hidden_state",
                left=left_step.transition.hidden_state,
                right=right_step.transition.hidden_state,
            )

    return MismatchReport(
        left_name=left_name,
        right_name=right_name,
        mismatches=tuple(mismatches),
        left_trace=left_trace,
        right_trace=right_trace,
    )


def format_mismatch_report(report: MismatchReport) -> str:
    """Render the mismatch report as user-facing text."""

    return report.to_text()


def _compare_value(
    mismatches: list[Mismatch],
    *,
    location: str,
    field: str,
    left: object,
    right: object,
) -> None:
    if not _values_match(left, right):
        mismatches.append(Mismatch(location=location, field=field, left=left, right=right))


def _values_match(left: object, right: object, *, atol: float = 1e-9) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), abs_tol=atol, rel_tol=0.0)
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return all(_values_match(lhs, rhs, atol=atol) for lhs, rhs in zip(left, right, strict=True))
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return all(_values_match(lhs, rhs, atol=atol) for lhs, rhs in zip(left, right, strict=True))
    if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        return all(_values_match(left[key], right[key], atol=atol) for key in left)
    return left == right
