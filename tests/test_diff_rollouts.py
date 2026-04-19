from typing import Sequence

from rddl2puffer.frontend.schema import Scalar
from rddl2puffer.testing.differential import ReferenceEnv, ResetResult, Transition
from rddl2puffer.testing.rollout_compare import compare_rollouts, format_mismatch_report


class CounterEnv(ReferenceEnv):
    def __init__(self, *, bonus: int = 0) -> None:
        self._bonus = bonus
        self._state = 0

    def reset(self, seed: int | None = None) -> ResetResult:
        self._state = 0 if seed is None else seed % 2
        return ResetResult(
            observation=(self._state,),
            hidden_state={"counter": self._state},
        )

    def step(self, action: Sequence[Scalar]) -> Transition:
        self._state += int(action[0]) + self._bonus
        return Transition(
            observation=(self._state,),
            reward=float(self._state),
            done=self._state >= 3,
            hidden_state={"counter": self._state},
        )


def test_compare_rollouts_matches_identical_envs() -> None:
    report = compare_rollouts(
        CounterEnv(),
        CounterEnv(),
        actions=[(1,), (1,), (1,)],
        seed=4,
        left_name="expected",
        right_name="candidate",
    )

    assert report.matches is True
    assert "matched exactly" in format_mismatch_report(report)


def test_compare_rollouts_reports_human_readable_mismatches() -> None:
    report = compare_rollouts(
        CounterEnv(),
        CounterEnv(bonus=1),
        actions=[(1,), (1,)],
        seed=7,
        left_name="expected",
        right_name="candidate",
    )

    text = format_mismatch_report(report)

    assert report.matches is False
    assert "step 0 observation" in text
    assert "expected=(2,)" in text
    assert "candidate=(3,)" in text

