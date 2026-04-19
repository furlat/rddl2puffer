from rddl2puffer.benchmarks.cartpole_rddl import (
    compile_cartpole_parity_program,
    make_cartpole_parity_compiled_env,
    make_cartpole_parity_reference_env,
)
from rddl2puffer.testing.rollout_compare import compare_rollouts


def test_cartpole_schema_preserves_source_order() -> None:
    program = compile_cartpole_parity_program()

    assert program.state_layout.offsets == {
        "pos": 0,
        "vel": 1,
        "ang-pos": 2,
        "ang-vel": 3,
    }
    assert program.action_layout.offsets == {"force-side": 0}
    assert program.observation_layout.offsets == {
        "pos": 0,
        "vel": 1,
        "ang-pos": 2,
        "ang-vel": 3,
    }


def test_compiled_cartpole_matches_pyrddlgym_reference_on_seeded_rollout() -> None:
    report = compare_rollouts(
        make_cartpole_parity_compiled_env(),
        make_cartpole_parity_reference_env(),
        actions=[(0,), (1,), (0,), (1,), (0,), (1,), (0,), (1,)],
        seed=123,
        left_name="compiled_rddl",
        right_name="pyrddlgym",
    )

    assert report.matches, report.to_text()


def test_compiled_cartpole_done_step_returns_reset_observation_and_zero_reward() -> None:
    env = make_cartpole_parity_compiled_env()
    env.reset(seed=123)

    transition = None
    for _ in range(256):
        transition = env.step((0,))
        if transition.done:
            break

    assert transition is not None
    assert transition.done is True
    assert transition.reward == 0.0
    assert transition.observation == (0.0, 0.0, 0.0, 0.0)
