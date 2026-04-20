from pathlib import Path
import random

from rddl2puffer.backends.puffer_c import render_env_bundle
from rddl2puffer.frontend.compile import compile_rddl_files
from rddl2puffer.ir.interpret import step_ir
from rddl2puffer.ir.nodes import NodeOp


def test_compile_raw_pong_domain_and_emit_c_bundle() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    domain_path = (
        repo_root
        / "third_party"
        / "rddlrepository"
        / "rddlrepository"
        / "archive"
        / "arcade"
        / "Pong"
        / "domain.rddl"
    )
    instance_path = domain_path.with_name("instance0.rddl")

    program = compile_rddl_files(domain_path, instance_path, env_name="rddl_pong_raw")

    assert program.state_layout.total_size == 5
    assert program.action_layout.total_size == 1
    assert program.observation_layout.total_size == 5
    assert [fluent.qualified_name for fluent in program.state_layout.fluents] == [
        "ball-x[b1]",
        "ball-y[b1]",
        "vel-x[b1]",
        "vel-y[b1]",
        "paddle-y",
    ]
    assert sum(1 for node in program.nodes if node.op is NodeOp.SAMPLE) == 1

    bundle = render_env_bundle(program, env_name="rddl_pong_raw")
    header = bundle["ocean/rddl_pong_raw/rddl_pong_raw.h"]
    assert "sample_uniform(&env->rng" in header
    assert "float ball_x_b1;" in header


def test_step_ir_supports_uniform_sample_nodes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    domain_path = (
        repo_root
        / "third_party"
        / "rddlrepository"
        / "rddlrepository"
        / "archive"
        / "arcade"
        / "Pong"
        / "domain.rddl"
    )
    instance_path = domain_path.with_name("instance0.rddl")

    program = compile_rddl_files(domain_path, instance_path, env_name="rddl_pong_raw")
    defaults = tuple(fluent.default for fluent in program.state_layout.fluents)
    result = step_ir(program, defaults, (0,), rng=random.Random(0))

    assert len(result.next_state) == 5
    assert len(result.observation) == 5
    assert isinstance(result.reward, float)


def test_compile_pong_puffer_parity_domain_and_emit_reset_observations() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    domain_path = repo_root / "examples" / "rddl_plus" / "pong_puffer_parity" / "domain.rddl"
    instance_path = domain_path.with_name("instance_deterministic_reset.rddl")

    program = compile_rddl_files(domain_path, instance_path, env_name="rddl_pong_parity")

    assert program.state_layout.total_size == 8
    assert program.observation_layout.total_size == 8
    assert program.metadata["runtime_semantics"]["reset_tick_on"] == "$reward_nonzero"
    assert program.metadata["puffer_codegen"]["step_template"] == "pong_native_loop"
    assert program.metadata["puffer_codegen"]["substeps"] == 8

    bundle = render_env_bundle(program, env_name="rddl_pong_parity")
    header = bundle["ocean/rddl_pong_parity/rddl_pong_parity.h"]
    ini = bundle["config/rddl_pong_parity.ini"]
    assert "static inline void reset_round(RddlPongParity* env)" in header
    assert "static inline void compute_observations(RddlPongParity* env)" in header
    assert "for (int i = 0; i < 8; i++) {" in header
    assert "env->observations[0] = (env->paddle_yl + 35.0f) / 640.0f;" in header
    assert "env->observations[2] = env->ball_x / 500.0f;" in header
    assert "env->log.perf += perf_den > 0.0f ? (env->score_r) / perf_den : 0.0f;" in header
    assert "float v_0_load_state_paddle_yl" not in header
    assert "use_rnn = 1" in ini
