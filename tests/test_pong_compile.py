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
