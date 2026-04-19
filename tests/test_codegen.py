from rddl2puffer.backends.puffer_c import render_env_bundle
from rddl2puffer.benchmarks.cartpole_rddl import compile_cartpole_parity_program
from rddl2puffer.frontend.compile import compile_rddl_sources


_STUB_DOMAIN = """
domain counter_codegen {
    requirements = { reward-deterministic };

    pvariables {
        counter : { state-fluent, int, default = 0 };
        delta : { action-fluent, int, default = 0 };
    };

    cpfs {
        counter' = counter + delta;
    };

    reward = 0.0;

    termination {
        counter + delta > 100;
    };

    action-preconditions {
        delta >= 0;
        delta <= 1;
    };
}
"""

_STUB_INSTANCE = """
non-fluents counter_codegen_nf {
    domain = counter_codegen;
}

instance counter_codegen_inst {
    domain = counter_codegen;
    non-fluents = counter_codegen_nf;
    init-state {
        counter = 0;
    };
    max-nondef-actions = pos-inf;
    horizon = 10;
    discount = 0.95;
}
"""


def test_render_env_bundle_contains_expected_files() -> None:
    program = compile_rddl_sources(_STUB_DOMAIN, _STUB_INSTANCE, env_name="stub_env")
    bundle = render_env_bundle(program, env_name="rddl_counter")

    assert set(bundle) == {
        "config/rddl_counter.ini",
        "ocean/rddl_counter/binding.c",
        "ocean/rddl_counter/rddl_counter.c",
        "ocean/rddl_counter/rddl_counter.h",
    }
    assert "#define RDDL_COUNTER_STATE_SIZE 1" in bundle["ocean/rddl_counter/rddl_counter.h"]
    assert "#define NUM_ATNS 1" in bundle["ocean/rddl_counter/binding.c"]
    assert "env_name = rddl_counter" in bundle["config/rddl_counter.ini"]
    assert "use_gpu = False" in bundle["config/rddl_counter.ini"]


def test_render_env_bundle_merges_metadata_config_overrides() -> None:
    program = compile_rddl_sources(
        _STUB_DOMAIN,
        _STUB_INSTANCE,
        env_name="stub_env",
        metadata_overrides={
            "puffer_config": {
                "vec": {"num_threads": 8},
                "train": {"total_timesteps": 123456},
                "sweep": {"downsample": 9},
            },
        },
    )
    bundle = render_env_bundle(program, env_name="rddl_counter")

    ini = bundle["config/rddl_counter.ini"]
    assert "num_threads = 8" in ini
    assert "total_timesteps = 123456" in ini
    assert "downsample = 9" in ini


def test_generated_cartpole_timeout_is_not_marked_terminal() -> None:
    program = compile_cartpole_parity_program()
    bundle = render_env_bundle(program, env_name="rddl_cartpole_discrete")

    header = bundle["ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h"]
    assert "bool terminated = false;" in header
    assert "bool truncated = false;" in header
    assert "truncated = (RDDL_CARTPOLE_DISCRETE_HORIZON > 0) && (env->tick >= RDDL_CARTPOLE_DISCRETE_HORIZON);" in header
    assert "done = terminated || truncated;" in header
    assert "env->terminals[0] = terminated ? 1.0f : 0.0f;" in header
    assert "float action_0 = env->actions[0];" in header
    assert "if (!isfinite(action_0)) { action_0 = 0.0f; }" in header
    assert "action_0 = fmaxf(action_0, 0.0f);" in header
    assert "action_0 = fminf(action_0, 1.0f);" in header
