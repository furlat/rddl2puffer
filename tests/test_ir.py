from rddl2puffer.frontend.compile import compile_rddl_sources
from rddl2puffer.ir.interpret import step_ir


_DOMAIN = """
domain toy_counter {
    requirements = { reward-deterministic };

    pvariables {
        counter : { state-fluent, int, default = 0 };
        delta : { action-fluent, int, default = 0 };
    };

    cpfs {
        counter' = counter + delta;
    };

    reward = if (counter + delta > 2) then 10 else 1;

    termination {
        counter + delta > 2;
    };

    action-preconditions {
        delta >= -2;
        delta <= 2;
    };
}
"""

_INSTANCE = """
non-fluents toy_counter_nf {
    domain = toy_counter;
}

instance toy_counter_inst {
    domain = toy_counter;
    non-fluents = toy_counter_nf;

    init-state {
        counter = 1;
    };

    max-nondef-actions = pos-inf;
    horizon = 5;
    discount = 1.0;
}
"""


def test_step_ir_evaluates_compiled_rddl_program() -> None:
    program = compile_rddl_sources(_DOMAIN, _INSTANCE, env_name="toy_counter")

    result = step_ir(program, state=(1,), action=(2,))

    assert result.next_state == (3,)
    assert result.observation == (3,)
    assert result.reward == 10.0
    assert result.done is True
