"""Emit a standalone local runner for a generated Ocean environment."""

from __future__ import annotations

from textwrap import dedent

from rddl2puffer.backends.puffer_c.model import build_env_spec
from rddl2puffer.ir.nodes import IRProgram


def emit_env_c(program: IRProgram, env_name: str) -> str:
    """Emit a standalone `.c` harness for local debugging."""

    spec = build_env_spec(program, env_name)

    return dedent(
        f"""\
        #include <stdio.h>
        #include <stdlib.h>
        #include "{env_name}.h"

        int main(void) {{
            {spec.struct_name} env = {{0}};
            float observations[{program.observation_layout.total_size}] = {{0}};
            float actions[{program.action_layout.total_size}] = {{0}};
            float rewards[1] = {{0}};
            float terminals[1] = {{0}};
            env.observations = observations;
            env.actions = actions;
            env.rewards = rewards;
            env.terminals = terminals;
            env.num_agents = 1;
            c_reset(&env);
            printf("Generated standalone for %s\\n", "{env_name}");
            printf("initial obs:");
            for (int i = 0; i < {program.observation_layout.total_size}; i++) {{
                printf(" %0.6f", env.observations[i]);
            }}
            printf("\\n");
            c_step(&env);
            printf("after one zero-action step:");
            for (int i = 0; i < {program.observation_layout.total_size}; i++) {{
                printf(" %0.6f", env.observations[i]);
            }}
            printf(" | reward=%0.6f terminal=%0.0f\\n", env.rewards[0], env.terminals[0]);
            return 0;
        }}
        """
    )
