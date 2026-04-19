"""Emit the training binding glue for generated Ocean environments."""

from __future__ import annotations

from rddl2puffer.backends.puffer_c.emit_helpers import sanitize_c_identifier
from rddl2puffer.backends.puffer_c.model import build_env_spec
from rddl2puffer.ir.nodes import IRProgram


def emit_binding_c(program: IRProgram, env_name: str) -> str:
    """Emit a minimal static Puffer binding for a generated env."""

    spec = build_env_spec(program, env_name)
    lines = [
        f'#include "{env_name}.h"',
        f"#define OBS_SIZE {program.observation_layout.total_size}",
        f"#define NUM_ATNS {spec.num_actions}",
        f"#define ACT_SIZES {spec.act_sizes_literal}",
        f"#define OBS_TENSOR_T {spec.observation_tensor_type}",
        "",
        f"#define Env {spec.struct_name}",
        '#include "vecenv.h"',
        "",
        "void my_init(Env* env, Dict* kwargs) {",
        "    (void)kwargs;",
        "    memset(&env->log, 0, sizeof(Log));",
        "    env->num_agents = 1;",
        "    env->episode_return = 0.0f;",
        "}",
        "",
        "void my_log(Log* log, Dict* out) {",
        '    dict_set(out, "perf", log->perf);',
        '    dict_set(out, "score", log->score);',
        '    dict_set(out, "episode_return", log->episode_return);',
        '    dict_set(out, "episode_length", log->episode_length);',
    ]
    lines.extend(
        f'    dict_set(out, "{counter.name}", log->{sanitize_c_identifier(counter.name)});'
        for counter in spec.log_counters
    )
    lines.extend(
        [
            '    dict_set(out, "n", log->n);',
            "}",
            "",
        ]
    )
    return "\n".join(lines)
