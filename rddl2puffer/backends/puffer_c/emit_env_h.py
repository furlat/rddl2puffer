"""Emit the core generated Ocean environment header."""

from __future__ import annotations

from textwrap import dedent, indent

from rddl2puffer.backends.puffer_c.emit_helpers import (
    _float_literal,
    action_var_name,
    emit_c_store_statements,
    emit_c_value_declarations,
    infer_c_node_values,
    lookup_var_name,
    required_node_ids,
    sanitize_c_identifier,
)
from rddl2puffer.backends.puffer_c.model import GeneratedEnvSpec, build_env_spec
from rddl2puffer.ir.nodes import IRProgram, NodeOp


def emit_env_header(program: IRProgram, env_name: str) -> str:
    """Emit a generic scalar-state Ocean environment from the ordered IR."""

    spec = build_env_spec(program, env_name)
    if spec.step_template == "pong_native_loop":
        return _emit_pong_native_loop_header(program, spec)

    inferred = infer_c_node_values(program)
    extra_roots = _metadata_node_roots(spec)
    live_nodes = required_node_ids(program, extra_roots=extra_roots)
    state_targets = {field.slot: f"env->{field.field_name}" for field in spec.state_fields}
    declarations = emit_c_value_declarations(
        program,
        inferred,
        state_targets=state_targets,
        required_node_ids=live_nodes,
    )
    next_state_targets = (
        {field.slot: f"next_{field.field_name}" for field in spec.state_fields}
        if spec.state_slots_fully_overwritten
        else None
    )
    observation_targets = (
        {slot: f"env->observations[{slot}]" for slot in range(program.observation_layout.total_size)}
        if spec.observation_slots_fully_overwritten
        else None
    )
    stores = emit_c_store_statements(
        program,
        next_state_targets=next_state_targets,
        observation_targets=observation_targets,
    )
    action_sanitizers = _emit_action_sanitizers(spec)
    counter_captures = _emit_counter_captures(program, spec)
    reward_postprocess = _emit_reward_postprocess(spec)
    tick_reset = _emit_tick_reset(program, spec)
    reset_state = "\n".join(
        f"env->{field.field_name} = {_float_literal(field.default_value)};"
        for field in spec.state_fields
    )
    reset_obs = "\n".join(
        (
            f"env->observations[{item.observation_slot}] = env->{item.state_field_name};"
            if item.state_field_name is not None
            else f"env->observations[{item.observation_slot}] = {_float_literal(item.default_value)};"
        )
        for item in spec.reset_observations
    )
    reset_log_fields = "\n".join(
        f"env->{_last_counter_field(counter.name)} = 0.0f;" for counter in spec.log_counters
    )

    reset_state_block = indent(reset_state, " " * 8) if reset_state else ""
    reset_obs_block = indent(reset_obs, " " * 8) if reset_obs else ""
    reset_log_block = indent(reset_log_fields, " " * 8) if reset_log_fields else ""
    action_block = indent(action_sanitizers, " " * 8) if action_sanitizers else ""
    declarations_block = indent(declarations, " " * 8) if declarations else ""
    stores_block = indent(stores, " " * 8) if stores else ""
    reward_block = indent(reward_postprocess, " " * 8) if reward_postprocess else ""
    counter_capture_block = indent(counter_captures, " " * 8) if counter_captures else ""
    tick_reset_block = indent(tick_reset, " " * 8) if tick_reset else ""
    state_scratch_block = indent(
        _emit_next_state_scratch(program, spec),
        " " * 8,
    ) if spec.state_slots_fully_overwritten else ""
    reset_obs_init = (
        ""
        if spec.reset_observation_slots_fully_overwritten
        else f"memset(env->observations, 0, sizeof(float) * {spec.prefix}_OBS_SIZE);"
    )
    reset_obs_init_block = indent(reset_obs_init, " " * 8) if reset_obs_init else ""
    step_state_setup = (
        ""
        if spec.state_slots_fully_overwritten
        else _emit_state_array_seed(spec)
    )
    step_obs_setup = (
        ""
        if spec.observation_slots_fully_overwritten
        else "memset(obs, 0, sizeof(obs));"
    )
    step_state_setup_block = indent(step_state_setup, " " * 8) if step_state_setup else ""
    step_obs_setup_block = indent(step_obs_setup, " " * 8) if step_obs_setup else ""
    state_commit_block = indent(_emit_state_commit(program, spec), " " * 8)
    observation_commit_block = (
        indent(
            f"memcpy(env->observations, obs, sizeof(obs));",
            " " * 8,
        )
        if not spec.observation_slots_fully_overwritten
        else ""
    )
    log_struct_fields = _emit_log_struct_fields(spec)
    env_counter_fields = _emit_env_counter_fields(spec)
    log_updates = _emit_add_log_body(spec)
    sample_helper_block = indent(_emit_sample_helpers(program), " " * 0)

    return dedent(
        f"""\
        #pragma once
        #include <math.h>
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdlib.h>
        #include <stdint.h>
        #include <string.h>

        #define {spec.prefix}_STATE_SIZE {program.state_layout.total_size}
        #define {spec.prefix}_ACTION_SIZE {program.action_layout.total_size}
        #define {spec.prefix}_OBS_SIZE {program.observation_layout.total_size}
        #define {spec.prefix}_HORIZON {spec.horizon}

        typedef struct {{
            float perf;
            float score;
            float episode_return;
            float episode_length;
            float n;
        {log_struct_fields}
        }} Log;

        typedef struct {{
            Log log;
            float* observations;
            float* actions;
            float* rewards;
            float* terminals;
            int num_agents;
            int tick;
            float episode_return;
            unsigned int rng;
        {_emit_state_struct_fields(spec)}
        {env_counter_fields}
        }} {spec.struct_name};

        {sample_helper_block}

        static inline void add_log({spec.struct_name}* env) {{
        {indent(log_updates, " " * 4)}
        }}

        static inline void c_reset({spec.struct_name}* env) {{
            env->tick = 0;
            env->episode_return = 0.0f;
        {reset_state_block}
        {reset_obs_init_block}
        {reset_obs_block}
        {reset_log_block}
        }}

        static inline void c_step({spec.struct_name}* env) {{
        {_emit_step_buffer_declarations(program, spec)}
            float reward = 0.0f;
            bool terminated = false;
            bool truncated = false;
            bool done = false;

        {state_scratch_block}
        {step_state_setup_block}
        {step_obs_setup_block}
            env->tick += 1;
        {action_block}
        {declarations_block}
        {stores_block}
            truncated = ({spec.prefix}_HORIZON > 0) && (env->tick >= {spec.prefix}_HORIZON);
            done = terminated || truncated;
        {reward_block}
        {counter_capture_block}
        {state_commit_block}
        {observation_commit_block}
            env->rewards[0] = reward;
            env->terminals[0] = terminated ? 1.0f : 0.0f;
            env->episode_return += reward;
        {tick_reset_block}

            if (done) {{
                add_log(env);
                c_reset(env);
            }}
        }}

        static inline void c_render({spec.struct_name}* env) {{
            (void)env;
        }}

        static inline void c_close({spec.struct_name}* env) {{
            (void)env;
        }}
        """
    )


def _emit_pong_native_loop_header(program: IRProgram, spec: GeneratedEnvSpec) -> str:
    state_fields = {field.name: field.field_name for field in spec.state_fields}
    required_states = (
        "paddle-yl",
        "paddle-yr",
        "ball-x",
        "ball-y",
        "ball-vx",
        "ball-vy",
        "score-l",
        "score-r",
    )
    missing = [name for name in required_states if name not in state_fields]
    if missing:
        raise ValueError(
            f"Pong native loop template requires state fluents {required_states}, missing {missing}."
        )
    obs_slots = {fluent.qualified_name: fluent.flat_index for fluent in program.observation_layout.fluents}
    required_obs = (
        "obs-paddle-yl",
        "obs-paddle-yr",
        "obs-ball-x",
        "obs-ball-y",
        "obs-ball-vx",
        "obs-ball-vy",
        "obs-score-l",
        "obs-score-r",
    )
    missing_obs = [name for name in required_obs if obs_slots.get(name) is None]
    if missing_obs:
        raise ValueError(
            f"Pong native loop template requires observation fluents {required_obs}, missing {missing_obs}."
        )

    paddle_yl = f"env->{state_fields['paddle-yl']}"
    paddle_yr = f"env->{state_fields['paddle-yr']}"
    ball_x = f"env->{state_fields['ball-x']}"
    ball_y = f"env->{state_fields['ball-y']}"
    ball_vx = f"env->{state_fields['ball-vx']}"
    ball_vy = f"env->{state_fields['ball-vy']}"
    score_l = f"env->{state_fields['score-l']}"
    score_r = f"env->{state_fields['score-r']}"
    sample_helper_block = indent(_emit_sample_helpers(program), " " * 0)
    log_struct_fields = _emit_log_struct_fields(spec)
    env_counter_fields = _emit_env_counter_fields(spec)
    log_updates = _emit_add_log_body(spec)
    reset_log_fields = "\n".join(
        f"env->{_last_counter_field(counter.name)} = 0.0f;" for counter in spec.log_counters
    )
    reset_log_block = indent(reset_log_fields, " " * 8) if reset_log_fields else ""

    obs_lines = [
        f"env->observations[{obs_slots['obs-paddle-yl']}] = ({paddle_yl} + 35.0f) / 640.0f;",
        f"env->observations[{obs_slots['obs-paddle-yr']}] = ({paddle_yr} + 35.0f) / 640.0f;",
        f"env->observations[{obs_slots['obs-ball-x']}] = {ball_x} / 500.0f;",
        f"env->observations[{obs_slots['obs-ball-y']}] = {ball_y} / 640.0f;",
        f"env->observations[{obs_slots['obs-ball-vx']}] = ({ball_vx} + 10.0f) / 20.0f;",
        f"env->observations[{obs_slots['obs-ball-vy']}] = ({ball_vy} + 13.0f) / 26.0f;",
        f"env->observations[{obs_slots['obs-score-l']}] = {score_l} / 21.0f;",
        f"env->observations[{obs_slots['obs-score-r']}] = {score_r} / 21.0f;",
    ]
    compute_observations_block = indent("\n".join(obs_lines), " " * 4)
    substeps = max(1, spec.substeps)

    return dedent(
        f"""\
        #pragma once
        #include <math.h>
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdlib.h>
        #include <stdint.h>
        #include <string.h>

        #define {spec.prefix}_STATE_SIZE {program.state_layout.total_size}
        #define {spec.prefix}_ACTION_SIZE {program.action_layout.total_size}
        #define {spec.prefix}_OBS_SIZE {program.observation_layout.total_size}
        #define {spec.prefix}_HORIZON {spec.horizon}

        typedef struct {{
            float perf;
            float score;
            float episode_return;
            float episode_length;
            float n;
        {log_struct_fields}
        }} Log;

        typedef struct {{
            Log log;
            float* observations;
            float* actions;
            float* rewards;
            float* terminals;
            int num_agents;
            int tick;
            float episode_return;
            unsigned int rng;
        {_emit_state_struct_fields(spec)}
        {env_counter_fields}
        }} {spec.struct_name};

        {sample_helper_block}

        static inline void add_log({spec.struct_name}* env) {{
        {indent(log_updates, " " * 4)}
        }}

        static inline void compute_observations({spec.struct_name}* env) {{
        {compute_observations_block}
        }}

        static inline void reset_round({spec.struct_name}* env) {{
            {paddle_yl} = 285.0f;
            {paddle_yr} = 285.0f;
            {ball_x} = 100.0f;
            {ball_y} = 304.0f;
            {ball_vx} = 10.0f;
            {ball_vy} = 1.0f;
            env->tick = 0;
        }}

        static inline void c_reset({spec.struct_name}* env) {{
            env->tick = 0;
            env->episode_return = 0.0f;
            reset_round(env);
            {score_l} = 0.0f;
            {score_r} = 0.0f;
            compute_observations(env);
        {reset_log_block}
        }}

        static inline void c_step({spec.struct_name}* env) {{
            float reward = 0.0f;
            bool terminated = false;
            bool truncated = false;
            bool done = false;

            env->tick += 1;
            env->rewards[0] = 0.0f;
            env->terminals[0] = 0.0f;

            float action_0 = env->actions[0];
            if (!isfinite(action_0)) {{ action_0 = 0.0f; }}
            action_0 = fmaxf(action_0, 0.0f);
            action_0 = fminf(action_0, 2.0f);
            env->actions[0] = action_0;

            float paddle_dir = 0.0f;
            if (action_0 == 1.0f) {{
                paddle_dir = 1.0f;
            }} else if (action_0 == 2.0f) {{
                paddle_dir = -1.0f;
            }}

            for (int i = 0; i < {substeps}; i++) {{
                {paddle_yr} += 8.0f * paddle_dir;

                float opp_paddle_delta = {ball_y} - ({paddle_yl} + 35.0f);
                opp_paddle_delta = fminf(fmaxf(opp_paddle_delta, -8.0f), 8.0f);
                {paddle_yl} += opp_paddle_delta;

                {paddle_yr} = fminf(fmaxf({paddle_yr}, -35.0f), 605.0f);
                {paddle_yl} = fminf(fmaxf({paddle_yl}, -35.0f), 605.0f);

                {ball_x} += {ball_vx};
                {ball_y} += {ball_vy};

                if ({ball_y} < 0.0f || {ball_y} + 32.0f > 640.0f) {{
                    {ball_vy} = -{ball_vy};
                }}

                if ({ball_x} < 0.0f) {{
                    if ({ball_y} + 32.0f > {paddle_yl} && {ball_y} < {paddle_yl} + 70.0f) {{
                        {ball_vx} = -{ball_vx};
                    }} else {{
                        {score_r} += 1.0f;
                        reward = 1.0f;
                        if ({score_r} == 21.0f) {{
                            terminated = true;
                            done = true;
                            env->rewards[0] = reward;
                            env->terminals[0] = 1.0f;
                            env->episode_return += reward;
                            add_log(env);
                            c_reset(env);
                            return;
                        }}
                        reset_round(env);
                        compute_observations(env);
                        env->rewards[0] = reward;
                        env->episode_return += reward;
                        return;
                    }}
                }}

                if ({ball_x} + 32.0f > 500.0f) {{
                    if ({ball_y} + 32.0f > {paddle_yr} && {ball_y} < {paddle_yr} + 70.0f) {{
                        {ball_vx} = -{ball_vx};
                        {ball_vy} += 3.0f * paddle_dir;
                        {ball_vy} = fminf(fmaxf({ball_vy}, -13.0f), 13.0f);
                        if (fabsf({ball_vy}) < 0.01f) {{
                            {ball_vy} = 3.0f;
                        }}
                    }} else {{
                        {score_l} += 1.0f;
                        reward = -1.0f;
                        if ({score_l} == 21.0f) {{
                            terminated = true;
                            done = true;
                            env->rewards[0] = reward;
                            env->terminals[0] = 1.0f;
                            env->episode_return += reward;
                            add_log(env);
                            c_reset(env);
                            return;
                        }}
                        reset_round(env);
                        compute_observations(env);
                        env->rewards[0] = reward;
                        env->episode_return += reward;
                        return;
                    }}
                    {ball_x} = fminf(fmaxf({ball_x}, 0.0f), 468.0f);
                    {ball_y} = fminf(fmaxf({ball_y}, 0.0f), 608.0f);
                }}

                compute_observations(env);
            }}

            truncated = ({spec.prefix}_HORIZON > 0) && (env->tick >= {spec.prefix}_HORIZON);
            done = terminated || truncated;
            env->rewards[0] = reward;
            env->terminals[0] = terminated ? 1.0f : 0.0f;
            env->episode_return += reward;

            if (done) {{
                add_log(env);
                c_reset(env);
            }}
        }}

        static inline void c_render({spec.struct_name}* env) {{
            (void)env;
        }}

        static inline void c_close({spec.struct_name}* env) {{
            (void)env;
        }}
        """
    )


def _emit_action_sanitizers(spec: GeneratedEnvSpec) -> str:
    lines: list[str] = []
    for head in spec.action_heads:
        action_var = action_var_name(head.flat_index)
        lines.append(f"float {action_var} = env->actions[{head.flat_index}];")
        lines.append(
            f"if (!isfinite({action_var})) {{ {action_var} = {_float_literal(head.default_value)}; }}"
        )
        if head.lower_bound is not None:
            lines.append(f"{action_var} = fmaxf({action_var}, {_float_literal(head.lower_bound)});")
        if head.upper_bound is not None:
            lines.append(f"{action_var} = fminf({action_var}, {_float_literal(head.upper_bound)});")
        lines.append(f"env->actions[{head.flat_index}] = {action_var};")
    return "\n".join(lines)


def _emit_reward_postprocess(spec: GeneratedEnvSpec) -> str:
    if not spec.zero_reward_on_done:
        return ""
    return "if (done) {\n    reward = 0.0f;\n}"


def _emit_log_struct_fields(spec: GeneratedEnvSpec) -> str:
    if not spec.log_counters:
        return ""
    return "\n" + "\n".join(f"    float {sanitize_c_identifier(counter.name)};" for counter in spec.log_counters)


def _emit_env_counter_fields(spec: GeneratedEnvSpec) -> str:
    if not spec.log_counters:
        return ""
    return "\n" + "\n".join(f"    float {_last_counter_field(counter.name)};" for counter in spec.log_counters)


def _emit_add_log_body(spec: GeneratedEnvSpec) -> str:
    lines: list[str] = []
    if spec.perf_mode == "state_ratio":
        numerator = _state_field_expr(spec, spec.perf_numerator_state)
        denominator_terms = " + ".join(_state_field_expr(spec, name) for name in spec.perf_denominator_states)
        lines.append(f"float perf_den = {denominator_terms};")
        lines.append(f"env->log.perf += perf_den > 0.0f ? ({numerator}) / perf_den : 0.0f;")
    elif spec.perf_mode == "positive_return_div_horizon":
        lines.append(
            "env->log.perf = env->episode_return > 0.0f ? "
            f"env->episode_return / (float)({spec.prefix}_HORIZON > 0 ? {spec.prefix}_HORIZON : 1) : 0.0f;"
        )
    else:
        lines.append(
            f"if (env->tick > 0) {{ env->log.perf += env->episode_return / (float)({spec.prefix}_HORIZON > 0 ? {spec.prefix}_HORIZON : 1); }}"
        )

    if spec.score_mode == "tick":
        lines.append("env->log.score += (float)env->tick;")
    else:
        lines.append("env->log.score += env->episode_return;")

    lines.append("env->log.episode_return += env->episode_return;")
    lines.append("env->log.episode_length += (float)env->tick;")
    for counter in spec.log_counters:
        lines.append(
            f"env->log.{sanitize_c_identifier(counter.name)} += env->{_last_counter_field(counter.name)};"
        )
    lines.append("env->log.n += 1.0f;")
    return "\n".join(lines)


def _emit_counter_captures(program: IRProgram, spec: GeneratedEnvSpec) -> str:
    lines: list[str] = []
    for counter in spec.log_counters:
        source = _counter_source_expr(program, counter.source)
        lines.append(f"env->{_last_counter_field(counter.name)} = {source};")
    return "\n".join(lines)


def _emit_tick_reset(program: IRProgram, spec: GeneratedEnvSpec) -> str:
    if not spec.reset_tick_on:
        return ""
    return f"if (!done && {_counter_source_expr(program, spec.reset_tick_on)}) {{ env->tick = 0; }}"


def _counter_source_expr(program: IRProgram, source: str) -> str:
    if source == "$reward_nonzero":
        return "(reward != 0.0f ? 1.0f : 0.0f)"
    if source == "$truncated":
        return "(truncated ? 1.0f : 0.0f)"
    if source == "$terminated":
        return "(terminated ? 1.0f : 0.0f)"
    if source == "$done":
        return "(done ? 1.0f : 0.0f)"
    return f"({lookup_var_name(program, source)} ? 1.0f : 0.0f)"


def _last_counter_field(name: str) -> str:
    return f"last_{sanitize_c_identifier(name)}"


def _state_field_expr(spec: GeneratedEnvSpec, state_name: str | None) -> str:
    if state_name is None:
        raise ValueError("Missing state name for state_ratio perf logging.")
    for field in spec.state_fields:
        if field.name == state_name:
            return f"env->{field.field_name}"
    raise ValueError(f"Unknown state fluent for state_ratio perf logging: {state_name}")


def _emit_step_buffer_declarations(program: IRProgram, spec: GeneratedEnvSpec) -> str:
    lines: list[str] = []
    if not spec.state_slots_fully_overwritten:
        lines.append(f"float next_state[{spec.prefix}_STATE_SIZE];")
    if not spec.observation_slots_fully_overwritten:
        lines.append(f"float obs[{spec.prefix}_OBS_SIZE];")
    return "\n".join(lines)


def _emit_next_state_scratch(program: IRProgram, spec: GeneratedEnvSpec) -> str:
    return "\n".join(
        f"float next_{field.field_name};"
        for field in spec.state_fields
    )


def _emit_state_commit(program: IRProgram, spec: GeneratedEnvSpec) -> str:
    if spec.state_slots_fully_overwritten:
        return "\n".join(
            f"env->{field.field_name} = next_{field.field_name};"
            for field in spec.state_fields
        )
    return "\n".join(
        f"env->{field.field_name} = next_state[{field.slot}];"
        for field in spec.state_fields
    )


def _emit_state_struct_fields(spec: GeneratedEnvSpec) -> str:
    return "\n" + "\n".join(f"    float {field.field_name};" for field in spec.state_fields)


def _emit_state_array_seed(spec: GeneratedEnvSpec) -> str:
    return "\n".join(
        f"next_state[{field.slot}] = env->{field.field_name};"
        for field in spec.state_fields
    )


def _emit_sample_helpers(program: IRProgram) -> str:
    if not any(node.op is NodeOp.SAMPLE for node in program.nodes):
        return ""
    return dedent(
        """\
        static inline float sample_uniform(unsigned int* rng, float low, float high) {
            float unit = (float)rand_r(rng) / (float)RAND_MAX;
            return low + (high - low) * unit;
        }
        """
    ).rstrip()


def _metadata_node_roots(spec: GeneratedEnvSpec) -> tuple[str, ...]:
    roots: list[str] = []
    if spec.reset_tick_on and not spec.reset_tick_on.startswith("$"):
        roots.append(spec.reset_tick_on)
    for counter in spec.log_counters:
        if not counter.source.startswith("$"):
            roots.append(counter.source)
    return tuple(roots)
