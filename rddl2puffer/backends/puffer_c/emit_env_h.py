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
    sanitize_c_identifier,
)
from rddl2puffer.backends.puffer_c.model import GeneratedEnvSpec, build_env_spec
from rddl2puffer.ir.nodes import IRProgram, NodeOp


def emit_env_header(program: IRProgram, env_name: str) -> str:
    """Emit a generic scalar-state Ocean environment from the ordered IR."""

    spec = build_env_spec(program, env_name)
    inferred = infer_c_node_values(program)
    state_targets = {field.slot: f"env->{field.field_name}" for field in spec.state_fields}
    declarations = emit_c_value_declarations(program, inferred, state_targets=state_targets)
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
