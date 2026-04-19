"""Emit a training config file for a generated Ocean environment."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

from rddl2puffer.backends.puffer_c.model import build_env_spec
from rddl2puffer.ir.nodes import IRProgram


def emit_ini(program: IRProgram, env_name: str) -> str:
    """Emit a compact default Puffer config for a generated env."""

    spec = build_env_spec(program, env_name)
    sections: OrderedDict[str, dict[str, object]] = OrderedDict(
        (
            ("base", {"env_name": env_name}),
            ("vec", {"total_agents": 4096}),
            ("policy", {"hidden_size": 64, "num_layers": 2}),
            (
                "env",
                {
                    "generated": 1,
                    "state_size": program.state_layout.total_size,
                    "action_size": program.action_layout.total_size,
                    "observation_size": program.observation_layout.total_size,
                    "num_action_heads": spec.num_actions,
                },
            ),
            (
                "train",
                {
                    "total_timesteps": 5_000_000,
                    "gamma": spec.discount,
                    "learning_rate": 0.0003,
                    "minibatch_size": 16384,
                    "horizon": 32,
                    "ent_coef": 0.01,
                },
            ),
            (
                "sweep",
                {
                    "use_gpu": False,
                },
            ),
        )
    )

    for section, values in _metadata_config_overrides(program).items():
        target = sections.setdefault(section, {})
        target.update(values)

    # Generated env metadata is authoritative for these values.
    sections["base"]["env_name"] = env_name
    sections["env"].update(
        {
            "generated": 1,
            "state_size": program.state_layout.total_size,
            "action_size": program.action_layout.total_size,
            "observation_size": program.observation_layout.total_size,
            "num_action_heads": spec.num_actions,
        }
    )
    sections["train"].setdefault("gamma", spec.discount)

    rendered_sections: list[str] = []
    for name, values in sections.items():
        rendered_sections.append(f"[{name}]")
        for key, value in values.items():
            rendered_sections.append(f"{key} = {_format_value(value)}")
        rendered_sections.append("")
    return "\n".join(rendered_sections).rstrip() + "\n"


def _metadata_config_overrides(program: IRProgram) -> dict[str, dict[str, object]]:
    raw = program.metadata.get("puffer_config", {})
    if not isinstance(raw, Mapping):
        return {}

    resolved: dict[str, dict[str, object]] = {}
    for section, values in raw.items():
        if not isinstance(section, str) or not isinstance(values, Mapping):
            continue
        _flatten_sections(resolved, section, values)
    return resolved


def _flatten_sections(
    resolved: dict[str, dict[str, object]],
    section_name: str,
    values: Mapping[str, object],
) -> None:
    scalars: dict[str, object] = {}
    for key, value in values.items():
        if isinstance(value, Mapping):
            _flatten_sections(resolved, f"{section_name}.{key}", value)
        else:
            scalars[str(key)] = value
    if scalars:
        target = resolved.setdefault(section_name, {})
        target.update(scalars)


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)
