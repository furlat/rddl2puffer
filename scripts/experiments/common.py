#!/usr/bin/env python3
"""Shared helpers for reproducible Puffertank experiment scripts."""

from __future__ import annotations

import ast
import configparser
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_REPO_ROOT = Path("/workspace/rddl2puffer")
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
PUFFERLIB_ROOT = THIRD_PARTY_ROOT / "pufferlib"
PUFFERTANK_ROOT = THIRD_PARTY_ROOT / "puffertank"
DEFAULT_IMAGE = "local/puffertank:4.0"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for site_packages in sorted((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages")):
    if str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))

from rddl2puffer.frontend.compile import compile_rddl_files
from rddl2puffer.workspace import discover_workspace


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def run_puffertank(
    command: str,
    *,
    image: str = DEFAULT_IMAGE,
    capture_output: bool = False,
) -> CommandResult:
    """Run one bash command inside the local Puffertank image."""

    inner = (
        "set -euo pipefail; "
        ". /root/.local/bin/env; "
        ". /puffertank/venv/bin/activate; "
        "export PYTHONPATH=/workspace/rddl2puffer${PYTHONPATH:+:$PYTHONPATH}; "
        "export MPLCONFIGDIR=/tmp/mpl; "
        "export TMPDIR=/tmp; "
        "cd /workspace/rddl2puffer; "
        f"{command}"
    )
    argv = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "--cgroupns=host",
        "-v",
        f"{PUFFERLIB_ROOT}:/puffertank/pufferlib",
        "-v",
        f"{REPO_ROOT}:/workspace/rddl2puffer",
        image,
        "bash",
        "-lc",
        inner,
    ]
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=capture_output,
        text=True,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            argv,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return CommandResult(
        argv=tuple(argv),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_shell(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def to_container_path(path: Path) -> str:
    """Translate one repo-local host path into the mounted container path."""

    resolved = path.resolve()
    relative = resolved.relative_to(REPO_ROOT)
    return str(CONTAINER_REPO_ROOT / relative)


def find_log_files(log_root: Path, env_name: str) -> list[Path]:
    return sorted((log_root / env_name).glob("*.json"))


def emit_pong_parity_workspace(env_name: str = "rddl_pong_parity") -> tuple[Path, ...]:
    """Compile the source-driven Pong parity env and write it into PufferLib."""

    domain_path = REPO_ROOT / "examples" / "rddl_plus" / "pong_puffer_parity" / "domain.rddl"
    instance_path = domain_path.with_name("instance_deterministic_reset.rddl")
    program = compile_rddl_files(domain_path, instance_path, env_name=env_name)
    workspace = discover_workspace(REPO_ROOT)
    return workspace.write_env_bundle(program, env_name=env_name)


def load_flag_types(env_name: str) -> dict[str, type]:
    """Load the CLI value types Puffer expects for one env config."""

    parser = configparser.ConfigParser()
    parser.read(
        [
            str(PUFFERLIB_ROOT / "config" / "default.ini"),
            str(PUFFERLIB_ROOT / "config" / f"{env_name}.ini"),
        ]
    )
    types: dict[str, type] = {}
    for section in parser.sections():
        for key, raw_value in parser[section].items():
            try:
                value = ast.literal_eval(raw_value)
            except Exception:
                value = raw_value
            dotted = key if section == "base" else f"{section}.{key}"
            types[dotted] = type(value)
    return types


def coerce_cli_value(value: Any, dtype: type) -> str:
    """Render one config value using the target CLI/default type."""

    if dtype is bool:
        return "True" if bool(value) else "False"
    if dtype is int:
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(int(value))
    if dtype is float:
        return format(float(value), ".17g")
    return str(value)


def native_replay_overrides(
    payload: dict[str, Any],
    *,
    generated_env_name: str,
) -> list[str]:
    """Extract the generated-env CLI overrides needed to replay one native run."""

    allowed_types = load_flag_types(generated_env_name)
    overrides: list[str] = []

    excluded = {
        "env",
        "env_name",
        "metrics",
        "base",
        "sweep",
        "checkpoint_dir",
        "log_dir",
        "load_model_path",
        "load_id",
        "wandb",
        "wandb_project",
        "wandb_group",
        "tag",
        "rank",
        "world_size",
        "gpu_id",
        "profile",
        "no_model_upload",
    }

    for key, value in payload.items():
        if key in excluded or isinstance(value, dict | list):
            continue
        if key in allowed_types:
            overrides.extend([f"--{key.replace('_', '-')}", coerce_cli_value(value, allowed_types[key])])

    for section in ("vec", "policy", "train", "torch"):
        mapping = payload.get(section)
        if not isinstance(mapping, dict):
            continue
        for key, value in mapping.items():
            dotted = f"{section}.{key}"
            if dotted not in allowed_types:
                continue
            overrides.extend(
                [f"--{dotted.replace('_', '-')}", coerce_cli_value(value, allowed_types[dotted])]
            )

    return overrides
