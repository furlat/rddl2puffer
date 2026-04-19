"""Helpers for integrating generated code with a local Puffer workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rddl2puffer.backends.puffer_c import render_env_bundle
from rddl2puffer.ir.nodes import IRProgram


@dataclass(frozen=True, slots=True)
class PufferWorkspace:
    """Resolved local workspace paths for compiler and runtime integration."""

    repo_root: Path
    third_party_root: Path
    puffertank_root: Path
    pufferlib_root: Path

    def write_env_bundle(self, program: IRProgram, env_name: str) -> tuple[Path, ...]:
        """Write a generated env bundle into the checked-out PufferLib tree."""

        bundle = render_env_bundle(program, env_name=env_name)
        written: list[Path] = []
        for relative_name, contents in bundle.items():
            destination = self.pufferlib_root / Path(relative_name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(contents, encoding="utf-8")
            written.append(destination)
        return tuple(written)

    def to_debug_dict(self) -> dict[str, str]:
        return {
            "repo_root": str(self.repo_root),
            "third_party_root": str(self.third_party_root),
            "puffertank_root": str(self.puffertank_root),
            "pufferlib_root": str(self.pufferlib_root),
        }


def discover_workspace(repo_root: Path | None = None) -> PufferWorkspace:
    """Resolve the expected local workspace layout for this project."""

    root = (repo_root or Path.cwd()).resolve()
    third_party_root = root / "third_party"
    puffertank_root = third_party_root / "puffertank"
    pufferlib_root = third_party_root / "pufferlib"

    missing = [path for path in (puffertank_root, pufferlib_root) if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing required Puffer workspace directories: {joined}. "
            "Run scripts/bootstrap_puffertank_workspace.sh first."
        )

    return PufferWorkspace(
        repo_root=root,
        third_party_root=third_party_root,
        puffertank_root=puffertank_root,
        pufferlib_root=pufferlib_root,
    )

