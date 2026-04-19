"""Readiness probes for the wrapper-first baseline plan."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from rddl2puffer.benchmarks.cartpole import CartPoleBenchmarkTrio, discover_cartpole_benchmark_trio


@dataclass(frozen=True, slots=True)
class ModuleAvailability:
    """Availability snapshot for an optional runtime dependency."""

    module_name: str
    importable: bool
    origin: str | None

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "module_name": self.module_name,
            "importable": self.importable,
            "origin": self.origin,
        }


@dataclass(frozen=True, slots=True)
class WrapperBaselineStatus:
    """Current status of the wrapper-first benchmark path."""

    repo_root: Path
    cartpole: CartPoleBenchmarkTrio
    pyrddlgym_module: ModuleAvailability
    pufferlib_module: ModuleAvailability
    vendored_pyrddlgym_root: Path
    vendored_rddlrepository_root: Path
    vendored_pufferlib_root: Path
    examples_reference_emulation: bool
    source_tree_has_emulation: bool
    notes: tuple[str, ...]

    @property
    def ready_for_local_wrapper_experiment(self) -> bool:
        return (
            self.cartpole.reference_inputs_present
            and self.pyrddlgym_module.importable
            and self.pufferlib_module.importable
        )

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "repo_root": str(self.repo_root),
            "ready_for_local_wrapper_experiment": self.ready_for_local_wrapper_experiment,
            "cartpole": self.cartpole.to_debug_dict(),
            "pyrddlgym_module": self.pyrddlgym_module.to_debug_dict(),
            "pufferlib_module": self.pufferlib_module.to_debug_dict(),
            "vendored_pyrddlgym_root": str(self.vendored_pyrddlgym_root),
            "vendored_rddlrepository_root": str(self.vendored_rddlrepository_root),
            "vendored_pufferlib_root": str(self.vendored_pufferlib_root),
            "examples_reference_emulation": self.examples_reference_emulation,
            "source_tree_has_emulation": self.source_tree_has_emulation,
            "notes": list(self.notes),
        }


def probe_wrapper_baseline(repo_root: Path | None = None) -> WrapperBaselineStatus:
    """Inspect the current workspace for the wrapper-baseline prerequisites."""

    root = (repo_root or Path.cwd()).resolve()
    cartpole = discover_cartpole_benchmark_trio(root)
    third_party_root = root / "third_party"
    pyrddlgym_root = third_party_root / "pyRDDLGym"
    rddlrepository_root = third_party_root / "rddlrepository"
    pufferlib_root = third_party_root / "pufferlib"

    pyrddlgym_module = _probe_module("pyRDDLGym")
    pufferlib_module = _probe_module("pufferlib")
    source_tree_has_emulation = (
        (pufferlib_root / "pufferlib" / "emulation.py").exists()
        or (pufferlib_root / "pufferlib" / "emulation" / "__init__.py").exists()
    )
    examples_reference_emulation = _directory_mentions(pufferlib_root / "examples", "pufferlib.emulation")

    notes: list[str] = [
        "Use wrapped RDDL discrete CartPole as the first benchmark before codegen.",
    ]
    if not pyrddlgym_module.importable and pyrddlgym_root.exists():
        notes.append(
            "Vendored pyRDDLGym is present but not installed in the local Python environment."
        )
    if not pufferlib_module.importable and pufferlib_root.exists():
        notes.append(
            "Vendored PufferLib is present but not installed in the local Python environment."
        )
    if examples_reference_emulation and not source_tree_has_emulation:
        notes.append(
            "PufferLib examples reference `pufferlib.emulation`, but the source tree checkout does not contain that module."
        )
    if not cartpole.reference_inputs_present:
        notes.append("Reference benchmark inputs are missing from `third_party/`.")
    else:
        notes.append("The local benchmark inputs for the CartPole trio are present.")

    return WrapperBaselineStatus(
        repo_root=root,
        cartpole=cartpole,
        pyrddlgym_module=pyrddlgym_module,
        pufferlib_module=pufferlib_module,
        vendored_pyrddlgym_root=pyrddlgym_root,
        vendored_rddlrepository_root=rddlrepository_root,
        vendored_pufferlib_root=pufferlib_root,
        examples_reference_emulation=examples_reference_emulation,
        source_tree_has_emulation=source_tree_has_emulation,
        notes=tuple(notes),
    )


def _probe_module(module_name: str) -> ModuleAvailability:
    spec = find_spec(module_name)
    return ModuleAvailability(
        module_name=module_name,
        importable=spec is not None,
        origin=str(spec.origin) if spec is not None and spec.origin is not None else None,
    )


def _directory_mentions(directory: Path, needle: str) -> bool:
    if not directory.exists():
        return False

    for path in directory.rglob("*.py"):
        try:
            contents = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if needle in contents:
            return True
    return False
