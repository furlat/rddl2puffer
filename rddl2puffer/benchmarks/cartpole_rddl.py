"""Source-driven helpers for the compiled CartPole parity benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rddl2puffer.frontend.compile import compile_rddl_files
from rddl2puffer.frontend.schema import EnvSchema
from rddl2puffer.ir.nodes import IRProgram
from rddl2puffer.testing.compiled_reference import CompiledProgramReferenceEnv
from rddl2puffer.testing.differential import ReferenceEnv
from rddl2puffer.testing.pyrddlgym_reference import PyRDDLGymReferenceEnv


@dataclass(frozen=True, slots=True)
class CartPoleRDDLSource:
    """Concrete source files for the repo-owned CartPole parity benchmark."""

    domain_path: Path
    instance_path: Path
    env_name: str = "rddl_cartpole_discrete"

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "domain_path": str(self.domain_path),
            "instance_path": str(self.instance_path),
            "env_name": self.env_name,
            "exists": self.domain_path.exists() and self.instance_path.exists(),
        }


def discover_cartpole_parity_source(repo_root: Path | None = None) -> CartPoleRDDLSource:
    """Resolve the repo-owned CartPole parity RDDL files."""

    root = (repo_root or Path.cwd()).resolve()
    source_root = root / "examples" / "rddl_plus" / "cartpole_puffer_parity"
    domain_path = source_root / "domain.rddl"
    instance_path = source_root / "instance_deterministic_reset.rddl"
    if not domain_path.is_file() or not instance_path.is_file():
        raise FileNotFoundError(
            f"Missing CartPole parity RDDL sources under {source_root}."
        )
    return CartPoleRDDLSource(domain_path=domain_path, instance_path=instance_path)


def compile_cartpole_parity_program(repo_root: Path | None = None, *, env_name: str | None = None) -> IRProgram:
    """Compile the repo-owned CartPole parity RDDL files into IR."""

    source = discover_cartpole_parity_source(repo_root)
    return compile_rddl_files(
        source.domain_path,
        source.instance_path,
        env_name=env_name or source.env_name,
    )


def make_cartpole_parity_compiled_env(repo_root: Path | None = None) -> ReferenceEnv:
    """Create the compiled-program reference runtime for CartPole parity."""

    return CompiledProgramReferenceEnv(compile_cartpole_parity_program(repo_root))


def make_cartpole_parity_reference_env(repo_root: Path | None = None) -> ReferenceEnv:
    """Create the pyRDDLGym oracle runtime from the same local source files."""

    source = discover_cartpole_parity_source(repo_root)
    program = compile_cartpole_parity_program(repo_root, env_name=source.env_name)
    schema = EnvSchema(
        state_layout=program.state_layout,
        action_layout=program.action_layout,
        observation_layout=program.observation_layout,
        metadata=program.metadata,
    )
    return PyRDDLGymReferenceEnv(
        domain=str(source.domain_path),
        instance=str(source.instance_path),
        schema=schema,
        auto_reset_on_done=True,
    )
