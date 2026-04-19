"""CartPole benchmark metadata for the wrapper/native comparison ladder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkFile:
    """One concrete file that participates in a benchmark target."""

    label: str
    path: Path
    required: bool = True

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "path": str(self.path),
            "required": self.required,
            "exists": self.exists,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkTarget:
    """One runtime target in the CartPole benchmark ladder."""

    key: str
    runtime: str
    env_name: str
    status: str
    description: str
    action_description: str
    observation_fields: tuple[str, ...]
    files: tuple[BenchmarkFile, ...]
    notes: tuple[str, ...] = ()

    @property
    def required_files_present(self) -> bool:
        return all(not file.required or file.exists for file in self.files)

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "runtime": self.runtime,
            "env_name": self.env_name,
            "status": self.status,
            "description": self.description,
            "action_description": self.action_description,
            "observation_fields": list(self.observation_fields),
            "required_files_present": self.required_files_present,
            "files": [file.to_debug_dict() for file in self.files],
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class CartPoleBenchmarkTrio:
    """The agreed first benchmark family for the project."""

    repo_root: Path
    wrapper_rddl: BenchmarkTarget
    generated_native: BenchmarkTarget
    native_puffer: BenchmarkTarget
    extra_reference_variants: tuple[BenchmarkTarget, ...] = ()
    semantic_risks: tuple[str, ...] = ()
    recommended_order: tuple[str, ...] = ()

    @property
    def reference_inputs_present(self) -> bool:
        targets = (self.wrapper_rddl, self.native_puffer, *self.extra_reference_variants)
        return all(target.required_files_present for target in targets)

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "repo_root": str(self.repo_root),
            "reference_inputs_present": self.reference_inputs_present,
            "wrapper_rddl": self.wrapper_rddl.to_debug_dict(),
            "generated_native": self.generated_native.to_debug_dict(),
            "native_puffer": self.native_puffer.to_debug_dict(),
            "extra_reference_variants": [
                variant.to_debug_dict() for variant in self.extra_reference_variants
            ],
            "semantic_risks": list(self.semantic_risks),
            "recommended_order": list(self.recommended_order),
        }


def discover_cartpole_benchmark_trio(repo_root: Path | None = None) -> CartPoleBenchmarkTrio:
    """Resolve the local benchmark files for the CartPole comparison ladder."""

    root = (repo_root or Path.cwd()).resolve()
    rddl_cartpole_root = (
        root / "third_party" / "rddlrepository" / "rddlrepository" / "archive" / "gym" / "CartPole"
    )
    discrete_root = rddl_cartpole_root / "Discrete"
    continuous_root = rddl_cartpole_root / "Continuous"
    pufferlib_root = root / "third_party" / "pufferlib"

    wrapper_rddl = BenchmarkTarget(
        key="wrapped_rddl_cartpole_discrete",
        runtime="pyRDDLGym through a Puffer wrapper path",
        env_name="CartPole_Discrete_gym",
        status="reference_available",
        description="Primary semantic baseline for the first wrapper experiment.",
        action_description="One discrete action fluent `force-side` in {0, 1}.",
        observation_fields=("pos", "vel", "ang-pos", "ang-vel"),
        files=(
            BenchmarkFile(label="rddl_discrete_domain", path=discrete_root / "domain.rddl"),
            BenchmarkFile(label="rddl_discrete_instance", path=discrete_root / "instance0.rddl"),
        ),
        notes=(
            "Use this as the first wrapper-baseline training target.",
            "This target should define the semantics for the generated native CartPole.",
        ),
    )

    generated_env_name = "rddl_cartpole_discrete"
    generated_native = BenchmarkTarget(
        key="generated_rddl_cartpole_discrete",
        runtime="generated native Puffer Ocean environment",
        env_name=generated_env_name,
        status="planned_output",
        description="Compiler target that should match the wrapped RDDL semantics.",
        action_description="One discrete action head mirroring `force-side`.",
        observation_fields=("pos", "vel", "ang-pos", "ang-vel"),
        files=(
            BenchmarkFile(
                label="generated_header",
                path=pufferlib_root / "ocean" / generated_env_name / f"{generated_env_name}.h",
                required=False,
            ),
            BenchmarkFile(
                label="generated_local_demo",
                path=pufferlib_root / "ocean" / generated_env_name / f"{generated_env_name}.c",
                required=False,
            ),
            BenchmarkFile(
                label="generated_config",
                path=pufferlib_root / "config" / f"{generated_env_name}.ini",
                required=False,
            ),
        ),
        notes=(
            "This env should be judged against pyRDDLGym semantics, not against native Puffer CartPole.",
            "Start with exact reset/step parity before training or performance claims.",
        ),
    )

    native_puffer = BenchmarkTarget(
        key="native_puffer_cartpole",
        runtime="hand-written native Puffer Ocean environment",
        env_name="cartpole",
        status="reference_available",
        description="Native performance/template reference already supported by upstream PufferLib.",
        action_description="One discrete action head with two choices.",
        observation_fields=("x", "x_dot", "theta", "theta_dot"),
        files=(
            BenchmarkFile(label="native_binding", path=pufferlib_root / "ocean" / "cartpole" / "binding.c"),
            BenchmarkFile(label="native_header", path=pufferlib_root / "ocean" / "cartpole" / "cartpole.h"),
            BenchmarkFile(label="native_config", path=pufferlib_root / "config" / "cartpole.ini"),
        ),
        notes=(
            "Treat this as a native template and performance reference.",
            "Do not treat this implementation as the semantic gold standard for RDDL CartPole.",
        ),
    )

    continuous_reference = BenchmarkTarget(
        key="rddl_cartpole_continuous",
        runtime="pyRDDLGym reference variant",
        env_name="CartPole_Continuous_gym",
        status="reference_available",
        description="Secondary RDDL benchmark once the discrete path is stable.",
        action_description="One real action fluent `force` with bounded magnitude.",
        observation_fields=("pos", "vel", "ang-pos", "ang-vel"),
        files=(
            BenchmarkFile(label="rddl_continuous_domain", path=continuous_root / "domain.rddl"),
            BenchmarkFile(label="rddl_continuous_instance", path=continuous_root / "instance0.rddl"),
        ),
        notes=(
            "Useful after the discrete trio is stable.",
            "Not directly comparable to upstream native Puffer CartPole because the action space differs.",
        ),
    )

    return CartPoleBenchmarkTrio(
        repo_root=root,
        wrapper_rddl=wrapper_rddl,
        generated_native=generated_native,
        native_puffer=native_puffer,
        extra_reference_variants=(continuous_reference,),
        semantic_risks=(
            "Native Puffer CartPole uses a random reset distribution; the RDDL instances reset deterministically.",
            "Reward and episode-end semantics differ between pyRDDLGym CartPole and native Puffer CartPole.",
            "The native Puffer CartPole dynamics appear to diverge materially from the classic/RDDL equations.",
        ),
        recommended_order=(
            "Train wrapped RDDL discrete CartPole first.",
            "Generate native RDDL discrete CartPole and match wrapped rollouts exactly.",
            "Only then compare throughput and training behavior against upstream native Puffer CartPole.",
        ),
    )
