"""Small CLI helpers for real source-driven smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rddl2puffer.backends.puffer_c import render_env_bundle
from rddl2puffer.baselines.wrapper import probe_wrapper_baseline
from rddl2puffer.benchmarks.cartpole import discover_cartpole_benchmark_trio
from rddl2puffer.benchmarks.cartpole_rddl import (
    compile_cartpole_parity_program,
    make_cartpole_parity_compiled_env,
    make_cartpole_parity_reference_env,
)
from rddl2puffer.frontend.compile import compile_rddl_sources
from rddl2puffer.ir.interpret import step_ir
from rddl2puffer.reporting import find_latest_cartpole_log, write_cartpole_reward_report
from rddl2puffer.reporting.cartpole_sweep_report import write_cartpole_sweep_comparison
from rddl2puffer.testing.rollout_compare import compare_rollouts, format_mismatch_report
from rddl2puffer.workspace import discover_workspace

_DEMO_DOMAIN = """
domain toy_counter {
    requirements = { reward-deterministic };

    pvariables {
        counter : { state-fluent, int, default = 0 };
        delta : { action-fluent, int, default = 0 };
    };

    cpfs {
        counter' = counter + delta;
    };

    reward = if (counter + delta > 2) then 10 else 1;

    termination {
        counter + delta > 2;
    };

    action-preconditions {
        delta >= -2;
        delta <= 2;
    };
}
"""

_DEMO_INSTANCE = """
non-fluents toy_counter_nf {
    domain = toy_counter;
}

instance toy_counter_inst {
    domain = toy_counter;
    non-fluents = toy_counter_nf;

    init-state {
        counter = 1;
    };

    max-nondef-actions = pos-inf;
    horizon = 5;
    discount = 1.0;
}
"""


def _build_demo_program():
    return compile_rddl_sources(
        _DEMO_DOMAIN,
        _DEMO_INSTANCE,
        env_name="toy_counter",
    )


def _run_schema_demo() -> int:
    program = _build_demo_program()
    payload = {
        "state_layout": program.state_layout.to_debug_dict(),
        "action_layout": program.action_layout.to_debug_dict(),
        "observation_layout": program.observation_layout.to_debug_dict(),
        "metadata": dict(program.metadata),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_ir_demo() -> int:
    program = _build_demo_program()
    result = step_ir(program, state=(1,), action=(2,))
    print(json.dumps(result.to_debug_dict(), indent=2, sort_keys=True))
    return 0


def _run_emit_demo() -> int:
    program = _build_demo_program()
    bundle = render_env_bundle(program, env_name="rddl_demo")
    payload = {name: text.splitlines()[:8] for name, text in bundle.items()}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_workspace_info(workspace_root: str | None) -> int:
    workspace = discover_workspace(Path(workspace_root) if workspace_root else None)
    print(json.dumps(workspace.to_debug_dict(), indent=2, sort_keys=True))
    return 0


def _run_emit_workspace_demo(workspace_root: str | None, env_name: str) -> int:
    workspace = discover_workspace(Path(workspace_root) if workspace_root else None)
    written = workspace.write_env_bundle(_build_demo_program(), env_name=env_name)
    print(
        json.dumps(
            {
                "workspace": workspace.to_debug_dict(),
                "written_files": [str(path) for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_cartpole_benchmark(repo_root: str | None) -> int:
    trio = discover_cartpole_benchmark_trio(Path(repo_root) if repo_root else None)
    print(json.dumps(trio.to_debug_dict(), indent=2, sort_keys=True))
    return 0


def _run_wrapper_baseline_info(repo_root: str | None) -> int:
    status = probe_wrapper_baseline(Path(repo_root) if repo_root else None)
    print(json.dumps(status.to_debug_dict(), indent=2, sort_keys=True))
    return 0


def _run_cartpole_ir_demo(repo_root: str | None) -> int:
    program = compile_cartpole_parity_program(Path(repo_root) if repo_root else None)
    print(json.dumps(program.to_debug_dict(), indent=2, sort_keys=True))
    return 0


def _run_emit_cartpole_workspace(workspace_root: str | None, env_name: str) -> int:
    root = Path(workspace_root) if workspace_root else None
    workspace = discover_workspace(root)
    program = compile_cartpole_parity_program(root, env_name=env_name)
    written = workspace.write_env_bundle(program, env_name=env_name)
    print(
        json.dumps(
            {
                "workspace": workspace.to_debug_dict(),
                "written_files": [str(path) for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_cartpole_compare(seed: int, actions: str, repo_root: str | None) -> int:
    action_values = [value.strip() for value in actions.split(",") if value.strip()]
    rollout_actions = [(int(value),) for value in action_values]
    resolved_repo_root = Path(repo_root) if repo_root else None
    report = compare_rollouts(
        make_cartpole_parity_compiled_env(resolved_repo_root),
        make_cartpole_parity_reference_env(resolved_repo_root),
        actions=rollout_actions,
        seed=seed,
        left_name="compiled_rddl",
        right_name="pyrddlgym",
    )
    print(format_mismatch_report(report))
    return 0 if report.matches else 1


def _run_cartpole_reward_report(
    output_dir: str,
    log_path: str | None,
    repo_root: str | None,
    episodes: int,
    seed: int,
) -> int:
    resolved_repo_root = Path(repo_root) if repo_root else None
    report = write_cartpole_reward_report(
        output_dir=Path(output_dir),
        log_path=Path(log_path) if log_path else None,
        repo_root=resolved_repo_root,
        episodes=episodes,
        seed=seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(output_dir).resolve()),
                "log_path": report.curve.log_path,
                "latest_detected_log": str(find_latest_cartpole_log(resolved_repo_root)),
                "summary": report.to_summary_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_cartpole_sweep_report(native_log_root: str, generated_log_root: str, output_dir: str) -> int:
    payload = write_cartpole_sweep_comparison(
        native_log_root=Path(native_log_root),
        generated_log_root=Path(generated_log_root),
        output_dir=Path(output_dir),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("schema-demo", help="Print a schema compiled from inline RDDL.")
    subparsers.add_parser("ir-demo", help="Run a deterministic inline-RDDL example.")
    subparsers.add_parser("emit-demo", help="Print the first lines of generated Puffer files from inline RDDL.")
    workspace_info = subparsers.add_parser("workspace-info", help="Print the detected Puffer workspace.")
    workspace_info.add_argument("--workspace-root", default=None, help="Repo root that contains third_party/.")
    emit_workspace = subparsers.add_parser(
        "emit-workspace-demo",
        help="Write the inline-RDDL demo environment bundle into a local PufferLib checkout.",
    )
    emit_workspace.add_argument("--workspace-root", default=None, help="Repo root that contains third_party/.")
    emit_workspace.add_argument("--env-name", default="rddl_demo", help="Generated environment name.")

    cartpole_benchmark = subparsers.add_parser(
        "cartpole-benchmark",
        help="Print the local CartPole benchmark trio metadata.",
    )
    cartpole_benchmark.add_argument("--repo-root", default=None, help="Repo root that contains third_party/.")

    wrapper_baseline = subparsers.add_parser(
        "wrapper-baseline-info",
        help="Print wrapper-baseline readiness for the CartPole trio.",
    )
    wrapper_baseline.add_argument("--repo-root", default=None, help="Repo root that contains third_party/.")

    cartpole_ir_demo = subparsers.add_parser(
        "cartpole-ir-demo",
        help="Print the compiled CartPole parity IR program.",
    )
    cartpole_ir_demo.add_argument("--repo-root", default=None, help="Repo root that contains examples/.")

    emit_cartpole_workspace = subparsers.add_parser(
        "emit-cartpole-workspace",
        help="Write the compiled CartPole parity env bundle into the local PufferLib checkout.",
    )
    emit_cartpole_workspace.add_argument("--workspace-root", default=None, help="Repo root that contains third_party/.")
    emit_cartpole_workspace.add_argument("--env-name", default="rddl_cartpole_discrete", help="Generated environment name.")

    cartpole_compare = subparsers.add_parser(
        "cartpole-compare",
        help="Compare compiled CartPole RDDL against pyRDDLGym on a fixed action sequence.",
    )
    cartpole_compare.add_argument("--repo-root", default=None, help="Repo root that contains examples/.")
    cartpole_compare.add_argument("--seed", type=int, default=123, help="Reset seed for the rollout.")
    cartpole_compare.add_argument("--actions", default="0,1,0,1,0,1", help="Comma-separated force-side actions to replay.")

    cartpole_reward_report = subparsers.add_parser(
        "cartpole-reward-report",
        help="Write reward-vs-random artifacts for the latest generated CartPole run.",
    )
    cartpole_reward_report.add_argument("--output-dir", default="artifacts/cartpole/generated_vs_random", help="Directory where JSON/CSV/SVG/Markdown artifacts should be written.")
    cartpole_reward_report.add_argument("--log-path", default=None, help="Specific Puffer JSON log to summarize. Defaults to the latest generated CartPole log.")
    cartpole_reward_report.add_argument("--repo-root", default=None, help="Repo root that contains examples/.")
    cartpole_reward_report.add_argument("--episodes", type=int, default=50_000, help="Number of random-policy episodes to estimate for the baseline.")
    cartpole_reward_report.add_argument("--seed", type=int, default=123, help="Random seed used for the baseline policy.")

    cartpole_sweep_report = subparsers.add_parser(
        "cartpole-sweep-report",
        help="Compare native and generated CartPole sweep batches.",
    )
    cartpole_sweep_report.add_argument("--native-log-root", required=True, help="Root directory of native sweep logs.")
    cartpole_sweep_report.add_argument("--generated-log-root", required=True, help="Root directory of generated sweep logs.")
    cartpole_sweep_report.add_argument("--output-dir", default="artifacts/cartpole/sweep_compare", help="Directory where comparison artifacts should be written.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "schema-demo":
        return _run_schema_demo()
    if args.command == "ir-demo":
        return _run_ir_demo()
    if args.command == "emit-demo":
        return _run_emit_demo()
    if args.command == "workspace-info":
        return _run_workspace_info(args.workspace_root)
    if args.command == "emit-workspace-demo":
        return _run_emit_workspace_demo(args.workspace_root, args.env_name)
    if args.command == "cartpole-benchmark":
        return _run_cartpole_benchmark(args.repo_root)
    if args.command == "wrapper-baseline-info":
        return _run_wrapper_baseline_info(args.repo_root)
    if args.command == "cartpole-ir-demo":
        return _run_cartpole_ir_demo(args.repo_root)
    if args.command == "emit-cartpole-workspace":
        return _run_emit_cartpole_workspace(args.workspace_root, args.env_name)
    if args.command == "cartpole-compare":
        return _run_cartpole_compare(args.seed, args.actions, args.repo_root)
    if args.command == "cartpole-reward-report":
        return _run_cartpole_reward_report(
            args.output_dir,
            args.log_path,
            args.repo_root,
            args.episodes,
            args.seed,
        )
    if args.command == "cartpole-sweep-report":
        return _run_cartpole_sweep_report(args.native_log_root, args.generated_log_root, args.output_dir)
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
