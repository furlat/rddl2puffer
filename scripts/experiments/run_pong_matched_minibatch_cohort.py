#!/usr/bin/env python3
"""Run a fixed-minibatch native Pong autosweep, replay generated Pong 1:1, and emit a report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import REPO_ROOT, render_shell, timestamp_slug, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minibatch-size", type=int, required=True, help="Fixed minibatch size for the cohort.")
    parser.add_argument("--max-runs", type=int, default=5, help="Number of native sweep trials.")
    parser.add_argument(
        "--train-total-timesteps",
        type=int,
        default=10_000_000,
        help="Total timesteps per trial.",
    )
    parser.add_argument("--downsample", type=int, default=128, help="Sweep log downsampling factor.")
    parser.add_argument("--tag", default=None, help="Base tag. Defaults to a timestamped minibatch slug.")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "artifacts" / "experiments"),
        help="Directory that stores manifests and reports.",
    )
    return parser


def _run(argv: list[str]) -> None:
    subprocess.run(argv, check=True, cwd=REPO_ROOT)


def main() -> int:
    args = build_parser().parse_args()
    base_tag = args.tag or f"pong_mb{args.minibatch_size}_{timestamp_slug()}"
    output_root = Path(args.output_root).resolve()
    native_tag = f"{base_tag}_native"
    generated_tag = f"{base_tag}_generated"

    native_script = REPO_ROOT / "scripts" / "experiments" / "run_pong_native_autosweep.py"
    replay_script = REPO_ROOT / "scripts" / "experiments" / "replay_pong_generated_from_native.py"
    report_script = REPO_ROOT / "scripts" / "experiments" / "report_pong_matched_replay.py"

    native_manifest = output_root / native_tag / "manifest.json"
    generated_manifest = output_root / generated_tag / "manifest.json"
    report_dir = output_root / generated_tag / "report"

    native_argv = [
        sys.executable,
        str(native_script),
        "--tag",
        native_tag,
        "--max-runs",
        str(args.max_runs),
        "--train-total-timesteps",
        str(args.train_total_timesteps),
        "--fixed-minibatch-size",
        str(args.minibatch_size),
        "--downsample",
        str(args.downsample),
        "--output-root",
        str(output_root),
    ]
    _run(native_argv)

    replay_argv = [
        sys.executable,
        str(replay_script),
        "--native-manifest",
        str(native_manifest),
        "--tag",
        generated_tag,
        "--output-root",
        str(output_root),
    ]
    _run(replay_argv)

    report_argv = [
        sys.executable,
        str(report_script),
        "--replay-manifest",
        str(generated_manifest),
        "--output-dir",
        str(report_dir),
    ]
    _run(report_argv)

    cohort_manifest = output_root / f"{base_tag}_cohort_manifest.json"
    write_json(
        cohort_manifest,
        {
            "experiment": "pong_matched_minibatch_cohort",
            "minibatch_size": args.minibatch_size,
            "max_runs": args.max_runs,
            "train_total_timesteps": args.train_total_timesteps,
            "downsample": args.downsample,
            "native_manifest": str(native_manifest),
            "generated_manifest": str(generated_manifest),
            "report_dir": str(report_dir),
            "command_shell": render_shell(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--minibatch-size",
                    str(args.minibatch_size),
                    "--max-runs",
                    str(args.max_runs),
                    "--train-total-timesteps",
                    str(args.train_total_timesteps),
                    "--downsample",
                    str(args.downsample),
                    "--output-root",
                    str(output_root),
                    *(["--tag", base_tag] if args.tag else []),
                ]
            ),
        },
    )
    print(cohort_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
