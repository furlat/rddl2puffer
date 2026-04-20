#!/usr/bin/env python3
"""Run a parity-locked native Pong autosweep inside Puffertank."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from common import (
    DEFAULT_IMAGE,
    REPO_ROOT,
    find_log_files,
    render_shell,
    run_puffertank,
    timestamp_slug,
    to_container_path,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Puffertank image tag.")
    parser.add_argument("--tag", default=None, help="Output tag. Defaults to a timestamped slug.")
    parser.add_argument("--max-runs", type=int, default=24, help="Number of sweep trials.")
    parser.add_argument(
        "--train-total-timesteps",
        type=int,
        default=50_000_000,
        help="Total timesteps per trial.",
    )
    parser.add_argument(
        "--fixed-minibatch-size",
        type=int,
        default=None,
        help="When set, lock both train.minibatch_size and the sweep range to this value.",
    )
    parser.add_argument("--downsample", type=int, default=128, help="Sweep log downsampling factor.")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "artifacts" / "experiments"),
        help="Directory that stores manifests and logs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tag = args.tag or f"pong_native_autosweep_{timestamp_slug()}"
    experiment_root = Path(args.output_root).resolve() / tag
    log_root = experiment_root / "logs"
    checkpoint_root = experiment_root / "checkpoints"
    container_log_root = to_container_path(log_root)
    container_checkpoint_root = to_container_path(checkpoint_root)
    manifest_path = experiment_root / "manifest.json"

    log_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    build_command = "cd /puffertank/pufferlib && bash build.sh pong"
    run_puffertank(build_command, image=args.image)

    sweep_command = (
        "cd /puffertank/pufferlib && "
        "puffer sweep pong "
        f"--log-dir {container_log_root} "
        f"--checkpoint-dir {container_checkpoint_root} "
        f"--train.total-timesteps {args.train_total_timesteps} "
        f"--sweep.max-runs {args.max_runs} "
        "--sweep.metric episode_return "
        f"--sweep.use-gpu False "
        "--sweep.gpus 1 "
        f"--sweep.downsample {args.downsample} "
        "--train.gpus 1 "
        "--train.use-rnn 1 "
        "--env.frameskip 8 "
        "--sweep.env.frameskip.min 8 "
        "--sweep.env.frameskip.max 8"
    )
    if args.fixed_minibatch_size is not None:
        # Keep the sweep inside valid heavy-minibatch regimes. Without this,
        # Protein spends most of its time proposing impossible configurations
        # where minibatch_size > total_agents * horizon and the run gets skipped.
        min_total_agents = 1024
        min_horizon = max(32, math.ceil(args.fixed_minibatch_size / min_total_agents))
        sweep_command += (
            f" --vec.total-agents {min_total_agents}"
            f" --train.horizon {min_horizon}"
            f" --train.minibatch-size {args.fixed_minibatch_size}"
            f" --sweep.train.minibatch-size.min {args.fixed_minibatch_size}"
            f" --sweep.train.minibatch-size.max {args.fixed_minibatch_size}"
            f" --sweep.vec.total-agents.min {min_total_agents}"
            f" --sweep.train.horizon.min {min_horizon}"
        )
    run_puffertank(sweep_command, image=args.image)

    logs = find_log_files(log_root, "pong")
    write_json(
        manifest_path,
        {
            "experiment": "pong_native_autosweep",
            "tag": tag,
            "image": args.image,
            "native_env_name": "pong",
            "generated_env_name": "rddl_pong_parity",
            "log_root": str(log_root),
            "checkpoint_root": str(checkpoint_root),
            "manifest_path": str(manifest_path),
            "max_runs": args.max_runs,
            "train_total_timesteps": args.train_total_timesteps,
            "fixed_minibatch_size": args.fixed_minibatch_size,
            "downsample": args.downsample,
            "frameskip_locked": 8,
            "metric_locked": "episode_return",
            "command": sweep_command,
            "command_shell": render_shell(
                [
                    "python3",
                    str(Path(__file__).resolve()),
                    "--tag",
                    tag,
                    "--max-runs",
                    str(args.max_runs),
                    "--train-total-timesteps",
                    str(args.train_total_timesteps),
                    *(
                        ["--fixed-minibatch-size", str(args.fixed_minibatch_size)]
                        if args.fixed_minibatch_size is not None
                        else []
                    ),
                    "--downsample",
                    str(args.downsample),
                    "--output-root",
                    str(Path(args.output_root).resolve()),
                ]
            ),
            "log_files": [str(path) for path in logs],
        },
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
