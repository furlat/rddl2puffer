#!/usr/bin/env python3
"""Replay every native Pong sweep trial 1:1 on the generated Pong parity env."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    DEFAULT_IMAGE,
    REPO_ROOT,
    emit_pong_parity_workspace,
    find_log_files,
    native_replay_overrides,
    render_shell,
    run_puffertank,
    timestamp_slug,
    to_container_path,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Puffertank image tag.")
    parser.add_argument("--native-manifest", required=True, help="Manifest JSON from the native autosweep script.")
    parser.add_argument("--tag", default=None, help="Replay tag. Defaults to a timestamped slug.")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "artifacts" / "experiments"),
        help="Directory that stores replay manifests and logs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    native_manifest_path = Path(args.native_manifest).resolve()
    native_manifest = json.loads(native_manifest_path.read_text(encoding="utf-8"))
    native_log_root = Path(native_manifest["log_root"]).resolve()

    tag = args.tag or f"pong_generated_replay_{timestamp_slug()}"
    experiment_root = Path(args.output_root).resolve() / tag
    log_root = experiment_root / "logs"
    checkpoint_root = experiment_root / "checkpoints"
    manifest_path = experiment_root / "manifest.json"
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    written = emit_pong_parity_workspace("rddl_pong_parity")
    build_command = "cd /puffertank/pufferlib && bash build.sh rddl_pong_parity"
    run_puffertank(build_command, image=args.image)

    entries: list[dict[str, object]] = []
    for native_log_path in find_log_files(native_log_root, "pong"):
        payload = json.loads(native_log_path.read_text(encoding="utf-8"))
        native_id = native_log_path.stem
        run_log_root = log_root / native_id
        run_checkpoint_root = checkpoint_root / native_id
        container_run_log_root = to_container_path(run_log_root)
        container_run_checkpoint_root = to_container_path(run_checkpoint_root)
        run_log_root.mkdir(parents=True, exist_ok=True)
        run_checkpoint_root.mkdir(parents=True, exist_ok=True)

        overrides = native_replay_overrides(payload, generated_env_name="rddl_pong_parity")
        command = (
            "cd /puffertank/pufferlib && puffer train rddl_pong_parity "
            f"--log-dir {container_run_log_root} "
            f"--checkpoint-dir {container_run_checkpoint_root} "
            f"--tag replay_{native_id} "
            f"--sweep.downsample {native_manifest['downsample']} "
            + " ".join(overrides)
        )
        run_puffertank(command, image=args.image)
        generated_logs = find_log_files(run_log_root, "rddl_pong_parity")
        if len(generated_logs) != 1:
            raise RuntimeError(
                f"Expected exactly one generated replay log under {run_log_root}, found {len(generated_logs)}."
            )
        entries.append(
            {
                "native_log_path": str(native_log_path),
                "generated_log_path": str(generated_logs[0]),
                "native_run_id": native_id,
                "overrides": overrides,
            }
        )

    write_json(
        manifest_path,
        {
            "experiment": "pong_generated_replay",
            "tag": tag,
            "image": args.image,
            "native_manifest": str(native_manifest_path),
            "native_log_root": str(native_log_root),
            "generated_log_root": str(log_root),
            "checkpoint_root": str(checkpoint_root),
            "generated_env_name": "rddl_pong_parity",
            "emitted_files": [str(path) for path in written],
            "build_command": build_command,
            "command_shell": render_shell(
                [
                    "python3",
                    str(Path(__file__).resolve()),
                    "--native-manifest",
                    str(native_manifest_path),
                    "--tag",
                    tag,
                    "--output-root",
                    str(Path(args.output_root).resolve()),
                ]
            ),
            "entries": entries,
        },
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
