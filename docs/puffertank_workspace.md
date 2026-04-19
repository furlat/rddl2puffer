# PufferTank Workspace

This repository is now wired for a Docker-first native Puffer workflow.

## Why this is the default

On this workstation, the GPU is visible inside WSL, but the host-side CUDA compiler is not yet present.

That means:

- direct host execution is possible later,
- but Docker via PufferTank is the cleaner and more reproducible starting point for native Ocean work.

## Local workspace layout

```text
rddl2puffer/
  third_party/
    puffertank/
    pufferlib/
```

Both upstream repos are treated as local dependencies and should stay ignored by git in this repo.

## Devcontainer flow

The root `.devcontainer/devcontainer.json` uses:

- `third_party/puffertank/puffertank.dockerfile` as the base image definition,
- `third_party/pufferlib` mounted into `/puffertank/pufferlib`,
- this repository mounted as the main workspace at `/workspace/rddl2puffer`.

Inside the container, the intended loop is:

```bash
. /root/.local/bin/env
. /puffertank/venv/bin/activate
uv pip install -e /workspace/rddl2puffer
uv run pytest
uv run rddl2puffer emit-workspace-demo --env-name rddl_demo
cd /puffertank/pufferlib
bash build.sh rddl_demo --local
```

## Why mount the local `pufferlib` checkout

The upstream PufferTank image clones its own `pufferlib` during image build.
For compiler development we want the generated files to persist in the local workspace, so the devcontainer mount shadows the image copy with the checked-out host copy.

That makes it easier to:

- inspect generated Ocean files from the host,
- diff generated code,
- run native builds in the container against local changes.

## Current state

- `third_party/puffertank` is checked out on branch `4.0`
- `third_party/pufferlib` is checked out on branch `4.0`
- the compiler can now emit a demo env bundle directly into the local PufferLib tree

## Next integration step

Replace the demo bundle emission with real RDDL-lowered code generation, then differential-test that output against a Python reference path before trying to train it.

