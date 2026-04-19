# Puffer Install Notes

This project targets the native Puffer workflow, so the install path matters.

## Official upstream paths checked on April 18, 2026

- Puffer docs: [puffer.ai/docs.html](https://puffer.ai/docs.html)
- PufferTank repo: [github.com/PufferAI/PufferTank](https://github.com/PufferAI/PufferTank)
- PufferTank installer script:
  [raw install.sh](https://raw.githubusercontent.com/PufferAI/PufferTank/refs/heads/4.0/install.sh)

## What the docs currently recommend

The docs list two main installation paths:

- Docker via PufferTank
- `uv` via the PufferTank `install.sh` script

The docs also give this installation smoke test:

```bash
bash build.sh breakout
puffer train breakout
puffer eval breakout --load-model-path latest
```

## What the current `install.sh` actually does

The current script assumes:

- Ubuntu 24.04
- CUDA drivers already installed

It then:

1. installs system packages with `apt-get`,
2. installs `uv`,
3. creates a Python 3.12 virtual environment,
4. clones `PufferLib` on branch `4.0`,
5. detects the CUDA version from `nvcc`,
6. installs PyTorch from the matching CUDA wheel index,
7. installs PufferLib in editable mode,
8. runs `bash build.sh breakout`.

The apt packages in the script currently include:

- `curl`
- `git`
- `build-essential`
- `clang`
- `htop`
- `gdb`
- `tmux`
- `ccache`
- `libomp-dev`
- `libglfw3`
- `libgl1-mesa-dev`
- `python3.12-dev`
- `libnccl2`
- `libnccl-dev`

## What matters for `rddl2puffer`

- We should assume Ubuntu 24.04 and Python 3.12 for local development.
- Native Puffer work is CUDA-first, with Docker as the preferred default when the host CUDA toolchain is incomplete.
- Generated Ocean environments should be debugged with `bash build.sh ENV_NAME --local` before trying to train them.
- Our backend should emit code that fits the documented Ocean layout: core logic in `.h`, standalone demo in `.c`, and environment config in `config/`.

## Practical implication

This repository uses `uv` for the compiler side, but the main intended runtime path is now a checked-out `PufferTank` + `PufferLib` workspace under `third_party/`, preferably opened through a devcontainer based on the upstream PufferTank Dockerfile.

