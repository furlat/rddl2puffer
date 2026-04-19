# rddl2puffer

`rddl2puffer` compiles RDDL domain/instance files into native Puffer Ocean/C environments.

## What This Project Does

For a supported RDDL model, this repository can:

1. parse the input `.rddl` files,
2. build a flat schema for state, action, and observation variables,
3. lower supported CPFs into an intermediate representation (IR),
4. emit Ocean/C and `.ini` files into a local `PufferLib` checkout,
5. build the generated environment with native Puffer tooling,
6. train or sweep the generated environment against an existing upstream native PufferLib environment.

## Terms Used In This Repo

- Input RDDL: the `.rddl` files under `examples/`
- Compiler or transpiler: the Python code under `rddl2puffer/`
- Generated environment: emitted files such as `third_party/pufferlib/ocean/rddl_cartpole_discrete/*`
- Existing upstream native environment: pre-existing C/Ocean code already in `third_party/pufferlib/ocean/cartpole/*`
- Python reference runtime: a semantic checker used for debugging; the generated-vs-native sweeps run native Puffer on C/Ocean code, not on a Python wrapper

## Current End-To-End Example

The clearest end-to-end path today is discrete CartPole:

- Input RDDL:
  `examples/rddl_plus/cartpole_puffer_parity/domain.rddl`
  and
  `examples/rddl_plus/cartpole_puffer_parity/instance_deterministic_reset.rddl`
- Generated Ocean/C output:
  `third_party/pufferlib/ocean/rddl_cartpole_discrete/`
- Existing upstream native CartPole implementation:
  `third_party/pufferlib/ocean/cartpole/`
- Fresh generated-vs-native sweep artifacts:
  `artifacts/cartpole/verify_sweep_20260419/`

Useful entry points:

- [CartPole side-by-side comparison](docs/cartpole_side_by_side.md)
- [Fresh generated-vs-native sweep plot](artifacts/cartpole/verify_sweep_20260419/comparison.svg)
- [Fresh generated-vs-native sweep summary](artifacts/cartpole/verify_sweep_20260419/report.md)

## Current Scope

This is not yet a full general RDDL compiler. The strongest path today is the deterministic scalar subset used in the CartPole experiment:

- direct RDDL parsing,
- schema construction,
- IR lowering and validation,
- Python IR execution for debugging,
- Ocean/C emission,
- Puffer build and sweep integration.

Still incomplete:

- full general grounding coverage,
- stochastic RDDL support end to end,
- broader domain coverage beyond the current CartPole-focused workflow.

## Quickstart

This repository is configured for `uv`.

```bash
uv sync --dev
uv run pytest
uv run rddl2puffer schema-demo
uv run rddl2puffer ir-demo
uv run rddl2puffer emit-demo
```

The Python package handles parsing, lowering, and code generation. Native training and sweeps are run through `PufferLib` after the generated Ocean/C files have been emitted.

The native CartPole pipeline requires the local `third_party/` workspace, especially:

- `third_party/pufferlib`
- `third_party/puffertank`

## Recommended Workspace

For this machine, the preferred path is Docker via PufferTank.

Why:

- the RTX 5090 is visible inside WSL,
- native Puffer expects a CUDA toolchain,
- PufferTank packages CUDA, cuDNN, Torch, and PufferLib in the upstream 4.0 workflow.

This repo assumes a local workspace shape like:

```text
rddl2puffer/
  third_party/
    puffertank/
    pufferlib/
```

Those checkouts are local workspace dependencies, not part of this repo history.

## Container Workflow

The root devcontainer configuration points at the checked-out PufferTank Dockerfile and mounts the local `third_party/pufferlib` checkout into `/puffertank/pufferlib` inside the container.

That gives this loop:

1. open this repo in the devcontainer,
2. run `uv pip install -e /workspace/rddl2puffer` inside the container,
3. emit a generated env bundle into `/puffertank/pufferlib`,
4. build it with `bash build.sh ENV_NAME`,
5. train or sweep it with `puffer train ENV_NAME` or `puffer sweep ENV_NAME`.

Example:

```bash
cd /workspace/rddl2puffer
uv pip install -e .
python - <<'PY'
from pathlib import Path
from rddl2puffer.frontend.compile import compile_rddl_files
from rddl2puffer.workspace import discover_workspace

root = Path("/workspace/rddl2puffer")
program = compile_rddl_files(
    root / "examples/rddl_plus/cartpole_puffer_parity/domain.rddl",
    root / "examples/rddl_plus/cartpole_puffer_parity/instance_deterministic_reset.rddl",
    env_name="rddl_cartpole_discrete",
)
discover_workspace(root).write_env_bundle(program, env_name="rddl_cartpole_discrete")
PY

cd /puffertank/pufferlib
bash build.sh rddl_cartpole_discrete
puffer sweep rddl_cartpole_discrete
```

## Project Layout

```text
rddl2puffer/
  frontend/        # Parsing, schema construction, grounding hooks
  rddl_plus/       # Local parser fork and language extensions
  ir/              # IR nodes, validation, interpreter, lowering
  backends/        # Ocean/C emission
  testing/         # Differential rollout and reference-runtime utilities
tests/             # Pytest coverage
docs/              # Design notes and comparison docs
scripts/           # Workspace bootstrap helpers
```

## Further Reading

- [PLAN.md](PLAN.md)
- [docs/puffer_install.md](docs/puffer_install.md)
- [docs/puffertank_workspace.md](docs/puffertank_workspace.md)
- [docs/rddl_plus_design.md](docs/rddl_plus_design.md)
- `docs/study/` contains older exploratory notes; treat those as historical context rather than the current project contract
