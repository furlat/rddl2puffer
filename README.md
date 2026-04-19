# rddl2puffer

`rddl2puffer` compiles RDDL domain/instance files into native Puffer Ocean/C environments.

## What This Project Does

For a supported RDDL model, this repository can:

1. parse the input `.rddl` files,
2. build a flat schema for state, action, and observation variables,
3. lower supported CPFs into generated native step/reset code,
4. emit Ocean/C and `.ini` files into a local `PufferLib` checkout,
5. build the generated environment with native Puffer tooling,
6. train or sweep the generated environment against an existing upstream native PufferLib environment.

## Current End-To-End Example

The clearest end-to-end path today is source-driven discrete CartPole:

- Input RDDL:
  `examples/rddl_plus/cartpole_puffer_parity/domain.rddl`
  and
  `examples/rddl_plus/cartpole_puffer_parity/instance_deterministic_reset.rddl`
- Generated Ocean/C output:
  `third_party/pufferlib/ocean/rddl_cartpole_discrete/`
- Existing upstream native CartPole implementation:
  `third_party/pufferlib/ocean/cartpole/`
- Current sweep parity artifacts:
  `artifacts/cartpole/source_apr19_sweep_compare/`
- Current reward-vs-random artifacts:
  `artifacts/cartpole/source_directives_vs_random/`

Useful entry points:

- [CartPole side-by-side comparison](docs/cartpole_side_by_side.md)
- [Current generated-vs-native sweep plot](artifacts/cartpole/source_apr19_sweep_compare/comparison.svg)
- [Current generated-vs-native sweep summary](artifacts/cartpole/source_apr19_sweep_compare/report.md)
- [Current reward-vs-random report](artifacts/cartpole/source_directives_vs_random/report.md)

## Current CartPole Status

The runnable CartPole path is now source-driven from RDDL files. There is no handwritten Python environment specification on the runnable path.

Current checkpoints worth knowing:

- Generated env source:
  `examples/rddl_plus/cartpole_puffer_parity/`
- Generated Ocean/C output:
  `third_party/pufferlib/ocean/rddl_cartpole_discrete/`
- Upstream native comparison target:
  `third_party/pufferlib/ocean/cartpole/`
- Latest compiled-source single-run result:
  `194.397` score at `7.807M SPS`
- Latest native single-run reference:
  `196.962` score at `7.523M SPS`
- Current 4-run sweep snapshot:
  generated mean final score `101.818`, native mean final score `103.560`

This means the current deterministic-reset CartPole compiler path is already close enough to do real native generated-vs-handwritten comparisons on both training outcome and throughput.

## Current Scope

This is not yet a full general RDDL compiler. The strongest path today is the deterministic scalar subset exercised by the CartPole parity experiment:

- direct RDDL parsing,
- schema construction,
- lowering into generated native update code,
- Ocean/C emission,
- Puffer build and sweep integration.

Still incomplete:

- full general grounding coverage,
- stochastic RDDL support end to end,
- broader domain coverage beyond the current CartPole-focused workflow.

The main missing semantic feature for exact native CartPole parity is stochastic `init-state`. The current source-driven CartPole example uses deterministic reset because that feature is not implemented yet.

## Ocean <-> RDDL Crossovers

The exact local environment-name overlap between `third_party/pufferlib/ocean/` and `third_party/rddlrepository/rddlrepository/archive/` is currently:

- `cartpole`
- `pong`
- `tetris`

`CartPole` is the first completed source-driven target. `Pong` is the best second target candidate because it exists in both codebases and is much smaller than `Tetris`, which is a significantly heavier systems and codegen problem.

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
puffer sweep rddl_cartpole_discrete --sweep.use-gpu False --sweep.gpus 1
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
