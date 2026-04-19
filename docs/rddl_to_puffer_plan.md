# RDDL to Puffer Native Environment Plan

## Goal

Build a toolchain that takes an RDDL domain plus instance and produces a native Puffer Ocean environment, not a Python wrapper.

The compiler pipeline for v1 should be:

```text
RDDL domain/instance
  -> grounded schema
  -> typed execution IR
  -> Python reference interpreter
  -> generated Ocean env bundle (.h, .c, .ini)
  -> Puffer build and train workflow
```

## Codebase State on April 18, 2026

The repo is no longer a greenfield scaffold. The following pieces already exist and are working:

- `rddl2puffer/frontend/schema.py`
  stable fluent metadata, canonical ordering, and flat layout generation
- `rddl2puffer/frontend/ground.py`
  `MockGroundedModel` placeholder that already builds an `EnvSchema`
- `rddl2puffer/ir/nodes.py`
  typed deterministic IR nodes and `IRProgram`
- `rddl2puffer/ir/interpret.py`
  deterministic Python interpreter for the current IR
- `rddl2puffer/testing/differential.py`
  abstract reference-runtime interface and mismatch reporting types
- `rddl2puffer/testing/rollout_compare.py`
  seeded rollout comparison utilities
- `rddl2puffer/backends/puffer_c/`
  stub emitters for Ocean `.h`, `.c`, and `.ini`
- `rddl2puffer/workspace.py`
  writes generated bundles into a local `third_party/pufferlib` checkout
- `.devcontainer/devcontainer.json`
  wired to `third_party/puffertank/puffertank.dockerfile` and mounts local `third_party/pufferlib`

Current verification in this WSL workspace:

- `env UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q` passes with `8 passed`
- `env UV_CACHE_DIR=/tmp/uv-cache uv run rddl2puffer workspace-info` resolves the expected local workspace
- `env UV_CACHE_DIR=/tmp/uv-cache uv run rddl2puffer emit-workspace-demo --env-name rddl_demo` writes a demo bundle into `third_party/pufferlib`

Current environment gap in this WSL shell:

- `docker` is not installed or not on `PATH`
- plain `pytest` is not on `PATH`, but `uv run pytest` works

## Refined Problem Statement

The main job is no longer "create the scaffold." The scaffold is here.

The real remaining milestones are:

1. replace mock grounding with a real RDDL grounding adapter,
2. extend the IR from deterministic arithmetic to staged CPF execution with RNG,
3. emit real Ocean environment logic instead of placeholder files,
4. validate generated behavior against a reference RDDL runtime,
5. run the native build and training loop inside Puffertank.

## v1 Scope

Keep v1 intentionally narrow:

- single-agent domains
- fully grounded execution
- fixed finite horizon
- deterministic expressions first
- seeded stochastic nodes second
- MDP support first
- POMDP observation fluents next
- flat or scalar observations
- discrete and box-style actions
- reward and terminal generation

Explicit non-goals for v1:

- multi-agent semantics
- lifted execution without grounding
- broad support for every RDDL operator or distribution
- planner-specific belief state features
- aggressive IR optimization before parity is stable

## Architecture Aligned to the Existing Repo

### 1. Front-end grounding

Existing anchor:

- `rddl2puffer/frontend/ground.py`
- `rddl2puffer/frontend/schema.py`

Next step:

- replace `MockGroundedModel` with a real grounded model adapter
- ingest domain, instance, and optional non-fluents
- produce canonical state, action, observation, intermediate, reward, and terminal specs
- preserve stable ordering so downstream offsets never drift across runs

Recommended output for debugging:

- `schema.json`
- CPF dependency dump
- ordered fluent table with bounds, dtype, shape, and flat offsets

### 2. Static IR

Existing anchor:

- `rddl2puffer/ir/nodes.py`
- `rddl2puffer/ir/lower.py`
- `rddl2puffer/ir/validate.py`
- `rddl2puffer/ir/interpret.py`

Next step:

- keep the IR backend-agnostic
- add explicit node support for intermediate values, reward, terminal, and observation stages
- add seeded RNG and distribution nodes without mixing sampling logic into the rest of the interpreter
- preserve a fully ordered execution schedule that is easy to interpret, diff, and emit

Recommended IR execution order:

1. derived and intermediate values
2. next-state writes
3. reward
4. terminal
5. observation

### 3. Python semantic oracle

Existing anchor:

- `rddl2puffer/ir/interpret.py`
- `rddl2puffer/testing/`

Next step:

- use the interpreter as the first semantic debugger
- add a thin adapter that wraps the grounded reference runtime behind `ReferenceEnv`
- compare exact seeded traces before touching C code generation

This keeps the critical debugging loop in Python until parity is solid.

### 4. Native Puffer backend

Existing anchor:

- `rddl2puffer/backends/puffer_c/emit_env_h.py`
- `rddl2puffer/backends/puffer_c/emit_env_c.py`
- `rddl2puffer/backends/puffer_c/emit_ini.py`
- `rddl2puffer/workspace.py`

Next step:

- replace placeholder emitters with real Ocean codegen
- lower layout metadata into fixed contiguous arrays
- emit reset and step logic from the ordered IR
- keep template code thin and push domain variation into generated constants and schedules

Generated layout target:

```text
third_party/pufferlib/
  ocean/rddl_<domain>/rddl_<domain>.h
  ocean/rddl_<domain>/rddl_<domain>.c
  config/rddl_<domain>.ini
```

### 5. Workspace and build path

Existing anchor:

- `.devcontainer/devcontainer.json`
- `docs/puffertank_workspace.md`
- `scripts/bootstrap_puffertank_workspace.sh`

Current repo assumption:

- local `third_party/puffertank` checkout exists
- local `third_party/pufferlib` checkout exists
- compiler-side Python work can happen directly in WSL with `uv`
- native build and train work should happen inside the Puffertank devcontainer once Docker is available in WSL

## Recommended Implementation Sequence

## Phase 0: Finish the workspace path

Goal:

- make the Docker-first path actually usable from this WSL environment

Tasks:

- install Docker Desktop with WSL integration, or install and expose a working Docker engine to this distro
- verify `docker --version` and `docker run --gpus all ...` from WSL
- open the repo through the existing devcontainer

Success criterion:

- the devcontainer starts using `third_party/puffertank/puffertank.dockerfile`
- `/puffertank/pufferlib` inside the container maps to local `third_party/pufferlib`

## Phase 1: Replace mock grounding

Goal:

- turn the current mock schema path into a real grounded RDDL frontend

Tasks:

- add a real frontend adapter in `rddl2puffer/frontend/parse.py` and `rddl2puffer/frontend/ground.py`
- define a non-mock grounded model type
- emit stable schema dumps for benchmark instances

Deliverables:

- real grounded schema object
- `schema.json`
- benchmark fixture inputs under `examples/domains/`

Success criterion:

- repeated runs on the same instance produce byte-stable schema dumps

## Phase 2: Lower grounded CPFs to IR

Goal:

- implement the real `lower_grounded_cpfs()` path

Tasks:

- map grounded expressions to typed nodes
- add stage-aware store nodes
- add seeded RNG nodes behind a narrow interface
- validate dependency ordering and slot coverage

Deliverables:

- `ir.json`
- pretty-printer for inspection
- unit tests for deterministic and stochastic lowering shape

Success criterion:

- the interpreter reproduces reference transitions on small domains

## Phase 3: Wire differential testing to a real reference runtime

Goal:

- turn the generic testing harness into a real parity checker

Tasks:

- implement a concrete `ReferenceEnv` adapter
- compare reset obs, step obs, reward, done, and optional hidden state
- generate readable mismatch reports with exact step locations

Deliverables:

- reference adapter
- seeded rollout regression tests

Success criterion:

- exact parity on chosen deterministic domains
- explainable parity on seeded stochastic domains

## Phase 4: Emit real Ocean environment logic

Goal:

- compile the IR into a real native Puffer environment

Tasks:

- emit real `step()` and `reset()` logic in the generated header
- emit a standalone local harness in the generated `.c`
- align emitted config with the actual Puffer config schema used by `third_party/pufferlib`
- keep codegen readable enough for diff-based debugging

Deliverables:

- generated Ocean env bundle for at least one benchmark domain

Success criterion:

- `bash build.sh rddl_<domain> --local` succeeds inside the Puffertank environment

## Phase 5: Native rollout parity

Goal:

- prove that generated native execution matches the Python reference path

Tasks:

- drive fixed action sequences through both runtimes
- compare obs, reward, done, and optional hidden state traces
- add debug hooks to expose internal state when parity fails

Success criterion:

- trace parity across multiple benchmark domains

## Phase 6: Training and cleanup

Goal:

- make the generated env usable in the normal Puffer loop

Tasks:

- train with `puffer train rddl_<domain>`
- inspect reward scale, reset correctness, and terminal handling
- only then optimize temporary buffers, branch shape, and zeroing strategy

Success criterion:

- generated env trains without obvious reset, metadata, or numerical issues

## Immediate File-Level Next Steps

These are the highest-value next edits from the current scaffold:

- `rddl2puffer/frontend/parse.py`
  define the real ingestion boundary for RDDL inputs
- `rddl2puffer/frontend/ground.py`
  replace `MockGroundedModel` with a real grounded adapter type
- `rddl2puffer/ir/lower.py`
  implement `lower_grounded_cpfs()`
- `rddl2puffer/ir/nodes.py`
  extend node types for seeded stochastic execution and explicit stage handling
- `rddl2puffer/ir/interpret.py`
  add RNG-backed execution semantics
- `rddl2puffer/testing/`
  add a concrete reference adapter and real parity tests
- `rddl2puffer/backends/puffer_c/`
  replace emitter stubs with actual Ocean code generation
- `tests/`
  add schema, lowering, stochastic, and codegen parity coverage

## Benchmark Strategy

Start with three very small domains:

1. a discrete MDP
2. a continuous MDP
3. a POMDP with explicit observation fluents

The right first domains are the ones that make debugging easy, not the most impressive ones.

## Definition of Done for v1

v1 is done when this repo can:

1. ingest a small RDDL domain plus instance,
2. ground it into a stable flat schema,
3. lower it into a typed IR,
4. reproduce seeded transitions in the Python interpreter,
5. emit a native Ocean environment into `third_party/pufferlib`,
6. build that env with the normal Puffer workflow,
7. and show rollout parity against the reference runtime on the same test cases.

## Practical Note for This WSL Session

Right now the compiler side is ready for iteration in WSL with:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
env UV_CACHE_DIR=/tmp/uv-cache uv run rddl2puffer workspace-info
env UV_CACHE_DIR=/tmp/uv-cache uv run rddl2puffer emit-workspace-demo --env-name rddl_demo
```

The native Puffer build side is not ready in this shell until Docker is installed or exposed to WSL. Once that is fixed, the existing devcontainer wiring should be the fastest way to continue without wrestling with host CUDA toolchain setup.
