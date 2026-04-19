# Tasks

## Current Milestone

- [x] Bootstrap a `uv`-managed Python package
- [x] Define canonical schema dataclasses and ordering rules
- [x] Define IR dataclasses and a deterministic interpreter
- [x] Add a differential-testing harness
- [x] Add baseline tests for schema, IR, and rollout comparison
- [x] Vendor local `PufferTank` and `PufferLib` checkouts under `third_party/`
- [x] Add a Docker-first workspace integration path
- [x] Study the CartPole benchmark family across RDDL and native Puffer
- [x] Codify the CartPole benchmark trio in the repo
- [x] Hand-lower discrete RDDL CartPole into the IR
- [x] Add a pyRDDLGym-backed oracle adapter and parity test for discrete CartPole
- [x] Emit a generated Ocean bundle including `binding.c`
- [ ] Build generated discrete CartPole through the native Puffer toolchain
- [ ] Run generated vs native CartPole comparisons through Puffer
- [ ] Preserve declared/source layout order instead of alphabetic canonicalization
- [ ] Design and implement stochastic `init-state` support in `RDDL+`
- [ ] Add reset-time sampling support to the IR and codegen
- [ ] Add a real RDDL grounding adapter
- [ ] Add stochastic IR nodes and seeded RNG semantics
- [ ] Generalize native code emission beyond the current scalar deterministic subset
- [ ] Validate against pyRDDLGym on seeded rollouts

## Suggested Next Domains

- discrete CartPole first,
- then one small continuous MDP,
- then one small POMDP with explicit observation fluents.

## Backend Checkpoints

- make schema dumps stable across runs,
- keep the IR backend-agnostic,
- preserve pyRDDLGym semantics before comparing against upstream native Puffer envs,
- use the interpreter as the semantic debugger,
- use generated `--local` Puffer builds before training builds,
- compare generated traces against the Python reference path before optimizing.
