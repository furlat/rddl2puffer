# RDDL+ Design Notes

## Purpose

This document records two immediate `RDDL+` design choices that came out of the CartPole audit:

1. `init-state` must support stochastic specification and be interpreted stochastically at reset time.
2. state and observation layout order must preserve source intent by default rather than being alphabetically canonicalized.

These two choices are tightly connected to the project objective:

- describe native `PufferLib` envs in a source language with strong RDDL ancestry
- generate native Ocean/C code
- match handcrafted `PufferLib` behavior closely enough that training and evaluation comparisons are meaningful

## Decision 1: Stochastic `init-state`

### Problem

The current standard init-state path we inspected behaves like a point-mass initial belief. In practice, this is too weak for many native `Puffer` environments and too weak for general POMDP-style initialization.

The current pyRDDLGym parser accepts `range_const` literals inside `init-state` entries:

- [parser.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/parser/parser.py:1066)
- [parser.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/parser/parser.py:1130)

At the same time, the parser already supports random-variable expressions elsewhere in the language:

- [parser.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/parser/parser.py:874)

So the issue is not that the language machinery cannot represent distributions. The issue is that `init-state` is currently restricted to literals.

### Choice

`RDDL+` should allow `init-state` entries to be stochastic.

This means:

- deterministic `init-state` entries remain valid
- stochastic `init-state` entries are sampled on every episode reset
- mixed deterministic and stochastic initialization is allowed
- the generator must never collapse a distribution to its mean unless explicitly asked to do so

### Intended Syntax

We should support one of these equivalent forms:

```text
init-state {
  pos = Uniform(-0.04, 0.04);
  vel = Uniform(-0.04, 0.04);
  ang-pos = Uniform(-0.04, 0.04);
  ang-vel = Uniform(-0.04, 0.04);
}
```

or, if we want an explicit sampling marker:

```text
init-state {
  pos ~ Uniform(-0.04, 0.04);
  vel ~ Uniform(-0.04, 0.04);
  ang-pos ~ Uniform(-0.04, 0.04);
  ang-vel ~ Uniform(-0.04, 0.04);
}
```

The assignment form is attractive because it reuses more of the existing grammar, but the `~` form is semantically clearer. We can choose one during parser implementation.

### Semantics

`init-state` should mean:

- this is the episode-start state specification
- if an entry is deterministic, copy it directly
- if an entry is stochastic, sample it at reset

This gives `RDDL+` a proper initial belief mechanism and lets us express native `Puffer` reset distributions without inventing a separate reset concept unless we later find we truly need one.

### Why This Matters

- It restores a sane POMDP story with nontrivial initial belief.
- It lets us model native `Puffer` reset behavior directly.
- It avoids fake “average point mass” approximations that would distort training.

## Decision 2: Preserve Source Order

### Problem

The current frontend canonicalization rule sorts by role and name:

- [schema.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/frontend/schema.py:134)

That means a source declaration order like:

- `pos`
- `ang-pos`
- `vel`
- `ang-vel`

becomes a generated flat order like:

- `ang-pos`
- `ang-vel`
- `pos`
- `vel`

This is not required by RDDL semantics. It is just our current implementation choice.

### Choice

By default, `RDDL+` should preserve declared/source order for:

- state layout
- observation layout
- action layout

We can still support explicit override metadata later, but the default should be source order, not alphabetical reordering.

### Why This Matters

This is especially important for later experiments where we want to:

- train in a generated environment
- test or compare in a canonical/reference environment
- transfer policies or compare representations without an avoidable coordinate scramble

If the frontend silently reorders the tensor layout, we create a fake mismatch that has nothing to do with the actual environment semantics.

## Current Parser Stack

The reference parser we have locally is written in Python and uses `PLY`:

- import site: [parser.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/parser/parser.py:8)
- lexer construction: [parser.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/parser/parser.py:227)
- parser construction: [parser.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/parser/parser.py:1286)

So yes: the parser is Python.

For this repo, we currently do not yet have a real frontend parser of our own. We still use mock grounding plus targeted benchmark builders. That means the next parser work will likely begin either by:

- extending a local pyRDDLGym-derived parser path, or
- implementing an adapter/forked frontend in Python with the same general structure

## Parser Expansion Plan

## Phase 1: Extend Grammar For Stochastic `init-state`

Goal:

- allow stochastic initialization in `init-state`

Planned change:

- replace the current `range_const` restriction in `pvar_inst_def` for `init-state` assignments with a broader initialization expression form

Practical options:

1. allow `expr`
2. allow a smaller `init_expr`
3. allow explicit `randomvar_expr` plus deterministic literals

Recommended approach:

- add a dedicated `init_expr`
- allow:
  - literals
  - unary/binary numeric expressions over literals
  - distribution expressions
- forbid references to state/action fluents unless we intentionally support them later

This keeps reset semantics clean and avoids weird circular initialization behavior.

## Phase 2: Frontend Data Model For Reset Semantics

Goal:

- represent reset-time behavior explicitly in our own frontend/schema layer

Needed additions:

- `InitSpec` or equivalent metadata for each state fluent
- support for:
  - deterministic reset value
  - stochastic reset distribution
- a clean distinction between:
  - reset-time sampling
  - step-time CPF execution

## Phase 3: Preserve Declared Order In Schema Construction

Goal:

- remove alphabetical canonicalization as the default layout rule

Needed changes:

- adjust [schema.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/frontend/schema.py) so layout construction preserves source/declaration order by default
- keep any secondary canonicalization only when explicitly requested
- make layout order part of the schema contract and debug dumps

## Phase 4: IR Support For Reset-Time Sampling

Goal:

- make stochastic init-state compile through the same pipeline as the rest of the environment

Needed changes:

- add reset-stage IR or reset nodes
- keep reset execution distinct from step execution
- ensure reset-time random draws are reproducible under seeding

## Phase 5: Native Codegen For Reset Sampling

Goal:

- emit reset-time sampling in generated Ocean/C code

Needed changes:

- extend generated env structs with whatever RNG state is required
- emit reset-time sampling logic in `c_reset`
- support the first useful distribution family:
  - `Uniform`
  - likely `Normal` after that

## Phase 6: Differential Validation

Goal:

- verify that stochastic reset semantics are actually preserved

Needed checks:

- seeded reset parity against a Python reference path
- empirical distribution checks on generated resets
- rollout parity after reset sampling

## Codegen Expansion Plan

## A. Source-Order Layout Preservation

First codegen-facing fix:

- preserve source order in the schema and generated layout

Expected repo impact:

- [schema.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/frontend/schema.py)
- frontend adapters and benchmark builders
- tests that currently assume alphabetical canonicalization

## B. Reset Sampling In Generated C

Second codegen-facing fix:

- support stochastic reset logic in emitted `c_reset`

Expected repo impact:

- IR nodes / validation
- codegen helpers
- [emit_env_h.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/backends/puffer_c/emit_env_h.py)

## C. Later Experiment Support

These two features together support an important later experiment:

- train in the generated environment
- evaluate or compare against a canonical/reference environment
- keep tensor conventions stable enough that differences reflect task semantics, not avoidable layout drift

That is one of the main reasons source-order preservation is not cosmetic. It is a real experimental-control requirement.

## Immediate Next Engineering Steps

1. change frontend layout construction to preserve source order by default
2. record layout order explicitly in debug artifacts
3. design the `init-state` grammar extension for stochastic values
4. add frontend representation for reset distributions
5. add reset-time sampling codegen in generated `c_reset`

## Working Rule

For both of these changes:

- if the issue is just our current implementation choice, fix the generator
- if the issue is a true standard-RDDL expressivity gap, extend `RDDL+`

These two items are examples of both:

- layout order is a generator/frontend choice
- stochastic `init-state` is a true language limitation that `RDDL+` should fix
