# PLAN

## Objective

Build `RDDL+`, a Puffer-native environment specification language and codegen pipeline that can reproduce handcrafted `PufferLib` environments in behavior, training stability, and throughput.

This is no longer a project about preserving standard RDDL purity.

The target is:

- source language with strong RDDL ancestry
- native Ocean/C codegen
- `PufferLib` parity as the gold standard

In short:

```text
RDDL / RDDL+ source
  -> frontend + semantic normalization
  -> Puffer-oriented IR
  -> native Ocean/C codegen
  -> Puffer build / train / benchmark
  -> parity against handcrafted Puffer envs
```

## Ground Truth

For this project, the truth is the native `PufferLib` implementation, not the historical benchmark RDDL files.

That means:

- handcrafted `PufferLib` env behavior is the semantic target
- handcrafted `PufferLib` training behavior is the stability target
- handcrafted `PufferLib` systems behavior is the performance target

Standard RDDL remains useful as a starting point, reference style, and baseline source language. But when standard RDDL cannot cleanly express the env behavior that `PufferLib` actually uses, we will extend it.

## Project Framing

The project is now:

- `RDDL+` language design
- native code generation
- environment parity work, one env at a time

We should think of the problem as:

1. describe a native `Puffer` env in a compact source language
2. compile it into Ocean/C
3. match the handwritten env closely enough that the generated version is effectively interchangeable

## What Counts As Success

An environment is "done" only when the generated version is close enough to handcrafted `PufferLib` on all three axes:

- task behavior
- training behavior
- systems behavior

More concretely:

- rollout traces are aligned enough to explain remaining differences
- training curves and sweep outcomes are in the same regime
- SPS and runtime behavior are competitive with the handwritten version

## Current State

As of April 18, 2026, the project has:

- working Docker + GPU `PufferTank` flow
- native `PufferLib` build loop working inside the container
- a local Python parser fork under `rddl2puffer/rddl_plus`
- generated native `rddl_cartpole_discrete` training on GPU
- working `puffer sweep` path in this workspace
- a real CartPole audit across:
  - RDDL definition
  - generated env
  - handwritten native `Puffer` env

Key CartPole artifacts:

- [RDDL definition](/home/tommaso/Dev/rddl2puffer/artifacts/cartpole/definition_audit/rddl_definition.md)
- [Three-way semantic audit](/home/tommaso/Dev/rddl2puffer/artifacts/cartpole/definition_audit/three_way_semantic_audit.md)
- [Generated vs native audit](/home/tommaso/Dev/rddl2puffer/docs/study/cartpole_generated_vs_native_audit.md)
- [RDDL+ design notes](/home/tommaso/Dev/rddl2puffer/docs/rddl_plus_design.md)

Key current conclusion:

- the generated env is already much closer to the RDDL definition than native `Puffer` CartPole is
- the biggest remaining mismatches are not all "compiler bugs"
- they split into:
  - native handwritten choices
  - arbitrary generator choices
  - backend limitations
  - true limits of current RDDL

## Taxonomy Of Differences

Every env mismatch should be classified into one of these buckets.

### 1. Codegen Choice

Definition:

- our current generator made a choice, but nothing in RDDL or `PufferLib` required it

Examples:

- alphabetical canonicalization of state / observation order
- missing action clamps or NaN repair
- missing diagnostic log fields

Rule:

- if native `Puffer` already made a good choice and RDDL does not forbid it, we should just do what `Puffer` does

### 2. RDDL Ambiguity Or Underspecification

Definition:

- the source language does not determine the exact runtime convention we need

Examples:

- flat observation order
- flat state layout
- logging field semantics

Rule:

- choose the convention that matches `PufferLib`
- encode that choice explicitly in the frontend or `RDDL+` metadata so it stops being implicit

### 3. Backend Limitation

Definition:

- the current Ocean / `vecenv` path cannot faithfully transport the semantic distinction we care about

Examples:

- no first-class truncation channel in the current static `vecenv` training path

Rule:

- work around it when possible
- document it clearly when it blocks exact parity
- upgrade backend contracts when the limitation becomes material

### 4. True RDDL Limitation

Definition:

- standard current RDDL cannot express behavior that native `Puffer` uses and that we care about preserving

Current best example:

- stochastic reset distributions at episode start

Rule:

- this is where `RDDL+` must extend the language

## Core Policy

This is the most important operational rule in the repo:

- if a mismatch is just our current codegen choice, make the generator behave like `Puffer`
- if a mismatch is due to RDDL ambiguity, resolve the ambiguity in favor of `Puffer`
- if a mismatch is a real RDDL limitation, add a deliberate `RDDL+` extension

That is the environment-by-environment workflow.

## Design Principles

- `PufferLib` is the gold standard.
- Standard RDDL is a starting point, not a constraint.
- The language may mutate if that is what parity requires.
- We prefer explicit semantics over implicit compiler behavior.
- We should keep a sharp distinction between:
  - strict source fidelity
  - native-parity benchmark mode
- We should fix generator arbitrariness before inventing new language features.
- We should add `RDDL+` features only when the need is real and demonstrated by parity work.

## Environment-By-Environment Method

Each environment should go through the same loop.

### Step 1. Choose the handwritten native target

Freeze the source of truth:

- native Ocean `.h`
- native `binding.c`
- native config
- training behavior and sweep behavior

### Step 2. Collect the ancestor definition

If a standard RDDL benchmark exists:

- collect it
- treat it as historical / semantic prior
- not as the final target unless it already matches native `Puffer`

### Step 3. Build a three-way audit

Compare:

- standard RDDL definition
- generated env
- native `Puffer` env

Classify every mismatch as:

- codegen choice
- ambiguity
- backend limitation
- true RDDL limitation

### Step 4. Fix the easy mismatches first

Before extending the language, remove generator arbitrariness:

- layout order
- diagnostics
- action handling
- reward / terminal boundary behavior when already source-controlled

### Step 5. Introduce `RDDL+` extensions only where needed

Extensions should be added when native parity depends on behavior that standard RDDL cannot express cleanly.

### Step 6. Re-run parity and training

Measure:

- rollout alignment
- train curve shape
- sweep behavior
- SPS

### Step 7. Freeze the env contract

Once parity is good enough:

- document the env contract
- keep it as a regression target
- move to the next environment

## Immediate Language Direction: RDDL+

The first `RDDL+` features should be driven by concrete parity failures, not by theory.

### Likely first-class extensions

- stochastic reset distributions
- explicit flat observation order
- explicit flat state order
- action sanitization / clamp policy
- reward-on-boundary semantics
- terminal vs truncation policy
- diagnostic log field declarations

## Near-Term Work Buckets

## A. Generator Fixes We Should Do First

These are high-confidence fixes because they are clearly our choices.

1. Preserve declared/source order instead of alphabetically canonicalizing state and observation layouts.
2. Add optional runtime action sanitation generated from action bounds.
3. Add native-style diagnostic termination counters and log fields.
4. Keep terminated vs truncated distinct internally everywhere.
5. Make generated runtime conventions configurable through explicit metadata instead of hidden defaults.

## B. RDDL+ Extensions We Probably Need Soon

These are the first serious candidates for language growth.

1. Stochastic `init-state` at episode start.
2. Explicit layout/order directives.
3. Optional action handling directives.
4. Optional log-field declarations.

## C. Backend Work

These are not first, but we should keep them in view.

1. Decide whether explicit truncation needs to flow into the training interface.
2. Make sure generated envs can expose the same useful diagnostics as handwritten envs.
3. Keep native performance competitive while adding flexibility.

## CartPole: Current Limitation Analysis

CartPole is the first full anchor for the project.

What we learned:

- timeout semantics were a real generator bug and have been fixed
- native `cartpole` dynamics are materially different from the classical/RDDL equations
- native `cartpole` uses a random reset distribution not directly expressible in current standard RDDL init-state syntax
- observation ordering is currently an arbitrary generator choice on our side
- several differences that affect stability are native handwritten choices, not DSL limits

So CartPole tells us exactly how to think about future envs:

- do not blame the DSL too early
- do not blame the generator too early
- classify each mismatch precisely

## Practical Milestones

## Milestone 1: CartPole Native-Parity Spec

Goal:

- define a `Puffer`-targeted CartPole source spec, effectively the first `RDDL+` parity benchmark

Deliverables:

- clear mapping from native handwritten CartPole to source semantics
- explicit list of required extensions
- generated env that is closer to native than the current strict-RDDL-flavored one

## Milestone 2: Generator De-Arbitrarization

Goal:

- remove the obvious arbitrary generator choices that currently create fake differences

Deliverables:

- source-order-preserving layout
- action sanitation
- richer logging

## Milestone 3: First RDDL+ Extension Set

Goal:

- add the smallest useful language extensions required for CartPole parity

Deliverables:

- stochastic `init-state` support
- explicit layout metadata

## Milestone 4: Second And Third Environments

Goal:

- prove that the method generalizes beyond CartPole

Likely sequence:

- one simple continuous-control env
- one env with richer diagnostics or reset logic

## Immediate Next Steps

1. Fix layout ordering so generated envs preserve declared/source order.
2. Add native-style action handling to generated envs.
3. Add native-style diagnostic logging to generated envs.
4. Extend `RDDL+` so `init-state` can be stochastic.
5. Draft the first CartPole `RDDL+` parity spec.
6. Re-run generated vs native comparisons after those fixes.

## Working Rule For Every New Environment

For each environment, ask:

1. Is this mismatch just our current generator choice?
2. Is this just an ambiguity that we can resolve in favor of `Puffer`?
3. Is this a backend transport limitation?
4. Is this a true limit of standard RDDL?

Then act accordingly:

- generator choice -> fix codegen
- ambiguity -> choose `Puffer` semantics
- backend limitation -> patch or document backend
- true limit -> extend `RDDL+`

That is the core method of the project.
