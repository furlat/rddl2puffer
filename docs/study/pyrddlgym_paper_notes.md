# pyRDDLGym Paper Notes

Source material:

- arXiv HTML:
  [pyRDDLGym: From RDDL to Gym Environments](https://arxiv.org/html/2211.05939v5)
- local cached HTML:
  `/home/tommaso/Dev/rddl2puffer/third_party/reference_material/papers/pyRDDLGym_2211.05939v5.html`

This document is a markdown study conversion of the paper for project planning. It is intentionally a distilled note set, not a verbatim mirror of the paper.

## Why The Paper Matters For `rddl2puffer`

The paper gives us the reference semantics for turning RDDL into an executable environment:

- parse domain, instance, and non-fluents
- ground lifted fluents into a concrete instance
- derive action and observation spaces
- compute dependency order for CPFs
- evaluate one step by sampling CPFs, reward, and terminal conditions

That is almost exactly the semantic lowering problem we need to solve. The difference is that we want to target Puffer Ocean instead of Gym.

## Key Takeaways

### 1. Grounded RDDL is the right semantic center

The paper frames a grounded RDDL instance as a factored MDP, or POMDP when observation fluents are present.

For us, this strongly supports:

- grounding first in v1
- keeping the core execution representation static
- separating internal state from external observation

### 2. CPF evaluation order is explicit compiler work

The paper describes automatic level reasoning:

- derived and intermediate fluents are ordered by dependency analysis
- the framework constructs a call graph from the CPFs
- it uses topological sorting to determine evaluation order
- it validates that the ordering forms a DAG

This maps directly onto our IR scheduler. We should own this logic in compiler form, but mirror the reference semantics exactly.

### 3. Reset is deterministic, step is the semantic hot path

The paper’s design section describes:

- `__init__`: parse, ground, build spaces, instantiate the CPF sampler
- `reset()`: revert to the initial state defined in the instance
- `step()`: evaluate the CPF transition logic

This is important because it tells us not to overcomplicate v1 reset behavior. Reset can stay simple and instance-defined. The hard part is step semantics.

### 4. POMDP support is observation-stage support

The paper is clear that if observation fluents exist, the environment emits observations rather than raw state.

For our compiler this means:

- state and observation must be different layouts
- observation generation should be a separate execution stage
- the backend should not assume "obs == state"

### 5. pyRDDLGym extends RDDL in ways we need to handle deliberately

Important extensions or behaviors highlighted by the paper:

- terminal states are added through a `termination` block
- fluent level declarations are deprecated in favor of automatic reasoning
- action preconditions can be enforced or treated more softly
- state/action bounds are inferred through constraints
- vectorized and multivariate distributions exist in the modern toolchain

We should not accidentally treat every pyRDDLGym extension as part of our v1 target. We need a written subset.

## Sections Most Relevant To `rddl2puffer`

### Introduction

Main relevance:

- hard-coded Python environments are painful to verify and reuse
- declarative model descriptions are valuable because they preserve structure
- RDDL can bridge planning and RL workflows

Implication:

- our native target is justified, but only if we preserve the model semantics rather than bypassing them

### RDDL Support and Extensions

Main relevance:

- the implementation omits or deprecates some older language forms
- automatic level reasoning replaces manual level specification
- terminal states are added to support Gym-style episode termination

Implication:

- our frontend adapter must record which semantics come from original RDDL and which come from pyRDDLGym conventions

### Design and Implementation

Main relevance:

- initialization parses and grounds the model
- action and observation spaces are derived from the grounded problem
- CPFs are evaluated in dependency order
- reward and terminal conditions are computed during step
- reset returns initial state for MDPs and special observation behavior for POMDPs

Implication:

- our compiler should mirror this semantic pipeline first, then change the data layout, not the meaning

### Beyond the Engine

Main relevance:

- pyRDDLGym is not just a parser, it is also an ecosystem:
  - example environments
  - a benchmark manager
  - tools such as dependency analysis and planners
- rddlrepository is the benchmark source we should use for regression domains

Implication:

- we should use the surrounding ecosystem to choose benchmarks and generate reference traces

## Practical Semantics To Preserve

These are the project-critical behaviors to preserve when lowering into our IR:

- parse domain, instance, and non-fluents together
- build a grounded representation with deterministic naming
- distinguish state, action, intermediate, next-state, observation, reward, and termination logic
- evaluate CPFs in dependency order
- update state only after next-state computation
- emit observation separately from internal state
- compute reward from the post-evaluation substitution environment
- terminate on the declared termination conditions

## Things To Treat Carefully

### Action preconditions

The paper describes configurable enforcement behavior. That is useful for a Python reference implementation, but a native backend needs a more explicit policy.

We should decide early whether v1 does:

- hard failure
- clamping / default filling
- mask generation
- or a domain-controlled behavior only

### Reset semantics for stochastic initial states

The paper’s reset description is simple and deterministic. If we later want richer reset behavior, that should be a deliberate extension rather than an accidental mismatch.

### Observations at time zero

The paper notes that for POMDPs the initial return differs from the MDP case. We should encode this explicitly in the differential harness.

## Most Important Insight For The Compiler

The paper validates the compiler architecture we want:

- RDDL is the source language
- grounding produces the concrete problem instance
- dependency analysis yields the execution order
- CPF evaluation defines the one-step semantics

So the real compiler center is:

```text
grounded model -> ordered execution representation
```

not:

```text
gym wrapper -> native backend somehow
```

## Follow-Up Research Questions

- Should our frontend adapter start from `RDDLLiftedModel` or from the explicit `RDDLGrounder` output?
- How much of pyRDDLGym’s modern language extension set should count as in-scope for v1?
- For stochastic semantics, which distributions are the minimum viable subset?
- Do we want exact pyRDDLGym reset and precondition behavior, or only exact transition/reward semantics?

