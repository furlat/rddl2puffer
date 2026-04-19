# Reference Code Map

This document maps the pulled reference code to the specific responsibilities we care about in `rddl2puffer`.

Reference roots:

- pyRDDLGym:
  `/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym`
- rddlrepository:
  `/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository`

## Big Picture

The current pyRDDLGym codebase has two closely related views of the world:

- a lifted planning model path used by the environment and simulator
- an explicit grounder path that still exists and is useful for inspection

That matters for us, because our compiler wants a stable grounded schema and an execution schedule, while pyRDDLGym’s runtime is comfortable staying somewhat more symbolic internally before tracing and evaluation.

## pyRDDLGym: Files That Matter Most

### Environment wrapper

- [env.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/env.py)

What it does:

- loads and parses domain plus instance
- creates an `RDDLLiftedModel`
- instantiates a simulator backend
- derives Gym observation and action spaces
- defines `reset()` and `step()` at the environment API level

Why we care:

- it shows the reference public semantics
- it shows how action preparation, reward, termination, and invariant checks are ordered

### Simulator

- [simulator.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/simulator.py)

What it does:

- compiles initial values
- computes CPF dependency levels
- traces expressions for later execution
- prepares action dictionaries
- checks preconditions, invariants, and termination
- executes one step by evaluating CPFs, reward, state update, and observation update

Why we care:

- this is the semantic hot path
- our Python IR interpreter should line up with this behavior
- our native backend should ultimately line up with the same ordering

### Dependency analysis

- [levels.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/compiler/levels.py)

What it does:

- builds the CPF call graph
- validates type-legal dependencies
- topologically orders CPF evaluation

Why we care:

- this is the clearest reference for our IR execution schedule
- it also tells us which cross-role dependencies are legal

### Planning model structures

- [model.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/compiler/model.py)

What it does:

- defines the core model abstractions:
  - `RDDLPlanningModel`
  - `RDDLGroundedModel`
  - `RDDLLiftedModel`
- defines naming conventions for grounded variables
- stores variable metadata, CPF metadata, constraints, and instance data

Why we care:

- it is the cleanest reference for what a canonical model object should carry
- it exposes the exact grounded naming scheme:
  - fluent separator: `___`
  - object separator: `__`
  - next-state suffix: `'`

### Explicit grounder

- [grounder.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/grounder.py)

What it does:

- expands lifted variables into grounded variables
- builds grounded state, action, observation, and CPF mappings
- constructs an `RDDLGroundedModel`

Why we care:

- even though the env path now starts from `RDDLLiftedModel`, this file is still the most direct reference for explicit grounding behavior
- this is a likely starting point for our real frontend adapter if we want a stable grounded schema first

### Example grounding script

- [run_ground.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/examples/run_ground.py)

What it does:

- constructs an env
- runs the explicit grounder
- decompiles and prints the grounded model

Why we care:

- this is the easiest “hello world” reference for turning a domain plus instance into a grounded artifact we can inspect

## rddlrepository: Files That Matter Most

### Repository manager

- [manager.py](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/core/manager.py)

What it does:

- indexes the benchmark archive
- lists problems and contexts
- resolves domain and instance paths
- supports registering new problems

Why we care:

- it is the cleanest benchmark-entry layer
- we should use it to select and standardize our regression suite

### Repository README

- [README.md](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/README.md)

What it gives us:

- usage expectations
- context naming conventions
- examples of how pyRDDLGym loads repository-backed problems

## Practical Takeaways For `rddl2puffer`

## 1. We should not start by writing our own parser

The reference stack already gives us:

- parsing
- lifted model creation
- explicit grounding
- dependency analysis
- benchmark discovery

The best near-term move is to define an adapter boundary that imports these semantics into our own schema and IR.

## 2. The real semantic path is narrower than the full pyRDDLGym feature set

The modern pyRDDLGym stack supports much more than we want in v1:

- external Python functions
- vectorized distributions
- matrix operations
- broader planner tooling

For the compiler, we should first mirror only the subset we can lower confidently into a static native runtime.

## 3. We need to decide between two frontend entry points

Option A:

- adapt from `RDDLLiftedModel`

Pros:

- aligned with the current env path

Cons:

- may leave more symbolic structure than we want during early compiler debugging

Option B:

- adapt from `RDDLGrounder.ground()`

Pros:

- naturally closer to our “grounded schema first” compiler architecture

Cons:

- may diverge a bit from the most modern pyRDDLGym runtime internals

Recommended short-term approach:

- use both as study tools, but build the first compiler adapter around explicit grounding because it better matches our current schema and IR design

## 4. The semantic hot path we need to mirror is stable

The runtime sequence in the simulator is straightforward:

1. prepare actions
2. check action count and optionally preconditions
3. evaluate CPFs in dependency order
4. evaluate reward
5. commit next-state values to current state
6. emit observation or state
7. check termination
8. separately check invariants / truncation at the env layer

That sequence should inform our IR stage boundaries.

## Benchmark Candidates Pulled Locally

Concrete local benchmark families now available in `rddlrepository` include:

- `gym/CartPole`
- `gym/MountainCar`
- `standalone/Elevators`
- `standalone/Reservoir`
- `standalone/UAV`
- competition POMDP families such as `Traffic`, `Navigation`, `Elevators`, and `Wildfire`

Suggested first ladder:

1. `CartPole` or `MountainCar`
2. `Elevators` or `Reservoir`
3. one POMDP traffic or navigation domain

## Recommended Study Order

1. paper notes
2. `env.py`
3. `simulator.py`
4. `levels.py`
5. `model.py`
6. `grounder.py`
7. `rddlrepository/core/manager.py`

That order moves from public semantics to execution internals to benchmark access.

