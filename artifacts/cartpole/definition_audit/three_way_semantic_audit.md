# CartPole Three-Way Semantic Audit

This artifact compares three different things:

1. the discrete RDDL CartPole definition
2. the generated native Ocean environment
3. upstream native Puffer `cartpole`

It answers two questions:

- are we expressing the same problem?
- if not, is the mismatch due to the DSL, the generator, the Puffer backend, or the native handwritten env?

Related sources:

- RDDL definition: [rddl_definition.md](/home/tommaso/Dev/rddl2puffer/artifacts/cartpole/definition_audit/rddl_definition.md)
- Generated env: [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h), [binding.c](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/binding.c)
- Native env: [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h), [binding.c](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/binding.c)
- Generator sources: [cartpole_discrete.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/benchmarks/cartpole_discrete.py), [schema.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/frontend/schema.py), [emit_env_h.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/backends/puffer_c/emit_env_h.py)

## Executive Read

The generated env is much closer to the RDDL definition than native `Puffer` CartPole is.

Most of the large semantic mismatches are not caused by the generator. They come from native `cartpole` being a different handwritten task:

- different reset distribution
- different reward on done
- different dynamics equations
- different observation order

The strongest real generator bug was timeout handling. That has now been fixed so that horizon exhaustion causes reset without being reported as a true terminal.

The biggest genuine DSL limitation exposed by this audit is stochastic reset: the current RDDL init-state grammar used by pyRDDLGym accepts constants, not random-variable expressions.

## Difference Matrix

| Topic | RDDL definition | Generated env | Native Puffer env | Classification |
| --- | --- | --- | --- | --- |
| Dynamics equations | Classical CartPole equations from [domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:69) | Matches the RDDL equations in [cartpole_discrete.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/benchmarks/cartpole_discrete.py:98) and emitted code [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h:70) | Uses different equations with `polemass_length = total_mass + pole_mass` in [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:170) | Native hardcoded divergence. This is expressible in RDDL; it is not a DSL limitation. |
| Initial state | Deterministic `pos=0, vel=0, ang-pos=0.1, ang-vel=0` from [instance0.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/instance0.rddl:11) | Matches the instance defaults in [cartpole_discrete.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/benchmarks/cartpole_discrete.py:20) and [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h:47) | Random uniform reset in `[-0.04, 0.04]` for all state components in [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:146) | Native hardcoded divergence. Not directly expressible in the current RDDL init-state grammar, so this is the clearest DSL limitation. |
| Reward on terminating step | Constant `1.0` from [domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:84) | Matches RDDL with `reward = 1.0f` in [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h:137) | Uses `0.0` on any done step in [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:188) | Native hardcoded divergence. Fully expressible in RDDL. |
| Termination vs time-limit | `terminated` and `truncated` are distinct in pyRDDLGym: [env.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/env.py:234), [env.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/env.py:257) | Now resets on timeout but only reports `terminals[0]` for true failure in [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h:138) | Same pattern in [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:183) | Previously a generator bug, now fixed. Remaining limitation is that current static `vecenv` transports only `terminals`, not an explicit truncation channel. |
| Observation tensor order | Not specified by RDDL semantics | Alphabetical canonical layout yields `[ang-pos, ang-vel, pos, vel]` via [schema.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/frontend/schema.py:134) and emitted stores in [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h:133) | Hardcoded `[x, x_dot, theta, theta_dot]` in [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:137) | Arbitrary generator/frontend choice versus arbitrary native choice. Not a DSL limitation. |
| Action sanitation | RDDL only specifies valid action set through preconditions in [domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:105) | No explicit clamp or NaN repair in emitted code; relies on discrete action head | Explicit clamp and finite check in [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:156) | Generator runtime choice. Not a DSL limitation. |
| Logging semantics | Not part of RDDL | `score += episode_return` in [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h:38) | `score += tick` in [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:64) | Backend choice, not DSL semantics. |
| Diagnostic termination counters | Not part of RDDL | Not emitted | Native exposes `x_threshold_termination`, `pole_angle_termination`, `max_steps_termination` in [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:17) and [binding.c](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/binding.c:22) | Generator feature gap, not DSL limitation. |
| Training defaults | Not part of RDDL | Generated config in [rddl_cartpole_discrete.ini](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/config/rddl_cartpole_discrete.ini) | Native config in [cartpole.ini](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/config/cartpole.ini) | Training setup difference, not task semantics. |

## Dynamics Are Literally Different

The dynamics mismatch is not subtle. Using the same state and action, the native handwritten env produces much larger accelerations than the RDDL/generated version.

Sample 1:

- state `(x=0.0, x_dot=0.0, theta=0.1, theta_dot=0.0)`
- action `1`

Results:

- generated `x_acc = 9.677810`
- native `x_acc = 60.104792`
- generated `theta_acc = -12.976640`
- native `theta_acc = -46.997518`

Sample 2:

- state `(x=0.0, x_dot=0.2, theta=0.05, theta_dot=-0.1)`
- action `0`

Results:

- generated `x_acc = -9.790078`
- native `x_acc = -71.178953`
- generated `theta_acc = 15.401458`
- native `theta_acc = 56.985758`

Sample 3:

- state `(x=0.5, x_dot=-0.3, theta=-0.12, theta_dot=0.4)`
- action `1`

Results:

- generated `x_acc = 9.830080`
- native `x_acc = 72.483995`
- generated `theta_acc = -16.398852`
- native `theta_acc = -58.550541`

Conclusion:

- native `cartpole` is not just a tuned implementation of the RDDL/classical problem
- it is a different dynamical system

## Which Differences Are Actually Generator Choices?

These are the places where the mismatch is on us rather than in the DSL or the handwritten native env.

### 1. Flat layout ordering

We define the CartPole schema in source order in [cartpole_discrete.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/benchmarks/cartpole_discrete.py:20), but then `canonical_fluent_key` sorts fluents by role and name in [schema.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/frontend/schema.py:134). That reorders:

- source order: `pos, ang-pos, vel, ang-vel`
- generated flat order: `ang-pos, ang-vel, pos, vel`

This is not required by RDDL. It is just our current canonicalization rule.

### 2. Action robustness

The generated env does not currently emit runtime clamps or NaN handling for actions. That is a codegen/runtime choice.

### 3. Diagnostic logging

The generated env exposes only generic log fields. Native `cartpole` exposes richer failure diagnostics. That is a generator capability gap, not a semantic necessity.

### 4. Trainer/config defaults

The generator copied many hyperparameters from native `cartpole`, but not all of them. For example:

- native sets `use_rnn = 1` in [cartpole.ini](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/config/cartpole.ini:38)
- generated does not

This is not about the problem definition. It is about the benchmark harness.

## Which Differences Are Native Handwritten Choices?

These are the places where the generated env is faithful to the RDDL problem and native `cartpole` is the outlier.

### 1. Physics

The native equations diverge from both the RDDL source and the generated code. This is expressible in the DSL, so the mismatch is not forced by RDDL.

### 2. Reset distribution

Native random reset is handwritten in C. The generated env follows the deterministic instance.

### 3. Terminal-step reward

Native zeroes reward on done. RDDL does not.

### 4. Observation order

Native chooses a Gym-like observation order. RDDL does not define tensor packing order.

## Which Differences Are Real DSL Limitations?

Only a small subset.

### 1. Stochastic reset distribution

This is the biggest one. The pyRDDLGym grammar accepts literal `range_const` values in `init-state`, not `randomvar_expr`, as seen in [parser.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/parser/parser.py:1130).

That means the native reset rule:

- `x, x_dot, theta, theta_dot ~ Uniform(-0.04, 0.04)`

is not directly representable as an episode-reset rule in the current RDDL instance syntax we are using.

You could approximate it with a different modeling convention, but not express it directly as written.

### 2. Flat tensor order is outside the DSL

RDDL is relational and symbolic. It does not define the order of a flattened neural-network observation vector. That mapping has to come from the frontend/backend contract.

## Which Differences Are Backend Limitations?

### Explicit truncation channel

The semantic distinction between `terminated` and `truncated` exists on the RDDL side and in pyRDDLGym, but the current static Puffer `vecenv` path copies only `terminals` to the learner in [vecenv.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/src/vecenv.h:265) and [vecenv.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/src/vecenv.h:627).

So the backend can currently:

- reset on timeout
- avoid marking timeout as terminal

but it cannot cleanly send a separate truncation signal through the generic training interface.

## Can We Express Precisely The Same Problem?

### RDDL definition vs generated env

Yes, almost entirely. The generated env is already close to a faithful native lowering of the discrete RDDL task, and the timeout bug is fixed.

Remaining generator-side mismatches are mostly implementation choices:

- flat order
- action robustness
- diagnostics

### Native Puffer env vs RDDL definition

Not exactly.

Most of native `cartpole` could be re-expressed in RDDL:

- alternative dynamics
- alternative reward rule
- same thresholds
- same horizon

But one important part is not directly expressible in the current RDDL init-state grammar:

- stochastic reset distribution at episode start

So if we want exact native semantic parity, we either need:

- a slightly extended modeling convention beyond current RDDL init-state syntax, or
- a benchmark-alignment mode in the generator/backend that intentionally departs from pure RDDL episode reset semantics

## Practical Conclusion

The current gap between generated and native CartPole is not evidence that RDDL is too weak to describe the important dynamics.

The more precise conclusion is:

- the RDDL DSL is strong enough for the core transition and reward dynamics
- the handwritten native env intentionally or accidentally defines a different task in several important places
- the generator still has some arbitrary backend choices that we should clean up
- the biggest genuine DSL limitation surfaced so far is direct stochastic reset specification at episode start

## Recommended Next Fixes

1. Preserve source order or explicit frontend order metadata instead of alphabetically canonicalizing state and observation tensors.
2. Add optional runtime action sanitation generated from action bounds.
3. Add native-style diagnostic counters to generated logs.
4. Decide whether we want:
   - pure RDDL fidelity mode
   - native-benchmark-aligned mode
5. If exact native parity matters, design an explicit strategy for reset distributions that current RDDL init-state syntax cannot encode directly.
