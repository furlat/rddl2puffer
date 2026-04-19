# CartPole Comparison

This note compares the local RDDL CartPole benchmarks against the native Puffer Ocean CartPole implementation.

Reference files:

- RDDL discrete domain:
  [domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl)
- RDDL discrete instance:
  [instance0.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/instance0.rddl)
- RDDL continuous domain:
  [domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Continuous/domain.rddl)
- RDDL continuous instance:
  [instance0.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Continuous/instance0.rddl)
- pyRDDLGym simulator semantics:
  [simulator.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/simulator.py)
- pyRDDLGym env wrapper semantics:
  [env.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/env.py)
- Puffer native CartPole:
  [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h)
- Puffer local demo:
  [cartpole.c](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.c)
- Puffer training binding:
  [binding.c](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/binding.c)
- Puffer config:
  [cartpole.ini](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/config/cartpole.ini)

## Short Answer

Yes, we do have CartPole in RDDL locally, and it is a very good benchmark candidate.

However, the cleanest direct comparison is only:

- RDDL `CartPole_Discrete_gym`
- native Puffer `cartpole`

and even that pair is not semantically identical out of the box.

The main reasons are:

- different reset semantics
- different reward semantics on failure
- different episode-ending semantics
- and, most importantly, the native Puffer CartPole dynamics appear to differ materially from the classic CartPole equations as written

## What Exists Locally

## RDDL side

Two benchmark variants are present in `rddlrepository`:

- `CartPole_Discrete_gym`
- `CartPole_Continuous_gym`

The names come from the problem `name` plus the `gym` context in the repository metadata:

- [Discrete/__init__.py](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/__init__.py)
- [Continuous/__init__.py](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Continuous/__init__.py)

## Puffer side

Native Ocean CartPole exists here:

- [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h)
- [binding.c](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/binding.c)
- [cartpole.ini](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/config/cartpole.ini)

The binding exposes one action head of size 2, so the training path is discrete:

- `NUM_ATNS = 1`
- `ACT_SIZES {2}`

from [binding.c](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/binding.c:2).

That means the native Puffer env is directly comparable only to the discrete RDDL variant, not the continuous one.

## Structural Comparison

## Observation / state layout

Both sides expose the same four logical state variables:

- cart position
- cart velocity
- pole angle
- pole angular velocity

RDDL state fluents:

- `pos`
- `vel`
- `ang-pos`
- `ang-vel`

from [Discrete/domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:52).

Puffer observation array order:

- `x`
- `x_dot`
- `theta`
- `theta_dot`

from [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:137).

So the state content lines up conceptually, and the native observation order is easy to map to the RDDL fluents.

## Action layout

RDDL discrete CartPole uses one integer action:

- `force-side : { action-fluent, int, default = 0 }`

from [Discrete/domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:58).

RDDL continuous CartPole uses one real-valued action:

- `force : { action-fluent, real, default = 0 }`

from [Continuous/domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Continuous/domain.rddl:57).

Puffer native CartPole uses one discrete action head with two choices, represented internally as one scalar chosen from the head:

- [binding.c](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/binding.c:3)

So the direct pair is:

- RDDL discrete
- Puffer native

## Horizon

Both RDDL variants use:

- `horizon = 200`

from the instances:

- [Discrete/instance0.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/instance0.rddl:19)
- [Continuous/instance0.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Continuous/instance0.rddl:19)

Puffer native uses:

- `MAX_STEPS 200`

from [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:11).

This part matches.

## Reset Semantics

## RDDL / pyRDDLGym

Both RDDL instances define a deterministic initial state:

- `pos = 0.0`
- `vel = 0.0`
- `ang-pos = 0.1`
- `ang-vel = 0.0`

from:

- [Discrete/instance0.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/instance0.rddl:11)
- [Continuous/instance0.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Continuous/instance0.rddl:11)

pyRDDLGym reset restores the instance-defined initial values:

- [simulator.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/simulator.py:402)

## Native Puffer

Native Puffer CartPole resets all four state values randomly in `[-0.04, 0.04]`:

- [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:144)

This means:

- the two envs do not start from the same state distribution
- exact rollout parity is impossible without normalization

## Dynamics Comparison

## RDDL dynamics

The RDDL discrete and continuous domains use the standard CartPole-style equations:

- temporary force term:
  [Discrete/domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:68)
- angular acceleration:
  [Discrete/domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:70)
- cart acceleration:
  [Discrete/domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:73)
- Euler update:
  [Discrete/domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:76)

The continuous variant uses the same dynamics, just with a real action:

- [Continuous/domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Continuous/domain.rddl:61)

## Native Puffer dynamics

Native Puffer computes:

- `total_mass = cart_mass + pole_mass`
- `polemass_length = total_mass + pole_mass`
- `temp = (force + polemass_length * theta_dot^2 * sin(theta)) / total_mass`
- `thetaacc = ... (4/3 - total_mass * cos^2(theta) / total_mass)`
- `xacc = temp - polemass_length * thetaacc * cos(theta) / total_mass`

from [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:169).

As written, this differs materially from the standard CartPole equations:

- `polemass_length` should usually be `pole_mass * pole_length`, not `total_mass + pole_mass`
- the denominator term simplifies to `4/3 - cos^2(theta)`, which drops the standard `pole_mass / total_mass` factor

I may be missing an intentional reparameterization, but as written the code does not match the classic equations used by the RDDL domains.

## One-step numerical check

Using:

- `x = 0`
- `x_dot = 0`
- `theta = 0.1`
- `theta_dot = 0`
- force to the right with magnitude `10`
- default masses and `dt = 0.02`

I computed:

- RDDL / classic:
  - `x_acc ~= 9.678`
  - `theta_acc ~= -12.977`
  - `x_dot_next ~= 0.1936`
  - `theta_dot_next ~= -0.2595`
- native Puffer as written:
  - `x_acc ~= 60.105`
  - `theta_acc ~= -46.998`
  - `x_dot_next ~= 1.2021`
  - `theta_dot_next ~= -0.9400`

So the native implementation is not just a little different. It is a substantially different system numerically.

## Reward Semantics

## RDDL discrete

The discrete RDDL reward is a constant:

- `reward = 1.0`

from [Discrete/domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:83)

## RDDL continuous

The continuous RDDL reward is:

- `1` if within limits
- `0` otherwise

from [Continuous/domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Continuous/domain.rddl:78)

But pyRDDLGym evaluates reward before committing next-state values back into the current state:

- CPF evaluation
- reward evaluation
- state update
- observation update
- terminal check

from [simulator.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/simulator.py:439)

Because the continuous reward expression uses unprimed state variables, it is effectively based on the pre-transition state, not the just-computed next state.

That means both RDDL variants behave like:

- reward `1` on the final failing step

assuming the previous state was still valid.

## Native Puffer

Native Puffer sets:

- `reward = 0` if `done`
- `reward = 1` otherwise

from [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:188)

So native Puffer gives `0` on the failure step, unlike the RDDL variants.

## Episode-End Semantics

## pyRDDLGym side

The simulator returns `done` based on termination conditions after state update:

- [simulator.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/simulator.py:494)

The env wrapper then separately computes `truncated` from state invariant violations:

- [env.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/env.py:219)

Since the CartPole RDDL domains encode the same bounds in both `termination` and `state-invariants`, failure can produce:

- `terminated = True`
- `truncated = True`

at the same time.

## Native Puffer side

Native Puffer computes:

- `terminated` for threshold violations
- `truncated` for `MAX_STEPS`
- `done = terminated || truncated`

from [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:183)

It then immediately auto-resets inside `c_step` when `done` is true:

- [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:192)

So the step-return behavior is not aligned with pyRDDLGym:

- pyRDDLGym expects an explicit reset by the caller
- native Puffer CartPole resets itself immediately inside the step

## What This Means For Benchmarking

## Good news

CartPole is still a great benchmark family for us because:

- it exists on both sides locally
- it is small and readable
- there is a direct discrete-action overlap
- it is easy to inspect, visualize, and debug

## But

We should not call the comparison "native vs wrapper for the same environment" without a normalization pass.

At minimum, we would need to account for:

- deterministic vs random reset
- reward-on-failure mismatch
- auto-reset vs explicit reset
- termination vs truncation conventions
- likely dynamics mismatch in the native Puffer code

## Recommendation

Use CartPole, but split the work into two distinct comparisons.

### Comparison A: Puffer wrapper baseline

Use:

- RDDL `CartPole_Discrete_gym`
- pyRDDLGym
- Puffer Python wrapper path

Goal:

- show that pyRDDLGym-generated envs can run and train in the Puffer stack
- measure the Python-wrapper baseline

### Comparison B: Native equivalence study

Use:

- generated native CartPole from RDDL semantics
- native Puffer CartPole only as an inspiration or secondary reference

Goal:

- compare our generated native env against pyRDDLGym semantics
- not against the current Ocean CartPole unless we explicitly decide to match its behavior

### Practical conclusion

If we want a faithful "same semantics, different runtime" comparison, the correct target is:

- pyRDDLGym CartPole
- our future generated native CartPole

not:

- pyRDDLGym CartPole
- current Puffer native CartPole as-is

The current Puffer native CartPole is still useful, but more as:

- a Puffer integration example
- a training-performance reference
- a native-environment implementation template

than as a semantic gold standard for classic CartPole.

