# RDDL CartPole Definition

This artifact restates the exact discrete CartPole problem we are currently targeting from the RDDL side.

Primary sources:

- Domain: [domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl)
- Instance: [instance0.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/instance0.rddl)
- Reference runtime semantics: [env.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/env.py), [simulator.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/simulator.py)

## Problem Summary

The RDDL instance defines a single-agent MDP with:

- deterministic continuous state
- one discrete binary action `force-side in {0, 1}`
- deterministic reward `1.0` on every transition
- termination when the cart position or pole angle leaves the allowed range
- time-limit truncation at horizon `200`
- deterministic initial state

## Constants

From [domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:33):

- `GRAVITY = 9.8`
- `FORCE-MAG = 10.0`
- `CART-MASS = 1.0`
- `POLE-MASS = 0.1`
- `POLE-LEN = 0.5`
- `TIME-STEP = 0.02`
- `POS-LIMIT = 2.4`
- `ANG-LIMIT = 0.2094395`

## State And Action Variables

From [domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:53):

- state `pos`
- state `ang-pos`
- state `vel`
- state `ang-vel`
- action `force-side`

Action bounds are expressed as preconditions in [domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:105):

- `force-side >= 0`
- `force-side <= 1`

## Initial State

From [instance0.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/instance0.rddl:11):

- `pos = 0.0`
- `vel = 0.0`
- `ang-pos = 0.1`
- `ang-vel = 0.0`

This is deterministic. In the current pyRDDLGym parser, `init-state` entries are restricted to `range_const` literals rather than random-variable expressions, as seen in [parser.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/parser/parser.py:1066) and [parser.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/parser/parser.py:1130).

## Dynamics

From [domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:64):

1. Signed force:

```text
force = +FORCE-MAG if force-side == 1 else -FORCE-MAG
```

2. Temporary term:

```text
temp = (force + POLE-LEN * POLE-MASS * ang-vel^2 * sin(ang-pos))
       / (CART-MASS + POLE-MASS)
```

3. Angular acceleration:

```text
ang-acc =
    (GRAVITY * sin(ang-pos) - cos(ang-pos) * temp)
    / (POLE-LEN * (4/3 - POLE-MASS * cos(ang-pos)^2 / (CART-MASS + POLE-MASS)))
```

4. Cart acceleration:

```text
acc = temp
      - (POLE-LEN * POLE-MASS * ang-acc * cos(ang-pos) / (CART-MASS + POLE-MASS))
```

5. Euler integration:

```text
pos'     = pos     + TIME-STEP * vel
ang-pos' = ang-pos + TIME-STEP * ang-vel
vel'     = vel     + TIME-STEP * acc
ang-vel' = ang-vel + TIME-STEP * ang-acc
```

## Reward

From [domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:84):

- `reward = 1.0`

Under pyRDDLGym, reward is evaluated before the state update and terminal check, but because it is a constant `1.0`, the result is still one reward unit for every transition, including the final terminating transition. See [simulator.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/simulator.py:455) and [simulator.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/simulator.py:498).

## Episode End Semantics

True termination conditions come from [domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/domain.rddl:86):

- `pos < -POS-LIMIT or pos > POS-LIMIT`
- `ang-pos < -ANG-LIMIT or ang-pos > ANG-LIMIT`

Time-limit truncation comes from [instance0.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/gym/CartPole/Discrete/instance0.rddl:19):

- `horizon = 200`

pyRDDLGym surfaces these separately:

- `terminated` from simulator terminal conditions: [env.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/env.py:234)
- `truncated` from invariant failure or horizon exhaustion: [env.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/env.py:242), [env.py](/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym/pyRDDLGym/core/env.py:257)

So the semantic intent of the RDDL task is:

- termination and time-limit are distinct
- the horizon is not itself a failure condition

## What The RDDL Definition Does Not Specify

The RDDL source specifies the control problem semantics, but not several backend details:

- flat observation tensor order
- flat state tensor order
- runtime logging fields such as `score` or `perf`
- auto-reset implementation details
- action sanitization and NaN handling in C

Those are backend or generator choices, not part of the domain definition itself.
