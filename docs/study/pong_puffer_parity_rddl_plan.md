# Pong Native-Parity In Plain RDDL

## Goal

Define a `pong_puffer_parity` source that matches the handwritten native
`Puffer` Pong implementation as closely as possible using plain RDDL, with only
one intentional miss:

- stochastic reset at episode/round start

Everything else should be treated as source work, not as an excuse to invent
backend-only semantics.

## Target Truth

The target semantics are the handwritten environment:

- [pong.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h)
- [binding.c](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/binding.c)
- [pong.ini](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/config/pong.ini)

## What We Can Match In Plain RDDL

These native Pong features are expressible without extending the language.

### State

Use explicit state fluents for all gameplay state:

- `paddle-yl`
- `paddle-yr`
- `ball-x`
- `ball-y`
- `ball-vx`
- `ball-vy`
- `score-l`
- `score-r`

Optional runtime/debug state if we want closer parity:

- `tick`
- `n-bounces`
- `win`

These are not required for gameplay parity, but they may help match logging.

### Actions

Use a single discrete action fluent:

- `move : { action-fluent, int, default = 0 }`

with preconditions:

- `move >= 0`
- `move <= 2`

Interpretation:

- `0 = still`
- `1 = up`
- `2 = down`

Then define an intermediate:

- `paddle-dir = if (move == 1) then 1 else if (move == 2) then -1 else 0`

This matches [pong.h:133-141](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:133).

### Geometry

All native geometry/config values can be non-fluents:

- `WIDTH`
- `HEIGHT`
- `PADDLE-WIDTH`
- `PADDLE-HEIGHT`
- `BALL-WIDTH`
- `BALL-HEIGHT`
- `PADDLE-SPEED`
- `BALL-INITIAL-SPEED-X`
- `BALL-INITIAL-SPEED-Y`
- `BALL-MAX-SPEED-Y`
- `BALL-SPEED-Y-INCREMENT`
- `MAX-SCORE`
- `FRAMESKIP`

### Opponent Policy

Native left-paddle tracking is expressible directly:

- `opp-paddle-delta = ball-y - (paddle-yl + PADDLE-HEIGHT / 2)`
- then clamp to `[-PADDLE-SPEED, +PADDLE-SPEED]`
- then update `paddle-yl`

This is just arithmetic plus `min/max`.

### Frameskip

Native Pong uses a `for` loop over `frameskip` in
[pong.h:144](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:144).

We can encode this in plain RDDL by unrolling a fixed number of substeps with
intermediate fluents:

- substep 1:
  - `paddle-yr-1`
  - `paddle-yl-1`
  - `ball-x-1`
  - `ball-y-1`
  - `ball-vx-1`
  - `ball-vy-1`
- ...
- substep 8:
  - `paddle-yr-8`
  - `paddle-yl-8`
  - `ball-x-8`
  - `ball-y-8`
  - `ball-vx-8`
  - `ball-vy-8`

Then the primed state becomes the substep-8 state unless a point-scoring round
reset happens first.

This is ugly, but it is still plain RDDL and it is honest.

### Ball / Paddle Collision

Native paddle collision is rectangle overlap logic:

- left side check in [pong.h:167-178](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:167)
- right side check in [pong.h:192-204](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:192)

These conditions are expressible with booleans and comparisons:

- left paddle contact
- right paddle contact
- top/bottom wall bounce
- left score event
- right score event

### Score and Round Reset

Native Pong resets the round on any score, and only terminates when one side
hits `MAX-SCORE`.

That can be encoded in state CPFs:

- `score-r' = if (left-miss) then score-r + 1 else score-r`
- `score-l' = if (right-miss) then score-l + 1 else score-l`

Then branch all state updates on whether a point was scored:

- if point scored:
  - paddles reset to center
  - ball resets to initial location
  - ball-vx resets to `BALL-INITIAL-SPEED-X`
  - ball-vy resets to deterministic surrogate
- else:
  - use substep-8 state

### Reward

Native reward is:

- `+1` when the agent scores
- `-1` when the agent loses the point
- `0` otherwise

That is trivial in plain RDDL:

```text
reward =
    if left-miss then 1.0
    else if right-miss then -1.0
    else 0.0;
```

### Termination

Native match termination is:

- `score-r == MAX-SCORE` or
- `score-l == MAX-SCORE`

This is straightforward in `termination`.

### Observations

Native Pong exposes 8 normalized observations in
[pong.h:96-104](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:96).

We can match this using observation fluents:

- `obs-paddle-yl`
- `obs-paddle-yr`
- `obs-ball-x`
- `obs-ball-y`
- `obs-ball-vx`
- `obs-ball-vy`
- `obs-score-l`
- `obs-score-r`

Each observation fluent is just a normalized transform of state.

## The One Acceptable Miss

The only allowed miss is stochastic reset sign for `ball-vy`.

Native reset uses:

- [pong.h:113](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:113)

```c
env->ball_vy = (rand_r(&env->rng) % 2 - 1) * env->ball_initial_speed_y;
```

If we stay in plain current RDDL, the deterministic surrogate should be:

- `ball-vy' = BALL-INITIAL-SPEED-Y`

or a fixed negative version if we want to test the opposite branch.

Everything else should still be matched.

## Recommended Plain-RDDL Design

### Domain Shape

Use a source like:

- `domain pong_puffer_parity { ... }`

with:

- all geometry/config as non-fluents
- explicit state fluents for game state and scores
- a single integer action `move`
- many intermediate substep fluents
- observation fluents for the 8 native observation channels

### Instance Shape

Deterministic reset instance:

- paddles centered
- ball at native reset location
- `ball-vx = BALL-INITIAL-SPEED-X`
- `ball-vy = BALL-INITIAL-SPEED-Y`
- `score-l = 0`
- `score-r = 0`
- horizon large enough to mimic a whole match

## Compiler/Backend Work Needed

This is not mostly a language problem. It is mostly source authoring plus
compiler throughput work.

What we need on our side:

- keep source-order preservation
- observation fluents lowered cleanly
- large intermediate chains emitted efficiently
- ideally common-subexpression elimination across repeated substep logic

What we do **not** need here:

- new math primitives beyond the current subset
- special backend-only gameplay semantics

## Immediate Implementation Sequence

1. create `examples/rddl_plus/pong_puffer_parity/domain.rddl`
2. create a deterministic-reset instance first
3. compile it directly with the current source path
4. differential-check compiled source semantics against a Python oracle
5. emit native C and train it
6. compare against handwritten native Pong
7. only after that, add stochastic reset as `RDDL+`

## Bottom Line

Plain RDDL can already express almost all of native Puffer Pong.

The main work is not language invention. It is:

- writing the right source
- unrolling frameskip honestly
- encoding round-reset/score semantics explicitly

The only major miss we should tolerate for the first native-parity Pong source
is the reset-time random sign of `ball-vy`.
