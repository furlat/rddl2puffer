# Pong Audit: Raw RDDL vs Native Puffer

## Scope

This note compares:

- raw Pong RDDL:
  - [domain.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/domain.rddl)
  - [instance0.rddl](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/instance0.rddl)
- native handwritten Puffer Pong:
  - [pong.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h)
  - [binding.c](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/binding.c)
  - [pong.ini](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/config/pong.ini)

The point is not to ask whether raw benchmark Pong is "good." The point is to separate:

- source-level task differences
- codegen/compiler bugs
- current compiler-subset limits
- true `RDDL+` language gaps

## Executive Summary

Raw RDDL Pong and native Puffer Pong are very different tasks.

The biggest differences are not codegen choices:

- raw RDDL Pong is a one-paddle dense-control benchmark
- native Puffer Pong is a two-paddle score-based arcade game

That matters because poor native-vs-raw training parity is expected even if the compiler is correct.

The current strongest sanity check is:

- compiled raw Pong source semantics matched `pyRDDLGym` exactly on a seeded rollout after fixing the stochastic test harness

So the main gap right now is not "the compiler is obviously wrong." It is that raw benchmark Pong is far from native Puffer Pong semantically.

## Difference Matrix

| Area | Raw RDDL Pong | Native Puffer Pong | Classification | Parity Path |
|---|---|---|---|---|
| Task shape | One controllable paddle and one ball | Two paddles, left opponent and right ego, one ball, score state | Source/task difference | Write a native-parity Pong source |
| Geometry | Unit square `[0, 1]` coordinates | Pixel court with width/height and object sizes | Source/task difference | Encode native geometry in source |
| Observation size | 5 values | 8 values | Source/task difference | Extend source spec to native obs schema |
| Action semantics | Integer `move in {-1, 0, 1}` scaled by `PADDLE-MAX-STEP` | Discrete action head `{still, up, down}` with `paddle_speed`; optional continuous mode | Source/task difference | Encode native action contract |
| Opponent | None | Left paddle tracks the ball | Source/task difference | Add opponent state/dynamics to parity spec |
| Reward | Dense `-(sum ball-x)` | Sparse `+1/-1` on point win/loss | Source/task difference | Replace reward with native scoring reward |
| Episode structure | Horizon `300`, no score state, no round reset | Round resets on miss, terminal at `max_score == 21` | Source/task difference | Add score and round lifecycle |
| Contact randomness | `Uniform(-NOISE-Y, NOISE-Y)` on paddle contact | No contact noise; deterministic velocity adjustment | Source/task difference | Match native contact semantics |
| Reset randomness | Deterministic defaults from pvariable defaults | Random initial vertical ball velocity sign on reset | True language gap today | Add stochastic `init-state` in `RDDL+` |
| Frameskip | One physics update per RL step | `frameskip` inner loop, default `8` | Source/task difference | Encode repeated substeps / macro-step semantics |
| Logging meaning | Reward-based dense control objective | Score / perf / episode logs tied to arcade scoring | Native runtime convention | Match native logging metadata |
| `NOISE-X` | Declared but unused | N/A | Source quirk | Ignore or remove in parity spec |

## Detailed Findings

### 1. High-Level Task Is Different

Raw RDDL Pong is not a direct textual version of native Puffer Pong.

Raw RDDL defines:

- one ball object type, with `instance0` containing only `{b1}` in [instance0.rddl:4-6](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/instance0.rddl:4)
- one controllable paddle state `paddle-y` in [domain.rddl:33](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/domain.rddl:33)
- no opponent paddle state
- no score state

Native Puffer Pong keeps:

- left and right paddles `paddle_yl`, `paddle_yr` in [pong.h:25-26](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:25)
- score state `score_l`, `score_r` in [pong.h:31-32](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:31)
- round counters and runtime state like `tick`, `n_bounces`, `win`, `frameskip` in [pong.h:48-52](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:48)

This is the biggest reason raw-vs-native training curves diverge.

### 2. Geometry and Units Are Different

Raw RDDL uses normalized coordinates:

- ball and paddle positions live in `[0, 1]`
- paddle size is a fraction `PADDLE-H = 0.2` in [domain.rddl:26](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/domain.rddl:26)
- paddle step is `PADDLE-MAX-STEP = 0.04` in [domain.rddl:27](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/domain.rddl:27)

Native Puffer uses pixel-space geometry loaded from config:

- width `500`, height `640`
- paddle width `20`, paddle height `70`
- ball width `32`, ball height `32`
- paddle speed `8`
- frameskip `8`

see [pong.ini](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/config/pong.ini) and the corresponding struct fields in [pong.h:33-52](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:33)

This difference is source-level, not a compiler bug.

### 3. Observation Schema Is Different

Raw RDDL state/observation semantics for `instance0` are effectively:

- `ball-x[b1]`
- `ball-y[b1]`
- `vel-x[b1]`
- `vel-y[b1]`
- `paddle-y`

Native Puffer writes 8 observations in [pong.h:96-104](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:96):

- left paddle y
- right paddle y
- ball x
- ball y
- normalized ball vx
- normalized ball vy
- left score
- right score

So raw RDDL Pong is missing:

- opponent paddle observation
- score observations
- native velocity normalization contract

### 4. Action Contract Is Similar in Spirit but Not the Same Contract

Raw RDDL uses an integer action:

- `move : { action-fluent, int, default = 0 }` in [domain.rddl:41](/home/tommaso/Dev/rddlrepository/rddlrepository/archive/arcade/Pong/domain.rddl:41)
- constrained by `move >= -1 ^ move <= 1` in [domain.rddl:91-93](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/domain.rddl:91)
- applied as `paddle-y' = max[min[paddle-y + move * PADDLE-MAX-STEP, 1.0 - PADDLE-H], 0.0]` in [domain.rddl:78](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/domain.rddl:78)

Native Puffer uses:

- one discrete action head of size `3` in [binding.c:4](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/binding.c:4)
- action interpretation `{still, up, down}` in [pong.h:133-141](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:133)
- optional continuous mode in [pong.h:130-132](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:130)

This is still source-level, not a language limit. Standard RDDL can express a 3-way discrete paddle action just fine.

### 5. Opponent Dynamics Exist Only in Native Puffer

Raw RDDL has no opponent.

Native Puffer updates the left paddle by chasing the ball each substep in [pong.h:147-150](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:147):

- `opp_paddle_delta = ball_y - (paddle_yl + paddle_height / 2)`
- then clamp by `paddle_speed`
- then move `paddle_yl`

This is a core gameplay difference and would need to be added to any parity source.

### 6. Reward Semantics Are Totally Different

Raw RDDL reward is dense:

- `reward = -(sum_{?b : ball} ball-x(?b));` in [domain.rddl:81](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/domain.rddl:81)

For `instance0`, that is just `-ball-x[b1]`.

Native Puffer reward is sparse and point-based:

- `+1` when the agent scores on the left side in [pong.h:175-179](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:175)
- `-1` when the agent misses on the right side in [pong.h:206-210](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:206)
- otherwise `0`

This is one of the biggest training-behavior differences.

### 7. Ball Dynamics Are Different Enough To Be A Different Game

Raw RDDL:

- predicts `new-x`, `new-y` first in [domain.rddl:47-48](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/domain.rddl:47)
- computes a projected paddle crossing `ball-crossing-y-raw` in [domain.rddl:51](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/domain.rddl:51)
- reflects on paddle contact or left wall bounce in [domain.rddl:64-75](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/domain.rddl:64)

Native Puffer:

- performs direct pixel-space integration every inner frameskip step in [pong.h:158-160](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:158)
- uses rectangle overlap checks for both paddles in [pong.h:169-170](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:169) and [pong.h:194-195](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:194)
- changes ball vertical speed based on agent paddle motion in [pong.h:199-203](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:199)

This is not a compiler issue. It is a different handcrafted environment design.

### 8. Stochasticity Is Different

Raw RDDL introduces stochasticity on paddle contact:

- `vel-y'(?b) = ... vel-y(?b) + Uniform(-NOISE-Y(?b), NOISE-Y(?b)) ...` in [domain.rddl:73](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/domain.rddl:73)

It also defines `NOISE-X(ball)` in [domain.rddl:24](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/domain.rddl:24), but raw Pong never uses it. That is why the generated C currently emits an unused constant warning.

Native Puffer stochasticity is instead in round reset:

- `ball_vy = (rand_r(&env->rng) % 2 - 1) * ball_initial_speed_y;` in [pong.h:113](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:113)

Native contact dynamics are deterministic apart from paddle-direction-based velocity adjustment.

This is a real place where `RDDL+` likely helps:

- episode-start stochastic reset is still a genuine gap in standard current `init-state`

### 9. Episode and Round Lifecycle Are Different

Raw RDDL Pong:

- uses `horizon = 300` in [instance0.rddl:13](/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository/rddlrepository/archive/arcade/Pong/instance0.rddl:13)
- has no score state
- has no terminal definition
- effectively runs one long dense-reward episode to horizon

Native Puffer Pong:

- has per-point round resets in [pong.h:107-116](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:107)
- calls `reset_round(env)` after a point unless the match is over in [pong.h:185-187](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:185) and [pong.h:215-217](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:215)
- terminates only when `score_r == max_score` or `score_l == max_score` in [pong.h:179-183](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:179) and [pong.h:210-214](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:210)

This is again a source semantics gap, not a codegen quirk.

### 10. Frameskip Matters A Lot

Raw RDDL Pong does one update per RL step.

Native Puffer wraps the whole physics update in:

- `for (int i = 0; i < env->frameskip; i++) { ... }` in [pong.h:144](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:144)

With the default config:

- `frameskip = 8` in [pong.ini:22](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/config/pong.ini:22)

This changes both dynamics and effective control frequency. It is fully expressible in principle, but the parity spec would need to encode macro-step semantics deliberately.

### 11. Logging Semantics Are Native-Specific

Native Puffer logs score/perf in terms of match results:

- `score = score_r - score_l` in [pong.h:88](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:88)
- `perf = score_r / (score_l + score_r)` in [pong.h:92](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:92)

Raw RDDL-generated Pong logs reward-shaped episode return, so direct score/perf comparisons are misleading unless we explicitly align the source.

## What Is A Compiler Bug vs What Is A Source Difference?

### Not A Compiler Bug

These differences are expected even if the compiler is perfect:

- one paddle vs two paddles
- dense reward vs sparse point reward
- no score state vs score-based rounds
- normalized box vs pixel court
- no opponent vs opponent paddle controller
- no frameskip vs frameskip `8`
- contact noise vs reset randomness
- 5-dim obs vs 8-dim obs

### Was A Real Harness Bug

We did have a real testing problem:

- the stochastic compiled reference runtime was not preserving a seeded RNG correctly
- the pyRDDLGym adapter was not mapping parameterized fluent names like `ball-x[b1]` to pyRDDLGym grounded keys like `ball-x___b1`

After fixing those, compiled raw Pong matched `pyRDDLGym` exactly on a seeded rollout.

That makes it much less likely that raw Pong’s training gap is caused by an obvious source-compiler mismatch.

## What Is Expressible In Standard RDDL vs What Likely Needs `RDDL+`?

### Expressible In Standard RDDL

The following native Pong concepts are expressible in principle:

- two paddles
- score state
- sparse `+1/-1` point reward
- `max_score` terminal
- opponent paddle chase logic
- pixel coordinates
- frameskip semantics via explicit macro-step modeling
- point-triggered round reset logic inside CPFs

### Likely Needs `RDDL+`

The clearest remaining real gap is:

- stochastic episode-start reset semantics

Native Puffer Pong randomizes initial vertical ball direction in [pong.h:113](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:113), and that is exactly the kind of thing we already identified as needing stochastic `init-state` / `RDDL+`.

## Recommended Next Move

Do not try to compare raw benchmark Pong to native Puffer Pong as if they are supposed to match.

Instead:

1. keep raw Pong as the "honest original RDDL benchmark" anchor
2. write a `pong_puffer_parity` source that matches native semantics
3. classify every missing piece as:
   - source rewrite
   - compiler subset expansion
   - true `RDDL+` extension
4. compare:
   - raw RDDL Pong
   - native-parity generated Pong
   - native handwritten Puffer Pong

That will give a meaningful parity story, whereas raw benchmark Pong vs native Pong mostly tells us that they are different environments.
