# Pong Puffer Parity Progress

## Current Source Path

The current path is now a real source-driven path:

- [domain.rddl](/home/tommaso/Dev/rddl2puffer/examples/rddl_plus/pong_puffer_parity/domain.rddl)
- [instance_deterministic_reset.rddl](/home/tommaso/Dev/rddl2puffer/examples/rddl_plus/pong_puffer_parity/instance_deterministic_reset.rddl)

These compile directly through:

- [compile.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/frontend/compile.py)
- [ground.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/frontend/ground.py)
- [parser.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/rddl_plus/parser/parser.py)

and emit native Ocean C into:

- [rddl_pong_parity.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_pong_parity/rddl_pong_parity.h)
- [binding.c](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_pong_parity/binding.c)
- [rddl_pong_parity.ini](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/config/rddl_pong_parity.ini)

## What Is Matching

- Native Pong geometry/config is encoded directly in the RDDL source.
- Two paddles, score state, sparse `+1/-1/0` point reward, and match termination at `MAX-SCORE`.
- Eight native-style normalized observations.
- Eight substep frameskip unrolled inside the source.
- Deterministic round reset after each point.
- Generated env builds in `PufferTank` with `bash build.sh rddl_pong_parity`.
- Generated env trains with real `puffer train`.
- Generated env now resets its round-local `tick` on non-terminal scoring steps through the source-driven directive:
  - `//% runtime_semantics.reset_tick_on = "$reward_nonzero"`
- Generated env now computes native-style perf from source-driven logging metadata:
  - `//% puffer_logging.perf_mode = "state_ratio"`
  - `//% puffer_logging.perf_numerator_state = "score-r"`
  - `//% puffer_logging.perf_denominator_states = "score-l, score-r"`

## Parser / RDDL+ Extension Added

To keep reset observations inside the source file, the local parser fork now accepts defaults on observation fluents:

- [parser.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/rddl_plus/parser/parser.py)

This is used so the generated env can reset to the correct initial observation vector without a backend-only hack.

## Current 1M-Step Smoke Comparison

Native `pong` reference log:

- [1776606583567.json](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/logs_pong_native_apr19_short/pong/1776606583567.json)

Generated `rddl_pong_parity` log:

- [1776611162282.json](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/logs_pong_parity_smoke3_apr19/rddl_pong_parity/1776611162282.json)

Updated generated `rddl_pong_parity` log after native-style perf logging:

- [1776611337954.json](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/logs_pong_parity_smoke4_apr19/rddl_pong_parity/1776611337954.json)

Final values at about `983,040` agent steps:

- Native `pong`
  - `SPS = 3,995,477`
  - `episode_return = -13.6621`
  - `episode_length = 51.4643`
  - `perf = 0.252369`
- Generated `rddl_pong_parity`
  - `SPS = 3,519,701`
  - `episode_return = -16.3834`
  - `episode_length = 52.2016`
  - `perf = 0.175038`

## Remaining Differences

### Accepted miss

- stochastic reset sign of `ball_vy`

### Remaining non-init mismatch

- the generated env still trails native on both `score` and `perf` in the 1M-step smoke
- generated SPS is still about `11.9%` lower than native in this short run

### Likely next target

- run longer native-vs-generated training curves on the same config
- inspect remaining semantic differences in point dynamics, not just logging
