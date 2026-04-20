# Pong Minibatch Code-Path Audit

## Objective

Audit native handwritten `PufferLib` Pong against generated `rddl_pong_parity` with one narrow question:

- what differences in the code path could plausibly explain the matched-run SPS deltas, especially the slow generated runs under large minibatches?

This is **not** a semantic-audit note in general. It is specifically a minibatch-sensitive performance audit.

Primary experiment context:

- matched native autosweep -> generated 1:1 replay report:
  [/home/tommaso/Dev/rddl2puffer/artifacts/experiments/pong_generated_replay_full_apr20/report](/home/tommaso/Dev/rddl2puffer/artifacts/experiments/pong_generated_replay_full_apr20/report)
- most useful diagnostic plots:
  - [sps_delta_distribution.svg](/home/tommaso/Dev/rddl2puffer/artifacts/experiments/pong_generated_replay_full_apr20/report/sps_delta_distribution.svg)
  - [sps_delta_vs_minibatch_size.svg](/home/tommaso/Dev/rddl2puffer/artifacts/experiments/pong_generated_replay_full_apr20/report/sps_delta_vs_minibatch_size.svg)
  - [sps_delta_vs_batch_size.svg](/home/tommaso/Dev/rddl2puffer/artifacts/experiments/pong_generated_replay_full_apr20/report/sps_delta_vs_batch_size.svg)
  - [paired_runs.csv](/home/tommaso/Dev/rddl2puffer/artifacts/experiments/pong_generated_replay_full_apr20/report/paired_runs.csv)

## High-Level Read

The matched data says:

- generated is usually a bit slower in SPS, but not catastrophically so on most runs
- the worst SPS losses cluster much more strongly with **large minibatch size** than with raw model size
- the median SPS delta is modest, but there are a few ugly outliers

From the matched report:

- mean `sps_delta`: `-120,227`
- median `sps_delta`: `-39,272.5`
- `18/24` runs slower, `6/24` faster
- strongest simple correlation in this batch:
  - `sps_delta` vs `minibatch_size`: `-0.477`
  - `sps_delta` vs `batch_size`: `-0.006`
  - `sps_delta` vs `model_size_proxy`: `+0.114`

Interpretation:

- the current slowdown signal is more “large minibatch regime hurts generated” than “bigger model hurts generated”
- but the code audit below suggests that the minibatch effect is probably **not** coming from a single minibatch-specific branch inside the generated env itself

## What Is Actually the Same

The core env/binding surface is already parity-aligned in the places most likely to affect trainer memory layout:

- both envs expose `OBS_SIZE 8`
- both envs expose `NUM_ATNS 1`
- both envs expose discrete action sizes `{3}`
- both use `FloatTensor` observations in the binding

Native binding:

- [third_party/pufferlib/ocean/pong/binding.c:2](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/binding.c:2)

Generated binding:

- [third_party/pufferlib/ocean/rddl_pong_parity/binding.c:2](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_pong_parity/binding.c:2)

This matters because it rules out the easiest explanation:

- this is **not** a case where generated Pong secretly has a bigger observation tensor, a different action-head layout, or a different dtype that would automatically blow up minibatch cost

## Where Minibatch Size Actually Bites in Puffer

The real minibatch-sensitive path is in the trainer, not the env step loop.

Important trainer code:

- [third_party/pufferlib/src/pufferlib.cu:1312](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/src/pufferlib.cu:1312)
- [third_party/pufferlib/src/pufferlib.cu:1333](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/src/pufferlib.cu:1333)
- [third_party/pufferlib/src/pufferlib.cu:1516](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/src/pufferlib.cu:1516)
- [third_party/pufferlib/src/pufferlib.cu:1576](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/src/pufferlib.cu:1576)

What scales directly with minibatch size:

- `minibatch_segments = minibatch_size / horizon`
- `B_TT = minibatch_segments * horizon`
- train activations and PPO buffers are registered with shapes derived from `minibatch_segments`
- every train minibatch does:
  - priority replay sampling
  - `select_copy`
  - optional RNN state zeroing
  - policy forward/backward
  - PPO loss forward/backward

So, if we see large-minibatch slowdown, the most direct place it can appear is:

- `perf/train_misc`
- `perf/train_forward`

That is the first reason not to assume “big SPS loss == env kernel problem”.

## Native vs Generated Env Code Shape

### Native Pong

The handwritten env is compact and loop-structured:

- precomputed paddle bounds in `init`: [pong.h:56](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:56)
- shared observation helper: [pong.h:96](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:96)
- `reset_round` with random `ball_vy` sign: [pong.h:107](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:107)
- `c_step` with explicit `for (i < frameskip)`: [pong.h:125](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:125)

Important structural traits:

- one compact inner loop
- helper reuse for observation writes
- compact branch structure
- fewer scalar temporaries live at once

### Generated Pong

The generated env is semantically faithful enough for parity experiments, but code-shape-wise it is still much more compiler-y:

- reset state and obs are fully inlined: [rddl_pong_parity.h:56](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_pong_parity/rddl_pong_parity.h:56)
- action sanitization is inlined: [rddl_pong_parity.h:97](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_pong_parity/rddl_pong_parity.h:97)
- the whole step body expands into a very large SSA-style block with hundreds of temporaries starting at [rddl_pong_parity.h:102](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_pong_parity/rddl_pong_parity.h:102)

Important structural traits:

- full frameskip logic is effectively unrolled
- repeated comparisons and select chains are materialized as many scalar temps
- no compact loop structure comparable to native
- much larger function body, likely worse i-cache and branch-prediction behavior on CPU

This is the strongest env-side performance suspect we currently have.

### Why This Likely Matters

Even though minibatch size itself does not change `c_step`, the autoswept large-minibatch regimes often also use larger rollout workloads:

- more agents
- larger horizon
- larger batch size

That means a slightly slower env step can show up more clearly in the same bad regimes where trainer cost is already high.

So the generated env code shape is probably a **secondary amplifier**, not necessarily the whole story.

## Concrete Outlier Reads

### Pure SPS loss without reward loss

Matched run `1776676451119`:

- config:
  - `total_agents=4096`
  - `horizon=64`
  - `batch_size=262144`
  - `minibatch_size=65536`
  - `hidden_size=64`
  - `num_layers=2`
- final reward/perf are basically identical
- but `sps_delta = -1,019,116`

Timing breakdown from the logs:

- native `perf/eval_env = 0.0118`
- generated `perf/eval_env = 0.0191`
- native `perf/train_forward = 0.0760`
- generated `perf/train_forward = 0.0917`

This is important:

- env step is slower
- trainer forward is also slower
- reward is not meaningfully different

So this is **not** “generated learns differently therefore it runs longer”.
It is a real code-path slowdown affecting both rollout and training time.

### Semantic failure without big SPS failure

Matched run `1776676727413`:

- config:
  - `total_agents=1024`
  - `horizon=128`
  - `batch_size=131072`
  - `minibatch_size=32768`
  - `hidden_size=64`
  - `num_layers=1`
- `episode_return_delta = -34.47`
- `perf_delta = -0.743`
- `sps_delta = -8,711`

This one says:

- there are still some semantic/training divergences that are **not** throughput divergences
- we should not collapse everything into one “performance bug”

### Generated can also win semantically while still losing some SPS

Matched run `1776676480992`:

- generated beats native strongly on reward/perf
- but still loses some SPS

This reinforces the same point:

- the throughput issue and the semantic issue are related but not identical

## What Is Ruled Out

### Not an obs/action tensor mismatch

The binding surfaces match on:

- observation width
- number of action heads
- discrete action cardinality
- observation dtype

So the large-minibatch slowdown is not explained by a larger generated tensor interface.

### Not an independent config mismatch

These runs were not separate sweeps.

They are matched replays of native-picked configs, using the script pipeline:

- [run_pong_native_autosweep.py](/home/tommaso/Dev/rddl2puffer/scripts/experiments/run_pong_native_autosweep.py)
- [replay_pong_generated_from_native.py](/home/tommaso/Dev/rddl2puffer/scripts/experiments/replay_pong_generated_from_native.py)

So the remaining differences are no longer “native used RNN and generated didn’t” style mistakes.

### Not purely an env-step issue

For the worst SPS outliers, the slowdown is not isolated to `perf/eval_env`.

We also see slower:

- `perf/eval_gpu`
- `perf/train_forward`

That means we should not waste time optimizing only the CPU `c_step` while ignoring the trainer-side path.

## What Still Differs Semantically

One semantic difference remains obvious in the code:

- native `reset_round` randomizes the sign of `ball_vy`: [pong.h:113](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/pong/pong.h:113)
- generated reset uses deterministic `ball_vy = 1.0f`: [rddl_pong_parity.h:64](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_pong_parity/rddl_pong_parity.h:64)

This is expected under the current project constraint that stochastic init/reset is the only acceptable miss for now.

This difference matters for learning dynamics.
It should **not** be treated as the main explanation for large SPS losses.

## Most Plausible Current Suspects

### Suspect 1: generated rollout code shape

The generated `c_step` is still a huge unrolled SSA kernel rather than a compact looped kernel.

Most plausible consequence:

- worse CPU instruction-cache behavior
- worse branch locality
- more scalar live range pressure
- more repeated observation/state update logic than native

This maps most directly to:

- slower `perf/eval_env`

### Suspect 2: trainer-side sensitivity amplified by generated trajectories

The trainer path that scales with minibatch size is identical in shape between envs, but it runs on env-produced observations, returns, terminals, and value targets.

Most plausible consequence:

- generated trajectories cause somewhat different rollout statistics
- same-shape train kernels still do the same operations, but some runs spend more time in the train path because the rollout path and capture warmup interact differently over the run

This maps most directly to:

- slower `perf/train_forward`
- slower `perf/train_misc`

This is not yet proven, but the worst outlier timings point this way.

### Suspect 3: large-minibatch regimes are stressing both paths at once

The matched batch suggests the ugliest SPS deltas live in:

- `minibatch_size = 65536`
- some `minibatch_size = 8192` and `32768` regimes

Grouped means from the matched table:

- `4096`: mean `sps_delta ≈ -36.7k`
- `8192`: mean `sps_delta ≈ -162.9k`
- `16384`: mean `sps_delta ≈ -11.0k`
- `32768`: mean `sps_delta ≈ -76.0k`
- `65536`: mean `sps_delta ≈ -688.8k`

So the current high-priority hotspot is the `65536` minibatch cohort.

## Recommended Next Pass

### 1. Keep the audit centered on the bad cohort

Use the matched table to isolate:

- `minibatch_size = 65536`
- then `32768`

Do not optimize against all 24 runs at once.

### 2. Attack the generated rollout kernel structure

Most concrete generator-side optimization target:

- stop emitting a giant fully-unrolled SSA block for Pong frameskip logic
- preserve loop structure when the source problem naturally has repeated substeps

This is the most promising env-side change because it directly targets the obvious native/generated code-shape gap.

### 3. Add deeper timing instrumentation for the bad cohort

For a few worst matched runs, collect:

- `perf/eval_gpu`
- `perf/eval_env`
- `perf/train_misc`
- `perf/train_forward`

Then compare before/after any codegen change.

### 4. Keep semantic debugging separate

Runs like `1776676727413` show there are still semantic/training divergences.

But that is a different bug class than the `65536` minibatch SPS collapse.

Do not mix these two investigations.

## Bottom Line

The current audit says:

- the matched SPS gap is real
- it is **not** explained by tensor-shape/config mismatches anymore
- the worst losses are most associated with large `minibatch_size`
- but the likely root is **not purely minibatch logic in the env**
- the strongest concrete env-side suspect is still the generated Pong step kernel shape
- the strongest trainer-side suspect is that large-minibatch regimes amplify rollout + forward-path differences together

So the next serious optimization move should be:

- make generated Pong emit a more native-like repeated-substep loop
- then rerun only the bad matched minibatch cohorts and compare `eval_env` and `train_forward` deltas
