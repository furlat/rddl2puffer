# Generated Vs Native CartPole Audit

This note compares the generated native RDDL CartPole env against upstream native Puffer `cartpole`, with a bias toward differences that could explain systematic stability gaps.

Reference files:

- Generated env header:
  [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h)
- Generated binding:
  [binding.c](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/binding.c)
- Generated config:
  [rddl_cartpole_discrete.ini](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/config/rddl_cartpole_discrete.ini)
- Native env header:
  [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h)
- Native binding:
  [binding.c](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/binding.c)
- Native config:
  [cartpole.ini](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/config/cartpole.ini)
- Local vecenv contract:
  [vecenv.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/src/vecenv.h)
- RDDL lowering source:
  [cartpole_discrete.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/benchmarks/cartpole_discrete.py)

## Bottom Line

The generated env is not merely a slower copy of native Puffer CartPole. It differs in several semantically important ways:

- reset state distribution,
- reward on terminal transition,
- horizon handling,
- observation ordering,
- and the underlying dynamics themselves.

So the current native-vs-generated learning gap is not evidence of one isolated compiler bug. Right now we are comparing two different tasks that happen to share the name "CartPole".

## Highest-Suspicion Differences

## 1. Horizon is treated as terminal in generated env but not in native env

This is the strongest generated-side stability risk.

Generated env:

- computes `horizon_done`
- folds it into `done`
- writes `env->terminals[0] = done ? 1.0f : 0.0f`

from:

- [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h:138)
- [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h:143)

Native env:

- computes `terminated`
- computes `truncated`
- auto-resets on either
- but only writes `env->terminals[0] = terminated ? 1 : 0`

from:

- [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:183)
- [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:190)

Why this matters:

- `vecenv` only copies `terminals` into the trainer interface; there is no standard truncation path in the current local static vecenv flow
- see [vecenv.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/src/vecenv.h:265) and [vecenv.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/src/vecenv.h:627)

So generated CartPole currently tells the learner that time-limit episodes are true terminals, while native CartPole does not.

That changes bootstrap targets and can easily affect stability.

## 2. Generated reset starts from a much harder state

Generated env always resets to the RDDL instance defaults:

- `theta = 0.1`
- `theta_dot = 0`
- `pos = 0`
- `vel = 0`

from:

- [cartpole_discrete.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/benchmarks/cartpole_discrete.py:18)
- [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h:47)

Native env resets all four values randomly in `[-0.04, 0.04]`:

- [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:144)

Why this matters:

- native almost always starts very close to upright
- generated always starts with a nontrivial pole angle, about `0.1 rad`
- failure threshold is about `0.2094 rad`

So generated starts at almost half the angular failure limit every episode, while native usually starts much closer to zero. That alone makes the generated task harder.

## 3. Reward on terminal step differs

Generated env always assigns reward `1.0` for the transition, even if it ends the episode:

- [cartpole_discrete.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/benchmarks/cartpole_discrete.py:120)
- [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h:127)

Native env assigns:

- `1.0` on non-done
- `0.0` on done

from:

- [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:188)

Why this matters:

- even if both envs report similar-looking episode lengths,
- the value function sees different boundary rewards,
- and generated `score` is closer to "episode length including terminal step reward"
- while native `episode_return` is one reward smaller on failures

This is another likely contributor to different learning dynamics.

## 4. Native and generated dynamics are not the same task

Generated env matches the classic/RDDL equations:

- `pole_mass_len = pole_len * pole_mass`
- denominator keeps the `pole_mass / total_mass` factor

from:

- [cartpole_discrete.py](/home/tommaso/Dev/rddl2puffer/rddl2puffer/benchmarks/cartpole_discrete.py:83)
- [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h:88)

Native env uses:

- `polemass_length = total_mass + pole_mass`
- denominator that simplifies to `4/3 - cos^2(theta)`

from:

- [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:169)

This materially changes the one-step updates. For the same state/action samples we checked, native produces substantially larger acceleration magnitudes than generated/classic CartPole.

So if native solves faster, that may be because it is literally a different dynamical system, not because the generated backend is unstable in the same task.

## Medium-Suspicion Differences

## 5. Observation ordering differs

Native observation order:

- `x`
- `x_dot`
- `theta`
- `theta_dot`

from:

- [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:131)

Generated observation order currently follows canonicalized fluent order:

- `theta`
- `theta_dot`
- `pos`
- `vel`

from:

- [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h:52)

This should not break an MLP outright, but it does mean we are not feeding the network the same coordinate layout as native CartPole.

## 6. Native and generated configs are only partly aligned

The good news:

- most core optimizer settings are aligned
- learning rate, gamma, entropy, minibatch, horizon, etc. were copied over

But there are still important differences:

- native default `total_timesteps = 5642560`
- generated default `total_timesteps = 52428800`
- native has `use_rnn = 1`
- generated does not set `use_rnn`

from:

- [cartpole.ini](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/config/cartpole.ini:20)
- [rddl_cartpole_discrete.ini](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/config/rddl_cartpole_discrete.ini:20)

Also, current `puffer sweep` launches the first two runs from defaults before suggesting alternatives:

- [pufferl.py](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/pufferlib/pufferl.py:401)

So any mismatch in default configs gets overweighted in a small sweep.

## Lower-Suspicion But Worth Cleaning Up

## 7. Generated log semantics are not identical to native log semantics

Generated `score` is accumulated from `episode_return`:

- [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h:42)

Native `score` is accumulated from `tick`:

- [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h:55)

These are close for CartPole, but not truly identical because native zeroes reward on done while generated does not.

## 8. Generated env lacks the richer native diagnostic fields

Native logs:

- `x_threshold_termination`
- `pole_angle_termination`
- `max_steps_termination`

Generated logs currently expose only:

- `perf`
- `score`
- `episode_return`
- `episode_length`

This does not cause instability directly, but it makes debugging harder because we cannot tell whether generated failures are mostly angle failures, x failures, or time-limit handling artifacts.

## Recommended Fix Order

If the goal is to compare generated vs native training stability in a meaningful way, the next fixes should be:

1. Align truncation semantics.
   Generated CartPole should not mark horizon timeouts as terminal if native CartPole does not.
2. Align reset semantics for the experiment.
   Either:
   - make generated use the native random reset distribution for a direct native comparison, or
   - make a second native-like generated variant for benchmarking while preserving the true RDDL variant separately.
3. Align terminal-step reward semantics.
   Decide whether the benchmark target is:
   - true RDDL/classic semantics, or
   - native Puffer CartPole semantics.
4. Add native-style diagnostic logging to generated CartPole.
5. Only after semantic alignment, rerun sweep comparisons and interpret stability gaps.

## Recommendation

Do not read the current native-vs-generated gap as "the compiler backend is unstable" yet.

The cleaner interpretation is:

- generated env seems trainable,
- native env is a different and somewhat easier task,
- and the strongest generated-side trainer mismatch is the terminal-vs-timeout handling.

That terminal handling issue is the first thing I would fix before doing any more sweep-based comparisons.
