# CartPole Side-By-Side

This document compares three concrete artifacts:

1. the input RDDL files that our compiler reads,
2. the generated Ocean/C files emitted by `rddl2puffer`,
3. the existing upstream native CartPole implementation already present in `PufferLib`.

The native training sweeps in `artifacts/cartpole/verify_sweep_20260419/` compare item 2 against item 3.

## Files Compared

- Input RDDL domain:
  [domain.rddl](/home/tommaso/Dev/rddl2puffer/examples/rddl_plus/cartpole_puffer_parity/domain.rddl)
- Input RDDL instance:
  [instance_deterministic_reset.rddl](/home/tommaso/Dev/rddl2puffer/examples/rddl_plus/cartpole_puffer_parity/instance_deterministic_reset.rddl)
- Generated Ocean header:
  [rddl_cartpole_discrete.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/rddl_cartpole_discrete.h)
- Generated Ocean binding:
  [binding.c](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/rddl_cartpole_discrete/binding.c)
- Existing upstream native CartPole:
  [cartpole.h](/home/tommaso/Dev/rddl2puffer/third_party/pufferlib/ocean/cartpole/cartpole.h)

## Reset And Initial State

<table>
  <tr>
    <th>Input RDDL</th>
    <th>Generated Ocean/C</th>
    <th>Existing Upstream Native Ocean/C</th>
  </tr>
  <tr>
    <td>
<pre><code>init-state {
    pos = 0.0;
    vel = 0.0;
    ang-pos = 0.0;
    ang-vel = 0.0;
};</code></pre>
    </td>
    <td>
<pre><code>env-&gt;state[0] = 0.0f;
env-&gt;state[1] = 0.0f;
env-&gt;state[2] = 0.0f;
env-&gt;state[3] = 0.0f;</code></pre>
    </td>
    <td>
<pre><code>env-&gt;x = rand(...) * 0.08f - 0.04f;
env-&gt;x_dot = rand(...) * 0.08f - 0.04f;
env-&gt;theta = rand(...) * 0.08f - 0.04f;
env-&gt;theta_dot = rand(...) * 0.08f - 0.04f;</code></pre>
    </td>
  </tr>
</table>

Takeaway:

- the generated environment currently follows the deterministic reset declared in the RDDL instance,
- the existing upstream native CartPole uses a random reset distribution.

## Action Mapping And Dynamics

<table>
  <tr>
    <th>Input RDDL</th>
    <th>Generated Ocean/C</th>
    <th>Existing Upstream Native Ocean/C</th>
  </tr>
  <tr>
    <td>
<pre><code>force = if (force-side == 1)
        then FORCE-MAG
        else -FORCE-MAG;

total-mass = CART-MASS + POLE-MASS;
polemass-length = total-mass + POLE-MASS;

temp = (force + polemass-length *
        pow[ang-vel, 2] * sin[ang-pos]) / total-mass;

ang-acc = (...);
acc = (...);

pos' = pos + TIME-STEP * vel;
vel' = vel + TIME-STEP * acc;
ang-pos' = ang-pos + TIME-STEP * ang-vel;
ang-vel' = ang-vel + TIME-STEP * ang-acc;</code></pre>
    </td>
    <td>
<pre><code>bool v_14_eq_14 = (... == 1.0f);
float v_16_select_16 =
    (v_14_eq_14 ? v_7_const_FORCE_MAG_0
                : v_15_neg_15);

float v_17_add_17 = (... + ...);
float v_18_add_18 = (... + ...);
float v_24_div_24 = (... / ...);
float v_39_div_39 = (... / ...);
float v_44_sub_44 = (... - ...);

next_state[0] = v_50_add_50;
next_state[1] = v_52_add_52;
next_state[2] = v_46_add_46;
next_state[3] = v_48_add_48;</code></pre>
    </td>
    <td>
<pre><code>float force = env-&gt;continuous ? a * env-&gt;force_mag
    : (a &gt; 0.5f ? env-&gt;force_mag : -env-&gt;force_mag);

float total_mass = env-&gt;cart_mass + env-&gt;pole_mass;
float polemass_length = total_mass + env-&gt;pole_mass;
float temp = (...);
float thetaacc = (...);
float xacc = (...);

env-&gt;x += env-&gt;tau * env-&gt;x_dot;
env-&gt;x_dot += env-&gt;tau * xacc;
env-&gt;theta += env-&gt;tau * env-&gt;theta_dot;
env-&gt;theta_dot += env-&gt;tau * thetaacc;</code></pre>
    </td>
  </tr>
</table>

Takeaway:

- the generated C is real emitted code from the compiler,
- the generated code uses compiler-generated temporary names,
- the existing upstream native CartPole is handwritten C that expresses similar equations directly.

## Reward, Termination, And Horizon

<table>
  <tr>
    <th>Input RDDL</th>
    <th>Generated Ocean/C</th>
    <th>Existing Upstream Native Ocean/C</th>
  </tr>
  <tr>
    <td>
<pre><code>reward = if ((pos + TIME-STEP * vel) &lt; -POS-LIMIT |
             (pos + TIME-STEP * vel) &gt; POS-LIMIT |
             (ang-pos + TIME-STEP * ang-vel) &lt; -ANG-LIMIT |
             (ang-pos + TIME-STEP * ang-vel) &gt; ANG-LIMIT)
         then 0.0
         else 1.0;

termination {
    pos &lt; -POS-LIMIT | pos &gt; POS-LIMIT;
    ang-pos &lt; -ANG-LIMIT | ang-pos &gt; ANG-LIMIT;
};

horizon = 200;</code></pre>
    </td>
    <td>
<pre><code>reward = (float)(v_72_select_72);
terminated = v_81_terminal_or_1;
truncated =
    (RDDL_CARTPOLE_DISCRETE_HORIZON &gt; 0) &amp;&amp;
    (env-&gt;tick &gt;= RDDL_CARTPOLE_DISCRETE_HORIZON);
done = terminated || truncated;

if (done) {
    reward = 0.0f;
}

env-&gt;terminals[0] = terminated ? 1.0f : 0.0f;</code></pre>
    </td>
    <td>
<pre><code>bool terminated =
    env-&gt;x &lt; -X_THRESHOLD || env-&gt;x &gt; X_THRESHOLD ||
    env-&gt;theta &lt; -THETA_THRESHOLD_RADIANS ||
    env-&gt;theta &gt; THETA_THRESHOLD_RADIANS;

bool truncated = env-&gt;tick &gt;= MAX_STEPS;
bool done = terminated || truncated;

env-&gt;rewards[0] = done ? 0.0f : 1.0f;
env-&gt;terminals[0] = terminated ? 1 : 0;</code></pre>
    </td>
  </tr>
</table>

Takeaway:

- the generated environment now mirrors the upstream native CartPole convention of zero reward on `done`,
- both generated and native distinguish `terminated` from `truncated`,
- both expose only `terminated` through `env->terminals[0]`.

## Logging And Sweep Configuration

<table>
  <tr>
    <th>Input RDDL</th>
    <th>Generated Ocean/C</th>
    <th>Existing Upstream Native Ocean/C</th>
  </tr>
  <tr>
    <td>
<pre><code>//% runtime_semantics.zero_reward_on_done = true
//% puffer_logging.score_mode = "tick"
//% puffer_logging.perf_mode =
//%     "positive_return_div_horizon"
//% puffer_config.vec.total_agents = 4096
//% puffer_config.policy.hidden_size = 32
//% puffer_config.policy.num_layers = 2
//% puffer_config.train.total_timesteps = 5642560
//% puffer_config.sweep.use_gpu = false</code></pre>
    </td>
    <td>
<pre><code>env-&gt;log.perf =
    env-&gt;episode_return / HORIZON;
env-&gt;log.score += (float)env-&gt;tick;
env-&gt;log.episode_return += env-&gt;episode_return;
env-&gt;log.episode_length += (float)env-&gt;tick;
env-&gt;log.n += 1.0f;</code></pre>
    </td>
    <td>
<pre><code>env-&gt;log.perf = env-&gt;episode_return / MAX_STEPS;
env-&gt;log.episode_length += env-&gt;tick;
env-&gt;log.score += env-&gt;tick;
env-&gt;log.x_threshold_termination += (...);
env-&gt;log.pole_angle_termination += (...);
env-&gt;log.max_steps_termination += (...);
env-&gt;log.n += 1;</code></pre>
    </td>
  </tr>
</table>

Takeaway:

- the top-of-file `//% ...` directives in the RDDL domain feed the generated `.ini` and some generated runtime semantics,
- the generated environment currently matches the upstream native `tick`-based score convention,
- the existing upstream native CartPole still exposes richer diagnostic counters.

## Current Generated-Vs-Native Sweep Plot

![Generated vs native CartPole sweep comparison](/home/tommaso/Dev/rddl2puffer/artifacts/cartpole/verify_sweep_20260419/comparison.svg)

Related artifacts:

- [Comparison report](/home/tommaso/Dev/rddl2puffer/artifacts/cartpole/verify_sweep_20260419/report.md)
- [Per-run CSV](/home/tommaso/Dev/rddl2puffer/artifacts/cartpole/verify_sweep_20260419/runs.csv)
- [Machine-readable summary](/home/tommaso/Dev/rddl2puffer/artifacts/cartpole/verify_sweep_20260419/summary.json)
