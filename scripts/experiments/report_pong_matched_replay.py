#!/usr/bin/env python3
"""Generate paired native-vs-generated Pong training curves from a replay manifest."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path
import statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-manifest", required=True, help="Replay manifest JSON.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the replay manifest directory plus /report.",
    )
    parser.add_argument("--points", type=int, default=160, help="Number of interpolation points.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = Path(args.replay_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else manifest_path.parent / "report"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for entry in manifest["entries"]:
        native_payload = json.loads(Path(entry["native_log_path"]).read_text(encoding="utf-8"))
        generated_payload = json.loads(Path(entry["generated_log_path"]).read_text(encoding="utf-8"))
        pairs.append((entry, native_payload, generated_payload))

    max_steps = min(
        min(native["metrics"]["agent_steps"][-1], generated["metrics"]["agent_steps"][-1])
        for _, native, generated in pairs
    )
    grid = [max_steps * index / (args.points - 1) for index in range(args.points)]

    native_return = [_sample_metric(native["metrics"], "env/episode_return", grid) for _, native, _ in pairs]
    generated_return = [_sample_metric(generated["metrics"], "env/episode_return", grid) for _, _, generated in pairs]
    native_perf = [_sample_metric(native["metrics"], "env/perf", grid) for _, native, _ in pairs]
    generated_perf = [_sample_metric(generated["metrics"], "env/perf", grid) for _, _, generated in pairs]
    native_sps = [_sample_metric(native["metrics"], "SPS", grid) for _, native, _ in pairs]
    generated_sps = [_sample_metric(generated["metrics"], "SPS", grid) for _, _, generated in pairs]

    pair_rows = _pair_rows(pairs)
    deltas = {
        "episode_return": [row["generated_final_episode_return"] - row["native_final_episode_return"] for row in pair_rows],
        "perf": [row["generated_final_perf"] - row["native_final_perf"] for row in pair_rows],
        "sps": [row["generated_final_sps"] - row["native_final_sps"] for row in pair_rows],
    }
    feature_correlations = {
        "episode_return_delta": _feature_correlations(
            pair_rows,
            metric_key="episode_return_delta",
            feature_keys=["batch_size", "minibatch_size", "hidden_size", "num_layers", "model_size_proxy", "replay_ratio"],
        ),
        "perf_delta": _feature_correlations(
            pair_rows,
            metric_key="perf_delta",
            feature_keys=["batch_size", "minibatch_size", "hidden_size", "num_layers", "model_size_proxy", "replay_ratio"],
        ),
        "sps_delta": _feature_correlations(
            pair_rows,
            metric_key="sps_delta",
            feature_keys=["batch_size", "minibatch_size", "hidden_size", "num_layers", "model_size_proxy", "replay_ratio"],
        ),
    }

    summary = {
        "num_pairs": len(pairs),
        "max_steps": max_steps,
        "native_final_episode_return_mean": statistics.fmean(series[-1] for series in native_return),
        "generated_final_episode_return_mean": statistics.fmean(series[-1] for series in generated_return),
        "native_final_perf_mean": statistics.fmean(series[-1] for series in native_perf),
        "generated_final_perf_mean": statistics.fmean(series[-1] for series in generated_perf),
        "native_final_sps_mean": statistics.fmean(series[-1] for series in native_sps),
        "generated_final_sps_mean": statistics.fmean(series[-1] for series in generated_sps),
        "delta": {
            "episode_return": _delta_summary(deltas["episode_return"]),
            "perf": _delta_summary(deltas["perf"]),
            "sps": _delta_summary(deltas["sps"]),
        },
        "feature_correlations": feature_correlations,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "paired_runs.csv").write_text(_pairs_csv(pair_rows), encoding="utf-8")
    (output_dir / "episode_return_ci.svg").write_text(
        _ci_svg(
            grid,
            native_return,
            generated_return,
            title="Pong Episode Return (Matched Replay)",
            y_label="episode_return",
            left_label="Native Pong",
            right_label="Generated Pong",
        ),
        encoding="utf-8",
    )
    (output_dir / "perf_ci.svg").write_text(
        _ci_svg(
            grid,
            native_perf,
            generated_perf,
            title="Pong Perf (Matched Replay)",
            y_label="perf",
            left_label="Native Pong",
            right_label="Generated Pong",
        ),
        encoding="utf-8",
    )
    (output_dir / "sps_ci.svg").write_text(
        _ci_svg(
            grid,
            native_sps,
            generated_sps,
            title="Pong SPS (Matched Replay)",
            y_label="SPS",
            left_label="Native Pong",
            right_label="Generated Pong",
        ),
        encoding="utf-8",
    )
    (output_dir / "episode_return_delta_distribution.svg").write_text(
        _delta_distribution_svg(
            pair_rows,
            metric_key="episode_return_delta",
            title="Matched Episode Return Delta",
            subtitle="generated - native",
        ),
        encoding="utf-8",
    )
    (output_dir / "perf_delta_distribution.svg").write_text(
        _delta_distribution_svg(
            pair_rows,
            metric_key="perf_delta",
            title="Matched Perf Delta",
            subtitle="generated - native",
        ),
        encoding="utf-8",
    )
    (output_dir / "sps_delta_distribution.svg").write_text(
        _delta_distribution_svg(
            pair_rows,
            metric_key="sps_delta",
            title="Matched SPS Delta",
            subtitle="generated - native",
        ),
        encoding="utf-8",
    )
    (output_dir / "episode_return_delta_by_pair.svg").write_text(
        _delta_by_pair_svg(
            pair_rows,
            metric_key="episode_return_delta",
            title="Episode Return Delta By Pair",
            subtitle="sorted by generated - native",
        ),
        encoding="utf-8",
    )
    (output_dir / "perf_delta_by_pair.svg").write_text(
        _delta_by_pair_svg(
            pair_rows,
            metric_key="perf_delta",
            title="Perf Delta By Pair",
            subtitle="sorted by generated - native",
        ),
        encoding="utf-8",
    )
    (output_dir / "sps_delta_by_pair.svg").write_text(
        _delta_by_pair_svg(
            pair_rows,
            metric_key="sps_delta",
            title="SPS Delta By Pair",
            subtitle="sorted by generated - native",
        ),
        encoding="utf-8",
    )
    scatter_specs = [
        ("sps_delta", "batch_size", "SPS Delta vs Batch Size", "generated - native", True, "batch_size = total_agents * horizon"),
        ("sps_delta", "minibatch_size", "SPS Delta vs Minibatch Size", "generated - native", True, "minibatch_size"),
        ("sps_delta", "model_size_proxy", "SPS Delta vs Model Size Proxy", "generated - native", True, "hidden_size * num_layers"),
        ("episode_return_delta", "batch_size", "Episode Return Delta vs Batch Size", "generated - native", True, "batch_size = total_agents * horizon"),
        ("episode_return_delta", "model_size_proxy", "Episode Return Delta vs Model Size Proxy", "generated - native", True, "hidden_size * num_layers"),
        ("perf_delta", "batch_size", "Perf Delta vs Batch Size", "generated - native", True, "batch_size = total_agents * horizon"),
        ("perf_delta", "hidden_size", "Perf Delta vs Hidden Size", "generated - native", True, "hidden_size"),
    ]
    for metric_key, feature_key, title, subtitle, log_x, x_label in scatter_specs:
        (output_dir / f"{metric_key}_vs_{feature_key}.svg").write_text(
            _scatter_svg(
                pair_rows,
                metric_key=metric_key,
                feature_key=feature_key,
                title=title,
                subtitle=subtitle,
                x_label=x_label,
                y_label=metric_key,
                log_x=log_x,
            ),
            encoding="utf-8",
        )
    (output_dir / "report.md").write_text(
        _report_markdown(summary, output_dir, pair_rows),
        encoding="utf-8",
    )
    print(output_dir)
    return 0


def _sample_metric(metrics: dict[str, list[float]], metric_key: str, grid: list[float]) -> list[float]:
    xs = [float(value) for value in metrics["agent_steps"]]
    ys = [float(value) for value in metrics[metric_key]]
    return [_interpolate(xs, ys, x) for x in grid]


def _interpolate(xs: list[float], ys: list[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    right = bisect_right(xs, x)
    left = right - 1
    x0 = xs[left]
    x1 = xs[right]
    y0 = ys[left]
    y1 = ys[right]
    if x1 <= x0:
        return y1
    ratio = (x - x0) / (x1 - x0)
    return y0 + ratio * (y1 - y0)


def _mean_and_ci(series_list: list[list[float]]) -> tuple[list[float], list[float], list[float]]:
    means: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    n = len(series_list)
    for values in zip(*series_list, strict=True):
        values_list = [float(value) for value in values]
        mean = statistics.fmean(values_list)
        if n > 1:
            stdev = statistics.stdev(values_list)
            margin = 1.96 * stdev / math.sqrt(n)
        else:
            margin = 0.0
        means.append(mean)
        lowers.append(mean - margin)
        uppers.append(mean + margin)
    return means, lowers, uppers


def _pair_rows(pairs: list[tuple[dict[str, object], dict[str, object], dict[str, object]]]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for entry, native, generated in pairs:
        native_final_episode_return = float(native["metrics"]["env/episode_return"][-1])
        generated_final_episode_return = float(generated["metrics"]["env/episode_return"][-1])
        native_final_perf = float(native["metrics"]["env/perf"][-1])
        generated_final_perf = float(generated["metrics"]["env/perf"][-1])
        native_final_sps = float(native["metrics"]["SPS"][-1])
        generated_final_sps = float(generated["metrics"]["SPS"][-1])
        overrides = _override_map(entry["overrides"])
        policy = native["policy"]
        train = native["train"]
        vec = native["vec"]
        hidden_size = _override_float(overrides, "--policy.hidden-size", float(policy["hidden_size"]))
        num_layers = _override_float(overrides, "--policy.num-layers", float(policy["num_layers"]))
        total_agents = _override_float(overrides, "--vec.total-agents", float(vec["total_agents"]))
        num_buffers = _override_float(overrides, "--vec.num-buffers", float(vec["num_buffers"]))
        horizon = _override_float(overrides, "--train.horizon", float(train["horizon"]))
        minibatch_size = _override_float(overrides, "--train.minibatch-size", float(train["minibatch_size"]))
        total_timesteps = _override_float(overrides, "--train.total-timesteps", float(train["total_timesteps"]))
        replay_ratio = _override_float(overrides, "--train.replay-ratio", float(train["replay_ratio"]))
        learning_rate = _override_float(overrides, "--train.learning-rate", float(train["learning_rate"]))
        rows.append(
            {
                "native_run_id": str(entry["native_run_id"]),
                "native_log_path": str(entry["native_log_path"]),
                "generated_log_path": str(entry["generated_log_path"]),
                "native_final_episode_return": native_final_episode_return,
                "generated_final_episode_return": generated_final_episode_return,
                "episode_return_delta": generated_final_episode_return - native_final_episode_return,
                "native_final_perf": native_final_perf,
                "generated_final_perf": generated_final_perf,
                "perf_delta": generated_final_perf - native_final_perf,
                "native_final_sps": native_final_sps,
                "generated_final_sps": generated_final_sps,
                "sps_delta": generated_final_sps - native_final_sps,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "model_size_proxy": hidden_size * num_layers,
                "total_agents": total_agents,
                "num_buffers": num_buffers,
                "horizon": horizon,
                "batch_size": total_agents * horizon,
                "minibatch_size": minibatch_size,
                "total_timesteps": total_timesteps,
                "replay_ratio": replay_ratio,
                "learning_rate": learning_rate,
            }
        )
    return rows


def _override_map(overrides: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for index in range(0, len(overrides), 2):
        key = overrides[index]
        if index + 1 < len(overrides):
            mapping[key] = overrides[index + 1]
    return mapping


def _override_float(mapping: dict[str, str], key: str, fallback: float) -> float:
    if key not in mapping:
        return fallback
    try:
        return float(mapping[key])
    except ValueError:
        return fallback


def _pairs_csv(rows: list[dict[str, float | str]]) -> str:
    lines = [
        "native_run_id,native_log_path,generated_log_path,native_final_episode_return,generated_final_episode_return,episode_return_delta,native_final_perf,generated_final_perf,perf_delta,native_final_sps,generated_final_sps,sps_delta,hidden_size,num_layers,model_size_proxy,total_agents,num_buffers,horizon,batch_size,minibatch_size,total_timesteps,replay_ratio,learning_rate"
    ]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row["native_run_id"]),
                    str(row["native_log_path"]),
                    str(row["generated_log_path"]),
                    f"{float(row['native_final_episode_return']):.6f}",
                    f"{float(row['generated_final_episode_return']):.6f}",
                    f"{float(row['episode_return_delta']):.6f}",
                    f"{float(row['native_final_perf']):.6f}",
                    f"{float(row['generated_final_perf']):.6f}",
                    f"{float(row['perf_delta']):.6f}",
                    f"{float(row['native_final_sps']):.6f}",
                    f"{float(row['generated_final_sps']):.6f}",
                    f"{float(row['sps_delta']):.6f}",
                    f"{float(row['hidden_size']):.6f}",
                    f"{float(row['num_layers']):.6f}",
                    f"{float(row['model_size_proxy']):.6f}",
                    f"{float(row['total_agents']):.6f}",
                    f"{float(row['num_buffers']):.6f}",
                    f"{float(row['horizon']):.6f}",
                    f"{float(row['batch_size']):.6f}",
                    f"{float(row['minibatch_size']):.6f}",
                    f"{float(row['total_timesteps']):.6f}",
                    f"{float(row['replay_ratio']):.6f}",
                    f"{float(row['learning_rate']):.9f}",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _delta_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p10": _percentile(ordered, 0.10),
        "p90": _percentile(ordered, 0.90),
        "nonnegative_count": sum(value >= 0.0 for value in values),
        "negative_count": sum(value < 0.0 for value in values),
    }


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = fraction * (len(sorted_values) - 1)
    left = int(math.floor(index))
    right = int(math.ceil(index))
    if left == right:
        return sorted_values[left]
    ratio = index - left
    return sorted_values[left] + ratio * (sorted_values[right] - sorted_values[left])


def _feature_correlations(
    rows: list[dict[str, float | str]],
    *,
    metric_key: str,
    feature_keys: list[str],
) -> dict[str, float]:
    correlations: dict[str, float] = {}
    for feature_key in feature_keys:
        correlations[feature_key] = _pearson(
            [float(row[feature_key]) for row in rows],
            [float(row[metric_key]) for row in rows],
        )
    return correlations


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _ci_svg(
    grid: list[float],
    left_series: list[list[float]],
    right_series: list[list[float]],
    *,
    title: str,
    y_label: str,
    left_label: str,
    right_label: str,
) -> str:
    width = 960
    height = 540
    pad_left = 90
    pad_right = 30
    pad_top = 50
    pad_bottom = 70
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom

    left_mean, left_low, left_high = _mean_and_ci(left_series)
    right_mean, right_low, right_high = _mean_and_ci(right_series)

    y_min = min(min(left_low), min(right_low))
    y_max = max(max(left_high), max(right_high))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    def x_px(value: float) -> float:
        return pad_left + (value / grid[-1]) * plot_width

    def y_px(value: float) -> float:
        return pad_top + (y_max - value) / (y_max - y_min) * plot_height

    def band_path(low: list[float], high: list[float]) -> str:
        forward = " ".join(f"L{x_px(x):.2f},{y_px(y):.2f}" for x, y in zip(grid[1:], high[1:], strict=True))
        backward = " ".join(
            f"L{x_px(x):.2f},{y_px(y):.2f}" for x, y in zip(reversed(grid), reversed(low), strict=True)
        )
        return f"M{x_px(grid[0]):.2f},{y_px(high[0]):.2f} {forward} {backward} Z"

    def line_path(values: list[float]) -> str:
        return " ".join(
            [f"M{x_px(grid[0]):.2f},{y_px(values[0]):.2f}"]
            + [f"L{x_px(x):.2f},{y_px(y):.2f}" for x, y in zip(grid[1:], values[1:], strict=True)]
        )

    ticks = 6
    y_ticks = [y_min + (y_max - y_min) * index / (ticks - 1) for index in range(ticks)]
    x_ticks = [grid[-1] * index / (ticks - 1) for index in range(ticks)]

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#f8fafc" />
<text x="{width/2:.0f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#0f172a">{title}</text>
<text x="{width/2:.0f}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">agent_steps</text>
<text x="24" y="{height/2:.0f}" transform="rotate(-90 24 {height/2:.0f})" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">{y_label}</text>
<rect x="{pad_left}" y="{pad_top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />
{''.join(f'<line x1="{pad_left}" x2="{pad_left+plot_width}" y1="{y_px(value):.2f}" y2="{y_px(value):.2f}" stroke="#e2e8f0" stroke-width="1" />' for value in y_ticks)}
{''.join(f'<line y1="{pad_top}" y2="{pad_top+plot_height}" x1="{x_px(value):.2f}" x2="{x_px(value):.2f}" stroke="#e2e8f0" stroke-width="1" />' for value in x_ticks)}
<path d="{band_path(left_low, left_high)}" fill="#2563eb" opacity="0.18" />
<path d="{band_path(right_low, right_high)}" fill="#16a34a" opacity="0.18" />
<path d="{line_path(left_mean)}" fill="none" stroke="#2563eb" stroke-width="3" />
<path d="{line_path(right_mean)}" fill="none" stroke="#16a34a" stroke-width="3" />
{''.join(f'<text x="{pad_left-10}" y="{y_px(value)+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#475569">{value:.3g}</text>' for value in y_ticks)}
{''.join(f'<text x="{x_px(value):.2f}" y="{pad_top+plot_height+20}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#475569">{value/1_000_000:.1f}M</text>' for value in x_ticks)}
<g transform="translate({pad_left + 20}, {pad_top + 20})">
  <rect x="0" y="0" width="14" height="14" fill="#2563eb" opacity="0.85" />
  <text x="22" y="12" font-family="sans-serif" font-size="13" fill="#0f172a">{left_label}</text>
  <rect x="200" y="0" width="14" height="14" fill="#16a34a" opacity="0.85" />
  <text x="222" y="12" font-family="sans-serif" font-size="13" fill="#0f172a">{right_label}</text>
</g>
</svg>
"""


def _delta_distribution_svg(
    rows: list[dict[str, float | str]],
    *,
    metric_key: str,
    title: str,
    subtitle: str,
) -> str:
    values = [float(row[metric_key]) for row in rows]
    width = 960
    height = 540
    pad_left = 80
    pad_right = 30
    pad_top = 60
    pad_bottom = 70
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom
    bins = min(12, max(6, int(math.sqrt(len(values))) + 1))
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    bin_width = (vmax - vmin) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        index = min(bins - 1, int((value - vmin) / bin_width))
        counts[index] += 1
    max_count = max(counts) or 1
    zero_x = pad_left + ((0.0 - vmin) / (vmax - vmin)) * plot_width if vmin <= 0.0 <= vmax else None

    bars = []
    for index, count in enumerate(counts):
        x0 = pad_left + (index / bins) * plot_width
        x1 = pad_left + ((index + 1) / bins) * plot_width
        bar_height = (count / max_count) * (plot_height - 10)
        y0 = pad_top + plot_height - bar_height
        color = "#16a34a" if (vmin + (index + 0.5) * bin_width) >= 0 else "#dc2626"
        bars.append(
            f'<rect x="{x0 + 2:.2f}" y="{y0:.2f}" width="{max(1.0, x1 - x0 - 4):.2f}" height="{bar_height:.2f}" fill="{color}" opacity="0.8" />'
        )

    tick_values = [vmin + (vmax - vmin) * i / 5 for i in range(6)]
    count_ticks = [max_count * i / 4 for i in range(5)]
    median = statistics.median(values)
    mean = statistics.fmean(values)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#f8fafc" />
<text x="{width/2:.0f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#0f172a">{title}</text>
<text x="{width/2:.0f}" y="48" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#475569">{subtitle}</text>
<rect x="{pad_left}" y="{pad_top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />
{''.join(f'<line x1="{pad_left}" x2="{pad_left+plot_width}" y1="{pad_top + plot_height - (tick/max_count)*plot_height:.2f}" y2="{pad_top + plot_height - (tick/max_count)*plot_height:.2f}" stroke="#e2e8f0" stroke-width="1" />' for tick in count_ticks)}
{f'<line x1="{zero_x:.2f}" x2="{zero_x:.2f}" y1="{pad_top}" y2="{pad_top+plot_height}" stroke="#0f172a" stroke-width="1.5" stroke-dasharray="6 4" />' if zero_x is not None else ''}
{''.join(bars)}
{''.join(f'<text x="{pad_left-10}" y="{pad_top + plot_height - (tick/max_count)*plot_height + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#475569">{tick:.0f}</text>' for tick in count_ticks)}
{''.join(f'<text x="{pad_left + ((value - vmin)/(vmax-vmin))*plot_width:.2f}" y="{pad_top + plot_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#475569">{value:.3g}</text>' for value in tick_values)}
<text x="{pad_left + 8}" y="{pad_top + 18}" font-family="sans-serif" font-size="13" fill="#0f172a">mean = {mean:.4g}</text>
<text x="{pad_left + 8}" y="{pad_top + 36}" font-family="sans-serif" font-size="13" fill="#0f172a">median = {median:.4g}</text>
</svg>
"""


def _delta_by_pair_svg(
    rows: list[dict[str, float | str]],
    *,
    metric_key: str,
    title: str,
    subtitle: str,
) -> str:
    ordered = sorted(rows, key=lambda row: float(row[metric_key]))
    values = [float(row[metric_key]) for row in ordered]
    width = 1080
    height = 520
    pad_left = 80
    pad_right = 30
    pad_top = 60
    pad_bottom = 80
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    limit = max(abs(vmin), abs(vmax))
    vmin = -limit
    vmax = limit

    def x_px(index: int) -> float:
        if len(values) == 1:
            return pad_left + plot_width / 2
        return pad_left + (index / (len(values) - 1)) * plot_width

    def y_px(value: float) -> float:
        return pad_top + (vmax - value) / (vmax - vmin) * plot_height

    zero_y = y_px(0.0)
    y_ticks = [vmin + (vmax - vmin) * i / 6 for i in range(7)]
    bars = []
    for index, row in enumerate(ordered):
        value = float(row[metric_key])
        x = x_px(index)
        y = y_px(value)
        color = "#16a34a" if value >= 0 else "#dc2626"
        top = min(y, zero_y)
        bar_height = abs(zero_y - y)
        bars.append(
            f'<rect x="{x - 6:.2f}" y="{top:.2f}" width="12" height="{max(1.0, bar_height):.2f}" fill="{color}" opacity="0.85" />'
        )

    x_labels = []
    if len(ordered) <= 12:
        label_indices = range(len(ordered))
    else:
        label_indices = sorted(set([0, len(ordered) // 4, len(ordered) // 2, (3 * len(ordered)) // 4, len(ordered) - 1]))
    for index in label_indices:
        x_labels.append(
            f'<text x="{x_px(index):.2f}" y="{pad_top + plot_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#475569">{ordered[index]["native_run_id"]}</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#f8fafc" />
<text x="{width/2:.0f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#0f172a">{title}</text>
<text x="{width/2:.0f}" y="48" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#475569">{subtitle}</text>
<rect x="{pad_left}" y="{pad_top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />
{''.join(f'<line x1="{pad_left}" x2="{pad_left+plot_width}" y1="{y_px(tick):.2f}" y2="{y_px(tick):.2f}" stroke="#e2e8f0" stroke-width="1" />' for tick in y_ticks)}
<line x1="{pad_left}" x2="{pad_left+plot_width}" y1="{zero_y:.2f}" y2="{zero_y:.2f}" stroke="#0f172a" stroke-width="1.5" />
{''.join(bars)}
{''.join(f'<text x="{pad_left-10}" y="{y_px(tick)+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#475569">{tick:.3g}</text>' for tick in y_ticks)}
{''.join(x_labels)}
</svg>
"""


def _scatter_svg(
    rows: list[dict[str, float | str]],
    *,
    metric_key: str,
    feature_key: str,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    log_x: bool,
) -> str:
    width = 960
    height = 540
    pad_left = 90
    pad_right = 30
    pad_top = 60
    pad_bottom = 70
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom

    points = []
    for row in rows:
        raw_x = float(row[feature_key])
        plot_x = math.log2(raw_x) if log_x else raw_x
        points.append((plot_x, float(row[metric_key]), str(row["native_run_id"])))

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    y_limit = max(abs(y_min), abs(y_max))
    y_min = -y_limit
    y_max = y_limit

    def x_px(value: float) -> float:
        return pad_left + (value - x_min) / (x_max - x_min) * plot_width

    def y_px(value: float) -> float:
        return pad_top + (y_max - value) / (y_max - y_min) * plot_height

    x_ticks_raw = sorted({float(row[feature_key]) for row in rows})
    if len(x_ticks_raw) > 6:
        indices = sorted(set([0, len(x_ticks_raw) // 4, len(x_ticks_raw) // 2, (3 * len(x_ticks_raw)) // 4, len(x_ticks_raw) - 1]))
        x_ticks_raw = [x_ticks_raw[index] for index in indices]
    x_ticks = [(math.log2(value) if log_x else value, value) for value in x_ticks_raw]
    y_ticks = [y_min + (y_max - y_min) * i / 6 for i in range(7)]
    zero_y = y_px(0.0)
    zero_x = x_px(0.0) if x_min <= 0.0 <= x_max else None
    mean_delta = statistics.fmean(y_values)

    dots = []
    for plot_x, plot_y, run_id in points:
        color = "#16a34a" if plot_y >= 0.0 else "#dc2626"
        dots.append(
            f'<circle cx="{x_px(plot_x):.2f}" cy="{y_px(plot_y):.2f}" r="5" fill="{color}" opacity="0.85"><title>run {run_id}: {feature_key}={2**plot_x if log_x else plot_x:.6g}, {metric_key}={plot_y:.6g}</title></circle>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#f8fafc" />
<text x="{width/2:.0f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#0f172a">{title}</text>
<text x="{width/2:.0f}" y="48" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#475569">{subtitle}</text>
<text x="{width/2:.0f}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">{x_label}{' (log2 scale)' if log_x else ''}</text>
<text x="24" y="{height/2:.0f}" transform="rotate(-90 24 {height/2:.0f})" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#334155">{y_label}</text>
<rect x="{pad_left}" y="{pad_top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />
{''.join(f'<line x1="{pad_left}" x2="{pad_left+plot_width}" y1="{y_px(value):.2f}" y2="{y_px(value):.2f}" stroke="#e2e8f0" stroke-width="1" />' for value in y_ticks)}
{''.join(f'<line y1="{pad_top}" y2="{pad_top+plot_height}" x1="{x_px(value):.2f}" x2="{x_px(value):.2f}" stroke="#e2e8f0" stroke-width="1" />' for value, _ in x_ticks)}
<line x1="{pad_left}" x2="{pad_left+plot_width}" y1="{zero_y:.2f}" y2="{zero_y:.2f}" stroke="#0f172a" stroke-width="1.5" />
{f'<line y1="{pad_top}" y2="{pad_top+plot_height}" x1="{zero_x:.2f}" x2="{zero_x:.2f}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="5 4" />' if zero_x is not None else ''}
{''.join(dots)}
{''.join(f'<text x="{pad_left-10}" y="{y_px(value)+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#475569">{value:.3g}</text>' for value in y_ticks)}
{''.join(f'<text x="{x_px(value):.2f}" y="{pad_top+plot_height+22}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#475569">{raw:.3g}</text>' for value, raw in x_ticks)}
<text x="{pad_left + 8}" y="{pad_top + 18}" font-family="sans-serif" font-size="13" fill="#0f172a">mean delta = {mean_delta:.4g}</text>
</svg>
"""


def _report_markdown(summary: dict[str, float], output_dir: Path, rows: list[dict[str, float | str]]) -> str:
    worst_return = min(rows, key=lambda row: float(row["episode_return_delta"]))
    worst_perf = min(rows, key=lambda row: float(row["perf_delta"]))
    worst_sps = min(rows, key=lambda row: float(row["sps_delta"]))
    best_return = max(rows, key=lambda row: float(row["episode_return_delta"]))
    return (
        "# Pong Matched Replay Report\n\n"
        f"- Pairs: `{summary['num_pairs']}`\n"
        f"- Native mean final `episode_return`: `{summary['native_final_episode_return_mean']:.4f}`\n"
        f"- Generated mean final `episode_return`: `{summary['generated_final_episode_return_mean']:.4f}`\n"
        f"- Native mean final `perf`: `{summary['native_final_perf_mean']:.4f}`\n"
        f"- Generated mean final `perf`: `{summary['generated_final_perf_mean']:.4f}`\n"
        f"- Native mean final `SPS`: `{summary['native_final_sps_mean']:.4f}`\n"
        f"- Generated mean final `SPS`: `{summary['generated_final_sps_mean']:.4f}`\n"
        f"- `episode_return` delta mean / median: `{summary['delta']['episode_return']['mean']:.4f}` / `{summary['delta']['episode_return']['median']:.4f}`\n"
        f"- `perf` delta mean / median: `{summary['delta']['perf']['mean']:.4f}` / `{summary['delta']['perf']['median']:.4f}`\n"
        f"- `SPS` delta mean / median: `{summary['delta']['sps']['mean']:.4f}` / `{summary['delta']['sps']['median']:.4f}`\n"
        f"- Strongest `episode_return` delta correlation: `{_top_correlation(summary['feature_correlations']['episode_return_delta'])}`\n"
        f"- Strongest `perf` delta correlation: `{_top_correlation(summary['feature_correlations']['perf_delta'])}`\n"
        f"- Strongest `SPS` delta correlation: `{_top_correlation(summary['feature_correlations']['sps_delta'])}`\n"
        f"- Best `episode_return` delta pair: `{best_return['native_run_id']}` -> `{float(best_return['episode_return_delta']):.4f}`\n"
        f"- Worst `episode_return` delta pair: `{worst_return['native_run_id']}` -> `{float(worst_return['episode_return_delta']):.4f}`\n"
        f"- Worst `perf` delta pair: `{worst_perf['native_run_id']}` -> `{float(worst_perf['perf_delta']):.4f}`\n"
        f"- Worst `SPS` delta pair: `{worst_sps['native_run_id']}` -> `{float(worst_sps['sps_delta']):.4f}`\n\n"
        f"Artifacts live in `{output_dir}`.\n"
    )


def _top_correlation(correlations: dict[str, float]) -> str:
    key, value = max(correlations.items(), key=lambda item: abs(item[1]))
    return f"{key} ({value:+.3f})"


if __name__ == "__main__":
    raise SystemExit(main())
