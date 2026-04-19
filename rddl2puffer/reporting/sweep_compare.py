"""Generic reporting for native-vs-generated Puffer sweep batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics


@dataclass(frozen=True, slots=True)
class SweepRun:
    """One completed sweep trial."""

    env_name: str
    log_path: str
    final_metric: float
    peak_metric: float
    final_sps: float
    final_episode_return: float | None
    final_episode_length: float | None
    final_perf: float | None
    final_steps: float
    hidden_size: float
    num_layers: float
    total_agents: float
    num_buffers: float
    gpu_percent: float
    vram_used_gb: float


@dataclass(frozen=True, slots=True)
class SweepBatchSummary:
    """Aggregate stats over one sweep family."""

    label: str
    metric_label: str
    runs: tuple[SweepRun, ...]

    @property
    def final_metrics(self) -> tuple[float, ...]:
        return tuple(run.final_metric for run in self.runs)

    @property
    def final_sps_values(self) -> tuple[float, ...]:
        return tuple(run.final_sps for run in self.runs)

    @property
    def best_run(self) -> SweepRun:
        return max(self.runs, key=lambda run: run.final_metric)

    @property
    def mean_final_metric(self) -> float:
        return statistics.fmean(self.final_metrics)

    @property
    def median_final_metric(self) -> float:
        return statistics.median(self.final_metrics)

    @property
    def mean_final_sps(self) -> float:
        return statistics.fmean(self.final_sps_values)

    def success_count(self, threshold: float | None) -> int | None:
        if threshold is None:
            return None
        return sum(metric >= threshold for metric in self.final_metrics)

    def to_summary_dict(self, *, success_threshold: float | None) -> dict[str, object]:
        return {
            "label": self.label,
            "metric_label": self.metric_label,
            "num_runs": len(self.runs),
            "mean_final_metric": self.mean_final_metric,
            "median_final_metric": self.median_final_metric,
            "mean_final_sps": self.mean_final_sps,
            "best_run": asdict(self.best_run),
            "success_threshold": success_threshold,
            "success_count": self.success_count(success_threshold),
            "runs": [asdict(run) for run in self.runs],
        }


def write_sweep_comparison(
    *,
    left_log_root: Path,
    right_log_root: Path,
    output_dir: Path,
    left_label: str,
    right_label: str,
    title: str,
    metric_key: str = "env/score",
    metric_label: str = "score",
    success_threshold: float | None = None,
) -> dict[str, object]:
    """Write a side-by-side comparison bundle for two sweep batches."""

    left = SweepBatchSummary(left_label, metric_label, _load_runs(left_log_root, metric_key))
    right = SweepBatchSummary(right_label, metric_label, _load_runs(right_log_root, metric_key))
    payload = {
        "left": left.to_summary_dict(success_threshold=success_threshold),
        "right": right.to_summary_dict(success_threshold=success_threshold),
        "delta": {
            "best_final_metric_gap": left.best_run.final_metric - right.best_run.final_metric,
            "mean_final_metric_gap": left.mean_final_metric - right.mean_final_metric,
            "median_final_metric_gap": left.median_final_metric - right.median_final_metric,
            "mean_final_sps_gap": left.mean_final_sps - right.mean_final_sps,
            "success_count_gap": (
                None
                if success_threshold is None
                else left.success_count(success_threshold) - right.success_count(success_threshold)
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (output_dir / "runs.csv").write_text(_runs_csv(left, right))
    (output_dir / "comparison.svg").write_text(_comparison_svg(left, right, title=title))
    (output_dir / "sps.svg").write_text(_sps_svg(left, right, title=title))
    (output_dir / "learning_curve.svg").write_text(
        _learning_curve_svg(left, right, title=title, metric_key=metric_key, value_label=metric_label)
    )
    (output_dir / "report.md").write_text(
        _report_markdown(left, right, title=title, success_threshold=success_threshold)
    )
    return payload


def _load_runs(log_root: Path, metric_key: str) -> tuple[SweepRun, ...]:
    files = sorted(log_root.glob("*/*.json"))
    if not files:
        raise FileNotFoundError(f"No sweep logs found under {log_root}")

    runs: list[SweepRun] = []
    for path in files:
        payload = json.loads(path.read_text())
        metrics = payload["metrics"]
        metric_series = metrics[metric_key]
        runs.append(
            SweepRun(
                env_name=str(payload["env_name"]),
                log_path=str(path),
                final_metric=float(metric_series[-1]),
                peak_metric=float(max(metric_series)),
                final_sps=float(metrics["SPS"][-1]),
                final_episode_return=_last_or_none(metrics.get("env/episode_return")),
                final_episode_length=_last_or_none(metrics.get("env/episode_length")),
                final_perf=_last_or_none(metrics.get("env/perf")),
                final_steps=float(metrics["agent_steps"][-1]),
                hidden_size=float(payload["policy"]["hidden_size"]),
                num_layers=float(payload["policy"]["num_layers"]),
                total_agents=float(payload["vec"]["total_agents"]),
                num_buffers=float(payload["vec"]["num_buffers"]),
                gpu_percent=float(metrics["util/gpu_percent"][-1]),
                vram_used_gb=float(metrics["util/vram_used_gb"][-1]),
            )
        )
    return tuple(runs)


def _last_or_none(values: object) -> float | None:
    if isinstance(values, list) and values:
        return float(values[-1])
    return None


def _runs_csv(left: SweepBatchSummary, right: SweepBatchSummary) -> str:
    lines = [
        "label,env_name,final_metric,peak_metric,final_sps,final_episode_return,final_episode_length,final_perf,final_steps,hidden_size,num_layers,total_agents,num_buffers,gpu_percent,vram_used_gb,log_path"
    ]
    for label, batch in ((left.label, left), (right.label, right)):
        for run in batch.runs:
            lines.append(
                ",".join(
                    [
                        label,
                        run.env_name,
                        f"{run.final_metric:.6f}",
                        f"{run.peak_metric:.6f}",
                        f"{run.final_sps:.6f}",
                        "" if run.final_episode_return is None else f"{run.final_episode_return:.6f}",
                        "" if run.final_episode_length is None else f"{run.final_episode_length:.6f}",
                        "" if run.final_perf is None else f"{run.final_perf:.6f}",
                        f"{run.final_steps:.0f}",
                        f"{run.hidden_size:.6f}",
                        f"{run.num_layers:.6f}",
                        f"{run.total_agents:.6f}",
                        f"{run.num_buffers:.6f}",
                        f"{run.gpu_percent:.6f}",
                        f"{run.vram_used_gb:.6f}",
                        run.log_path,
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def _report_markdown(
    left: SweepBatchSummary,
    right: SweepBatchSummary,
    *,
    title: str,
    success_threshold: float | None,
) -> str:
    left_success = left.success_count(success_threshold)
    right_success = right.success_count(success_threshold)
    success_lines = ""
    if success_threshold is not None:
        success_lines = (
            f"- {left.label} runs >= {success_threshold:g}: `{left_success}`\n"
            f"- {right.label} runs >= {success_threshold:g}: `{right_success}`\n"
        )
    return (
        f"# {title}\n\n"
        f"- Left runs: `{len(left.runs)}`\n"
        f"- Right runs: `{len(right.runs)}`\n"
        f"- Left best final {left.metric_label}: `{left.best_run.final_metric:.3f}`\n"
        f"- Right best final {right.metric_label}: `{right.best_run.final_metric:.3f}`\n"
        f"- Left mean final {left.metric_label}: `{left.mean_final_metric:.3f}`\n"
        f"- Right mean final {right.metric_label}: `{right.mean_final_metric:.3f}`\n"
        f"- Left mean final SPS: `{_format_millions(left.mean_final_sps)}`\n"
        f"- Right mean final SPS: `{_format_millions(right.mean_final_sps)}`\n"
        f"{success_lines}\n"
        "Artifacts in this directory:\n\n"
        "- `summary.json`: machine-readable batch summary\n"
        "- `runs.csv`: per-run sweep results\n"
        "- `comparison.svg`: final-metric bar chart\n"
        "- `sps.svg`: final-SPS bar chart\n"
        "- `learning_curve.svg`: overlaid best-run learning curves\n"
    )


def _comparison_svg(left: SweepBatchSummary, right: SweepBatchSummary, *, title: str) -> str:
    return _bar_chart_svg(
        left,
        right,
        title=title,
        value_getter=lambda run: run.final_metric,
        value_label=left.metric_label,
        left_color="#127fbf",
        right_color="#0b6e4f",
    )


def _sps_svg(left: SweepBatchSummary, right: SweepBatchSummary, *, title: str) -> str:
    return _bar_chart_svg(
        left,
        right,
        title=f"{title}: final SPS",
        value_getter=lambda run: run.final_sps / 1_000_000.0,
        value_label="SPS (M)",
        left_color="#cc7a00",
        right_color="#7d3cff",
    )


def _bar_chart_svg(
    left: SweepBatchSummary,
    right: SweepBatchSummary,
    *,
    title: str,
    value_getter,
    value_label: str,
    left_color: str,
    right_color: str,
) -> str:
    width = 1100
    height = 560
    margin_left = 80
    margin_right = 24
    margin_top = 52
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    all_runs = [(left.label, run) for run in left.runs] + [(right.label, run) for run in right.runs]
    min_value = min(value_getter(run) for _, run in all_runs)
    max_value = max(value_getter(run) for _, run in all_runs)
    if min_value == max_value:
        padding = max(1.0, abs(max_value) * 0.1)
        y_min = min_value - padding
        y_max = max_value + padding
    else:
        y_min = min(0.0, min_value)
        y_max = max(0.0, max_value)
        spread = y_max - y_min
        padding = max(1.0, spread * 0.08)
        y_min -= padding
        y_max += padding
    bar_width = plot_width / max(1, len(all_runs) * 1.5)

    def map_y(value: float) -> float:
        return margin_top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    grid = []
    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        level = y_min + (y_max - y_min) * ratio
        y = map_y(level)
        grid.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#d5d8dc" stroke-width="1" />'
        )
        grid.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="monospace" font-size="12" fill="#4a5568">{level:.1f}</text>'
        )

    bars = []
    baseline_y = map_y(0.0)
    for idx, (label, run) in enumerate(all_runs):
        x = margin_left + (idx + 0.5) * (plot_width / len(all_runs))
        value = value_getter(run)
        y = map_y(value)
        color = left_color if label == left.label else right_color
        bars.append(
            f'<rect x="{x - bar_width / 2:.2f}" y="{min(y, baseline_y):.2f}" width="{bar_width:.2f}" height="{abs(baseline_y - y):.2f}" fill="{color}" />'
        )
        bars.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 18:.2f}" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a5568">{("L" if label == left.label else "R")}{idx + 1 if label == left.label else idx + 1 - len(left.runs)}</text>'
        )
        bars.append(
            f'<text x="{x:.2f}" y="{(y - 8 if value >= 0 else y + 16):.2f}" text-anchor="middle" font-family="monospace" font-size="10" fill="{color}">{value:.2f}</text>'
        )

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fcfcfb" />',
            f'<text x="80" y="28" font-family="monospace" font-size="22" fill="#1f2933">{title}</text>',
            f'<text x="80" y="46" font-family="monospace" font-size="12" fill="#52606d">Left batch = {left.label}. Right batch = {right.label}.</text>',
            *grid,
            f'<line x1="{margin_left}" y1="{baseline_y:.2f}" x2="{width - margin_right}" y2="{baseline_y:.2f}" stroke="#1f2933" stroke-width="2" />',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#1f2933" stroke-width="2" />',
            *bars,
            f'<text x="{width / 2:.2f}" y="{height - 12}" text-anchor="middle" font-family="monospace" font-size="12" fill="#4a5568">Sweep runs</text>',
            f'<text x="20" y="{height / 2:.2f}" transform="rotate(-90 20 {height / 2:.2f})" text-anchor="middle" font-family="monospace" font-size="12" fill="#4a5568">{value_label}</text>',
            "</svg>",
        ]
    )


def _learning_curve_svg(
    left: SweepBatchSummary,
    right: SweepBatchSummary,
    *,
    title: str,
    metric_key: str,
    value_label: str,
) -> str:
    width = 1100
    height = 560
    margin_left = 88
    margin_right = 30
    margin_top = 58
    margin_bottom = 74
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    left_steps, left_values = _load_metric_curve(Path(left.best_run.log_path), metric_key)
    right_steps, right_values = _load_metric_curve(Path(right.best_run.log_path), metric_key)
    max_step = max(max(left_steps), max(right_steps))
    min_value = min(min(left_values), min(right_values))
    max_value = max(max(left_values), max(right_values))

    if min_value == max_value:
        padding = max(1.0, abs(max_value) * 0.1)
        y_min = min_value - padding
        y_max = max_value + padding
    else:
        spread = max_value - min_value
        padding = max(1.0, spread * 0.08)
        y_min = min_value - padding
        y_max = max_value + padding

    def map_x(step: float) -> float:
        if max_step == 0:
            return margin_left
        return margin_left + (step / max_step) * plot_width

    def map_y(value: float) -> float:
        return margin_top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    grid = []
    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        level = y_min + (y_max - y_min) * ratio
        y = map_y(level)
        grid.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#d5d8dc" stroke-width="1" />'
        )
        grid.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="monospace" font-size="12" fill="#4a5568">{level:.1f}</text>'
        )

    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        step = max_step * ratio
        x = map_x(step)
        grid.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" stroke="#eef2f7" stroke-width="1" />'
        )
        grid.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 20:.2f}" text-anchor="middle" font-family="monospace" font-size="12" fill="#4a5568">{step / 1_000_000:.2f}M</text>'
        )

    left_points = " ".join(f"{map_x(step):.2f},{map_y(value):.2f}" for step, value in zip(left_steps, left_values))
    right_points = " ".join(f"{map_x(step):.2f},{map_y(value):.2f}" for step, value in zip(right_steps, right_values))

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fcfcfb" />',
            f'<text x="80" y="28" font-family="monospace" font-size="22" fill="#1f2933">{title}: overlaid best-run learning curves</text>',
            f'<text x="80" y="46" font-family="monospace" font-size="12" fill="#52606d">Blue = {left.label}. Green = {right.label}. X axis = agent steps.</text>',
            *grid,
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#1f2933" stroke-width="2" />',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#1f2933" stroke-width="2" />',
            f'<polyline fill="none" stroke="#127fbf" stroke-width="3" points="{left_points}" />',
            f'<polyline fill="none" stroke="#0b6e4f" stroke-width="3" points="{right_points}" />',
            f'<circle cx="{map_x(left_steps[-1]):.2f}" cy="{map_y(left_values[-1]):.2f}" r="4" fill="#127fbf" />',
            f'<circle cx="{map_x(right_steps[-1]):.2f}" cy="{map_y(right_values[-1]):.2f}" r="4" fill="#0b6e4f" />',
            f'<text x="{width / 2:.2f}" y="{height - 12}" text-anchor="middle" font-family="monospace" font-size="12" fill="#4a5568">Agent steps</text>',
            f'<text x="20" y="{height / 2:.2f}" transform="rotate(-90 20 {height / 2:.2f})" text-anchor="middle" font-family="monospace" font-size="12" fill="#4a5568">{value_label}</text>',
            "</svg>",
        ]
    )


def _load_metric_curve(log_path: Path, metric_key: str) -> tuple[list[float], list[float]]:
    payload = json.loads(log_path.read_text())
    metrics = payload["metrics"]
    return [float(value) for value in metrics["agent_steps"]], [float(value) for value in metrics[metric_key]]


def _format_millions(value: float) -> str:
    return f"{value / 1_000_000:.3f}M"
