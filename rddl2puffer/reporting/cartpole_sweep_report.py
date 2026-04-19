"""Comparison reporting for native-vs-generated CartPole sweep batches."""

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
    final_score: float
    peak_score: float
    final_steps: float
    hidden_size: float
    num_layers: float
    total_agents: float
    num_buffers: float
    gpu_percent: float
    vram_used_gb: float


@dataclass(frozen=True, slots=True)
class SweepBatchSummary:
    """Aggregate stats over a family of sweep runs."""

    label: str
    runs: tuple[SweepRun, ...]

    @property
    def final_scores(self) -> tuple[float, ...]:
        return tuple(run.final_score for run in self.runs)

    @property
    def peak_scores(self) -> tuple[float, ...]:
        return tuple(run.peak_score for run in self.runs)

    @property
    def best_run(self) -> SweepRun:
        return max(self.runs, key=lambda run: run.final_score)

    @property
    def mean_final_score(self) -> float:
        return statistics.fmean(self.final_scores)

    @property
    def median_final_score(self) -> float:
        return statistics.median(self.final_scores)

    @property
    def success_count_100(self) -> int:
        return sum(score >= 100.0 for score in self.final_scores)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "num_runs": len(self.runs),
            "mean_final_score": self.mean_final_score,
            "median_final_score": self.median_final_score,
            "best_run": asdict(self.best_run),
            "success_count_100": self.success_count_100,
            "runs": [asdict(run) for run in self.runs],
        }


def write_cartpole_sweep_comparison(
    native_log_root: Path,
    generated_log_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Write a side-by-side comparison bundle for two sweep batches."""

    native = SweepBatchSummary("native_cartpole", _load_runs(native_log_root))
    generated = SweepBatchSummary("generated_rddl_cartpole", _load_runs(generated_log_root))
    payload = {
        "native": native.to_summary_dict(),
        "generated": generated.to_summary_dict(),
        "delta": {
            "best_final_score_gap": native.best_run.final_score - generated.best_run.final_score,
            "mean_final_score_gap": native.mean_final_score - generated.mean_final_score,
            "median_final_score_gap": native.median_final_score - generated.median_final_score,
            "success_count_100_gap": native.success_count_100 - generated.success_count_100,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (output_dir / "runs.csv").write_text(_runs_csv(native, generated))
    (output_dir / "comparison.svg").write_text(_comparison_svg(native, generated))
    (output_dir / "report.md").write_text(_report_markdown(native, generated))
    return payload


def _load_runs(log_root: Path) -> tuple[SweepRun, ...]:
    files = sorted(log_root.glob("*/*.json"))
    if not files:
        raise FileNotFoundError(f"No sweep logs found under {log_root}")

    runs: list[SweepRun] = []
    for path in files:
        payload = json.loads(path.read_text())
        metrics = payload["metrics"]
        runs.append(
            SweepRun(
                env_name=str(payload["env_name"]),
                log_path=str(path),
                final_score=float(metrics["env/score"][-1]),
                peak_score=float(max(metrics["env/score"])),
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


def _runs_csv(native: SweepBatchSummary, generated: SweepBatchSummary) -> str:
    lines = [
        "label,env_name,final_score,peak_score,final_steps,hidden_size,num_layers,total_agents,num_buffers,gpu_percent,vram_used_gb,log_path"
    ]
    for label, batch in (("native", native), ("generated", generated)):
        for run in batch.runs:
            lines.append(
                ",".join(
                    [
                        label,
                        run.env_name,
                        f"{run.final_score:.6f}",
                        f"{run.peak_score:.6f}",
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


def _report_markdown(native: SweepBatchSummary, generated: SweepBatchSummary) -> str:
    return (
        "# CartPole Sweep Comparison\n\n"
        f"- Native runs: `{len(native.runs)}`\n"
        f"- Generated runs: `{len(generated.runs)}`\n"
        f"- Native best final score: `{native.best_run.final_score:.3f}`\n"
        f"- Generated best final score: `{generated.best_run.final_score:.3f}`\n"
        f"- Native mean final score: `{native.mean_final_score:.3f}`\n"
        f"- Generated mean final score: `{generated.mean_final_score:.3f}`\n"
        f"- Native runs >= 100: `{native.success_count_100}`\n"
        f"- Generated runs >= 100: `{generated.success_count_100}`\n\n"
        "Artifacts in this directory:\n\n"
        "- `summary.json`: machine-readable batch summary\n"
        "- `runs.csv`: per-run sweep results\n"
        "- `comparison.svg`: final-score bar chart\n"
    )


def _comparison_svg(native: SweepBatchSummary, generated: SweepBatchSummary) -> str:
    width = 1100
    height = 560
    margin_left = 80
    margin_right = 24
    margin_top = 52
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    all_runs = [("native", run) for run in native.runs] + [("generated", run) for run in generated.runs]
    max_score = max(run.final_score for _, run in all_runs)
    y_max = max(200.0, ((int(max_score) // 20) + 1) * 20.0)
    bar_width = plot_width / max(1, len(all_runs) * 1.5)

    def map_y(score: float) -> float:
        return margin_top + plot_height - (score / y_max) * plot_height

    grid = []
    for level in [0, 50, 100, 150, 200]:
        y = map_y(level)
        grid.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#d5d8dc" stroke-width="1" />'
        )
        grid.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="monospace" font-size="12" fill="#4a5568">{level}</text>'
        )

    bars = []
    for idx, (label, run) in enumerate(all_runs):
        x = margin_left + (idx + 0.5) * (plot_width / len(all_runs))
        y = map_y(run.final_score)
        color = "#127fbf" if label == "native" else "#0b6e4f"
        bars.append(
            f'<rect x="{x - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{margin_top + plot_height - y:.2f}" fill="{color}" />'
        )
        bars.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 18:.2f}" text-anchor="middle" font-family="monospace" font-size="10" fill="#4a5568">{"N" if label == "native" else "G"}{idx + 1 if label == "native" else idx + 1 - len(native.runs)}</text>'
        )
        bars.append(
            f'<text x="{x:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-family="monospace" font-size="10" fill="{color}">{run.final_score:.1f}</text>'
        )

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fcfcfb" />',
            '<text x="80" y="28" font-family="monospace" font-size="22" fill="#1f2933">CartPole sweep comparison: native vs generated</text>',
            '<text x="80" y="46" font-family="monospace" font-size="12" fill="#52606d">Final score per sweep run. Blue = native cartpole, green = generated RDDL CartPole.</text>',
            *grid,
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#1f2933" stroke-width="2" />',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#1f2933" stroke-width="2" />',
            *bars,
            f'<text x="{width / 2:.2f}" y="{height - 12}" text-anchor="middle" font-family="monospace" font-size="12" fill="#4a5568">Sweep runs</text>',
            f'<text x="20" y="{height / 2:.2f}" transform="rotate(-90 20 {height / 2:.2f})" text-anchor="middle" font-family="monospace" font-size="12" fill="#4a5568">Final score</text>',
            "</svg>",
        ]
    )
