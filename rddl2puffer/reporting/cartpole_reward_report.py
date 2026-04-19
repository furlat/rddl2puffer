"""Reward-vs-random reporting utilities for generated discrete CartPole."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
from typing import Iterable

from rddl2puffer.benchmarks.cartpole_rddl import make_cartpole_parity_compiled_env


@dataclass(frozen=True, slots=True)
class CurvePoint:
    """One logged training point."""

    agent_steps: float
    score: float
    episode_return: float


@dataclass(frozen=True, slots=True)
class TrainingCurve:
    """Parsed training metrics from one Puffer log."""

    env_name: str
    log_path: str
    points: tuple[CurvePoint, ...]

    @property
    def peak_point(self) -> CurvePoint:
        return max(self.points, key=lambda point: point.score)

    @property
    def final_point(self) -> CurvePoint:
        return self.points[-1]


@dataclass(frozen=True, slots=True)
class RandomBaseline:
    """Monte Carlo estimate for the random policy on the same semantics."""

    episodes: int
    seed: int
    mean_return: float
    stdev_return: float
    mean_length: float
    min_return: float
    max_return: float
    quantile_05: float
    quantile_25: float
    quantile_50: float
    quantile_75: float
    quantile_95: float


@dataclass(frozen=True, slots=True)
class CartPoleRewardReport:
    """High-level reward report for one generated CartPole training run."""

    curve: TrainingCurve
    random_baseline: RandomBaseline

    @property
    def peak_improvement_vs_random(self) -> float:
        return self.curve.peak_point.score / self.random_baseline.mean_return

    @property
    def final_improvement_vs_random(self) -> float:
        return self.curve.final_point.score / self.random_baseline.mean_return

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "env_name": self.curve.env_name,
            "log_path": self.curve.log_path,
            "num_points": len(self.curve.points),
            "random_baseline": asdict(self.random_baseline),
            "peak_logged_point": asdict(self.curve.peak_point),
            "final_logged_point": asdict(self.curve.final_point),
            "peak_improvement_vs_random": self.peak_improvement_vs_random,
            "final_improvement_vs_random": self.final_improvement_vs_random,
        }


def find_latest_cartpole_log(repo_root: Path | None = None) -> Path:
    """Locate the newest generated-CartPole log produced by the local workspace."""

    root = (repo_root or Path.cwd()).resolve()
    search_roots = (
        root / "third_party" / "pufferlib" / "logs_reward_runs_dense" / "rddl_cartpole_discrete",
        root / "third_party" / "pufferlib" / "logs_reward_runs" / "rddl_cartpole_discrete",
        root / "third_party" / "pufferlib" / "logs" / "rddl_cartpole_discrete",
    )

    candidates: list[Path] = []
    for directory in search_roots:
        if not directory.is_dir():
            continue
        candidates.extend(path for path in directory.glob("*.json") if path.is_file())

    if not candidates:
        raise FileNotFoundError("Could not find any generated CartPole logs under third_party/pufferlib.")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_training_curve(log_path: Path) -> TrainingCurve:
    """Parse the score curve from one Puffer JSON log."""

    payload = json.loads(log_path.read_text())
    metrics = payload["metrics"]
    steps = metrics["agent_steps"]
    scores = metrics["env/score"]
    returns = metrics.get("env/episode_return", scores)

    if not steps or not scores:
        raise ValueError(f"No training points found in {log_path}.")
    if len(steps) != len(scores) or len(scores) != len(returns):
        raise ValueError(f"Mismatched metric lengths in {log_path}.")

    points = tuple(
        CurvePoint(agent_steps=float(step), score=float(score), episode_return=float(episode_return))
        for step, score, episode_return in zip(steps, scores, returns, strict=True)
    )
    return TrainingCurve(
        env_name=str(payload.get("env_name", "rddl_cartpole_discrete")),
        log_path=str(log_path),
        points=points,
    )


def estimate_random_policy_baseline(
    episodes: int = 50_000,
    seed: int = 123,
    repo_root: Path | None = None,
) -> RandomBaseline:
    """Roll out a random policy against the IR oracle and summarize returns."""

    env = make_cartpole_parity_compiled_env(repo_root)
    rng = random.Random(seed)
    returns: list[float] = []
    lengths: list[int] = []

    for episode_index in range(episodes):
        env.reset(seed=episode_index)
        total_reward = 0.0
        episode_length = 0
        while True:
            transition = env.step((rng.randint(0, 1),))
            total_reward += transition.reward
            episode_length += 1
            if transition.done:
                returns.append(total_reward)
                lengths.append(episode_length)
                break

    sorted_returns = sorted(returns)
    return RandomBaseline(
        episodes=episodes,
        seed=seed,
        mean_return=statistics.fmean(returns),
        stdev_return=statistics.pstdev(returns),
        mean_length=statistics.fmean(lengths),
        min_return=min(returns),
        max_return=max(returns),
        quantile_05=_quantile(sorted_returns, 0.05),
        quantile_25=_quantile(sorted_returns, 0.25),
        quantile_50=_quantile(sorted_returns, 0.50),
        quantile_75=_quantile(sorted_returns, 0.75),
        quantile_95=_quantile(sorted_returns, 0.95),
    )


def write_cartpole_reward_report(
    output_dir: Path,
    log_path: Path | None = None,
    repo_root: Path | None = None,
    episodes: int = 50_000,
    seed: int = 123,
) -> CartPoleRewardReport:
    """Write JSON/CSV/SVG/Markdown artifacts for one generated CartPole run."""

    resolved_log_path = log_path or find_latest_cartpole_log(repo_root)
    curve = load_training_curve(resolved_log_path)
    baseline = estimate_random_policy_baseline(episodes=episodes, seed=seed, repo_root=repo_root)
    report = CartPoleRewardReport(curve=curve, random_baseline=baseline)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(report.to_summary_dict(), indent=2, sort_keys=True))
    (output_dir / "training_curve.csv").write_text(_curve_csv(curve))
    (output_dir / "random_baseline.json").write_text(json.dumps(asdict(baseline), indent=2, sort_keys=True))
    (output_dir / "reward_vs_random.svg").write_text(_reward_plot_svg(report))
    (output_dir / "report.md").write_text(_report_markdown(report))
    return report


def _curve_csv(curve: TrainingCurve) -> str:
    lines = ["agent_steps,score,episode_return"]
    for point in curve.points:
        lines.append(f"{point.agent_steps:.0f},{point.score:.6f},{point.episode_return:.6f}")
    return "\n".join(lines) + "\n"


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile of an empty sample.")
    index = int((len(sorted_values) - 1) * q)
    return float(sorted_values[index])


def _report_markdown(report: CartPoleRewardReport) -> str:
    peak = report.curve.peak_point
    final = report.curve.final_point
    baseline = report.random_baseline
    return (
        "# Generated CartPole Reward Report\n\n"
        f"- Environment: `{report.curve.env_name}`\n"
        f"- Source log: `{report.curve.log_path}`\n"
        f"- Random baseline episodes: `{baseline.episodes}`\n"
        f"- Random mean return: `{baseline.mean_return:.3f}`\n"
        f"- Best logged score: `{peak.score:.3f}` at `{_format_millions(peak.agent_steps)}` agent steps\n"
        f"- Final logged score: `{final.score:.3f}` at `{_format_millions(final.agent_steps)}` agent steps\n"
        f"- Peak vs random: `{report.peak_improvement_vs_random:.2f}x`\n"
        f"- Final vs random: `{report.final_improvement_vs_random:.2f}x`\n\n"
        "Artifacts in this directory:\n\n"
        "- `summary.json`: machine-readable summary\n"
        "- `training_curve.csv`: step-by-step score curve\n"
        "- `random_baseline.json`: Monte Carlo random-policy summary\n"
        "- `reward_vs_random.svg`: score curve with random baseline overlay\n"
    )


def _reward_plot_svg(report: CartPoleRewardReport) -> str:
    width = 960
    height = 560
    margin_left = 80
    margin_right = 24
    margin_top = 48
    margin_bottom = 64
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    max_steps = max(point.agent_steps for point in report.curve.points)
    max_score = max(
        max(point.score for point in report.curve.points),
        report.random_baseline.quantile_95,
        report.random_baseline.mean_return,
    )
    y_max = max(200.0, math.ceil(max_score / 20.0) * 20.0)

    def map_x(step: float) -> float:
        return margin_left + (step / max_steps) * plot_width

    def map_y(score: float) -> float:
        return margin_top + plot_height - (score / y_max) * plot_height

    grid_levels = [0, 50, 100, 150, 200]
    grid_svg = []
    for level in grid_levels:
        y = map_y(level)
        grid_svg.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" '
            'stroke="#d5d8dc" stroke-width="1" />'
        )
        grid_svg.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="monospace" font-size="12" fill="#4a5568">'
            f"{level}</text>"
        )

    curve_points = " ".join(f"{map_x(point.agent_steps):.2f},{map_y(point.score):.2f}" for point in report.curve.points)
    baseline_y = map_y(report.random_baseline.mean_return)
    peak = report.curve.peak_point
    final = report.curve.final_point

    x_ticks = [0.0, 0.25 * max_steps, 0.5 * max_steps, 0.75 * max_steps, max_steps]
    tick_svg = []
    for tick in x_ticks:
        x = map_x(tick)
        tick_svg.append(
            f'<line x1="{x:.2f}" y1="{margin_top + plot_height}" x2="{x:.2f}" y2="{margin_top + plot_height + 6}" '
            'stroke="#4a5568" stroke-width="1" />'
        )
        tick_svg.append(
            f'<text x="{x:.2f}" y="{height - 18}" text-anchor="middle" '
            'font-family="monospace" font-size="12" fill="#4a5568">'
            f"{_format_millions(tick)}</text>"
        )

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fcfcfb" />',
            '<text x="80" y="28" font-family="monospace" font-size="22" fill="#1f2933">Generated RDDL CartPole: reward vs random</text>',
            '<text x="80" y="48" font-family="monospace" font-size="12" fill="#52606d">Score curve from Puffer training log with Monte Carlo random-policy baseline.</text>',
            *grid_svg,
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#1f2933" stroke-width="2" />',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#1f2933" stroke-width="2" />',
            *tick_svg,
            f'<line x1="{margin_left}" y1="{baseline_y:.2f}" x2="{width - margin_right}" y2="{baseline_y:.2f}" stroke="#d64545" stroke-width="2" stroke-dasharray="8 8" />',
            f'<polyline fill="none" stroke="#147d64" stroke-width="4" points="{curve_points}" />',
            f'<circle cx="{map_x(peak.agent_steps):.2f}" cy="{map_y(peak.score):.2f}" r="5" fill="#0b6e4f" />',
            f'<circle cx="{map_x(final.agent_steps):.2f}" cy="{map_y(final.score):.2f}" r="5" fill="#1f2933" />',
            f'<text x="{map_x(peak.agent_steps) + 8:.2f}" y="{map_y(peak.score) - 10:.2f}" font-family="monospace" font-size="12" fill="#0b6e4f">peak {peak.score:.1f}</text>',
            f'<text x="{map_x(final.agent_steps) - 8:.2f}" y="{map_y(final.score) - 10:.2f}" text-anchor="end" font-family="monospace" font-size="12" fill="#1f2933">final {final.score:.1f}</text>',
            f'<text x="{width - margin_right}" y="{baseline_y - 8:.2f}" text-anchor="end" font-family="monospace" font-size="12" fill="#d64545">random mean {report.random_baseline.mean_return:.1f}</text>',
            f'<text x="{width / 2:.2f}" y="{height - 4}" text-anchor="middle" font-family="monospace" font-size="12" fill="#4a5568">agent steps (M)</text>',
            f'<text x="20" y="{height / 2:.2f}" transform="rotate(-90 20 {height / 2:.2f})" text-anchor="middle" font-family="monospace" font-size="12" fill="#4a5568">episode return / score</text>',
            f'<text x="{width - margin_right}" y="28" text-anchor="end" font-family="monospace" font-size="12" fill="#1f2933">peak/random {report.peak_improvement_vs_random:.2f}x</text>',
            f'<text x="{width - margin_right}" y="44" text-anchor="end" font-family="monospace" font-size="12" fill="#1f2933">final/random {report.final_improvement_vs_random:.2f}x</text>',
            "</svg>",
        ]
    )


def _format_millions(value: float) -> str:
    return f"{value / 1_000_000:.1f}M"
