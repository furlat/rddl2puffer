import json
from pathlib import Path

from rddl2puffer.reporting.cartpole_reward_report import (
    estimate_random_policy_baseline,
    find_latest_cartpole_log,
    load_training_curve,
    write_cartpole_reward_report,
)


def test_load_training_curve_reads_score_points(tmp_path: Path) -> None:
    log_path = tmp_path / "run.json"
    log_path.write_text(
        json.dumps(
            {
                "env_name": "rddl_cartpole_discrete",
                "metrics": {
                    "agent_steps": [1_000_000, 2_000_000],
                    "env/score": [10.0, 20.0],
                    "env/episode_return": [10.0, 20.0],
                },
            }
        )
    )

    curve = load_training_curve(log_path)

    assert curve.env_name == "rddl_cartpole_discrete"
    assert len(curve.points) == 2
    assert curve.peak_point.score == 20.0
    assert curve.final_point.agent_steps == 2_000_000


def test_find_latest_cartpole_log_prefers_newest_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "third_party" / "pufferlib" / "logs_reward_runs_dense" / "rddl_cartpole_discrete"
    log_dir.mkdir(parents=True)
    older = log_dir / "older.json"
    newer = log_dir / "newer.json"
    older.write_text("{}")
    newer.write_text("{}")
    older.touch()
    newer.touch()

    assert find_latest_cartpole_log(tmp_path) == newer


def test_write_cartpole_reward_report_writes_artifacts(tmp_path: Path) -> None:
    log_path = tmp_path / "run.json"
    log_path.write_text(
        json.dumps(
            {
                "env_name": "rddl_cartpole_discrete",
                "metrics": {
                    "agent_steps": [1_000_000, 2_000_000, 3_000_000],
                    "env/score": [15.0, 30.0, 45.0],
                    "env/episode_return": [15.0, 30.0, 45.0],
                },
            }
        )
    )

    output_dir = tmp_path / "artifacts"
    report = write_cartpole_reward_report(output_dir=output_dir, log_path=log_path, episodes=250, seed=7)

    assert report.curve.final_point.score == 45.0
    assert output_dir.joinpath("summary.json").is_file()
    assert output_dir.joinpath("training_curve.csv").is_file()
    assert output_dir.joinpath("random_baseline.json").is_file()
    assert output_dir.joinpath("reward_vs_random.svg").is_file()
    assert output_dir.joinpath("report.md").is_file()

    summary = json.loads(output_dir.joinpath("summary.json").read_text())
    assert summary["env_name"] == "rddl_cartpole_discrete"
    assert summary["peak_logged_point"]["score"] == 45.0


def test_estimate_random_policy_baseline_returns_reasonable_stats() -> None:
    baseline = estimate_random_policy_baseline(episodes=200, seed=11)

    assert baseline.episodes == 200
    assert baseline.mean_return > 0.0
    assert baseline.quantile_95 >= baseline.quantile_50 >= baseline.quantile_05
