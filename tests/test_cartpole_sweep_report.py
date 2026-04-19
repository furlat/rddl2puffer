import json
from pathlib import Path

from rddl2puffer.reporting.cartpole_sweep_report import write_cartpole_sweep_comparison


def test_write_cartpole_sweep_comparison_writes_artifacts(tmp_path: Path) -> None:
    native_root = tmp_path / "native" / "cartpole"
    generated_root = tmp_path / "generated" / "rddl_cartpole_discrete"
    native_root.mkdir(parents=True)
    generated_root.mkdir(parents=True)

    _write_log(native_root / "native_a.json", "cartpole", 120.0, 32, 2, 4096, 4, 24.0)
    _write_log(native_root / "native_b.json", "cartpole", 180.0, 64, 3, 2048, 2, 26.0)
    _write_log(generated_root / "generated_a.json", "rddl_cartpole_discrete", 90.0, 32, 2, 4096, 4, 23.0)
    _write_log(generated_root / "generated_b.json", "rddl_cartpole_discrete", 140.0, 128, 4, 1024, 3, 25.0)

    output_dir = tmp_path / "artifacts"
    payload = write_cartpole_sweep_comparison(tmp_path / "native", tmp_path / "generated", output_dir)

    assert payload["native"]["best_run"]["final_score"] == 180.0
    assert payload["generated"]["best_run"]["final_score"] == 140.0
    assert output_dir.joinpath("summary.json").is_file()
    assert output_dir.joinpath("runs.csv").is_file()
    assert output_dir.joinpath("comparison.svg").is_file()
    assert output_dir.joinpath("report.md").is_file()


def _write_log(
    path: Path,
    env_name: str,
    score: float,
    hidden_size: float,
    num_layers: float,
    total_agents: float,
    num_buffers: float,
    gpu_percent: float,
) -> None:
    payload = {
        "env_name": env_name,
        "policy": {"hidden_size": hidden_size, "num_layers": num_layers},
        "vec": {"total_agents": total_agents, "num_buffers": num_buffers},
        "metrics": {
            "env/score": [score - 10.0, score],
            "agent_steps": [1_000_000, 2_000_000],
            "util/gpu_percent": [gpu_percent, gpu_percent],
            "util/vram_used_gb": [1.9, 1.9],
        },
    }
    path.write_text(json.dumps(payload))
