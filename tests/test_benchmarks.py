from importlib.machinery import ModuleSpec
from pathlib import Path

from rddl2puffer.baselines.wrapper import probe_wrapper_baseline
from rddl2puffer.benchmarks.cartpole import discover_cartpole_benchmark_trio


def test_discover_cartpole_benchmark_trio_resolves_local_targets(tmp_path: Path) -> None:
    _seed_cartpole_reference_tree(tmp_path)

    trio = discover_cartpole_benchmark_trio(tmp_path)

    assert trio.reference_inputs_present is True
    assert trio.wrapper_rddl.env_name == "CartPole_Discrete_gym"
    assert trio.generated_native.env_name == "rddl_cartpole_discrete"
    assert trio.native_puffer.env_name == "cartpole"
    assert trio.generated_native.required_files_present is True
    assert trio.extra_reference_variants[0].env_name == "CartPole_Continuous_gym"
    assert "semantic gold standard" in trio.native_puffer.notes[1]


def test_probe_wrapper_baseline_reports_emulation_gap(tmp_path: Path, monkeypatch) -> None:
    _seed_cartpole_reference_tree(tmp_path)
    (tmp_path / "third_party" / "pyRDDLGym").mkdir(parents=True)
    example = tmp_path / "third_party" / "pufferlib" / "examples" / "gymnasium_env.py"
    example.parent.mkdir(parents=True, exist_ok=True)
    example.write_text("import pufferlib.emulation\n", encoding="utf-8")

    def fake_find_spec(name: str):
        if name in {"pyRDDLGym", "pufferlib"}:
            return None
        return ModuleSpec(name, loader=None)

    monkeypatch.setattr("rddl2puffer.baselines.wrapper.find_spec", fake_find_spec)

    status = probe_wrapper_baseline(tmp_path)

    assert status.ready_for_local_wrapper_experiment is False
    assert status.examples_reference_emulation is True
    assert status.source_tree_has_emulation is False
    assert any("not installed" in note for note in status.notes)
    assert any("does not contain that module" in note for note in status.notes)


def _seed_cartpole_reference_tree(root: Path) -> None:
    discrete_root = (
        root
        / "third_party"
        / "rddlrepository"
        / "rddlrepository"
        / "archive"
        / "gym"
        / "CartPole"
        / "Discrete"
    )
    continuous_root = discrete_root.parent / "Continuous"
    puffer_cartpole_root = root / "third_party" / "pufferlib" / "ocean" / "cartpole"
    puffer_config_root = root / "third_party" / "pufferlib" / "config"

    for path in (
        discrete_root / "domain.rddl",
        discrete_root / "instance0.rddl",
        continuous_root / "domain.rddl",
        continuous_root / "instance0.rddl",
        puffer_cartpole_root / "binding.c",
        puffer_cartpole_root / "cartpole.h",
        puffer_config_root / "cartpole.ini",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")
