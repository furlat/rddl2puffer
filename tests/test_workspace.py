from pathlib import Path

from rddl2puffer.frontend.compile import compile_rddl_sources
from rddl2puffer.workspace import discover_workspace


_DOMAIN = """
domain workspace_demo {
    requirements = { reward-deterministic };

    pvariables {
        counter : { state-fluent, int, default = 0 };
    };

    cpfs {
        counter' = counter + 1;
    };

    reward = 0.0;

    termination {
        counter > 10;
    };
}
"""

_INSTANCE = """
non-fluents workspace_demo_nf {
    domain = workspace_demo;
}

instance workspace_demo_inst {
    domain = workspace_demo;
    non-fluents = workspace_demo_nf;
    init-state {
        counter = 0;
    };
    max-nondef-actions = pos-inf;
    horizon = 4;
    discount = 1.0;
}
"""


def test_discover_workspace_and_write_bundle(tmp_path: Path) -> None:
    (tmp_path / "third_party" / "puffertank").mkdir(parents=True)
    (tmp_path / "third_party" / "pufferlib").mkdir(parents=True)

    workspace = discover_workspace(tmp_path)
    program = compile_rddl_sources(_DOMAIN, _INSTANCE, env_name="demo")
    written = workspace.write_env_bundle(program, env_name="rddl_counter")

    assert workspace.pufferlib_root == tmp_path / "third_party" / "pufferlib"
    assert (tmp_path / "third_party" / "pufferlib" / "config" / "rddl_counter.ini").exists()
    assert (tmp_path / "third_party" / "pufferlib" / "ocean" / "rddl_counter" / "binding.c").exists()
    assert len(written) == 4
