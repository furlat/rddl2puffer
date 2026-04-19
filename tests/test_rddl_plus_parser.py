from pathlib import Path

from rddl2puffer.frontend import parse_rddl_files, parse_rddl_sources


def test_parse_rddl_files_parses_local_cartpole_domain() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    domain_path = (
        repo_root
        / "third_party"
        / "rddlrepository"
        / "rddlrepository"
        / "archive"
        / "gym"
        / "CartPole"
        / "Discrete"
        / "domain.rddl"
    )
    instance_path = domain_path.with_name("instance0.rddl")

    parsed = parse_rddl_files(domain_path, instance_path)

    assert parsed.ast.domain.name == "cart_pole_discrete"
    assert parsed.ast.instance.name == "inst_cart_pole_disc_0"


def test_parse_rddl_sources_parses_in_memory_text() -> None:
    parsed = parse_rddl_sources(
        domain_text="""
        domain mini {
            requirements = { reward-deterministic };
            pvariables {
                x : { state-fluent, real, default = 0.0 };
            };
            cpfs {
                x' = x;
            };
            reward = 0.0;
        }
        """,
        instance_text="""
        non-fluents mini_nf {
            domain = mini;
        }

        instance mini_inst {
            domain = mini;
            non-fluents = mini_nf;
            init-state {
                x = 1.0;
            };
            max-nondef-actions = pos-inf;
            horizon = 4;
            discount = 1.0;
        }
        """,
    )

    assert parsed.ast.domain.name == "mini"
    assert parsed.ast.instance.name == "mini_inst"
