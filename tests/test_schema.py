from rddl2puffer.frontend.schema import FluentDType, FluentRole, FluentSpec, build_layout, canonicalize_fluents


def test_canonicalize_fluents_is_stable_and_lexicographic() -> None:
    fluents = (
        FluentSpec(name="position", role=FluentRole.STATE, dtype=FluentDType.REAL, parameters=("b",)),
        FluentSpec(name="position", role=FluentRole.STATE, dtype=FluentDType.REAL, parameters=("a",)),
        FluentSpec(name="battery", role=FluentRole.STATE, dtype=FluentDType.REAL),
    )

    ordered = canonicalize_fluents(fluents)

    assert [fluent.qualified_name for fluent in ordered] == [
        "battery",
        "position[a]",
        "position[b]",
    ]


def test_build_layout_assigns_contiguous_offsets() -> None:
    layout = build_layout(
        "state",
        (
            FluentSpec(name="vector", role=FluentRole.STATE, dtype=FluentDType.REAL, shape=(3,)),
            FluentSpec(name="flag", role=FluentRole.STATE, dtype=FluentDType.BOOL),
        ),
    )

    assert layout.total_size == 4
    assert layout.offsets == {"vector": 0, "flag": 3}
