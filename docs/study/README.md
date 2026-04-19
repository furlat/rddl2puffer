# Study Notes

This directory is the tracked study layer for `rddl2puffer`.

The purpose is to keep the implementation plan grounded in the actual reference semantics and codebases, instead of relying on memory or high-level summaries.

## Local Reference Material

These are intentionally stored under `third_party/` so they stay out of repo history:

- paper HTML:
  `/home/tommaso/Dev/rddl2puffer/third_party/reference_material/papers/pyRDDLGym_2211.05939v5.html`
- pyRDDLGym code:
  `/home/tommaso/Dev/rddl2puffer/third_party/pyRDDLGym`
- rddlrepository code and benchmark corpus:
  `/home/tommaso/Dev/rddl2puffer/third_party/rddlrepository`

## Tracked Study Docs

- [pyrddlgym_paper_notes.md](/home/tommaso/Dev/rddl2puffer/docs/study/pyrddlgym_paper_notes.md)
- [reference_code_map.md](/home/tommaso/Dev/rddl2puffer/docs/study/reference_code_map.md)
- [cartpole_comparison.md](/home/tommaso/Dev/rddl2puffer/docs/study/cartpole_comparison.md)

## Why This Exists

The compiler path we care about is not:

```text
RDDL -> wrapper -> maybe parity later
```

It is:

```text
RDDL -> understood semantics -> owned compiler IR -> native backend
```

These notes are the bridge between the reference world and the compiler we actually want to build.
