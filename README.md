# MicroTexGS Workspace

This workspace is intentionally kept minimal. Use the current code below as the
source of truth; removed historical copies must not be used for experiments.

## Active Code

- `MicroTexGS/`
  - MicroTexGS baseline implementation.
  - Use `scripts/official_compare/` for all comparison training scripts.
- `GS3_release/`
  - GS^3 release code path used for the GS^3 baseline.
  - Contains the patched release CUDA path needed for our environment.
- `RNG_release_aligned/`
  - RNG code path used for reproduced RNG experiments.
  - Contains our alignment patches for camera optimization, HDR/gamma handling,
    resolution control, and platform builds.

## Experiment Entry Point

Use only:

```bash
MicroTexGS/scripts/official_compare/
```

The root-level historical training scripts and old duplicated source trees have
been removed to avoid accidental use.

## Local Experimental Copies

`MicroTexGS_shadow_spatial_cuda/` is reserved for shadow-spatial CUDA experiments
and is intentionally ignored by git. Do not run paper baselines from that local
copy.

## Preserved Non-Code Directories

- `data/`: local data, ignored by git.
- `local_runs/`, `local_outputs/`, `output/`, `remote_artifacts/`: previous
  outputs and checkpoints, ignored by git.
- `_handoff/`: metrics, figures, platform handoff files, and audit artifacts,
  ignored by git.
