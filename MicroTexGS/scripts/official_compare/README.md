# Official Comparison Scripts

Use only this directory for the comparison runs.

## Methods

- GS^3 baseline: `run_gs3_release.sh`
  - Code: `GS3_release`
  - Uses the GS^3 release training path and the patched release light rasterizer.
  - Does not pass `--detach_shadow`.
  - Does not pass MicroTexGS texture flags.

- MicroTexGS: `run_microtexgs.sh`
  - Code: `MicroTexGS`
  - Uses the same shared GS^3 scene schedule, then adds the MicroTexGS flags.
  - Default texture setting: fixed `4x4`, `uvshadow_specular_lobe`, start iteration `0`.

- RNG forward: `run_rng_forward.sh`
  - Code: `RNG_release_aligned`
  - Runs the 100k forward-shading recipe.
  - Uses `--max_reso 0` to avoid hidden resolution downsampling.
  - Defaults `--max_training_images` to `VIEW_NUM` so RNG sees the same number of train views as GS^3/MicroTexGS.

- RNG deferred: `run_rng_deferred.sh`
  - Code: `RNG_release_aligned`
  - Runs stage 1 forward 30k, then stage 2 deferred 100k from `chkpnt30000.pth`.
  - Uses the official deferred flags and `--depth_mlp_modifier 1.0` by default.
  - Uses `--max_reso 0` to avoid hidden resolution downsampling.
  - Defaults `--max_training_images` to `VIEW_NUM` so RNG sees the same number of train views as GS^3/MicroTexGS.

- All methods for one scene: `run_scene_all.sh`
  - Runs `preflight_alignment.sh` first and fails before training if RNG train-view count and `VIEW_NUM` are not aligned.

## Data Rules

- GS^3-only scenes use `GSRELIGHT_ROOT`.
- Shared NRHints scenes must set `NRHINTS_ROOT` explicitly.
- `NRHINTS_ROOT` must point to the original NRHints dataset, not a silent fallback under `GSRELIGHT_ROOT`.

## Required Variables

- `SCENE`, for example `LightStage/Container` or `NRHints/Pikachu`.
- `VIEW_NUM`, default `2000`; shared by GS^3/MicroTexGS and RNG train-view count.
- `ROOT`, default `/ssdwork/liuhaohan/RTS`.
- `GSRELIGHT_ROOT`, default `/ssdwork/liuhaohan/datasets/gsrelight`.
- `NRHINTS_ROOT`, required for `NRHints/*`.
- `RUN_ROOT`, default timestamped directory under `/ssdwork/liuhaohan/outputs/compare_runs`.

## Alignment Guard

By default, `ALIGN_TRAIN_VIEWS=1`. RNG runs fail if `RNG_MAX_TRAINING_IMAGES`
does not match `VIEW_NUM`. Set `ALIGN_TRAIN_VIEWS=0` only for a deliberate
official-recipe probe, not for the main comparison table.

## Smoke Test

Set `SMOKE=1` to reduce each method to a short run before submitting long training.
