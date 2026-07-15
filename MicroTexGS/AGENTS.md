# Current MicroTexGS / LoFT-GS Handoff

This file describes the current paper implementation. Older differentiable-
shadow supervision work is no longer the project mainline.

## Current Method

The active method is a 2DGS relighting model with factorized local transport:

- Gaussian level: geometry, opacity, local frame, RGB specular coefficient,
  scalar ASG response, neural shadow correction, and neural RGB residual.
- Local level: absolute RGB diffuse, dynamic per-light visibility, scalar
  specular gain, and scalar lobe scale.
- Final deferred composition: I(x) = B(x) * S(x) + O(x).

The active model path is:

    train.py
      -> GaussianModel2DGSAdapter
      -> _NativeTextureAdapter
      -> render_2dgs_texture_deferred
      -> gaussian_renderer/texture_branch.py

The local-grid camera pass uses submodules/surfel-texture-deferred. The
light-space local-visibility pass uses
submodules/diff-surfel-rasterization-shadow. That shadow extension has no
autograd backward; do not claim that local visibility back-propagates through
the light-space pass to geometry or opacity.

## Formal Baseline Configuration

The formal entry point is:

    scripts/official_compare/run_microtexgs.sh

Current baseline:

- total iterations: 100k;
- fixed local resolution: 4x4;
- local grid active from iteration 0;
- Gaussian densification through 80k;
- effect mode: uvshadow_specular_lobe;
- camera and point-light optimization enabled;
- local normal disabled;
- texture alpha disabled in render and shadow;
- shadow-spatial resolution and hole filling disabled.

Scene-specific HDR/background and GS3-derived transport parameters are defined
in scripts/official_compare/common.sh. Do not infer paper settings from old
commands or copied repositories.

## Reverse Densification

The formal entry point is:

    scripts/reverse_densification/run_scene_rtd_from80k.sh

It resumes a fixed-4x4 80k checkpoint, compacts at 85k, 90k, and 95k, and
recovers through 100k. Resolution changes one level at a time (4->3->2->1).
Transition thresholds are 0.65, 0.45, and 0.30; each event changes at most 25%
of eligible Gaussians. Diffuse, visibility, and specular errors have weight 1;
local normal has weight 0. The joint score is their maximum.

## Evaluation

- Test images only are scored; valid trajectories have no GT.
- Render at native resolution with the training background.
- HDR/EXR predictions and GT use the same gamma/display transform.
- Use saved optimized camera/light states for GS3 and MicroTexGS.
- Compute per-frame PSNR, SSIM, and LPIPS with
  scripts/evaluation/unified_image_metrics.py, then average per scene.

The paper benchmark contains LightStage, NRHints, and Synthetic, four scenes
each. RenderCapture is not part of the current main table.

## Inherited Versus New

Inherited: 2DGS geometry/densification, GS3-style ASG and neural residual
backbone, point-light shadow rendering, and deferred B*S+O composition.

New mainline: factorized local transport, texture-aware local visibility and
camera sampling, packed variable-resolution local grids, and
transport-constrained reverse densification.

MicroTexGS_shadow_spatial_cuda is a separate experiment. Local normal,
texture-alpha, shadow-spatial, hole-fill, RMD, and TOR paths are not the
default paper method unless an experiment explicitly enables them.

## Reading Rules

1. Use this repository and the two formal shell entry points above.
2. Do not use historical _handoff files to decide method behavior.
3. For a method question, inspect the active call chain before searching
   alternative renderers or old experiments.
4. For a metric question, read only the named result manifest/CSV; never
   recursively enumerate rendered frames.
5. If a paper statement conflicts with the formal runner, the runner and its
   saved command manifest determine the experiment.
