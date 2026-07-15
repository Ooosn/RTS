# MicroTexGS / LoFT-GS

This repository contains the current factorized local-transport implementation
for relightable 2D Gaussian splatting.

## Formal Entry Points

- Baseline training: scripts/official_compare/run_microtexgs.sh
- Shared scene configuration: scripts/official_compare/common.sh
- Reverse densification: scripts/reverse_densification/run_scene_rtd_from80k.sh
- Unified image metrics: scripts/evaluation/unified_image_metrics.py

The current baseline uses a fixed 4x4 local grid from iteration 0, trains for
100k iterations, and stops Gaussian densification at 80k. Camera and
point-light optimization are enabled by the formal scene configuration.

The active local factors are diffuse, dynamic visibility, specular gain, and
lobe scale. Local normal, texture alpha, shadow-spatial resolution, and hole
filling are not enabled in the default method.

See AGENTS.md for the active code path, method boundaries, RTD schedule, and
evaluation contract.
