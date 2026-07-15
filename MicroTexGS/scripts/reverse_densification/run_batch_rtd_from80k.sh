#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCENES_DEFAULT=(
  "NRHints/Fish"
  "NRHints/CupFabric"
  "LightStage/Container"
  "Synthetic/AnisoMetal"
  "Synthetic/FurBall"
  "LightStage/Boot"
)

if [[ -n "${SCENES:-}" ]]; then
  read -r -a SCENES_TO_RUN <<< "${SCENES}"
else
  SCENES_TO_RUN=("${SCENES_DEFAULT[@]}")
fi

BASELINE_ROOT="${BASELINE_ROOT:-/ssdwork/liuhaohan/outputs/compare_runs/20260616154502_a/microtexgs}"
RUN_ROOT="${RUN_ROOT:-/ssdwork/liuhaohan/outputs/experiments/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_ROOT}"

{
  echo "run_root=${RUN_ROOT}"
  echo "baseline_root=${BASELINE_ROOT}"
  echo "scenes=${SCENES_TO_RUN[*]}"
  echo "host=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-0}"
  echo "started=$(date -Iseconds)"
} > "${RUN_ROOT}/batch_manifest.txt"

for scene in "${SCENES_TO_RUN[@]}"; do
  if [[ "${scene}" == RenderCapture/* ]]; then
    echo "RenderCapture scenes are excluded from this experiment: ${scene}" >&2
    exit 2
  fi
  checkpoint="${BASELINE_ROOT}/${scene}/chkpnt80000.pth"
  echo "========== ${scene} =========="
  SCENE="${scene}" \
  START_CHECKPOINT="${checkpoint}" \
  RUN_ROOT="${RUN_ROOT}" \
    bash "${SCRIPT_DIR}/run_scene_rtd_from80k.sh"
done

echo "finished=$(date -Iseconds)" >> "${RUN_ROOT}/batch_manifest.txt"
