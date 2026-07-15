#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DIR="$(cd "${SCRIPT_DIR}/../official_compare" && pwd)"

ITERATIONS="${ITERATIONS:-100000}"
export ITERATIONS
source "${OFFICIAL_DIR}/common.sh"

START_CHECKPOINT="${START_CHECKPOINT:?START_CHECKPOINT must point to chkpnt80000.pth}"
SRC="$(scene_data_path "${SCENE}")"
assert_dataset "${SRC}"
scene_args "${SCENE}"

if [[ ! -f "${START_CHECKPOINT}" ]]; then
  echo "Missing start checkpoint: ${START_CHECKPOINT}" >&2
  exit 2
fi

COMPACT_FROM="${COMPACT_FROM:-85000}"
COMPACT_UNTIL="${COMPACT_UNTIL:-95000}"
COMPACT_INTERVAL="${COMPACT_INTERVAL:-5000}"
THRESHOLD_4TO3="${THRESHOLD_4TO3:-0.65}"
THRESHOLD_3TO2="${THRESHOLD_3TO2:-0.45}"
THRESHOLD_2TO1="${THRESHOLD_2TO1:-0.30}"
MAX_FRACTION="${MAX_FRACTION:-0.25}"
SHADOW_SAMPLE_INTERVAL="${SHADOW_SAMPLE_INTERVAL:-1}"
TEXTURE_EFFECT_MODE="${TEXTURE_EFFECT_MODE:-uvshadow_specular_lobe}"
RTD_KD_WEIGHT="${RTD_KD_WEIGHT:-1.0}"
RTD_SHADOW_WEIGHT="${RTD_SHADOW_WEIGHT:-1.0}"
RTD_SPECULAR_WEIGHT="${RTD_SPECULAR_WEIGHT:-1.0}"
RTD_NORMAL_WEIGHT="${RTD_NORMAL_WEIGHT:-0.0}"

OUT="${RUN_ROOT}/microtexgs/${SCENE}"
LOG="${RUN_ROOT}/logs/microtexgs/$(safe_scene_name "${SCENE}").log"
MANIFEST="${RUN_ROOT}/manifests/$(safe_scene_name "${SCENE}").txt"
mkdir -p "$(dirname "${MANIFEST}")"

{
  echo "scene=${SCENE}"
  echo "source=${SRC}"
  echo "start_checkpoint=${START_CHECKPOINT}"
  echo "output=${OUT}"
  echo "iterations=${ITERATIONS}"
  echo "compact_from=${COMPACT_FROM}"
  echo "compact_until=${COMPACT_UNTIL}"
  echo "compact_interval=${COMPACT_INTERVAL}"
  echo "threshold_4to3=${THRESHOLD_4TO3}"
  echo "threshold_3to2=${THRESHOLD_3TO2}"
  echo "threshold_2to1=${THRESHOLD_2TO1}"
  echo "max_fraction=${MAX_FRACTION}"
  echo "shadow_sample_interval=${SHADOW_SAMPLE_INTERVAL}"
  echo "texture_effect_mode=${TEXTURE_EFFECT_MODE}"
  echo "rtd_kd_weight=${RTD_KD_WEIGHT}"
  echo "rtd_shadow_weight=${RTD_SHADOW_WEIGHT}"
  echo "rtd_specular_weight=${RTD_SPECULAR_WEIGHT}"
  echo "rtd_normal_weight=${RTD_NORMAL_WEIGHT}"
  echo "git_commit=$(git -C "${MICROTEX_DIR}" rev-parse HEAD 2>/dev/null || printf 'unversioned')"
} > "${MANIFEST}"

MICRO_PYTHONPATH="${MICROTEX_DIR}:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization_light:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization_hgs:${MICROTEX_DIR}/submodules/v_3dgs:${MICROTEX_DIR}/submodules/v_3dgs_ortho:${MICROTEX_DIR}/submodules/diff-surfel-rasterization:${MICROTEX_DIR}/submodules/diff-surfel-rasterization-shadow:${MICROTEX_DIR}/submodules/surfel-texture:${MICROTEX_DIR}/submodules/surfel-texture-deferred:${MICROTEX_DIR}/submodules/simple-knn:${MICROTEX_DIR}/submodules/gsplat-1.1.1"

run_logged "${MICROTEX_DIR}" "${LOG}" \
  env PYTHONPATH="${MICRO_PYTHONPATH}" "${PYTHON}" train.py \
    -s "${SRC}" \
    -m "${OUT}" \
    --start_checkpoint "${START_CHECKPOINT}" \
    "${ARGS[@]}" \
    --rasterizer 2dgs \
    --sh_degree 0 \
    --resolution 1 \
    --use_textures \
    --texture_resolution 4 \
    --texture_dynamic_resolution \
    --texture_min_resolution 1 \
    --texture_max_resolution 4 \
    --texture_effect_mode "${TEXTURE_EFFECT_MODE}" \
    --texture_start_iter 0 \
    --texture_specular_lr_scale 1.0 \
    --texture_normal_lr_scale 1.0 \
    --texture_rtd_enabled \
    --texture_rtd_compress_from_iter "${COMPACT_FROM}" \
    --texture_rtd_compress_until_iter "${COMPACT_UNTIL}" \
    --texture_rtd_compress_interval "${COMPACT_INTERVAL}" \
    --texture_rtd_min_resolution 1 \
    --texture_rtd_step_mode step \
    --texture_rtd_error_threshold "${THRESHOLD_4TO3}" \
    --texture_rtd_error_threshold_4to3 "${THRESHOLD_4TO3}" \
    --texture_rtd_error_threshold_3to2 "${THRESHOLD_3TO2}" \
    --texture_rtd_error_threshold_2to1 "${THRESHOLD_2TO1}" \
    --texture_rtd_kd_weight "${RTD_KD_WEIGHT}" \
    --texture_rtd_shadow_weight "${RTD_SHADOW_WEIGHT}" \
    --texture_rtd_specular_weight "${RTD_SPECULAR_WEIGHT}" \
    --texture_rtd_normal_weight "${RTD_NORMAL_WEIGHT}" \
    --texture_rtd_max_fraction "${MAX_FRACTION}" \
    --texture_rtd_shadow_sample_interval "${SHADOW_SAMPLE_INTERVAL}"
