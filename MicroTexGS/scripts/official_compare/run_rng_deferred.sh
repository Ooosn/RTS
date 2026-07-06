#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

SRC="$(scene_data_path "${SCENE}")"
assert_dataset "${SRC}"

RNG_PYTHONPATH="${RNG_DIR}:${RNG_DIR}/submodules/diff-gaussian-rasterization:${RNG_DIR}/submodules/simple-knn"
OUT="${RUN_ROOT}/rng_deferred/${SCENE}"
LOG1="${RUN_ROOT}/logs/rng_deferred/$(safe_scene_name "${SCENE}")_stage1.log"
LOG2="${RUN_ROOT}/logs/rng_deferred/$(safe_scene_name "${SCENE}")_stage2.log"
RNG_MAX_RESO="${RNG_MAX_RESO:-0}"
RNG_DEPTH_MLP_MODIFIER="${RNG_DEPTH_MLP_MODIFIER:-1.0}"
RNG_MAX_TRAINING_IMAGES="${RNG_MAX_TRAINING_IMAGES:-${VIEW_NUM}}"
RNG_DATA_DEVICE="${RNG_DATA_DEVICE:-cpu}"
require_rng_train_view_alignment "${RNG_MAX_TRAINING_IMAGES}"
RNG_LESS_ARGS=()
if [[ -n "${RNG_LESS:-}" ]]; then
  RNG_LESS_ARGS+=(--less "${RNG_LESS}")
fi
RNG_HDR_ARGS=()
RNG_BG_ARGS=()
case "${SCENE}" in
  LightStage/*|Synthetic/*|RenderCapture/*) RNG_HDR_ARGS+=(--hdr) ;;
esac
case "${SCENE}" in
  Synthetic/*|RenderCapture/*) RNG_BG_ARGS+=(--white_background) ;;
esac
export OPENCV_IO_ENABLE_OPENEXR=1

append_unique_iter() {
  local array_name="$1"
  local value="$2"
  local existing
  eval "existing=(\"\${${array_name}[@]:-}\")"
  for item in "${existing[@]}"; do
    [[ "${item}" == "${value}" ]] && return 0
  done
  eval "${array_name}+=(\"${value}\")"
}

if [[ "${SMOKE:-0}" == "1" ]]; then
  STAGE1_ITERS=20
  STAGE2_ITERS=25
  STAGE1_TEST=(20)
  STAGE2_TEST=(25)
  STAGE1_SAVE=(20)
  STAGE2_SAVE=(25)
else
  STAGE1_ITERS="${RNG_STAGE1_ITERS:-30000}"
  STAGE2_ITERS="${RNG_STAGE2_ITERS:-100000}"
  STAGE1_TEST=()
  STAGE1_SAVE=()
  if (( STAGE1_ITERS >= 7000 )); then
    append_unique_iter STAGE1_TEST 7000
    append_unique_iter STAGE1_SAVE 7000
  fi
  for ((iter = 10000; iter < STAGE1_ITERS; iter += 10000)); do
    append_unique_iter STAGE1_TEST "${iter}"
  done
  append_unique_iter STAGE1_TEST "${STAGE1_ITERS}"
  append_unique_iter STAGE1_SAVE "${STAGE1_ITERS}"

  STAGE2_TEST=()
  STAGE2_SAVE=("${STAGE2_ITERS}")
  for ((iter = 10000; iter < STAGE2_ITERS; iter += 10000)); do
    append_unique_iter STAGE2_TEST "${iter}"
  done
  append_unique_iter STAGE2_TEST "${STAGE2_ITERS}"
fi

run_logged "${RNG_DIR}" "${LOG1}" \
  env PYTHONPATH="${RNG_PYTHONPATH}" "${PYTHON}" train.py \
    -s "${SRC}" \
    -m "${OUT}" \
    --iterations "${STAGE1_ITERS}" \
    --test_iterations "${STAGE1_TEST[@]}" \
    --save_iterations "${STAGE1_SAVE[@]}" \
    --densify_until_iter "${STAGE1_ITERS}" \
    --eval \
    --json \
    --color_mlp \
    --in_channels 16 \
    --max_training_images "${RNG_MAX_TRAINING_IMAGES}" \
    --max_reso "${RNG_MAX_RESO}" \
    --data_device "${RNG_DATA_DEVICE}" \
    "${RNG_LESS_ARGS[@]}" \
    "${RNG_HDR_ARGS[@]}" \
    "${RNG_BG_ARGS[@]}" \
    --loss l1 \
    --cam_opt

CKPT="${OUT}/chkpnt${STAGE1_ITERS}.pth"
if [[ "${DRY_RUN}" != "1" && ! -f "${CKPT}" ]]; then
  echo "Missing RNG stage1 checkpoint: ${CKPT}" >&2
  exit 2
fi

run_logged "${RNG_DIR}" "${LOG2}" \
  env PYTHONPATH="${RNG_PYTHONPATH}" "${PYTHON}" train.py \
    -s "${SRC}" \
    -m "${OUT}" \
    --iterations "${STAGE2_ITERS}" \
    --test_iterations "${STAGE2_TEST[@]}" \
    --save_iterations "${STAGE2_SAVE[@]}" \
    --load_pc "${CKPT}" \
    --eval \
    --json \
    --color_mlp \
    --defer_shading \
    --in_channels 16 \
    --max_training_images "${RNG_MAX_TRAINING_IMAGES}" \
    --max_reso "${RNG_MAX_RESO}" \
    --data_device "${RNG_DATA_DEVICE}" \
    "${RNG_LESS_ARGS[@]}" \
    "${RNG_HDR_ARGS[@]}" \
    "${RNG_BG_ARGS[@]}" \
    --shadow_map \
    --shadow_grad \
    --depth_mlp \
    --depth_mlp_modifier "${RNG_DEPTH_MLP_MODIFIER}" \
    --encoding_levels_each 2 \
    --encoding_levels_shadow 8 \
    --crop_pc 1.0 \
    --loss l1 \
    --cam_opt
