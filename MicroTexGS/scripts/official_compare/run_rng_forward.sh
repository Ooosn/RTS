#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

SRC="$(scene_data_path "${SCENE}")"
assert_dataset "${SRC}"

RNG_PYTHONPATH="${RNG_DIR}:${RNG_DIR}/submodules/diff-gaussian-rasterization:${RNG_DIR}/submodules/simple-knn"
OUT="${RUN_ROOT}/rng_forward100k/${SCENE}"
LOG="${RUN_ROOT}/logs/rng_forward100k/$(safe_scene_name "${SCENE}").log"
RNG_MAX_RESO="${RNG_MAX_RESO:-0}"
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
export HDR_GAMMA="${HDR_GAMMA:-1}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  RNG_ITERS=20
  RNG_TEST_ITERS=(20)
  RNG_SAVE_ITERS=(20)
else
  RNG_ITERS="${RNG_FORWARD_ITERS:-100000}"
  RNG_TEST_ITERS=()
  RNG_SAVE_ITERS=()
  for ((iter = 10000; iter < RNG_ITERS; iter += 10000)); do
    RNG_TEST_ITERS+=("${iter}")
    RNG_SAVE_ITERS+=("${iter}")
  done
  RNG_TEST_ITERS+=("${RNG_ITERS}")
  RNG_SAVE_ITERS+=("${RNG_ITERS}")
fi

run_logged "${RNG_DIR}" "${LOG}" \
  env HDR_GAMMA="${HDR_GAMMA}" PYTHONPATH="${RNG_PYTHONPATH}" "${PYTHON}" train.py \
    -s "${SRC}" \
    -m "${OUT}" \
    --iterations "${RNG_ITERS}" \
    --densify_until_iter "${RNG_ITERS}" \
    --test_iterations "${RNG_TEST_ITERS[@]}" \
    --save_iterations "${RNG_SAVE_ITERS[@]}" \
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
