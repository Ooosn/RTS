#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

SRC="$(scene_data_path "${SCENE}")"
assert_dataset "${SRC}"
scene_args "${SCENE}"

GS3_PYTHONPATH="${GS3_RELEASE_DIR}:${GS3_RELEASE_DIR}/submodules/diff-gaussian-rasterization_light:${GS3_RELEASE_DIR}/submodules/simple-knn:${MICROTEX_DIR}/submodules/gsplat-1.1.1"
OUT="${RUN_ROOT}/gs3_release/${SCENE}"
LOG="${RUN_ROOT}/logs/gs3_release/$(safe_scene_name "${SCENE}").log"

run_logged "${GS3_RELEASE_DIR}" "${LOG}" \
  env PYTHONPATH="${GS3_PYTHONPATH}" "${PYTHON}" train.py \
    -s "${SRC}" \
    -m "${OUT}" \
    "${ARGS[@]}"
