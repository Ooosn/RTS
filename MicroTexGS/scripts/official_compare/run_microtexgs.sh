#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

SRC="$(scene_data_path "${SCENE}")"
assert_dataset "${SRC}"
scene_args "${SCENE}"

TEXTURE_START_ITER="${TEXTURE_START_ITER:-0}"
TEXTURE_RESOLUTION="${TEXTURE_RESOLUTION:-4}"
TEXTURE_EFFECT_MODE="${TEXTURE_EFFECT_MODE:-uvshadow_specular_lobe}"

MICRO_PYTHONPATH="${MICROTEX_DIR}:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization_light:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization_hgs:${MICROTEX_DIR}/submodules/v_3dgs:${MICROTEX_DIR}/submodules/v_3dgs_ortho:${MICROTEX_DIR}/submodules/diff-surfel-rasterization:${MICROTEX_DIR}/submodules/diff-surfel-rasterization-shadow:${MICROTEX_DIR}/submodules/surfel-texture:${MICROTEX_DIR}/submodules/surfel-texture-deferred:${MICROTEX_DIR}/submodules/simple-knn:${MICROTEX_DIR}/submodules/gsplat-1.1.1"
OUT="${RUN_ROOT}/microtexgs/${SCENE}"
LOG="${RUN_ROOT}/logs/microtexgs/$(safe_scene_name "${SCENE}").log"

run_logged "${MICROTEX_DIR}" "${LOG}" \
  env PYTHONPATH="${MICRO_PYTHONPATH}" "${PYTHON}" train.py \
    -s "${SRC}" \
    -m "${OUT}" \
    "${ARGS[@]}" \
    --rasterizer 2dgs \
    --sh_degree 0 \
    --resolution 1 \
    --use_textures \
    --texture_resolution "${TEXTURE_RESOLUTION}" \
    --texture_effect_mode "${TEXTURE_EFFECT_MODE}" \
    --texture_start_iter "${TEXTURE_START_ITER}" \
    --texture_specular_lr_scale 1.0 \
    --texture_normal_lr_scale 1.0
