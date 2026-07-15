#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICROTEX_DIR="${MICROTEX_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"
RUN_ROOTS="${RUN_ROOTS:?RUN_ROOTS must be a colon-separated list of completed RTD roots}"
SCENES="${SCENES:-NRHints/Fish,NRHints/CupFabric,LightStage/Container,Synthetic/AnisoMetal,Synthetic/FurBall,LightStage/Boot}"
EVAL_ROOT="${EVAL_ROOT:-/ssdwork/liuhaohan/outputs/experiments/20260713_1545_eval}"
BASELINE_ROOT="${BASELINE_ROOT:-/ssdwork/liuhaohan/outputs/compare_runs/20260616154502_a/microtexgs}"
INCLUDE_FIXED="${INCLUDE_FIXED:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:/opt/conda/envs/gs/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export PYTHONPATH="${MICROTEX_DIR}:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization_light:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization_hgs:${MICROTEX_DIR}/submodules/v_3dgs:${MICROTEX_DIR}/submodules/v_3dgs_ortho:${MICROTEX_DIR}/submodules/diff-surfel-rasterization:${MICROTEX_DIR}/submodules/diff-surfel-rasterization-shadow:${MICROTEX_DIR}/submodules/surfel-texture:${MICROTEX_DIR}/submodules/surfel-texture-deferred:${MICROTEX_DIR}/submodules/simple-knn:${MICROTEX_DIR}/submodules/gsplat-1.1.1"

mkdir -p "${EVAL_ROOT}/logs"
MANIFEST="${EVAL_ROOT}/manifest.jsonl"
: > "${MANIFEST}"

IFS=':' read -r -a ROOT_ARRAY <<< "${RUN_ROOTS}"
IFS=',' read -r -a SCENE_ARRAY <<< "${SCENES}"

find_model() {
  local scene="$1"
  local root candidate
  for root in "${ROOT_ARRAY[@]}"; do
    candidate="${root}/microtexgs/${scene}"
    if [[ -f "${candidate}/chkpnt100000.pth" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

render_and_register() {
  local scene="$1"
  local method="$2"
  local model="$3"
  local required
  for required in cfg_args chkpnt100000.pth point_cloud/iteration_100000/point_cloud.ply; do
    [[ -f "${model}/${required}" ]] || {
      echo "Missing ${model}/${required}" >&2
      exit 2
    }
  done

  local render_args=(
    "${PYTHON}" render.py
    -m "${model}"
    --load_iteration 100000
    --skip_train
    --write_images
    --force_save
    --opt_pose
  )
  case "${scene}" in
    LightStage/*|Synthetic/*) render_args+=(--hdr --gamma) ;;
  esac

  local log="${EVAL_ROOT}/logs/${scene//\//_}_${method}.log"
  printf '[render] %s %s\n' "${scene}" "${method}" | tee "${log}"
  printf 'CMD:' | tee -a "${log}"
  printf ' %q' "${render_args[@]}" | tee -a "${log}"
  printf '\n' | tee -a "${log}"
  (
    cd "${MICROTEX_DIR}"
    "${render_args[@]}"
  ) 2>&1 | tee -a "${log}"

  local renders="${model}/test/ours_100000/renders/volume_final_image"
  local gt="${model}/test/ours_100000/gt"
  local render_count gt_count
  render_count="$(find "${renders}" -maxdepth 1 -type f -name '*.png' | wc -l)"
  gt_count="$(find "${gt}" -maxdepth 1 -type f -name '*.png' | wc -l)"
  if [[ "${render_count}" -le 0 || "${render_count}" -ne "${gt_count}" ]]; then
    echo "Invalid image counts for ${scene} ${method}: renders=${render_count}, gt=${gt_count}" >&2
    exit 2
  fi

  "${PYTHON}" - "${MANIFEST}" "${scene}" "${method}" "${renders}" "${gt}" <<'PY'
import json
import sys

manifest, scene, method, renders, gt = sys.argv[1:]
with open(manifest, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "scene": scene,
        "method": method,
        "renders": renders,
        "gt": gt,
    }) + "\n")
PY
}

for scene in "${SCENE_ARRAY[@]}"; do
  if [[ "${scene}" == RenderCapture/* ]]; then
    echo "RenderCapture is excluded from this experiment: ${scene}" >&2
    exit 2
  fi
  model="$(find_model "${scene}")" || {
    echo "No completed RTD checkpoint found for ${scene}" >&2
    exit 2
  }
  if [[ "${INCLUDE_FIXED}" == "1" ]]; then
    render_and_register "${scene}" "MicroTexGS-fixed" "${BASELINE_ROOT}/${scene}"
  fi
  render_and_register "${scene}" "MicroTexGS-RTD" "${model}"
  if [[ "${INCLUDE_FIXED}" == "1" ]]; then
    fixed_gt="${BASELINE_ROOT}/${scene}/test/ours_100000/gt"
    compact_gt="${model}/test/ours_100000/gt"
    if ! diff -u \
      <(cd "${fixed_gt}" && sha256sum ./*.png | sort -k2) \
      <(cd "${compact_gt}" && sha256sum ./*.png | sort -k2) \
      > "${EVAL_ROOT}/logs/${scene//\//_}_gt_hash.diff"; then
      echo "Fixed and compact GT frames differ for ${scene}" >&2
      exit 2
    fi
  fi
done

"${PYTHON}" "${MICROTEX_DIR}/scripts/evaluation/unified_image_metrics.py" \
  --manifest "${MANIFEST}" \
  --out "${EVAL_ROOT}/metrics" \
  --device cuda \
  --batch-size "${EVAL_BATCH_SIZE:-4}"
