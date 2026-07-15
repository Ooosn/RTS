#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/ssdwork/liuhaohan/RTS}"
GS3_RELEASE_DIR="${GS3_RELEASE_DIR:-${ROOT}/GS3_release}"
MICROTEX_DIR="${MICROTEX_DIR:-${ROOT}/MicroTexGS}"
RNG_DIR="${RNG_DIR:-${ROOT}/RNG_release_aligned}"

GSRELIGHT_ROOT="${GSRELIGHT_ROOT:-/ssdwork/liuhaohan/datasets/gsrelight}"
NRHINTS_ROOT="${NRHINTS_ROOT:-}"
RUN_ROOT="${RUN_ROOT:-/ssdwork/liuhaohan/outputs/compare_runs/run_$(date +%Y%m%d_%H%M%S)}"
SCENE="${SCENE:?SCENE must be set, e.g. LightStage/Container or NRHints/Pikachu}"
PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VIEW_NUM="${VIEW_NUM:-2000}"
ALIGN_TRAIN_VIEWS="${ALIGN_TRAIN_VIEWS:-1}"
DENSIFY_UNTIL="${DENSIFY_UNTIL:-80000}"

export CUDA_VISIBLE_DEVICES
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:/opt/conda/envs/gs/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

BASE_TEST_ITERS=(2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000)
BASE_SAVE_ITERS=(7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000)
BASE_CKPT_ITERS=(7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000)

if [[ "${SMOKE:-0}" == "1" ]]; then
  ITERATIONS=20
else
  ITERATIONS="${ITERATIONS:-100000}"
fi

filter_iters() {
  local max_iter="$1"
  shift
  local value
  local out=()
  for value in "$@"; do
    if (( value <= max_iter )); then
      out+=("${value}")
    fi
  done
  if [[ "${#out[@]}" -eq 0 || "${out[-1]}" != "${max_iter}" ]]; then
    out+=("${max_iter}")
  fi
  printf '%s\n' "${out[@]}"
}

mapfile -t TEST_ITERS < <(filter_iters "${ITERATIONS}" "${BASE_TEST_ITERS[@]}")
mapfile -t SAVE_ITERS < <(filter_iters "${ITERATIONS}" "${BASE_SAVE_ITERS[@]}")
mapfile -t CKPT_ITERS < <(filter_iters "${ITERATIONS}" "${BASE_CKPT_ITERS[@]}")

log() {
  echo "[$(date '+%F %T')] $*"
}

run_logged() {
  local workdir="$1"
  local log_file="$2"
  shift 2
  mkdir -p "$(dirname "${log_file}")"
  log "WORKDIR ${workdir}" | tee "${log_file}"
  printf 'CMD:' | tee -a "${log_file}"
  printf ' %q' "$@" | tee -a "${log_file}"
  printf '\n' | tee -a "${log_file}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  (
    cd "${workdir}"
    "$@"
  ) 2>&1 | tee -a "${log_file}"
  local status=${PIPESTATUS[0]}
  if [[ ${status} -ne 0 ]]; then
    echo "FAILED status=${status}: ${log_file}" >&2
    return "${status}"
  fi
}

scene_data_path() {
  local key="$1"
  local group="${key%%/*}"
  local scene="${key#*/}"
  if [[ "${group}" == "NRHints" ]]; then
    if [[ -z "${NRHINTS_ROOT}" ]]; then
      echo "NRHINTS_ROOT must point to the original NRHints dataset root." >&2
      return 2
    fi
    if [[ "${scene}" == "Cup-Fabric" ]]; then
      scene="CupFabric"
    fi
    local base="${NRHINTS_ROOT}/${scene}"
    if [[ -f "${base}/transforms_train.json" ]]; then
      printf '%s\n' "${base}"
    elif [[ -f "${base}/${scene}/transforms_train.json" ]]; then
      printf '%s\n' "${base}/${scene}"
    else
      echo "Cannot find NRHints dataset for ${key} under ${NRHINTS_ROOT}" >&2
      return 2
    fi
    return 0
  fi
  printf '%s\n' "${GSRELIGHT_ROOT}/${group}/${scene}"
}

assert_dataset() {
  local path="$1"
  if [[ ! -f "${path}/transforms_train.json" ]]; then
    echo "Missing transforms_train.json: ${path}" >&2
    return 2
  fi
}

add_schedule_args() {
  ARGS+=(--iterations "${ITERATIONS}")
  ARGS+=(--test_iterations "${TEST_ITERS[@]}")
  ARGS+=(--save_iterations "${SAVE_ITERS[@]}")
  ARGS+=(--checkpoint_iterations "${CKPT_ITERS[@]}")
}

add_transport_args() {
  local position_lr_steps="$1"
  local densify_until="$2"
  ARGS+=(
    --view_num "${VIEW_NUM}"
    --asg_freeze_step 22000
    --spcular_freeze_step 9000
    --fit_linear_step 7000
    --asg_lr_freeze_step 40000
    --asg_lr_max_steps 50000
    --asg_lr_init 0.01
    --asg_lr_final 0.0001
    --local_q_lr_freeze_step 40000
    --local_q_lr_init 0.01
    --local_q_lr_final 0.0001
    --local_q_lr_max_steps 50000
    --neural_phasefunc_lr_init 0.001
    --neural_phasefunc_lr_final 0.00001
    --freeze_phasefunc_steps 50000
    --neural_phasefunc_lr_max_steps 50000
    --position_lr_max_steps "${position_lr_steps}"
    --densify_until_iter "${densify_until}"
  )
  add_schedule_args
  ARGS+=(--unfreeze_iterations 5000 --use_nerual_phasefunc --eval)
}

require_rng_train_view_alignment() {
  local rng_train_views="$1"
  if [[ "${ALIGN_TRAIN_VIEWS}" != "1" ]]; then
    return 0
  fi
  if [[ "${rng_train_views}" != "${VIEW_NUM}" ]]; then
    echo "RNG_MAX_TRAINING_IMAGES (${rng_train_views}) must match VIEW_NUM (${VIEW_NUM}) for aligned comparison." >&2
    echo "Set ALIGN_TRAIN_VIEWS=0 only for an intentional official-recipe probe." >&2
    return 2
  fi
}

scene_args() {
  local key="$1"
  ARGS=()
  case "${key}" in
    LightStage/Container) ARGS+=(--hdr --data_device cpu); add_transport_args 70000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt --densify_grad_threshold 0.0001) ;;
    LightStage/Boot) ARGS+=(--hdr --data_device cpu); add_transport_args 70000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt --densify_grad_threshold 0.00006) ;;
    LightStage/Fox) ARGS+=(--hdr --data_device cpu); add_transport_args 70000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt --densify_grad_threshold 0.0001) ;;
    LightStage/Nefertiti) ARGS+=(--hdr --data_device cpu); add_transport_args 70000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt --densify_grad_threshold 0.0001) ;;
    NRHints/Pikachu) ARGS+=(--data_device cpu); add_transport_args 70000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt --densify_grad_threshold 0.00015) ;;
    NRHints/Fish) ARGS+=(--data_device cpu); add_transport_args 70000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt --densify_grad_threshold 0.00012) ;;
    NRHints/Cat) ARGS+=(--data_device cpu); add_transport_args 70000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt --densify_grad_threshold 0.0002) ;;
    NRHints/CupFabric|NRHints/Cup-Fabric) ARGS+=(--data_device cpu); add_transport_args 70000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt --densify_grad_threshold 0.0001) ;;
    Synthetic/Hotdog) ARGS+=(--hdr --white_background); add_transport_args 50000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt) ;;
    Synthetic/FurBall) ARGS+=(--hdr --white_background); add_transport_args 50000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt) ;;
    Synthetic/AnisoMetal) ARGS+=(--hdr --white_background); add_transport_args 50000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt) ;;
    Synthetic/Drums)
      ARGS+=(--hdr --white_background)
      add_transport_args 50000 "${DENSIFY_UNTIL}"
      for i in "${!ARGS[@]}"; do
        [[ "${ARGS[$i]}" == "--asg_lr_freeze_step" ]] && ARGS[$((i + 1))]=30000
        [[ "${ARGS[$i]}" == "--asg_lr_max_steps" ]] && ARGS[$((i + 1))]=70000
        [[ "${ARGS[$i]}" == "--asg_lr_final" ]] && ARGS[$((i + 1))]=0.0008
      done
      ARGS+=(--cam_opt --pl_opt --densify_grad_threshold 0.00013)
      ;;
    RenderCapture/MaterialBalls) ARGS+=(--hdr --white_background); add_transport_args 50000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt) ;;
    RenderCapture/Fabric) ARGS+=(--hdr --white_background); add_transport_args 50000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt) ;;
    RenderCapture/Cup) ARGS+=(--hdr --white_background); add_transport_args 50000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt) ;;
    RenderCapture/Tower) ARGS+=(--hdr --white_background); add_transport_args 50000 "${DENSIFY_UNTIL}"; ARGS+=(--cam_opt --pl_opt) ;;
    *) echo "Unknown scene: ${key}" >&2; return 2 ;;
  esac
}

safe_scene_name() {
  printf '%s\n' "$1" | tr '/' '_'
}
