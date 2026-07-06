#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/ssdwork/liuhaohan/RTS_shadow_spatial_20260706}"
MICROTEX_DIR="${MICROTEX_DIR:-${ROOT}/MicroTexGS}"
GSRELIGHT_ROOT="${GSRELIGHT_ROOT:-/ssdwork/liuhaohan/datasets/gsrelight}"
NRHINTS_ROOT="${NRHINTS_ROOT:-/ssdwork/liuhaohan/datasets/nrhints_original/Real}"
PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"
RUN_ROOT="${RUN_ROOT:-/ssdwork/liuhaohan/outputs/shadow_spatial/pikachu_40k_$(date +%Y%m%d_%H%M%S)}"

export ROOT MICROTEX_DIR GSRELIGHT_ROOT NRHINTS_ROOT PYTHON RUN_ROOT
export SCENE="${SCENE:-NRHints/Pikachu}"
export ITERATIONS="${ITERATIONS:-40000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SHADOW_HOLE_FILL="${SHADOW_HOLE_FILL:-local}"
export SHADOW_TEXEL_SIZE="${SHADOW_TEXEL_SIZE:-0.01}"
export SHADOW_MIN_RES="${SHADOW_MIN_RES:-4}"
export SHADOW_MAX_RES="${SHADOW_MAX_RES:-32}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DIR="${MICROTEX_DIR}/scripts/official_compare"
SPATIAL_DIR="${MICROTEX_DIR}/scripts/shadow_spatial"
SUMMARY="${RUN_ROOT}/shadow_spatial_summary.tsv"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/render_logs"

echo "[job] start $(date -Is)"
echo "[job] root=${ROOT}"
echo "[job] microtex=${MICROTEX_DIR}"
echo "[job] run_root=${RUN_ROOT}"
echo "[job] scene=${SCENE}"
echo "[job] iterations=${ITERATIONS}"
echo "[job] shadow_texel_size=${SHADOW_TEXEL_SIZE} range=${SHADOW_MIN_RES}-${SHADOW_MAX_RES} fill=${SHADOW_HOLE_FILL}"
nvidia-smi -L

echo "[job] preflight"
bash "${OFFICIAL_DIR}/preflight_alignment.sh" 2>&1 | tee "${RUN_ROOT}/logs/preflight_alignment.log"

echo "[job] baseline train start $(date -Is)"
bash "${OFFICIAL_DIR}/run_microtexgs.sh"
echo "[job] baseline train done $(date -Is)"

echo "[job] spatial train start $(date -Is)"
bash "${SPATIAL_DIR}/run_microtexgs_shadow_spatial.sh"
echo "[job] spatial train done $(date -Is)"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[job] dry-run done; skip render and summary"
  exit 0
fi

micro_pythonpath="${MICROTEX_DIR}:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization_light:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization_hgs:${MICROTEX_DIR}/submodules/v_3dgs:${MICROTEX_DIR}/submodules/v_3dgs_ortho:${MICROTEX_DIR}/submodules/diff-surfel-rasterization:${MICROTEX_DIR}/submodules/diff-surfel-rasterization-shadow:${MICROTEX_DIR}/submodules/surfel-texture:${MICROTEX_DIR}/submodules/surfel-texture-deferred:${MICROTEX_DIR}/submodules/simple-knn:${MICROTEX_DIR}/submodules/gsplat-1.1.1"

has_split_frames() {
  local split="$1"
  "${PYTHON}" - "${split}" <<'PY'
import json
import os
import sys
from pathlib import Path

split = sys.argv[1]
scene = os.environ["SCENE"]
roots = [
    Path(os.environ["NRHINTS_ROOT"]) / scene.split("/")[-1],
    Path(os.environ["GSRELIGHT_ROOT"]) / scene,
]
candidates = {
    "valid": ["transforms_val.json", "transforms_valid.json"],
    "test": ["transforms_test.json"],
    "train": ["transforms_train.json"],
}[split]
for root in roots:
    for name in candidates:
        path = root / name
        if not path.is_file():
            continue
        try:
            frames = json.loads(path.read_text()).get("frames", [])
        except Exception:
            frames = []
        if len(frames) > 0:
            print(len(frames))
            raise SystemExit(0)
raise SystemExit(1)
PY
}

render_one() {
  local name="$1"
  local model_dir="$2"
  local log_prefix="${RUN_ROOT}/render_logs/${name}"
  echo "[job] render ${name} test $(date -Is)"
  (
    cd "${MICROTEX_DIR}"
    env PYTHONPATH="${micro_pythonpath}" "${PYTHON}" render.py \
      -m "${model_dir}" \
      --load_iteration "${ITERATIONS}" \
      --skip_train \
      --write_images \
      --force_save \
      --opt_pose
  ) 2>&1 | tee "${log_prefix}_test.log"

  if has_split_frames valid >/dev/null; then
    echo "[job] render ${name} valid $(date -Is)"
    (
      cd "${MICROTEX_DIR}"
      env PYTHONPATH="${micro_pythonpath}" "${PYTHON}" render.py \
        -m "${model_dir}" \
        --load_iteration "${ITERATIONS}" \
        --valid \
        --skip_train \
        --skip_test \
        --write_images \
        --force_save \
        --opt_pose
    ) 2>&1 | tee "${log_prefix}_valid.log"
  else
    echo "[job] skip ${name} valid: no valid frames in source dataset"
  fi
}

render_one "baseline" "${RUN_ROOT}/microtexgs/${SCENE}"
render_one "spatial" "${RUN_ROOT}/microtexgs_shadow_spatial/${SCENE}"

"${PYTHON}" - <<'PY' "${RUN_ROOT}" "${SUMMARY}"
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
summary = Path(sys.argv[2])
rows = [("method", "iteration", "split", "psnr", "ssim", "lpips", "l1", "num_views")]
for method, rel in [
    ("baseline", "microtexgs/NRHints/Pikachu/convergence/eval_metrics.jsonl"),
    ("spatial", "microtexgs_shadow_spatial/NRHints/Pikachu/convergence/eval_metrics.jsonl"),
]:
    path = run_root / rel
    if not path.is_file():
        rows.append((method, "missing", "", "", "", "", "", ""))
        continue
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rows.append((
            method,
            str(rec.get("iteration", "")),
            str(rec.get("split", "")),
            f'{rec.get("psnr", 0):.6f}',
            f'{rec.get("ssim", 0):.6f}',
            f'{rec.get("lpips", 0):.6f}',
            f'{rec.get("l1", 0):.6f}',
            str(rec.get("num_views", "")),
        ))
summary.parent.mkdir(parents=True, exist_ok=True)
summary.write_text("\n".join("\t".join(r) for r in rows) + "\n")
print(summary)
PY

echo "[job] summary"
cat "${SUMMARY}"
echo "[job] done $(date -Is)"
