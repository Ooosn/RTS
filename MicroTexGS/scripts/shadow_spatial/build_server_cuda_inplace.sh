#!/usr/bin/env bash
set -euo pipefail

MICROTEX_DIR="${MICROTEX_DIR:-/ssdwork/liuhaohan/RTS_shadow_spatial_20260706/MicroTexGS}"
PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:/opt/conda/envs/gs/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

modules=(
  submodules/diff-gaussian-rasterization
  submodules/diff-gaussian-rasterization_light
  submodules/diff-gaussian-rasterization_hgs
  submodules/v_3dgs
  submodules/v_3dgs_ortho
  submodules/diff-surfel-rasterization
  submodules/diff-surfel-rasterization-shadow
  submodules/surfel-texture
  submodules/surfel-texture-deferred
  submodules/simple-knn
)

echo "[build] start $(date -Is)"
echo "[build] microtex=${MICROTEX_DIR}"
"${PYTHON}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY
nvcc --version || true

for rel in "${modules[@]}"; do
  if [[ ! -f "${MICROTEX_DIR}/${rel}/setup.py" ]]; then
    echo "[build] skip missing ${rel}"
    continue
  fi
  echo "[build] ${rel}"
  (
    cd "${MICROTEX_DIR}/${rel}"
    rm -rf build
    "${PYTHON}" setup.py build_ext --inplace
  )
done

echo "[build] import check"
cd "${MICROTEX_DIR}"
PYTHONPATH="${MICROTEX_DIR}:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization_light:${MICROTEX_DIR}/submodules/diff-gaussian-rasterization_hgs:${MICROTEX_DIR}/submodules/v_3dgs:${MICROTEX_DIR}/submodules/v_3dgs_ortho:${MICROTEX_DIR}/submodules/diff-surfel-rasterization:${MICROTEX_DIR}/submodules/diff-surfel-rasterization-shadow:${MICROTEX_DIR}/submodules/surfel-texture:${MICROTEX_DIR}/submodules/surfel-texture-deferred:${MICROTEX_DIR}/submodules/simple-knn:${MICROTEX_DIR}/submodules/gsplat-1.1.1" \
  "${PYTHON}" - <<'PY'
import gaussian_renderer
import scene
print("import_ok")
PY
echo "[build] done $(date -Is)"
