#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/ssdwork/liuhaohan/RTS}"
GS3_RELEASE_DIR="${GS3_RELEASE_DIR:-${ROOT}/GS3_release}"
PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"

cd "${GS3_RELEASE_DIR}/submodules/diff-gaussian-rasterization_light"
"${PYTHON}" setup.py build_ext --inplace
cd "${GS3_RELEASE_DIR}/submodules/simple-knn"
"${PYTHON}" setup.py build_ext --inplace
