#!/usr/bin/env bash
set -euo pipefail

export MICROTEX_DIR=/ssdwork/liuhaohan/RTS/experiments/20260713_1545/src
export NRHINTS_ROOT=/ssdwork/liuhaohan/datasets/nrhints_original/Real
export GSRELIGHT_ROOT=/ssdwork/liuhaohan/datasets/gsrelight
export BASELINE_ROOT=/ssdwork/liuhaohan/outputs/compare_runs/20260616154502_a/microtexgs
export RUN_ROOT=/ssdwork/liuhaohan/outputs/experiments/20260713_1545_b
export PYTHON=/opt/conda/envs/gs/bin/python
export CUDA_VISIBLE_DEVICES=0
export SCENES="NRHints/CupFabric Synthetic/FurBall LightStage/Boot"

hostname
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
bash "${MICROTEX_DIR}/scripts/reverse_densification/run_batch_rtd_from80k.sh"
