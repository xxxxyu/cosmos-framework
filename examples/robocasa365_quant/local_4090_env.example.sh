#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Example local RTX 4090 environment for direct-load replay validation.
# Source this file, then override paths for the target host as needed.

set -euo pipefail

export LOCAL_4090_ROOT="${LOCAL_4090_ROOT:-$HOME/cosmos_ws/local_4090_validation}"
export COSMOS_REPO="${COSMOS_REPO:-$HOME/cosmos_ws/cosmos-framework-product}"
export COSMOS_PYTHON="${COSMOS_PYTHON:-$LOCAL_4090_ROOT/envs/python313_pkg/bin/python}"

export QUANT_BUNDLE_DIR="${QUANT_BUNDLE_DIR:-$LOCAL_4090_ROOT/self_contained/attention_w8_v2}"
export REPLAY_CAPTURE_DIR="${REPLAY_CAPTURE_DIR:-$LOCAL_4090_ROOT/data/closefridge_action_parity_v1}"
export RUN_DIR="${RUN_DIR:-$LOCAL_4090_ROOT/logs/direct_replay_$(date +%Y%m%d_%H%M%S)}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-5577}"
export REPLAY_LIMIT="${REPLAY_LIMIT:-32}"
export SERVED_ACTION_STEPS="${SERVED_ACTION_STEPS:-8}"

# Optional package overlays from the validated local workspace.
export DINGXIN_SITE="${DINGXIN_SITE:-$LOCAL_4090_ROOT/envs/dingxin_py313_site}"
export BACKEND_PROBE_SITE="${BACKEND_PROBE_SITE:-$LOCAL_4090_ROOT/envs/py313_backend_probe_site}"
export CUDNN_HOME="${CUDNN_HOME:-$DINGXIN_SITE/nvidia/cudnn}"
export NVRTC_HOME="${NVRTC_HOME:-$DINGXIN_SITE/nvidia/cuda_nvrtc}"
export CURAND_HOME="${CURAND_HOME:-$DINGXIN_SITE/nvidia/curand}"

export PYTHONPATH="$BACKEND_PROBE_SITE:$COSMOS_REPO:$COSMOS_REPO/packages/transformers-cosmos3/src:$COSMOS_REPO/packages/diffusers-cosmos3/src:$COSMOS_REPO/packages/vllm-cosmos3:$DINGXIN_SITE:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$LOCAL_4090_ROOT/envs/python313_pkg/lib:$CUDNN_HOME/lib:$NVRTC_HOME/lib:$CURAND_HOME/lib:$DINGXIN_SITE/nvidia/cublas/lib:$DINGXIN_SITE/nvidia/cuda_runtime/lib:$DINGXIN_SITE/nvidia/nccl/lib:$DINGXIN_SITE/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
