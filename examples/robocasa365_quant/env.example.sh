#!/usr/bin/env bash
# Example environment for Cosmos3 RoboCasa365 quantized policy serving.
#
# Copy this file outside the repo and edit paths for the target machine.
# Do not commit machine-specific absolute paths or credentials.

set -euo pipefail

# Cosmos checkout that contains this branch.
export COSMOS_REPO="${COSMOS_REPO:-$PWD}"

# Python from the Cosmos environment. For fresh setup, create it with one of:
#   uv sync --all-extras --group=cu128-train --group=vllm
#   uv sync --all-extras --group=cu130-train --group=vllm
export COSMOS_PYTHON="${COSMOS_PYTHON:-$COSMOS_REPO/.venv/bin/python}"

# Optional external package overlays used by the current Cosmos3 action-policy
# experiments. Keep empty for a self-contained fresh install.
export PYTHONPATH="$COSMOS_REPO:$COSMOS_REPO/packages/transformers-cosmos3/src:$COSMOS_REPO/packages/diffusers-cosmos3/src:$COSMOS_REPO/packages/vllm-cosmos3:${PYTHONPATH:-}"

# RoboCasa/RLDX rollout environment. This is only needed for closed-loop rollout;
# direct replay benchmark does not need the simulator.
export RLDX_REPO="${RLDX_REPO:-/path/to/RLDX-1}"
export ROBOCASA365_PYTHON="${ROBOCASA365_PYTHON:-$RLDX_REPO/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python}"
export ROBOCASA365_ROLLOUT_SCRIPT="${ROBOCASA365_ROLLOUT_SCRIPT:-$RLDX_REPO/rldx/eval/rollout_policy.py}"

# Validated rollout gate for CloseFridge.
export N_EPISODES="${N_EPISODES:-50}"
export N_ENVS="${N_ENVS:-5}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1200}"
export USE_TASK_HORIZON="${USE_TASK_HORIZON:-0}"
export DISABLE_VIDEO="${DISABLE_VIDEO:-1}"
