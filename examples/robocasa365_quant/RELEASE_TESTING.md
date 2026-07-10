# RoboCasa365 Quantized 4090 Release Testing

This document is the handoff runbook for the `robocasa365-quant-v0.1-4090`
release line. It is written so another engineer or coding agent can check out
the repo, validate the quantized policy, and understand what is still required
before real-robot use.

## Release Scope

Target model:

- Cosmos 3 Nano RoboCasa365 CloseFridge action policy.
- Checkpoint used in validation: `iter_000008000`.
- Runtime target: single RTX 4090 24GB.
- Quantization type: weight-only W4A16/W8A16. Activations remain BF16.
- Runtime backend: direct-load packed vLLM Marlin quantized linear modules.
- Default inference path: eager PyTorch, `--no-torch-compile`.

Published tag:

```bash
git clone git@github.com:xxxxyu/cosmos-framework.git
cd cosmos-framework
git checkout robocasa365-quant-v0.1-4090
```

If you need this runbook inside the checkout, use
`robocasa365-quant-v0.1.1-4090-docs` or the newest
`feature/robocasa365-quant-pipeline` branch. Do not move the already-published
`robocasa365-quant-v0.1-4090` tag.

## Validated Strategies

The release supports four fixed strategies. `attention_w8` is the recommended
4090 default because it matches the best rollout success observed here while
leaving comfortable memory headroom.

| Strategy | Policy | RoboCasa M13 success | 4090 peak allocated/reserved | Replay generate p50/p95 |
|---|---|---:|---:|---:|
| `full_w8` | all language Linear W8A16 | 0.96 | 19.20/19.50GB | 1015/1098 ms |
| `full_w4` | all language Linear W4A16 | 0.92 | 12.47/12.75GB | 1108/1191 ms |
| `attention_w8` | self-attention W8A16, MLP W4A16 | 0.96 | 13.93/14.28GB | 1029/1143 ms |
| `gen_branch_w8` | MoT generation branch W8A16, rest W4A16 | 0.94 | 15.84/16.11GB | 1048/1138 ms |

Use these numbers as sanity checks, not as exact pass/fail thresholds. Fresh
machines, drivers, CUDA builds, and Python package overlays can shift latency.

## Required Inputs

A tester needs these artifacts on the target machine:

- Cosmos checkout on this branch/tag.
- A Python environment that imports `torch`, `vllm._C`, and local
  `cosmos_framework`.
- One self-contained schema-v2 quant bundle. It includes packed quant tensors,
  residual non-quantized model state, runtime config, tokenizer, and Wan VAE.
- Optional direct replay capture directory, e.g.
  `closefridge_action_parity_v1`, for fast validation without simulator.
- RoboCasa/RLDX simulator only for closed-loop rollout.

The bundle is not stored in git. If it is unavailable, build it from the BF16
DCP and training-set calibration captures as described in `README.md`. The DCP
and standalone config are export-time inputs and must not be required on a
deployment host.

## Environment Smoke Test

From the Cosmos checkout:

```bash
python - <<'PY'
import torch
import vllm._C
import cosmos_framework
import diffusers_cosmos3
import transformers_cosmos3
import vllm_cosmos3
print("torch", torch.__version__)
print("cuda", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("Cosmos package overlays and vllm._C ok")
PY
```

Expected:

- `vllm._C` imports successfully.
- All three Cosmos package overlays import from the checked-out repository or
  its environment, not from a stale checkout path.
- CUDA sees the target GPU.
- No `LD_LIBRARY_PATH` conflict with unrelated CUDA/cuDNN installs.

## Artifact Validation

Validate manifest and tensor files before serving:

```bash
python -m cosmos_framework.scripts.robocasa365_quant_pipeline \
  validate-artifact \
  --quant-artifact-dir /path/to/quant_bundles/attention_w8 \
  --strategy attention_w8 \
  --require-self-contained \
  --check-tensors
```

Expected:

- `modules` is 504 for the validated CloseFridge artifact.
- `attention_w8` should report 216 W4 modules and 288 W8 modules.

## Fast Direct Replay Test

Direct replay is the preferred first test because it does not need RoboCasa.
It checks model load, packed quant import, ZMQ serving, preprocessing, action
postprocessing, latency, memory, and open-loop action parity.

```bash
source examples/robocasa365_quant/local_4090_env.example.sh

STRATEGY=attention_w8 \
QUANT_BUNDLE_DIR=/path/to/quant_bundles/attention_w8 \
REPLAY_CAPTURE_DIR=/path/to/closefridge_action_parity_v1 \
CUDA_VISIBLE_DEVICES=0 \
REPLAY_LIMIT=32 \
RUN_DIR=/tmp/cosmos3_attention_w8_replay32 \
examples/robocasa365_quant/run_direct_replay_4090.sh
```

Inspect:

```bash
cat /tmp/cosmos3_attention_w8_replay32/profile_summary.json
cat /tmp/cosmos3_attention_w8_replay32/replay/metrics.json
```

Sanity checks:

- Peak allocated memory is below 24GB.
- `attention_w8` generate p50 is around 1.0-1.2s on a validated 4090 stack.
- Replay action deviations are in the same range as prior validation for the
  same capture set.
- No server error appears in `server.log`.
- The server command and runtime config do not reference the source DCP or
  machine-specific config paths.

## Closed-Loop Rollout Test

Use the validated rollout gate:

```text
N_EPISODES=50
N_ENVS=5
N_ACTION_STEPS=8
MAX_EPISODE_STEPS=1200
USE_TASK_HORIZON=0
DISABLE_VIDEO=1
```

Do not use `USE_TASK_HORIZON=1`; it can restore the shorter task horizon.

Recommended sequence:

1. Run direct replay first.
2. Run a small rollout smoke, e.g. 2-5 episodes, to verify simulator wiring.
3. Run the full 50-episode gate only after smoke passes.
4. Record strategy, checkpoint, quant artifact path, config file, action steps,
   max episode steps, seed if any, GPU type, driver, CUDA, and git commit/tag.

## Real-Robot Deployment Gap

The release is a quantized policy server and evaluation pipeline. It is not a
complete real-robot controller by itself. If real robot data is available, first
use it as an offline replay/capture set. Online robot use needs the following
adapter and safety work.

### Observation Adapter

The current server expects RoboCasa-style observations and builds the model
sample from:

- wrist camera RGB frame,
- left third-person RGB frame,
- right third-person RGB frame,
- task text,
- optional reset/options fields.

For a real robot, implement an adapter that produces the same semantic fields,
camera order, RGB `uint8` layout, and resize behavior. Verify with saved
captures before sending actions to hardware.

Key risks:

- Camera extrinsics/viewpoint mismatch with RoboCasa training data.
- RGB/BGR or normalization mismatch.
- Different crop/resize policy.
- Missing or changed language instruction text.

### Action Adapter

The server outputs split action fields:

- `action.base_motion`
- `action.control_mode`
- `action.end_effector_position`
- `action.end_effector_rotation`
- `action.gripper_close`

The real robot controller must map these to its own command protocol. Confirm:

- coordinate frame conventions,
- position and rotation units,
- gripper open/close polarity,
- control mode semantics,
- action rate,
- whether actions are absolute, delta, or normalized by the training transform,
- how many actions from each chunk are executed before the next inference call.

### Timing

The 4090 direct replay path is around 1s per request for the validated
strategies. Real robot control should not assume a blocking 20Hz policy call.
Use a chunked/asynchronous controller if the robot executes multiple actions per
policy request, and define behavior for missed deadlines.

Minimum safety requirements:

- action clipping and workspace limits,
- velocity/acceleration limits,
- stale-action timeout,
- emergency stop path outside Python,
- dry-run mode that logs actions without commanding hardware,
- one-step or low-speed guarded test before autonomous execution.

### Recommended Real-Robot Bring-Up

1. Capture real robot observations without executing policy actions.
2. Run offline replay through the quantized server and inspect predicted action
   magnitudes, gripper behavior, and latency.
3. Compare BF16 and quantized outputs on the same real captures.
4. Run a no-op or log-only online policy loop.
5. Run single-step guarded execution with safety limits.
6. Only then run chunked autonomous execution.

## Known Non-Goals Of This Release

- No activation quantization.
- No W4A8/W8A8/FP8 kernel path.
- No TensorRT engine export.
- No ExLlama backend integration.
- No real-robot hardware adapter.
- No default `torch.compile` acceleration. The graph-break compile ablation was
  tested and did not improve steady 4090 latency for the current Marlin path.

## Handoff Prompt For Another Coding Agent

Give the agent this instruction:

```text
Read examples/robocasa365_quant/README.md and
examples/robocasa365_quant/RELEASE_TESTING.md. Do not modify user-owned or
external artifact directories. First validate the Python/CUDA/vllm._C
environment, then validate the quant artifact manifest, then run direct replay
with attention_w8. Only after direct replay passes should you run RoboCasa
rollout. For real robot use, implement observation/action adapters and safety
checks; do not command hardware directly from the RoboCasa eval server without
adapter validation.
```
