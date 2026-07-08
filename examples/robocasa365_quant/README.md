# Cosmos3 RoboCasa365 Quantization Pipeline

This overlay is the first productized surface for the Cosmos3 Nano
RoboCasa365 CloseFridge quantization work.

For release testing and real-robot handoff notes, read
`examples/robocasa365_quant/RELEASE_TESTING.md`.

It supports four fixed weight-only strategies:

| Strategy | Plan | M13 success | Direct replay peak alloc |
|---|---|---:|---:|
| `full_w8` | all language Linear W8A16 | 0.96 | 19.20GB |
| `full_w4` | all language Linear W4A16 | 0.92 | 12.48GB |
| `attention_w8` | self-attention W8A16, language MLP W4A16 | 0.96 | 13.94GB |
| `gen_branch_w8` | MoT generation branch W8A16, rest W4A16 | 0.94 | 15.84GB |

All strategies are weight-only. Activation quantization is not used.

## Install Overlay

If this directory is already in a cosmos-framework checkout, no overlay step is
needed. Otherwise, from the overlay root:

```bash
rsync -a ./ /path/to/cosmos-framework/
```

Then make sure the Python environment can import:

- `torch`
- `vllm._C` with Marlin kernels
- the local `cosmos_framework`
- Cosmos3 package overlays used by the checkpoint

## Cosmos Environment

Use the repo's `uv.lock` and select the CUDA group that matches the machine.
For RTX 4090 / Ada, CUDA 12.8 is the conservative first target unless the
machine has a validated CUDA 13 stack:

```bash
uv sync --all-extras --group=cu128-train --group=vllm
source .venv/bin/activate
export LD_LIBRARY_PATH=
```

For H100 environments that already use CUDA 13:

```bash
uv sync --all-extras --group=cu130-train --group=vllm
source .venv/bin/activate
export LD_LIBRARY_PATH=
```

`vllm._C` must import successfully because the direct-load backend calls Marlin
kernels through `torch.ops._C`.

## RoboCasa/RLDX Environment

Closed-loop rollout needs the external RoboCasa365/RLDX simulator environment.
Direct replay benchmark does not.

Start from `env.example.sh`:

```bash
cp examples/robocasa365_quant/env.example.sh /tmp/robocasa365_quant_env.sh
source /tmp/robocasa365_quant_env.sh
```

Then set:

```bash
export RLDX_REPO=/path/to/RLDX-1
export ROBOCASA365_PYTHON=$RLDX_REPO/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python
export ROBOCASA365_ROLLOUT_SCRIPT=$RLDX_REPO/rldx/eval/rollout_policy.py
```

## Write Strategy Configs

```bash
python -m cosmos_framework.scripts.robocasa365_quant_pipeline \
  write-strategy-configs \
  --output-dir examples/robocasa365_quant/configs
```

## Export A Quant Artifact

Use training-set calibration captures. The validated experiments used
`calib_limit=128` and `calib_alpha=0.5`.

```bash
python -m cosmos_framework.scripts.robocasa365_quant_pipeline \
  print-export-command \
  --strategy attention_w8 \
  --plan-dir examples/robocasa365_quant/configs \
  --checkpoint-path /path/to/checkpoints/iter_000008000 \
  --config-file /path/to/config.yaml \
  --calib-capture-dir /path/to/train_calib_capture \
  --quant-export-dir /path/to/quant_artifacts/attention_w8 \
  --output-dir /tmp/cosmos3_attention_w8_export
```

The command printed by this helper starts
`cosmos_framework.scripts.action_policy_server_robocasa365_quant` with
`--quant-export-dir`. The server exits after writing the quant artifact
manifest.

After export, attach metadata:

```bash
python -m cosmos_framework.scripts.robocasa365_quant_pipeline \
  write-artifact-metadata \
  --strategy attention_w8 \
  --quant-artifact-dir /path/to/quant_artifacts/attention_w8 \
  --checkpoint-path /path/to/checkpoints/iter_000008000 \
  --config-file /path/to/config.yaml \
  --calib-capture-dir /path/to/train_calib_capture
```

## Serve A Quant Artifact

```bash
python -m cosmos_framework.scripts.robocasa365_quant_pipeline \
  print-serve-command \
  --checkpoint-path /path/to/checkpoints/iter_000008000 \
  --config-file /path/to/config.yaml \
  --quant-import-dir /path/to/quant_artifacts/attention_w8 \
  --output-dir /tmp/cosmos3_attention_w8_server \
  --port 5577
```

The direct-load path inserts packed W4/W8 modules before the DCP BF16 weight
load, so it avoids materializing the full BF16 language model on GPU.

## Replay Benchmark

Start the server, then run replay against captured requests:

```bash
python -m cosmos_framework.scripts.robocasa365_quant_pipeline \
  print-replay-command \
  --capture-dir /path/to/closefridge_action_parity_v1 \
  --output-dir /tmp/replay_attention_w8 \
  --port 5577 \
  --limit 32
```

For local RTX 4090 validation, source the environment template and use the
wrapper:

```bash
source examples/robocasa365_quant/local_4090_env.example.sh
STRATEGY=full_w4 CUDA_VISIBLE_DEVICES=2 \
  examples/robocasa365_quant/run_direct_replay_4090.sh
```

The wrapper validates the packed artifact manifest, starts the direct-load
server, waits for the ZMQ port, runs replay, writes `profile_events.jsonl`, and
summarizes server profile events into `profile_summary.json`.

## Validate Artifact

```bash
python -m cosmos_framework.scripts.robocasa365_quant_pipeline \
  validate-artifact \
  --quant-artifact-dir /path/to/quant_artifacts/attention_w8 \
  --strategy attention_w8
```

Use `--check-tensors` when you want to open every tensor payload as well as
checking manifest structure and file existence.

## Profile Replay

Use the profiling wrapper when investigating backend latency:

```bash
source examples/robocasa365_quant/local_4090_env.example.sh
STRATEGY=attention_w8 \
QUANT_ARTIFACT_DIR=$LOCAL_4090_ROOT/quant_artifacts/attention_w8 \
CUDA_VISIBLE_DEVICES=2 \
PROFILE_TOOL=nsys \
REPLAY_LIMIT=8 \
examples/robocasa365_quant/profile_direct_replay_4090.sh
```

`PROFILE_TOOL` may be:

| Tool | Purpose |
|---|---|
| `none` | server/replay timing only |
| `nsys` | CUDA kernel/API attribution and condensed category summary |
| `ncu-marlin` | targeted Nsight Compute profile for a small number of Marlin kernels |

Set `TORCH_COMPILE=1` only for model/graph-level experiments. The default is
`TORCH_COMPILE=0` so operator-level and eager baseline profiles remain
comparable.

For quantized direct-load experiments, the default MoT compile path uses
`fullgraph=True` and cannot trace Marlin custom ops. A graph-break experiment
can be run with:

```bash
TORCH_COMPILE=1 \
COSMOS3_MOT_COMPILE_FULLGRAPH=0 \
COSMOS3_DYNAMO_DISABLE_QUANT_LINEAR=1 \
examples/robocasa365_quant/profile_direct_replay_4090.sh
```

Treat this as a model/graph-level ablation, not an operator benchmark.

For targeted Marlin kernel profiling:

```bash
PROFILE_TOOL=ncu-marlin \
NCU_KERNEL_NAME='regex:.*marlin::Marlin.*' \
NCU_LAUNCH_SKIP=32 \
NCU_LAUNCH_COUNT=8 \
NCU_SET=basic \
examples/robocasa365_quant/profile_direct_replay_4090.sh
```

Start with `NCU_SET=basic`. Heavier sets should be used only on `REPLAY_LIMIT=1`
or `2`, because Nsight Compute can replay selected kernels and substantially
slow execution.

Set `NCU_KERNEL_NAME` to a narrower Nsight Compute kernel-name filter when a
single replay contains multiple Marlin variants and you need kernel-internal
metrics for one shape at a time.

If Nsight Compute reports `ERR_NVGPUCTRPERM`, GPU performance counters are
restricted on that host. Use the `nsys` path for launch/API attribution, or ask
an administrator to enable performance counter access before collecting
kernel-internal metrics. When running the profiler under `sudo`, explicitly set
`NCU_BIN` if root's `PATH` does not include CUDA tools, for example:

```bash
NCU_BIN=/usr/local/cuda-12.4/bin/ncu PROFILE_TOOL=ncu-marlin ...
```

For operator-level benchmarking, run the profiling wrapper once with
`LINEAR_SHAPE_PROFILE=1`. This writes `linear_shapes.jsonl`, a quantized-linear
shape/call-count summary from the real replay:

```bash
LINEAR_SHAPE_PROFILE=1 \
PROFILE_TOOL=none \
examples/robocasa365_quant/profile_direct_replay_4090.sh

python -m cosmos_framework.scripts.quant_backend_microbench \
  --shape-file /path/to/profile_run/linear_shapes.jsonl \
  --max-shapes 12 \
  --backends bf16,vllm_gptq_marlin_w4a16,vllm_gptq_marlin_w8a16,vllm_allspark_w8a16 \
  --chain none \
  --output /tmp/cosmos3_quant_backend_microbench.json
```

## Rollout Gate

The validated long-horizon gate uses:

```text
N_EPISODES=50
N_ENVS=5
N_ACTION_STEPS=8
MAX_EPISODE_STEPS=1200
USE_TASK_HORIZON=0
disable_video=1
```

Do not use `USE_TASK_HORIZON=1` for this gate; it can silently restore the
CloseFridge horizon to 600.

## Current Runtime Notes

Memory is within the 24GB target for all four quantized strategies. On the
validated RTX 4090 stack, direct replay generate latency is around 1.0-1.2s for
the recommended `attention_w8` path and similar W8-heavy strategies.

Further operator-level optimization is possible but not included in this stable
release. Real-shape microbenchmarks show that most quantized-linear time is
already dominated by large MLP shapes where the current Marlin backend is
competitive; shape-aware backend selection has only modest expected upside for
this release.

For deployment, benchmark and optimize runtime on the target GPU, especially
RTX 4090 / Ada. H100 speed is useful but not definitive for 4090.
