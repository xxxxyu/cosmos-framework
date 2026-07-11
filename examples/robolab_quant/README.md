# Cosmos3 RoboLab Quantized Pipeline

This is the release entry point for downloading the public
`nvidia/Cosmos3-Nano-Policy-DROID` model, building a self-contained packed
W4A16/W8A16 bundle, and running it against RoboLab on RTX 4090.

Read [BENCHMARKS.md](BENCHMARKS.md) before changing the default strategy or
sampler.

## 1. Setup

```bash
CUDA_VISIBLE_DEVICES=0 examples/robolab_quant/pipeline.sh setup
```

The shared locked environment contains only the policy runtime. RoboLab and
IsaacSim remain in their own environment or container.

## 2. Build From The Public Policy

The default command downloads an immutable DROID policy revision, only the
Qwen tokenizer files needed at runtime, and the Wan VAE. It then stream-packs
`full_w8` without loading the complete BF16 policy on GPU.

```bash
HF_TOKEN=... \
ASSET_DIR=/data/cosmos3_quant/sources \
BUNDLE_DIR=/data/cosmos3_quant/robolab_full_w8 \
STRATEGY=full_w8 \
POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh build-public
```

The defaults pin DROID to `6706d7680581c255ff61e0f3bb49d90eac55c79e`,
Qwen to `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, and Wan to
`921dbaf3f1674a56f47e83fb80a34bac8a8f203e`. `sources.json` records requested
and resolved revisions, which are also embedded in the bundle manifest.
Reusing `ASSET_DIR` resumes downloads. The output directory must not exist.

W4 or mixed strategies additionally require DROID training calibration stats:

```bash
STRATEGY=attention_w8 \
CALIBRATION_STATS=/path/to/droid_train128_input_amax.pt \
ASSET_DIR=/data/cosmos3_quant/sources \
BUNDLE_DIR=/data/cosmos3_quant/robolab_attention_w8 \
POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh build-public
```

## 3. Validate Or Transfer

```bash
BUNDLE_DIR=/path/to/robolab_full_w8 STRATEGY=full_w8 \
examples/robolab_quant/pipeline.sh validate
```

This CPU-only command verifies all hashes, opens all 504 packed payloads, and
checks the strategy precision map. Serving a completed bundle never reads the
source checkpoint or calibration dataset.

## 4. Replay

```bash
BUNDLE_DIR=/path/to/robolab_full_w8 \
CAPTURE_DIR=/path/to/openpi_replay_requests \
RUN_DIR=/data/cosmos3_runs/robolab_replay \
POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh replay
```

The general deployment default is guidance 3.0 / two UniPC steps. Set
`GUIDANCE=3.0 NUM_STEPS=4` for the conservative rollback sampler.

## 5. Rollout

Use separate physical GPUs for the policy server and IsaacSim. Merely passing
`--device` is insufficient because Vulkan may initialize on physical GPU 0.

```bash
BUNDLE_DIR=/path/to/robolab_full_w8 \
ROBOLAB_DIR=/path/to/RoboLab \
ROBOLAB_PYTHON=/path/to/robolab/python \
RUN_DIR=/data/cosmos3_runs/robolab_banana \
POLICY_GPU=0 SIM_GPU=1 \
TASK=BananaInBowlTask NUM_ENVS=1 NUM_RUNS=1 \
examples/robolab_quant/pipeline.sh rollout
```

`ROBOLAB_PYTHON` may be the interpreter inside the official RoboLab container
or a validated host installation. The policy server uses the locked Cosmos
runtime, never the simulator interpreter.

## Observation Contract

RoboLab renders three 1280x720 views. The client constructs one 640x540 RGB
image with a 640x360 wrist view above two 320x180 exterior views. The server
maps this to the model's 736x544 bucket. Requests also contain joint position,
gripper state, and task text; responses contain a 32x8 action chunk.

## Strategy Selection

| Strategy | W4/W8 modules | Peak reserved | Deployment role |
|---|---:|---:|---|
| `full_w8` | 0/504 | 21.42GB | General default |
| `attention_w8` | 216/288 | 16.21GB | Banana-only low-memory option at g3/s4 |
| `gen_branch_w8` | 252/252 | 18.03GB | Banana-specific; failed cross-task transfer |
| `full_w4` | 504/0 | 14.67GB | Experimental; failed quality gate |

Use `full_w8`, guidance 3.0, two steps for general RoboLab deployment. It
reached 45/50 on Banana and 25/30 across three additional task sets. Do not
promote a mixed strategy to another task without paired rollout validation.

## DROID Training Calibration

W4/mixed bundles use 128 frames from 128 distinct episodes of
`nvidia/Cosmos3-DROID`, revision
`5c11a20accb11497270a5247a7f1e66ad04c956c`, split `train/success`.
Calibration is embodiment/input-contract specific, not Banana-task specific.

To reproduce it, export requests with
`export_robolab_train_calibration_requests`, replay all 128 requests through a
`full_w8` server using `--calibration-stats-output`, then pass the resulting
stats to `build-public`. See the benchmark's Training Calibration Protocol for
the exact sample and view contract.

## Serving Only

```bash
BUNDLE_DIR=/path/to/robolab_full_w8 POLICY_GPU=0 \
examples/robolab_quant/pipeline.sh serve
```

The OpenPI WebSocket endpoint defaults to `0.0.0.0:8000`. Roll back by using
the same immutable `full_w8` bundle with guidance 3.0 / four steps.
