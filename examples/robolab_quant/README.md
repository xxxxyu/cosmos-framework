# Cosmos3 RoboLab Quantized Deployment

This directory is the deployment entry point for the self-contained packed
Cosmos3 Nano DROID policy. Read [BENCHMARKS.md](BENCHMARKS.md) before selecting
a quantization or denoising strategy.

## Artifact Contract

A RoboLab bundle contains all runtime model state, 504 packed W4/W8 Linear
payloads, the Qwen tokenizer, the Wan VAE, a portable config, and a SHA256/size
manifest. Serving a bundle does not read the source DROID checkpoint or the
calibration dataset.

The fixed precision strategies are:

| Strategy | W4 modules | W8 modules | Deployment status |
|---|---:|---:|---|
| `full_w8` | 0 | 504 | General RoboLab default |
| `full_w4` | 504 | 0 | Experimental memory floor; failed current rollout gate |
| `attention_w8` | 216 | 288 | Low-memory Banana option at guidance 3 / 4 steps |
| `gen_branch_w8` | 252 | 252 | Banana-specific option; requires per-task validation |

These are weight-only W4A16/W8A16 bundles. Activations remain BF16. W4 export
requires DROID training calibration statistics unless the explicitly unsafe
`--allow-uncalibrated-w4` flag is used.

The paired 50-episode `BananaInBowlTask` result at guidance 3 / 4 steps is
43/50 for `full_w8`, 26/50 for `full_w4`, 42/50 for `attention_w8`, and 45/50
for `gen_branch_w8`. The general deployment default is `full_w8` with guidance
3 / 2 steps: it reached 45/50 on Banana and 25/30 across three additional task
sets. Use [BENCHMARKS.md](BENCHMARKS.md) before choosing a bundle or sampler.

## Observation Resolution

RoboLab renders the wrist, left shoulder, and right shoulder cameras at
1280x720 each. The Cosmos3 client resizes each view to 640x360, keeps the wrist
view on top, and places two 320x180 exterior views side by side underneath.
The OpenPI request therefore contains one 640x540 RGB image. The serving
transform maps it to the resolution-480 4:3 model bucket, 736x544. The
`resolution=480` setting is a bucket name, not a 480x480 tensor.

## Validate a Bundle

Run the strong validation once after export or transfer. It verifies all file
hashes, opens all packed tensor payloads, and checks the precision map.

```bash
python -m cosmos_framework.scripts.robolab_quant_pipeline validate \
  --bundle-dir /path/to/bundle \
  --expected-strategy attention_w8 \
  --check-hashes \
  --check-tensors
```

The command is CPU-only. A deployment must not proceed if validation fails.

## Build Training Calibration Requests

Download a versioned subset of the `nvidia/Cosmos3-DROID` **training split**.
The local subset must contain parquet metadata/data and enough matching wrist,
left, and right RGB video files for 128 distinct episodes. Evaluation captures
must not be used as W4 calibration data.

Ubuntu's FFmpeg with the `libdav1d` AV1 decoder is required for this data
preparation step. Export deterministic OpenPI requests:

```bash
PYTHONPATH=/path/to/openpi/packages/openpi-client/src:$PYTHONPATH \
python -m cosmos_framework.scripts.export_robolab_train_calibration_requests \
  --dataset-root /path/to/Cosmos3-DROID/success \
  --dataset-revision <immutable-huggingface-commit> \
  --output-dir /path/to/droid_train128_requests \
  --samples 128 \
  --seed 0 \
  --ffmpeg /usr/bin/ffmpeg
```

The exporter fails if fewer than 128 episodes have all three local RGB views.
It emits one client-composed 640x540 W x H request per episode plus a manifest
and per-request dataset/episode/frame/video/SHA256 provenance.

Start a `full_w8` server with calibration hooks, then replay all requests:

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  cosmos_framework.scripts.action_policy_server_robolab \
  --quant-import-dir /path/to/full_w8_bundle \
  --port 8000 \
  --output-dir /path/to/calibration_server_run \
  --calibration-stats-output /path/to/droid_train128_input_amax.pt \
  --deterministic-seed \
  --guidance 3.0 \
  --num-steps 4

python -m cosmos_framework.scripts.robolab_policy_replay \
  --capture-dir /path/to/droid_train128_requests \
  --output-dir /path/to/droid_train128_responses \
  --port 8000 \
  --limit 128
```

The stats file contains a one-dimensional input-channel amax tensor for each of
the 504 packed Linear modules.

## Export a Bundle

Export is streamed one source shard and one quantized Linear at a time. It does
not instantiate the full BF16 model on GPU.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  cosmos_framework.scripts.robolab_quant_pipeline build-bundle \
  --strategy attention_w8 \
  --source-checkpoint /path/to/Cosmos3-Nano-Policy-DROID \
  --tokenizer-dir /path/to/Qwen3-VL-8B-Instruct \
  --vae-path /path/to/Wan2.2_VAE.pth \
  --calibration-stats /path/to/droid_train128_input_amax.pt \
  --calibration-alpha 0.5 \
  --output-dir /path/to/attention_w8_bundle \
  --device cuda:0
```

Use a new output directory for every export. The exporter refuses to overwrite
an existing artifact, writes through a temporary directory, and renames only
after the manifest is complete.

## Start the Policy Server

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  cosmos_framework.scripts.action_policy_server_robolab \
  --quant-import-dir /path/to/full_w8_bundle \
  --host 0.0.0.0 \
  --port 8000 \
  --output-dir /path/to/server_run \
  --profile-jsonl /path/to/server_run/profile.jsonl \
  --guidance 3.0 \
  --num-steps 2
```

Production defaults keep guardrails and `torch.compile` disabled, use a
480-resolution Wan VAE encode chunk of 8 frames, and add exact duration 33 for
the 32-action chunk. These settings avoid first-request compilation/network
access and keep the full-W8 path below 24GB allocated memory.

## Run RoboLab

Use separate physical GPUs for the server and IsaacSim. Setting only
`--device cuda:1` is insufficient because Isaac/Vulkan can still initialize on
physical GPU 0. Process-level `CUDA_VISIBLE_DEVICES` isolation is mandatory.

```bash
cd /path/to/RoboLab
CUDA_VISIBLE_DEVICES=1 python policies/cosmos3/run.py \
  --task BananaInBowlTask \
  --remote-host 127.0.0.1 \
  --remote-port 8000 \
  --num-envs 1 \
  --num-runs 1 \
  --device cuda:0 \
  --headless \
  --video-mode none \
  --output-folder-name cosmos3_quant_smoke
```

Inside the simulator process, physical GPU 1 is remapped to `cuda:0`. Keep the
task, seed/initialization protocol, episode horizon, action chunk, sampler, and
bundle manifest hash with every reported result.

## Validation Order and Rollback

For every new bundle, task, checkpoint, or sampler setting:

1. Run strong artifact validation.
2. Run deterministic replay on the same request set and inspect finite actions,
   latency, memory, and action error against the chosen reference.
3. Run a one-episode end-to-end smoke.
4. Run repeated closed-loop rollouts under one fixed protocol.

Quantization and sampler changes are independently configurable but are not
quality-orthogonal. The Banana winner `gen_branch_w8` g3/s2 fell from 50/50 on
Banana to 17/30 across three other tasks, while `full_w8` g3/s2 reached 25/30.
Validate every combination. Roll back by switching `--quant-import-dir` to the
immutable `full_w8` bundle and restoring guidance `3.0`, UniPC steps `4`.
Bundles are never modified in place.
