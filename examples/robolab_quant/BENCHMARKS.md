# RoboLab Quantization and Inference Benchmarks

This is the deployment-oriented benchmark record for Cosmos3 Nano Policy DROID
on RTX 4090. It separates open-loop diagnostics from closed-loop task success.
For commands and artifact requirements, see [README.md](README.md).

## Measurement Rules

Unless a table states otherwise:

- GPU: RTX 4090 24GB on `4090-NX-1`.
- Model: public Cosmos3 Nano Policy DROID Diffusers checkpoint.
- Quantization: packed vLLM Marlin weight-only W4A16/W8A16.
- Batch size: one policy request; output action chunk: 32 x 8.
- Resolution/views: 540 x 640 concat3 (wrist above left/right exterior).
- VAE encode chunk: 8 frames; exact duration: 33.
- `torch.compile`: disabled; deterministic model seed: 0.
- Default sampler: guidance 3.0, 4 UniPC steps, shift 5.0.

**Generate latency** is timed around model generation inside the server.
**Request latency** additionally includes preprocessing and response handling.
**Policy inference timing** in RoboLab is client-observed and amortized over
simulator steps, so it is not directly comparable to request latency.

Replay action error measures divergence from saved reference actions:

- `L1 mean`: mean absolute error over all action timesteps and dimensions.
- `Linf p95`: for each request, take its largest absolute action error, then
  report the 95th percentile across requests.

These are diagnostics, not success-rate proxies. Closed-loop success is the
quality gate.

## End-to-End Deployment Smoke

The first self-contained `full_w8` rollout used a policy server isolated on
physical GPU 0 and IsaacSim isolated on physical GPU 1.

```text
task: BananaInBowlTask
episodes: 1
max episode steps: 750
open-loop action horizon: 32
guidance/steps: 3.0/4
```

| Strategy | Guidance | Steps | Success | Completion step | Policy requests | Policy inference total |
|---|---:|---:|---:|---:|---:|---:|
| `full_w8` | 3.0 | 4 | 1/1 | 248/750 | 8 | 38.663s |

Memory after load was 17.88GB allocated. Peak allocated/reserved was
20.95/21.42GB. This verifies direct bundle load, WebSocket serving, real
RoboLab observation preprocessing, action execution, and task completion under
the 24GB target.

Raw artifacts:

```text
/mnt/lixiangyu/cosmos_ws/RoboLab/output/robolab_full_w8_g3s4_smoke3_20260711
/mnt/lixiangyu/cosmos_ws/robolab_validation/runs/full_w8_g3s4_rollout_smoke3
/mnt/lixiangyu/cosmos_ws/robolab_validation/data/banana_in_bowl_full_w8_g3s4_smoke3
```

## Full-W8 Sampler Replay

Eight requests captured from the successful rollout were replayed against the
same bundle and deterministic seed. Errors are relative to guidance 3.0 / 4
steps.

| Guidance | Steps | Denoiser forwards | Request p50/p95 | L1 mean | Linf p95 |
|---:|---:|---:|---:|---:|---:|
| 3.0 | 4 | 8 | 4113/4120ms | 0 | 0 |
| 1.0 | 4 | 4 | 2415/2672ms | 0.0704 | 0.868 |
| 3.0 | 2 | 4 | 2394/2674ms | 0.0376 | 0.303 |
| 1.0 | 2 | 2 | 1576/1838ms | 0.0523 | 0.676 |

For this model, denoiser forwards per request are approximately:

```text
num_steps * (2 if guidance > 1 else 1)
```

Guidance 1 removes the unconditional classifier-free-guidance pass. Two steps
retain fewer points in the denoising integration. Both are behavioral changes.

## Full-W8 Sampler Closed-Loop Smoke

All rows use the same `BananaInBowlTask`, one environment, one episode, 750-step
horizon, and bundle as the end-to-end reference. One episode establishes
connectivity only; it is too small for quality ranking.

| Guidance | Steps | Success | Completion step | Policy inference total | Policy inference / simulator step |
|---:|---:|---:|---:|---:|---:|
| 3.0 | 4 | 1/1 | 248 | 38.663s | not reported by this run |
| 1.0 | 4 | 0/1 | 750 | 75.635s | 100.8ms |
| 3.0 | 2 | 1/1 | 129 | 14.987s | 116.2ms |
| 1.0 | 2 | 1/1 | 355 | 27.402s | 77.2ms |

Raw results:

```text
/mnt/lixiangyu/cosmos_ws/RoboLab/output/robolab_full_w8_g1s4_smoke_20260711
/mnt/lixiangyu/cosmos_ws/RoboLab/output/robolab_full_w8_g3s2_smoke_20260711
/mnt/lixiangyu/cosmos_ws/RoboLab/output/robolab_full_w8_g1s2_smoke_20260711
```

The replay error and smoke result together make guidance 3.0 / 2 steps the
current sampler candidate for repeated rollout. It preserves CFG and had the
lowest replay action error among accelerated settings. This is not yet evidence
of quality parity.

## Training Calibration Protocol

W4 and mixed bundles use 128 distinct episodes from the official DROID training
split, not RoboLab evaluation captures:

```text
dataset: nvidia/Cosmos3-DROID
revision: 5c11a20accb11497270a5247a7f1e66ad04c956c
split: train/success
samples: 128 frames from 128 episodes
selection: stratified episodes, one deterministic central-80% frame each
views: wrist + left/right exterior
requests: 128 unique payloads, 97 unique selected prompts
calibration sampler: guidance 3.0, 4 UniPC steps, seed 0
calibration method: per-Linear input-channel amax, W4 alpha 0.5
```

Local request validation found image means from 67.99 to 108.24 and no blank,
non-finite, malformed, duplicate-episode, or duplicate-payload sample.

```text
/home/lixiangyu/cosmos_ws/robolab_validation/calibration/requests_train128_rev5c11a20
/mnt/lixiangyu/cosmos_ws/robolab_validation/calibration/requests_train128_rev5c11a20
```

## Quantization Strategy Comparison

All four bundles passed manifest SHA256 validation and packed-tensor loading.
The replay rows use the same eight requests captured from the successful
`full_w8` rollout. Every server used guidance 3.0, four UniPC steps, and model
seed 0 on the same RTX 4090 machine.

| Strategy | W4/W8 modules | Bundle bytes | Peak alloc/reserved | Request p50/p95 | Generate p50/p95 | L1 mean | Linf p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full_w8` | 0/504 | 20,444,946,152 | 20.95/21.42GB | 4092/4301ms | 4052/4257ms | 0 | 0 |
| `full_w4` | 504/0 | 13,716,089,316 | 14.23/14.67GB | 4228/4438ms | 4188/4397ms | 0.0505 | 0.532 |
| `attention_w8` | 216/288 | 15,177,881,780 | 15.69/16.21GB | 4123/4336ms | 4081/4291ms | 0.0437 | 0.472 |
| `gen_branch_w8` | 252/252 | 17,080,516,726 | 17.59/18.03GB | 4072/4310ms | 4032/4269ms | 0.0266 | 0.257 |

Quantization changes memory much more than latency for these batch-one shapes.
`gen_branch_w8` has the lowest action divergence among the W4/W8 mixed
strategies. Open-loop error alone does not establish closed-loop quality.

## Quantization Strategy Closed-Loop Rollout

The four processes used the same protocol and were run concurrently on one
eight-GPU RTX 4090 machine. Each policy server and IsaacSim process had its own
physical GPU.

```text
task: BananaInBowlTask
instruction: Pick up the banana and place it in the bowl
episodes per strategy: 5
num_envs: 1
max episode steps: 750
RoboLab process seed: 0
action chunk: 32 x 8
guidance/steps: 3.0/4
```

| Strategy | Success | Episode steps by run | Successful completion median | Server requests | Request p50/p95 | Policy time / simulator step |
|---|---:|---|---:|---:|---:|---:|
| `full_w8` | 5/5 | 237, 159, 284, 155, 381 | 237 | 39 | 4104/4120ms | 155.6ms |
| `full_w4` | 2/5 | 750F, 166, 750F, 220, 750F | 193 | 85 | 4237/4243ms | 159.4ms |
| `attention_w8` | 5/5 | 247, 189, 462, 329, 375 | 329 | 52 | 4147/4156ms | 158.2ms |
| `gen_branch_w8` | 5/5 | 311, 179, 118, 152, 223 | 179 | 32 | 4089/4096ms | 156.5ms |

`F` marks a timeout failure at the 750-step horizon. Server request latency is
from each rollout server's request profile. Policy time per simulator step is
`sum(policy_inference_s) / sum(episode_step)` and includes client-side image
preparation, serialization, and transport that are outside the server timer.
Different request counts are a consequence of different episode lengths, with
one policy request approximately every 32 simulator steps.

The five-episode result is sufficient to reject `full_w4` as the default for
this task: its memory advantage came with three observed closed-loop failures.
It is not sufficient to rank the other three strategies statistically; a 5/5
result still has a wide success-rate confidence interval. Retain all three W8
options and select by deployment memory budget:

- `full_w8`: quality-first reference; highest memory, still below 24GB.
- `gen_branch_w8`: current balanced default; 18.03GB peak reserved and the
  lowest mixed-precision replay error.
- `attention_w8`: memory-focused retained option; 16.21GB peak reserved.
- `full_w4`: experimental memory floor only; it failed this rollout gate.

Guidance 3.0 / two steps remains a separate latency candidate. Its one-episode
smoke is not enough to combine it with a quantization recommendation yet.

Raw artifacts:

```text
/mnt/lixiangyu/cosmos_ws/robolab_validation/replay/benchmark_<strategy>_g3s4_replay8
/mnt/lixiangyu/cosmos_ws/robolab_validation/runs/benchmark_<strategy>_g3s4_replay8
/mnt/lixiangyu/cosmos_ws/robolab_validation/runs/rollout5_<strategy>_g3s4
/mnt/lixiangyu/cosmos_ws/RoboLab/output/robolab_<strategy>_g3s4_rollout5_20260711
```
