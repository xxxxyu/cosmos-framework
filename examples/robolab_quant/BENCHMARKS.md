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
- Client request: 640x540 W x H (array H x W is 540x640), concat3 RGB.
- Model spatial bucket: 736x544 W x H (tensor H x W is 544x736).
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

The replay error and smoke result motivated guidance 3.0 / 2 steps as the
candidate for repeated rollout. It preserves CFG and had the lowest replay
action error among accelerated settings. The 50-episode and multi-task gates
below provide the closed-loop evidence used for the final decision.

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

## Paired 50-Episode Quantization Gate

The earlier five-episode rollout was a connectivity smoke and produced an
unstable ranking. The release comparison below uses 50 paired initial states.
Every server and simulator had an exclusive physical RTX 4090.

```text
task: BananaInBowlTask
episodes per strategy: 50
num_envs: 1
max episode steps: 750
RoboLab process seed: 0
model seed: 0 for every request
action chunk: 32 x 8
guidance/steps: 3.0/4
```

All 200 HDF5 initial-state hashes matched by run across the four strategies.
The hash includes robot, object, and camera state. Confidence intervals are
Wilson 95% intervals. A paired win means the candidate succeeded when
`full_w8` failed; a paired loss means the reverse. The p-value is an exact
two-sided McNemar test and is not adjusted for multiple comparisons.

| Strategy | Success (Wilson 95% CI) | Successful step median | Paired wins/losses | McNemar p | Episode wall p50 | 50-episode wall |
|---|---:|---:|---:|---:|---:|---:|
| `full_w8` | 43/50 = 0.86 [0.738, 0.930] | 213 | reference | - | 78.9s | 85.6min |
| `full_w4` | 26/50 = 0.52 [0.385, 0.652] | 292.5 | 4/21 | 0.00091 | 203.4s | 140.6min |
| `attention_w8` | 42/50 = 0.84 [0.715, 0.917] | 373.5 | 4/5 | 1.000 | 137.4s | 115.3min |
| `gen_branch_w8` | 45/50 = 0.90 [0.786, 0.957] | 208 | 6/4 | 0.754 | 81.1s | 81.8min |

Wall time is the sum of per-episode wall time and excludes one-time model and
IsaacSim startup. Failed episodes run to the 750-step horizon, so lower policy
quality also increases total evaluation time.

| Strategy | Peak alloc/reserved | Request p50/p95 | Generate p50/p95 | Policy time / simulator step |
|---|---:|---:|---:|---:|
| `full_w8` | 20.95/21.42GB | 4110/4120ms | 4071/4080ms | 157.0ms |
| `full_w4` | 14.23/14.67GB | 4248/4257ms | 4207/4214ms | 159.3ms |
| `attention_w8` | 15.69/16.21GB | 4153/4162ms | 4111/4118ms | 156.2ms |
| `gen_branch_w8` | 17.59/18.03GB | 4104/4116ms | 4062/4071ms | 156.6ms |

Quantization changes memory, not request latency, at these batch-one shapes.
`full_w4` is statistically and operationally worse than `full_w8` on this
task. The three W8-containing strategies are not distinguishable by 50 Banana
episodes alone.

## Paired 50-Episode Sampler Gate

The sampler gate fixes the bundle to `full_w8`. Initial-state hashes matched
for all 50 runs across all four settings.

| Guidance | Steps | Success (Wilson 95% CI) | Paired wins/losses vs g3/s4 | McNemar p | Request p50/p95 | Episode wall p50 | Total wall / speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.0 | 4 | 43/50 = 0.86 [0.738, 0.930] | reference | - | 4110/4120ms | 78.9s | 85.6min / 1.00x |
| 3.0 | 2 | 45/50 = 0.90 [0.786, 0.957] | 5/3 | 0.727 | 2403/2413ms | 67.0s | 71.8min / 1.19x |
| 1.0 | 4 | 32/50 = 0.64 [0.501, 0.759] | 3/14 | 0.0127 | 2431/2440ms | 93.0s | 98.0min / 0.87x |
| 1.0 | 2 | 40/50 = 0.80 [0.670, 0.888] | 5/8 | 0.581 | 1565/1574ms | 111.7s | 101.1min / 0.85x |

Guidance 3.0 / two steps is the only retained accelerated sampler. It reduces
request latency by 1.71x while preserving CFG and did not show a quality loss.
Its end-to-end speedup is 1.19x because rendering/environment work remains and
its successful trajectory median increased from 213 to 239 steps. Removing CFG
reduced request latency but caused enough failures and longer trajectories to
make both guidance-1 settings slower end to end.

## Quantization And Sampler Interaction

Quantization and denoising controls are independent runtime knobs but are not
quality-orthogonal. All rows use guidance 3.0 / two steps and the same 50
paired Banana initial states.

| Strategy | Success (Wilson 95% CI) | Successful step median | Paired wins/losses vs full-W8 | McNemar p | Request p50/p95 | Peak reserved | Episode p50 / total wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full_w8` | 45/50 = 0.90 [0.786, 0.957] | 239 | reference | - | 2403/2413ms | 21.42GB | 67.0s / 71.8min |
| `gen_branch_w8` | 50/50 = 1.00 [0.929, 1.000] | 219 | 5/0 | 0.0625 | 2433/2442ms | 18.03GB | 59.2s / 54.1min |
| `attention_w8` | 39/50 = 0.78 [0.648, 0.872] | 348 | 4/10 | 0.180 | 2422/2433ms | 16.21GB | 106.7s / 97.1min |

`gen_branch_w8` is a strong Banana-specific configuration, but the cross-task
test below shows that its 50/50 result does not generalize. `attention_w8` lost
quality when combined with the two-step sampler and is not retained for that
combination.

## Paired Multi-Task Coverage

Three configurations were evaluated on 10 paired runs of three DROID-aligned
RoboLab tasks. The task horizons were 600 steps for RubiksCube, 450 for
MustardInRightBin, and 900 for SpoonInMug. All 90 initial-state hashes matched.

| Strategy / sampler | RubiksCube | MustardInRightBin | SpoonInMug | Aggregate (Wilson 95% CI) | Aggregate wall |
|---|---:|---:|---:|---:|---:|
| `full_w8` g3/s4 | 10/10 | 8/10 | 4/10 | 22/30 = 0.73 [0.556, 0.858] | 96.5min |
| `full_w8` g3/s2 | 10/10 | 8/10 | 7/10 | 25/30 = 0.83 [0.664, 0.927] | 70.5min |
| `gen_branch_w8` g3/s2 | 10/10 | 6/10 | 1/10 | 17/30 = 0.57 [0.392, 0.726] | 99.8min |

For `full_w8`, g3/s2 had seven paired wins and four losses against g3/s4
across the 30 runs (McNemar p=0.549). There is no observed cross-task quality
penalty, while aggregate wall time fell by 1.37x. Against `full_w8` g3/s2,
`gen_branch_w8` had two paired wins and ten losses (p=0.0386); on Spoon alone
it had zero wins and six losses (p=0.0313). The mixed strategy is therefore not
a general RoboLab default.

## Deployment Selection

| Scope | Recommended configuration | Evidence and tradeoff |
|---|---|---|
| General RoboLab default | `full_w8`, guidance 3.0, 2 steps | 21.42GB reserved; Banana 45/50; multi-task 25/30; ~2.4s request |
| Conservative rollback | `full_w8`, guidance 3.0, 4 steps | No sampler approximation; ~4.1s request; multi-task 22/30 |
| Validated Banana-only deployment | `gen_branch_w8`, guidance 3.0, 2 steps | 18.03GB; Banana 50/50; not transferable without per-task rollout |
| Low-memory Banana option | `attention_w8`, guidance 3.0, 4 steps | 16.21GB; Banana 42/50; longer trajectories |
| Experimental only | `full_w4` | 14.67GB but Banana 26/50; failed paired quality gate |

Do not deploy guidance 1.0 from these results. Do not choose a quantization
strategy from replay error alone. For a new task or robot, roll back to
`full_w8` g3/s4, validate replay and paired closed-loop behavior, then enable
g3/s2 and any mixed strategy independently.

Raw artifacts:

```text
/mnt/lixiangyu/cosmos_ws/robolab_validation/replay/benchmark_<strategy>_g3s4_replay8
/mnt/lixiangyu/cosmos_ws/robolab_validation/runs/q50_<strategy>_g3s4_20260711_r1
/mnt/lixiangyu/cosmos_ws/robolab_validation/runs/s50_full_w8_<sampler>_20260711
/mnt/lixiangyu/cosmos_ws/robolab_validation/runs/i50_<strategy>_g3s2_20260711
/mnt/lixiangyu/cosmos_ws/robolab_validation/runs/mt_<strategy>_<sampler>_20260711
/mnt/lixiangyu/cosmos_ws/RoboLab/output/robolab_<matching-run-name>
```
