# RoboCasa365 Latency Motivation Study

This note records the first single-robot latency study after the stable
weight-only quantized release. It focuses on optimizations that could plausibly
beat small kernel/backend gains.

Branch:

```text
experiment/robocasa365-a8-backends
```

Local 4090 run root:

```text
/home/lixiangyu/cosmos_ws/local_4090_validation/logs/token_motivation_20260708_234136
```

Artifact:

```text
/home/lixiangyu/cosmos_ws/local_4090_validation/quant_artifacts/attention_w8
```

Replay set:

```text
/home/lixiangyu/cosmos_ws/local_4090_validation/data/closefridge_action_parity_v1
```

All runs used `REPLAY_LIMIT=8`, `PROFILE_TOOL=none`,
`LINEAR_SHAPE_PROFILE=1`, and the local iter8000/config setup.

## Bottleneck Check

The policy server now supports optional denoiser profiling through:

```bash
COSMOS3_DENOISER_PROFILE_JSONL=/path/to/denoiser_profile.jsonl
```

Baseline:

| Setting | Value |
|---|---|
| `guidance` | `3.0` |
| `num_steps` | `4` |
| `action_chunk_size` | `32` |
| `served_action_steps` | `8` |
| `resolution` | `256` |
| `camera_size` | `256` |
| `view_mode` | `concat3` |

Baseline result:

| Metric | Value |
|---|---:|
| Steady generate p50 | `1178.6 ms` |
| Steady request p50 | `1272.2 ms` |
| Denoiser forwards | `64` over 8 requests |
| Velocity steps | `32` over 8 requests |
| Denoiser forward p50 | `128.6 ms` |

Interpretation:

- `num_steps=4` with `guidance=3.0` issues 8 denoiser forwards per request
  because CFG runs conditional and unconditional passes.
- Denoiser/DiT compute is the primary single-robot latency bottleneck.
- This makes CFG removal/distillation and step reduction higher-value than
  additional FP8 kernel work.

## Motivation Variants

| Variant | Generate p50 | Request p50 | Generate speedup | Denoiser forwards | Velocity steps | Linear work ratio | L1 mean | Linf p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_concat3` | 1178.6 | 1272.2 | 1.00x | 64 | 32 | 1.00 | 0.0547 | 0.579 |
| `guidance1` | 630.4 | 989.9 | 1.87x | 32 | 32 | 0.53 | 0.0526 | 0.611 |
| `steps2` | 671.7 | 1007.7 | 1.75x | 32 | 16 | 0.50 | 0.0526 | 0.576 |
| `wrist256` | 1092.8 | 1125.3 | 1.08x | 64 | 32 | 1.16 | 0.0793 | 0.750 |
| `wrist_left` | 1088.6 | 1305.9 | 1.08x | 64 | 32 | 0.94 | 0.0671 | 0.650 |
| `chunk16` | 1224.5 | 1408.9 | 0.96x | 64 | 32 | 0.59 | 0.0456 | 0.395 |
| `chunk8` | 1253.0 | 1342.8 | 0.94x | 64 | 32 | 0.38 | 0.0598 | 0.446 |
| `camera192` | 1162.8 | 1340.8 | 1.01x | 64 | 32 | 1.00 | 0.0534 | 0.585 |

`Linear work ratio` is:

```text
sum(count * batch_tokens * in_features * out_features) / baseline
```

Notes:

- `resolution=192` and `resolution=128` are not currently valid for this model
  config. The framework rejected them because `VIDEO_RES_SIZE_INFO` only
  contains `256`, `480`, `704`, `720`, `768`, `1080`, `1280`, `2048`, and
  `gt_2048`.
- `camera192` only pre-resizes camera frames before composing the image. The
  model still receives resolution `256`, so it is not a true token-resolution
  reduction.
- `wrist` and `wrist_left` change the camera composition. They do not provide a
  compelling speed/quality tradeoff in this quick open-loop study.
- `chunk16` and `chunk8` reduce linear work proxy but do not reduce latency.
  This suggests action-token count is not the dominant runtime driver for the
  current path.

## Conclusions

1. CFG is the strongest immediate optimization target.
   - `guidance=1.0` halves denoiser forward count and gives `1.87x` generate
     p50 speedup.
   - Open-loop action deviation does not explode on this tiny replay set, but
     rollout is required before treating CFG-free inference as valid.

2. Step reduction is also high-value.
   - `num_steps=2` halves velocity steps and gives `1.75x` generate p50 speedup.
   - The measured gain is below ideal 2x because request/build overhead and
     non-denoiser work remain.

3. Simple token compression is not yet a proven path.
   - Camera/view ablations do not show clean speedup and increase action error.
   - Action chunk reduction lowers the linear work proxy but does not lower
     observed latency.
   - True sub-256 visual token studies require model/config support for smaller
     `VIDEO_RES_SIZE_INFO` entries or a separate trained setting.

4. The next engineering experiment should be quality-gated CFG/step ablation:
   - `guidance=1.0`, `num_steps=4`
   - `guidance=3.0`, `num_steps=2`
   - `guidance=1.0`, `num_steps=2`
   - Compare replay32 parity and short rollout before any 50-episode run.

## CFG / Step Quality Gate

Date: 2026-07-09.

Common configuration:

- checkpoint: `iter_000008000`
- config: `robocasa365_repro/config.yaml`
- quant strategy/artifact: `attention_w8`
- action chunk: `32`, served action steps: `8`
- replay capture: `closefridge_action_parity_v1`, limit `32`
- rollout protocol: `CloseFridge`, `target`, `N_EPISODES=5`, `N_ENVS=1`,
  `MAX_EPISODE_STEPS=1200`, `USE_TASK_HORIZON=0`, video disabled

Replay32 was run on a local RTX 4090. It is the open-loop parity and local
latency gate.

| Variant | Denoiser forwards/request | Generate p50/p95 | Request p50/p95 | Generate speedup | L1 mean/p95 | Linf mean/p95 |
|---|---:|---:|---:|---:|---:|---:|
| `guidance3_steps4` baseline | 8 | 1063/1190 ms | 1250/1419 ms | 1.00x | 0.0428/0.0847 | 0.306/0.773 |
| `guidance1_steps4` | 4 | 614/666 ms | 771/985 ms | 1.73x | 0.0404/0.0830 | 0.289/0.599 |
| `guidance3_steps2` | 4 | 633/694 ms | 793/951 ms | 1.68x | 0.0435/0.0744 | 0.284/0.527 |
| `guidance1_steps2` | 2 | 382/444 ms | 532/695 ms | 2.78x | 0.0412/0.0743 | 0.271/0.520 |

Replay32 run root:
`/home/lixiangyu/cosmos_ws/local_4090_validation/logs/guidance_step_gate_20260709_003628`.

Short rollout was run on `aigc28` H100. These timings are not directly
comparable to local 4090 timings, but they confirm rollout-path behavior and
relative denoising cost.

| Variant | Success | Requests | Generate p50/p95 | Request p50/p95 | Peak alloc/reserved |
|---|---:|---:|---:|---:|---:|
| `guidance1_steps4` | 5/5 | 274 | 419/444 ms | 433/473 ms | 13.94/14.19 GB |
| `guidance3_steps2` | 5/5 | 278 | 430/445 ms | 455/481 ms | 13.94/14.19 GB |
| `guidance1_steps2` | 4/5 | 326 | 242/262 ms | 255/284 ms | 13.94/14.19 GB |

Short rollout shared run root:
`/mnt/100T/lixiangyu/cosmos3_robocasa365_opt/rollouts/guidance_step_short_*_20260709_003737`.

Interpretation:

- All three variants passed replay32 parity on the fixed capture set.
- `guidance1_steps2` is the fastest setting, but its first short rollout was
  `4/5`; do not promote it directly to a 50-episode gate without at least one
  repeat.
- `guidance1_steps4` and `guidance3_steps2` both passed the short rollout
  smoke at `5/5`. They are the safer candidates for a full 50-episode repeat.
- Because `guidance1_steps4` and `guidance3_steps2` both use four denoiser
  forwards per request, their rollout latency is similar. On the local 4090
  replay gate, `guidance1_steps4` was slightly faster.

## Student Policy / Distillation Survey

This direction is a research project, but it is the most plausible route to
order-of-magnitude single-robot speedups.

| Work | Main idea | Reported relevance |
|---|---|---|
| OneDP: Fast Visuomotor Policies via Diffusion Distillation | Distill a pretrained diffusion policy into a one-step action generator with KL matching along the diffusion chain | Reports `2%-10%` extra pretraining cost and action frequency from `1.5 Hz` to `62 Hz` |
| Consistency Policy | Distill a diffusion policy into a few-step consistency policy | Reports order-of-magnitude faster inference while maintaining competitive success |
| ManiCM | Consistency model for 3D diffusion policy, one-step action inference | Reports about `10x` average inference speedup on manipulation tasks |
| LightDP / On-Device Diffusion Transformer Policy | Compress denoising modules and reduce sampling steps for on-device deployment | Targets real-time mobile deployment with pruning/retraining plus consistency distillation |

Sources:

- OneDP project: <https://research.nvidia.com/labs/cosmos-lab/onedp/>
- OneDP paper: <https://arxiv.org/abs/2410.21257>
- Consistency Policy: <https://arxiv.org/abs/2405.07503>
- Consistency Policy code: <https://github.com/Aaditya-Prasad/Consistency-Policy/>
- ManiCM: <https://arxiv.org/abs/2406.01586>
- ManiCM project/code: <https://manicm-fast.github.io/> and <https://github.com/ManiCM-fast/ManiCM>
- LightDP: <https://arxiv.org/abs/2508.00697>

For Cosmos/RoboCasa365, a reasonable research plan is:

1. Treat Cosmos as the teacher and collect teacher action chunks on RoboCasa365
   training/replay observations.
2. Train a small student with the same observation/action contract first, before
   trying one-step diffusion distillation.
3. Use the existing open-loop replay parity harness as a cheap filter.
4. Run short rollouts, then 50-episode protocol only for promising students.
5. If a plain student underfits, add consistency/diffusion distillation rather
   than returning to low-level kernel work.
