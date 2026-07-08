# A8 Backend Exploration

This is an experimental branch note for activation-quantized inference backend
prototyping. It is not part of the stable `robocasa365-quant-v0.1-4090`
deployment path.

## Scope

Branch:

```text
experiment/robocasa365-a8-backends
```

Goal:

- Check whether RTX 4090 native INT8/FP8 matmul paths are fast enough to justify
  a later calibrated W4A8/W8A8 policy implementation.
- Use real Cosmos MoT linear shapes from RoboCasa365 direct replay.
- Keep this speed-only until the backend shows clear latency upside.

Non-goals for this branch:

- No rollout validation.
- No activation calibration.
- No artifact format change.
- No server integration.
- No real-robot deployment change.

## Local API Findings

Validated local stack:

- PyTorch `2.10.0+cu128`
- RTX 4090, CUDA capability `(8, 9)`

Available and runnable:

- `torch.float8_e4m3fn`
- `torch._scaled_mm`
- `torch._int_mm`

Important details:

- `torch._scaled_mm` requires A row-major and B column-major layouts for the
  tested FP8 path. A contiguous `(K, N)` B matrix fails; use
  `q_weight.t().contiguous().t()` to produce a `(K, N)` column-major view.
- `torch.float8_e5m2` x `torch.float8_e5m2` was not accepted by
  `torch._scaled_mm` in this environment.
- `torch._int_mm` works for INT8 x INT8 -> INT32, but failed for
  `batch_tokens=10` with `self.size(0) needs to be greater than 16`.
- PyTorch exposes `float4_e2m1fn_x2` and int4 pack APIs, but this branch does
  not yet implement W4A8. That likely needs a separate packed-layout study or a
  mature CUTLASS/TensorRT path.

## Implemented Microbench Backends

The following are speed-only prototypes in
`cosmos_framework/scripts/quant_backend_microbench.py`:

| Backend | Description |
|---|---|
| `torch_scaled_mm_fp8_w8a8` | Tensor-wise FP8 W8A8; dynamic activation quantization in forward; `torch._scaled_mm` |
| `torch_int8_mm_w8a8` | Tensor-wise INT8 W8A8; dynamic activation quantization in forward; `torch._int_mm` plus dequant |

These use BF16 inputs and return BF16 outputs so they can be compared with the
existing microbench contract. They are not calibrated and should not be used as
robot policy backends.

## Reproduce Real-Shape Benchmark

First collect real linear shapes from direct replay:

```bash
source examples/robocasa365_quant/local_4090_env.example.sh

STRATEGY=attention_w8 \
QUANT_ARTIFACT_DIR=/path/to/quant_artifacts/attention_w8 \
CHECKPOINT_PATH=/path/to/iter_000008000 \
CONFIG_FILE=/path/to/config.yaml \
REPLAY_CAPTURE_DIR=/path/to/closefridge_action_parity_v1 \
CUDA_VISIBLE_DEVICES=0 \
REPLAY_LIMIT=1 \
LINEAR_SHAPE_PROFILE=1 \
PROFILE_TOOL=none \
RUN_DIR=/tmp/cosmos3_shape_profile_attention_w8 \
examples/robocasa365_quant/profile_direct_replay_4090.sh
```

Then run the A8 benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 "$COSMOS_PYTHON" \
  -m cosmos_framework.scripts.quant_backend_microbench \
  --shape-file /tmp/cosmos3_shape_profile_attention_w8/linear_shapes.jsonl \
  --max-shapes 8 \
  --backends bf16,vllm_gptq_marlin_w4a16,vllm_gptq_marlin_w8a16,vllm_allspark_w8a16,torch_scaled_mm_fp8_w8a8,torch_int8_mm_w8a8 \
  --chain none \
  --warmup 5 \
  --iters 20 \
  --output /tmp/cosmos3_operator_microbench_attention_w8_top8_a8.json
```

## Initial Results

Run:

```text
/tmp/cosmos3_operator_microbench_attention_w8_top8_a8_20260708_210726.json
```

Weighted top-8 result:

| Backend | Benchmarked shapes | Weighted candidate ms | Weighted ratio vs BF16 |
|---|---:|---:|---:|
| BF16 | 8 | 0.1818 | 0.920 |
| vLLM Marlin W4A16 | 8 | 0.1424 | 0.849 |
| vLLM Marlin W8A16 | 8 | 0.1470 | 0.875 |
| vLLM AllSpark W8A16 | 8 | 0.1583 | 0.938 |
| FP8 W8A8 dynamic | 8 | 0.1636 | 0.739 |
| INT8 W8A8 dynamic | 6 | 0.3274 | 1.736 |

Interpretation:

- Full FP8 W8A8 replacement is slower than the current Marlin W4/W8 top-8
  weighted result.
- INT8 W8A8 through `torch._int_mm` is not competitive in this implementation
  and does not support the `batch_tokens=10` shape.
- FP8 is promising for selected large shapes, especially
  `518 x 4096 -> 12288`, where it was about 23% faster than current Marlin W4.
- FP8 is much worse on small attention projection shapes such as
  `10/82 x 4096 -> 1024`.

Selective FP8 upper bound on this top-8 set:

- Current policy weighted sum: `453.25`
- Use FP8 only when faster than current backend: `401.88`
- Top-8 quant-linear saving: `11.33%`
- Rough end-to-end GPU-kernel-time upper bound using prior Marlin share
  `65.9%`: about `7.5%`

This is enough to justify a narrow follow-up, but not enough to replace the
stable W4/W8A16 release path.

## Recommended Next Steps

1. Add a kernel-only FP8 microbench variant that separates activation
   quantization overhead from matmul speed.
2. Test row-wise or block-wise FP8 scales if `torch._scaled_mm` supports a
   layout compatible with the Cosmos shapes.
3. If FP8 large-shape speed remains strong, prototype a per-shape runtime
   policy for only the high-payoff MLP shapes.
4. Only after speed is clearly positive, add activation calibration and compare
   BF16 vs quantized actions on replay captures.
5. W4A8 should be a separate investigation using a mature CUTLASS/TensorRT or
   PyTorch float4/int4 packed path; do not hand-roll it into the policy server
   without a standalone benchmark first.

