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
| `torch_scaled_mm_fp8_w8a8_kernel_only` | Tensor-wise FP8 W8A8; fixed benchmark activation is prequantized; times `torch._scaled_mm` only |
| `torch_int8_mm_w8a8` | Tensor-wise INT8 W8A8; dynamic activation quantization in forward; `torch._int_mm` plus dequant |

These use BF16 inputs and return BF16 outputs so they can be compared with the
existing microbench contract. They are not calibrated and should not be used as
robot policy backends.

The `kernel_only` FP8 backend is even narrower: it stores a prequantized copy of
the fixed benchmark input. It is only for separating FP8 matmul kernel speed
from dynamic activation quantization overhead.

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
  --backends bf16,vllm_gptq_marlin_w4a16,vllm_gptq_marlin_w8a16,vllm_allspark_w8a16,torch_scaled_mm_fp8_w8a8,torch_scaled_mm_fp8_w8a8_kernel_only,torch_int8_mm_w8a8 \
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

## FP8 Kernel-Only Follow-Up

Run:

```text
/tmp/cosmos3_operator_microbench_attention_w8_top8_fp8_kernel_only_20260708_215407.json
```

Weighted top-8 result:

| Backend | Benchmarked shapes | Weighted candidate ms | Weighted ratio vs BF16 |
|---|---:|---:|---:|
| BF16 | 8 | 0.1698 | 0.973 |
| vLLM Marlin W4A16 | 8 | 0.1440 | 0.849 |
| vLLM Marlin W8A16 | 8 | 0.1492 | 0.880 |
| FP8 W8A8 dynamic | 8 | 0.1664 | 0.981 |
| FP8 W8A8 kernel-only | 8 | 0.0882 | 0.522 |

Shape-level result:

| Count | Example | Shape | BF16 ms | Marlin W4 ms | Marlin W8 ms | FP8 dyn ms | FP8 kernel ms | Dynamic quant overhead ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 576 | `mlp_moe_gen.gate_proj` | `518x4096->12288` | 0.4183 | 0.3520 | 0.3651 | 0.2362 | 0.1952 | 0.0410 |
| 576 | `self_attn.k_proj_moe_gen` | `518x4096->1024` | 0.0422 | 0.0467 | 0.0480 | 0.1383 | 0.0328 | 0.1055 |
| 576 | `self_attn.o_proj_moe_gen` | `518x4096->4096` | 0.1407 | 0.1228 | 0.1273 | 0.1327 | 0.0740 | 0.0587 |
| 288 | `mlp.gate_proj` | `10x4096->12288` | 0.1192 | 0.0347 | 0.0415 | 0.1167 | 0.0310 | 0.0857 |
| 288 | `mlp.gate_proj` | `82x4096->12288` | 0.1242 | 0.0846 | 0.0867 | 0.1211 | 0.0474 | 0.0737 |
| 288 | `mlp_moe_gen.down_proj` | `518x12288->4096` | 0.3735 | 0.3454 | 0.3581 | 0.3448 | 0.2172 | 0.1276 |
| 288 | `self_attn.k_proj` | `10x4096->1024` | 0.0302 | 0.0345 | 0.0340 | 0.1022 | 0.0331 | 0.0691 |
| 288 | `self_attn.k_proj` | `82x4096->1024` | 0.0177 | 0.0413 | 0.0402 | 0.1308 | 0.0380 | 0.0928 |

Interpretation:

- Ada FP8 matmul itself is fast enough to matter: the kernel-only top-8
  weighted average is `0.0882 ms`, about `39%` faster than Marlin W4 on the
  same shape mix.
- The current dynamic FP8 path is not deployable for speed. Its weighted
  average is `0.1664 ms`, slower than Marlin W4 and W8 because activation
  quantization is implemented as eager PyTorch pointwise work before
  `_scaled_mm`.
- A production A8 path must use fused or otherwise low-overhead activation
  quantization. Without that, adding A8 into the policy server would likely
  increase latency despite good FP8 Tensor Core throughput.
- The highest-value follow-up is not more model-level routing yet. It is an
  operator-level implementation study for fused/blockwise activation quant plus
  FP8 GEMM on the large MLP shapes.

## Recommended Next Steps

1. Test row-wise or block-wise FP8 scales if `torch._scaled_mm` supports a
   layout compatible with the Cosmos shapes.
2. Survey and benchmark mature fused A8 paths on RTX 4090, especially
   CUTLASS/cuBLASLt/TensorRT-style FP8 GEMM integrations that avoid eager
   activation quant overhead.
3. If FP8 large-shape speed remains strong with fused activation quant,
   prototype a per-shape runtime
   policy for only the high-payoff MLP shapes.
4. Only after speed is clearly positive, add activation calibration and compare
   BF16 vs quantized actions on replay captures.
5. W4A8 should be a separate investigation using a mature CUTLASS/TensorRT or
   PyTorch float4/int4 packed path; do not hand-roll it into the policy server
   without a standalone benchmark first.
