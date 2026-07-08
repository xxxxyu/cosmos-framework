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
| `torch_scaled_mm_fp8_w8a8_static_scale` | Tensor-wise FP8 W8A8; fixed activation scale; eager quantization; `torch._scaled_mm` |
| `torch_scaled_mm_fp8_w8a8_kernel_only` | Tensor-wise FP8 W8A8; fixed benchmark activation is prequantized; times `torch._scaled_mm` only |
| `vllm_cutlass_fp8_w8a8_dynamic` | Tensor-wise FP8 W8A8; vLLM `scaled_fp8_quant` dynamic activation quantization; vLLM CUTLASS scaled-mm |
| `vllm_cutlass_fp8_w8a8_static_scale` | Tensor-wise FP8 W8A8; fixed activation scale; vLLM `scaled_fp8_quant`; vLLM CUTLASS scaled-mm |
| `vllm_cutlass_fp8_w8a8_kernel_only` | Tensor-wise FP8 W8A8; fixed benchmark activation is prequantized; times vLLM CUTLASS scaled-mm only |
| `torch_int8_mm_w8a8` | Tensor-wise INT8 W8A8; dynamic activation quantization in forward; `torch._int_mm` plus dequant |

These use BF16 inputs and return BF16 outputs so they can be compared with the
existing microbench contract. They are not calibrated and should not be used as
robot policy backends.

The `kernel_only` FP8 backend is even narrower: it stores a prequantized copy of
the fixed benchmark input. It is only for separating FP8 matmul kernel speed
from dynamic activation quantization overhead.

The vLLM CUTLASS experiments call vLLM's low-level `_custom_ops` directly. The
high-level `QuantFP8` class imports broader vLLM model/config dependencies in
this local lightweight environment, so the benchmark avoids that extra import
surface and uses `scaled_fp8_quant` plus `cutlass_scaled_mm`.

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

## vLLM CUTLASS and Static-Scale Follow-Up

Run:

```text
/tmp/cosmos3_operator_microbench_attention_w8_top8_a8_vllm_cutlass_20260708_223859.json
```

This run adds:

- PyTorch `_scaled_mm` with a fixed activation scale, simulating calibrated
  static per-tensor activation scaling.
- vLLM `scaled_fp8_quant` plus vLLM CUTLASS scaled-mm, both dynamic and static.
- vLLM CUTLASS kernel-only, to compare CUTLASS GEMM speed against PyTorch
  `_scaled_mm` without activation quantization.

Weighted top-8 result:

| Backend | Weighted candidate ms | Saving vs current top-8 policy |
|---|---:|---:|
| Current policy (`source_backend_class`) | 0.1419 | 0.0% |
| vLLM Marlin W4A16 everywhere | 0.1392 | 1.9% |
| vLLM Marlin W8A16 everywhere | 0.1435 | -1.1% |
| PyTorch FP8 dynamic | 0.1439 | -1.4% |
| PyTorch FP8 static-scale | 0.1156 | 18.6% |
| PyTorch FP8 kernel-only | 0.0896 | 36.9% |
| vLLM CUTLASS FP8 dynamic | 0.1422 | -0.2% |
| vLLM CUTLASS FP8 static-scale | 0.1335 | 5.9% |
| vLLM CUTLASS FP8 kernel-only | 0.1222 | 13.9% |

Large MLP subset:

| Count | Example | Shape | Current Marlin W4 ms | PyTorch FP8 dyn ms | PyTorch FP8 static ms | PyTorch FP8 kernel ms | vLLM CUTLASS dyn ms | vLLM CUTLASS static ms | vLLM CUTLASS kernel ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 576 | `mlp_moe_gen.gate_proj` | `518x4096->12288` | 0.3501 | 0.2333 | 0.2181 | 0.1945 | 0.2625 | 0.2558 | 0.2476 |
| 288 | `mlp.gate_proj` | `82x4096->12288` | 0.0918 | 0.1052 | 0.0591 | 0.0435 | 0.0706 | 0.0642 | 0.0616 |
| 288 | `mlp_moe_gen.down_proj` | `518x12288->4096` | 0.3289 | 0.2967 | 0.2809 | 0.2098 | 0.3596 | 0.3631 | 0.3290 |

Weighted large MLP result:

| Backend | Weighted candidate ms | Saving vs current Marlin W4 |
|---|---:|---:|
| Current Marlin W4 | 0.2802 | 0.0% |
| PyTorch FP8 dynamic | 0.2171 | 22.5% |
| PyTorch FP8 static-scale | 0.1941 | 30.7% |
| PyTorch FP8 kernel-only | 0.1605 | 42.7% |
| vLLM CUTLASS FP8 dynamic | 0.2388 | 14.8% |
| vLLM CUTLASS FP8 static-scale | 0.2347 | 16.2% |
| vLLM CUTLASS FP8 kernel-only | 0.2214 | 21.0% |

If only the large MLP subset is replaced and other top-8 shapes keep the current
policy, the top-8 weighted saving is:

| Replacement for large MLP only | Top-8 saving | Rough end-to-end upper bound using prior 65.9% Marlin kernel share |
|---|---:|---:|
| PyTorch FP8 dynamic | 16.2% | 10.7% |
| PyTorch FP8 static-scale | 22.1% | 14.5% |
| PyTorch FP8 kernel-only | 30.7% | 20.2% |
| vLLM CUTLASS FP8 dynamic | 10.6% | 7.0% |
| vLLM CUTLASS FP8 static-scale | 11.7% | 7.7% |
| vLLM CUTLASS FP8 kernel-only | 15.1% | 9.9% |

Interpretation:

- Static activation scale matters. Removing the runtime `amax` reduction improves
  PyTorch FP8 on the large MLP subset from `22.5%` to `30.7%` saving vs Marlin
  W4.
- The current vLLM CUTLASS path is not the best 4090 backend for these shapes.
  Even its kernel-only large-MLP result is slower than PyTorch `_scaled_mm`
  kernel-only (`0.2214 ms` vs `0.1605 ms` weighted).
- vLLM's low-level `scaled_fp8_quant` is useful as a mature static/dynamic
  quantization reference, but its CUTLASS GEMM wrapper is not a drop-in speed
  win for this Ada setup.
- The best speed-only next candidate is calibrated static activation scale plus
  PyTorch `_scaled_mm` for selected large MLP layers. This still needs action
  parity validation because static activation scales can saturate outlier
  inputs if calibration coverage is insufficient.

## Recommended Next Steps

1. Add calibration collection for per-layer static FP8 activation scales on
   training/replay data and measure saturation rate.
2. Compare action parity for large-MLP-only PyTorch FP8 static-scale against the
   BF16/current-quant baselines.
3. If parity is acceptable, prototype a per-shape runtime policy for only the
   high-payoff MLP shapes.
4. Keep vLLM CUTLASS as a reference, but do not prioritize it for 4090 unless a
   newer kernel or layout variant beats PyTorch `_scaled_mm` on these shapes.
5. W4A8 should be a separate investigation using a mature CUTLASS/TensorRT or
   PyTorch float4/int4 packed path; do not hand-roll it into the policy server
   without a standalone benchmark first.
