# RoboCasa365 Quant Productization Notes

Date: 2026-07-07.

## Why Quantized Marlin Is Not Faster Yet

The current implementation proves memory viability first. It wraps individual
`nn.Linear` modules with packed Marlin W4/W8 backends and calls
`torch.ops._C.marlin_gemm` module by module.

Direct replay showed memory wins but no latency win over BF16:

| Strategy | Peak alloc | Generate p50/p95 |
|---|---:|---:|
| BF16 | 33.09GB | 702 / 883 ms |
| full_w8 | 19.20GB | 794 / 887 ms |
| full_w4 | 12.48GB | 860 / 1052 ms |
| attention_w8 | 13.94GB | 766 / 887 ms |
| gen_branch_w8 | 15.84GB | 815 / 997 ms |

Likely causes:

1. The action policy request path has small effective batch/token shapes. Marlin
   dequant and packing metadata overhead is harder to amortize than in high
   throughput LLM serving.
2. The integration replaces individual Linear modules. It does not fuse
   attention projections, MLP blocks, MoE branch routing, residuals, or norm
   operations.
3. Each quantized Linear still pays Python module overhead, reshape/contiguous
   handling, optional input-scale application, BF16 output conversion, and bias
   addition.
4. BF16 on H100 uses very strong cuBLAS/Tensor Core kernels. W8 weight-only is
   mainly a memory optimization here and can lose on latency when dequant
   overhead dominates.
5. The RoboCasa server currently evaluates each environment item through
   `_infer_one` rather than batching all env observations into one model call.
6. The current path disables torch compile and does not use CUDA graphs.

## 4090 Work Split

Inference speed optimization can move to RTX 4090 once quant artifacts exist.
The direct-load inference path for all retained candidates is below 24GB, so
local replay latency and memory validation are the right next target.

Caveats:

- Exporting new quant artifacts from BF16 may still need H100 or a CPU/streaming
  export path if the exporter materializes the full BF16 model on GPU.
- A fresh 4090 deployment must validate that the selected `torch`/`vllm` build
  exposes compatible Marlin kernels for Ada / SM89.
- Rollout can move local only if the RoboCasa/RLDX simulator environment is
  reproducible on that machine. Direct replay is enough for kernel/runtime
  iteration.

## Productization Boundary

This overlay is intended to become a cosmos-framework feature branch. It
contains:

- a quant-capable RoboCasa365 policy server;
- fixed strategy names and plan generation;
- command generation for export, serve, replay, and rollout;
- documentation of the validated protocol and current runtime caveats.

Next implementation work should focus on:

- reducing per-Linear Python overhead;
- batching per-env inference inside the policy server;
- evaluating CUDA graph capture for fixed replay shapes;
- comparing Marlin vs alternative W8/W4 kernels on 4090;
- adding a CPU/streaming export path if local 4090 quantization from BF16 is
  required.
