# Quantized Robot Policy Runtime

This directory owns the locked, minimal policy-server environment shared by
the RoboCasa365 and RoboLab quantization pipelines.

For the reviewed release, use the immutable
`cosmos3-robot-policy-quant-v0.3-4090` tag. Do not deploy from an arbitrary
moving branch.

## Supported Runtime

- Linux x86-64, Python 3.13, NVIDIA driver compatible with CUDA 12.8.
- RTX 4090 / Ada is the release target; Ampere or newer is required by Marlin.
- PyTorch 2.10.0+cu128, vLLM 0.19.1, FlashAttention/NATTEN CUDA 12.8
  kernels, and OpenPI server/client 0.1.x.
- Packed weight-only W4A16/W8A16. Activations remain BF16.

The lock does not install Cosmos training extras, LeRobot, Megatron, RLDX,
RoboCasa, RoboLab, or IsaacSim. Simulator environments remain separate from
the policy server.

## Install

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 examples/quantized_robot_policy/setup.sh
```

This creates `examples/quantized_robot_policy/.venv` from the local
`uv.lock`, then verifies:

- all Cosmos plugin packages import from this checkout;
- `vllm._C` loads and registers `gptq_marlin_gemm`;
- OpenPI and ZMQ import;
- CUDA is visible.

Do not add ad hoc `PYTHONPATH` entries or copy site-packages from another
machine. A fresh checkout must pass this command before bundle validation.
The pipeline exports `COSMOS_TRAINING=0` so inference does not import optional
training, cloud-storage, or dataset backends.

## Choose A Pipeline

- [RoboCasa365](../robocasa365_quant/README.md): start from a fine-tuned BF16
  DCP checkpoint and training-set calibration captures.
- [RoboLab](../robolab_quant/README.md): start directly from the public
  `nvidia/Cosmos3-Nano-Policy-DROID` checkpoint.

Both pipelines expose the same lifecycle:

```text
setup -> build -> validate -> replay -> rollout
```

Deployment hosts need only this checkout, the locked runtime, and one
self-contained quantized bundle. Source checkpoints, calibration data,
tokenizer downloads, and VAE downloads are export-time inputs only.

## Artifact Rules

- Never modify a completed bundle in place.
- Always run strong validation after building or transferring a bundle.
- Record the bundle manifest hash, git tag, strategy, sampler, GPU, and rollout
  protocol with every result.
- Keep the simulator on a separate physical GPU when it uses CUDA/Vulkan.
- Roll back by switching to a previously validated immutable bundle; do not
  patch weights on a deployment host.

See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) for the tag gate.
