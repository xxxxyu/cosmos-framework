#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Command helpers for Cosmos3 RoboCasa365 weight-only quantization.

This module intentionally keeps the first productized surface small:

- fixed strategy names that map to reviewed W4/W8 quantization plans;
- artifact manifests with enough metadata for deployment selection;
- generated commands for export, serving, replay, and rollout.

The heavy lifting is done by ``action_policy_server_robocasa365_quant.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cosmos_framework.scripts.robocasa365_quant_bundle import (
    BUNDLE_SCHEMA_VERSION,
    build_self_contained_bundle,
    validate_self_contained_bundle,
)

_BACKEND_BITS = {
    "VllmGptqMarlinW4A16Linear": 4,
    "VllmGptqMarlinW8A16Linear": 8,
}
_BACKEND_CLASSES = {
    "vllm_gptq_marlin_w4a16": "VllmGptqMarlinW4A16Linear",
    "vllm_gptq_marlin_w8a16": "VllmGptqMarlinW8A16Linear",
}
_SUPPORTED_FORMATS = {"vllm_marlin_wna16"}


@dataclass(frozen=True)
class QuantStrategy:
    name: str
    description: str
    plan: list[dict[str, str]]
    expected_peak_alloc_gb: float | None
    m13_success_rate: float | None


STRATEGIES: dict[str, QuantStrategy] = {
    "full_w8": QuantStrategy(
        name="full_w8",
        description="All language-layer Linear modules use packed Marlin W8A16.",
        plan=[
            {
                "prefix": "net.language_model.model.layers",
                "backend": "vllm_gptq_marlin_w8a16",
            }
        ],
        expected_peak_alloc_gb=19.20,
        m13_success_rate=0.96,
    ),
    "full_w4": QuantStrategy(
        name="full_w4",
        description="All language-layer Linear modules use packed Marlin W4A16.",
        plan=[
            {
                "prefix": "net.language_model.model.layers",
                "backend": "vllm_gptq_marlin_w4a16",
            }
        ],
        expected_peak_alloc_gb=12.48,
        m13_success_rate=0.92,
    ),
    "attention_w8": QuantStrategy(
        name="attention_w8",
        description="Language self-attention Linear modules use W8A16; remaining language Linear modules use W4A16.",
        plan=[
            {
                "prefix": "net.language_model.model.layers",
                "backend": "vllm_gptq_marlin_w4a16",
            },
            {
                "prefix": "net.language_model.model.layers",
                "name_regex": "\\.self_attn\\.",
                "backend": "vllm_gptq_marlin_w8a16",
            },
        ],
        expected_peak_alloc_gb=13.94,
        m13_success_rate=0.96,
    ),
    "gen_branch_w8": QuantStrategy(
        name="gen_branch_w8",
        description="MoT generation-branch attention and MLP Linear modules use W8A16; remaining language Linear modules use W4A16.",
        plan=[
            {
                "prefix": "net.language_model.model.layers",
                "backend": "vllm_gptq_marlin_w4a16",
            },
            {
                "prefix": "net.language_model.model.layers",
                "name_regex": "\\.self_attn\\.(q_proj_moe_gen|k_proj_moe_gen|v_proj_moe_gen|o_proj_moe_gen)$",
                "backend": "vllm_gptq_marlin_w8a16",
            },
            {
                "prefix": "net.language_model.model.layers",
                "name_regex": "\\.mlp_moe_gen\\.",
                "backend": "vllm_gptq_marlin_w8a16",
            },
        ],
        expected_peak_alloc_gb=15.84,
        m13_success_rate=0.94,
    ),
}


def _quote_parts(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _strategy(name: str) -> QuantStrategy:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        raise SystemExit(f"Unknown strategy {name!r}. Choose one of: {', '.join(sorted(STRATEGIES))}") from exc


def write_strategy_configs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for strategy in STRATEGIES.values():
        path = output_dir / f"{strategy.name}.json"
        path.write_text(json.dumps(strategy.plan, indent=2) + "\n")
    manifest = {
        strategy.name: {
            "description": strategy.description,
            "config": f"{strategy.name}.json",
            "expected_peak_alloc_gb": strategy.expected_peak_alloc_gb,
            "m13_success_rate": strategy.m13_success_rate,
        }
        for strategy in STRATEGIES.values()
    }
    (output_dir / "strategies_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_artifact_manifest(args: argparse.Namespace) -> None:
    strategy = _strategy(args.strategy)
    root = Path(args.quant_artifact_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "strategy": strategy.name,
        "description": strategy.description,
        "checkpoint_path": args.checkpoint_path,
        "config_file": args.config_file,
        "calib_capture_dir": args.calib_capture_dir,
        "calib_limit": args.calib_limit,
        "calib_alpha": args.calib_alpha,
        "weight_only": True,
        "activation_quant": False,
        "runtime_backend": "vllm_marlin_wna16",
        "plan": strategy.plan,
        "expected_peak_alloc_gb": strategy.expected_peak_alloc_gb,
        "m13_success_rate": strategy.m13_success_rate,
    }
    (root / "cosmos3_quant_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"quant artifact manifest field {field!r} must be a non-empty string")
    return value


def _backend_for_name(plan: list[dict[str, str]], name: str) -> str:
    selected = "none"
    for row in plan:
        prefix = row.get("prefix", "")
        if prefix and not name.startswith(prefix):
            continue
        name_regex = row.get("name_regex", "")
        if name_regex and not re.search(name_regex, name):
            continue
        exclude_regex = row.get("exclude_regex", "")
        if exclude_regex and re.search(exclude_regex, name):
            continue
        selected = row.get("backend", "none")
    return selected


def validate_quant_artifact(
    quant_artifact_dir: str | Path,
    *,
    expected_strategy: str | None = None,
    check_tensors: bool = False,
    require_self_contained: bool = False,
) -> dict[str, Any]:
    """Validate a packed quant artifact manifest before serving or replay.

    The default check is intentionally metadata-only so it is safe to run in
    CI and before large local replays. ``check_tensors=True`` additionally
    verifies that each tensor file can be opened by ``torch.load``.
    """

    root = Path(quant_artifact_dir).expanduser()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing quant artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise TypeError(f"quant artifact manifest must be a dict, got {type(manifest)}")

    schema_version = manifest.get("schema_version")
    if require_self_contained and schema_version != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            "Quantized deployment requires a self-contained schema-v2 bundle; "
            f"artifact has schema_version={schema_version!r}"
        )
    bundle_validation: dict[str, Any] | None = None
    if schema_version == BUNDLE_SCHEMA_VERSION:
        bundle_validation = validate_self_contained_bundle(root, check_hashes=check_tensors)
    elif schema_version != 1:
        raise ValueError(
            f"Unsupported quant artifact schema_version={schema_version!r}; "
            f"expected 1 or {BUNDLE_SCHEMA_VERSION}"
        )

    modules = manifest.get("modules")
    if not isinstance(modules, list) or not modules:
        raise TypeError(f"quant artifact manifest modules must be a non-empty list, got {type(modules)}")

    expected_plan: list[dict[str, str]] | None = None
    if expected_strategy is not None:
        expected_plan = _strategy(expected_strategy).plan

    metadata_path = root / "cosmos3_quant_metadata.json"
    metadata: dict[str, Any] | None = None
    if metadata_path.is_file():
        loaded_metadata = json.loads(metadata_path.read_text())
        if not isinstance(loaded_metadata, dict):
            raise TypeError(f"quant artifact metadata must be a dict, got {type(loaded_metadata)}")
        metadata = loaded_metadata
        if expected_strategy is not None and metadata.get("strategy") != expected_strategy:
            raise ValueError(
                f"quant artifact strategy mismatch: expected {expected_strategy!r}, "
                f"metadata has {metadata.get('strategy')!r}"
            )
    elif schema_version == BUNDLE_SCHEMA_VERSION:
        quantization = manifest.get("quantization", {})
        if isinstance(quantization, dict):
            metadata = quantization
        if expected_strategy is not None and quantization.get("strategy") != expected_strategy:
            raise ValueError(
                f"quant artifact strategy mismatch: expected {expected_strategy!r}, "
                f"bundle has {quantization.get('strategy')!r}"
            )

    seen_names: set[str] = set()
    seen_files: set[str] = set()
    counts: dict[str, int] = {}
    for idx, entry in enumerate(modules):
        if not isinstance(entry, dict):
            raise TypeError(f"quant artifact module entry {idx} must be a dict, got {type(entry)}")
        name = _require_string(entry.get("name"), f"modules[{idx}].name")
        if name in seen_names:
            raise ValueError(f"Duplicate quant artifact module name: {name}")
        seen_names.add(name)

        backend_class = _require_string(entry.get("backend_class"), f"modules[{idx}].backend_class")
        fmt = _require_string(entry.get("format"), f"modules[{idx}].format")
        if fmt not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported quant artifact format for {name}: {fmt!r}")
        if backend_class not in _BACKEND_BITS:
            raise ValueError(f"Unsupported quant artifact backend_class for {name}: {backend_class!r}")
        if expected_plan is not None:
            expected_backend = _backend_for_name(expected_plan, name)
            expected_backend_class = _BACKEND_CLASSES.get(expected_backend)
            if expected_backend_class != backend_class:
                raise ValueError(
                    f"Quant artifact {name} has backend_class={backend_class!r}, "
                    f"but strategy {expected_strategy!r} expects {expected_backend_class!r}"
                )

        expected_bits = _BACKEND_BITS[backend_class]
        if int(entry.get("num_bits", -1)) != expected_bits:
            raise ValueError(
                f"Quant artifact {name} has num_bits={entry.get('num_bits')!r}, "
                f"but {backend_class} expects {expected_bits}"
            )
        group_size = int(entry.get("group_size", 0))
        if expected_bits == 4 and group_size <= 0:
            raise ValueError(f"Quant artifact {name} has invalid W4 group_size={entry.get('group_size')!r}")
        if expected_bits == 8 and group_size not in {-1, 0} and group_size <= 0:
            raise ValueError(f"Quant artifact {name} has invalid W8 group_size={entry.get('group_size')!r}")
        if int(entry.get("size_k", 0)) <= 0 or int(entry.get("size_n", 0)) <= 0:
            raise ValueError(
                f"Quant artifact {name} has invalid shape size_k={entry.get('size_k')!r}, "
                f"size_n={entry.get('size_n')!r}"
            )

        tensor_file = _require_string(entry.get("tensor_file"), f"modules[{idx}].tensor_file")
        rel_tensor_path = Path(tensor_file)
        if rel_tensor_path.is_absolute() or ".." in rel_tensor_path.parts:
            raise ValueError(f"Quant artifact tensor_file must stay under artifact root: {tensor_file!r}")
        tensor_path = root / tensor_file
        if tensor_file in seen_files:
            raise ValueError(f"Duplicate quant artifact tensor_file: {tensor_file}")
        seen_files.add(tensor_file)
        if not tensor_path.is_file():
            raise FileNotFoundError(f"Missing quant artifact tensor file for {name}: {tensor_path}")

        counts[backend_class] = counts.get(backend_class, 0) + 1

    if check_tensors:
        import torch

        required_payload_keys = {"qweight", "scales"}
        for entry in modules:
            tensor_path = root / str(entry["tensor_file"])
            payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise TypeError(f"Quant tensor payload must be a dict: {tensor_path}")
            missing = sorted(required_payload_keys - set(payload))
            if missing:
                raise KeyError(f"Quant tensor payload {tensor_path} missing keys: {missing}")

    return {
        "root": str(root),
        "manifest": manifest,
        "metadata": metadata,
        "modules": len(modules),
        "counts": counts,
        "self_contained": schema_version == BUNDLE_SCHEMA_VERSION,
        "state_keys": bundle_validation["state_keys"] if bundle_validation is not None else None,
        "file_bytes": bundle_validation["file_bytes"] if bundle_validation is not None else None,
    }


def validate_artifact_command(args: argparse.Namespace) -> None:
    result = validate_quant_artifact(
        args.quant_artifact_dir,
        expected_strategy=args.strategy,
        check_tensors=args.check_tensors,
        require_self_contained=args.require_self_contained,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "manifest"}, indent=2, sort_keys=True))


def build_bundle_command(args: argparse.Namespace) -> None:
    validate_quant_artifact(args.quant_artifact_dir, expected_strategy=args.strategy)
    strategy = _strategy(args.strategy)
    result = build_self_contained_bundle(
        strategy=args.strategy,
        strategy_description=strategy.description,
        strategy_plan=strategy.plan,
        quant_artifact_dir=args.quant_artifact_dir,
        checkpoint_path=args.checkpoint_path,
        config_file=args.config_file,
        tokenizer_dir=args.tokenizer_dir,
        vae_path=args.vae_path,
        output_dir=args.output_dir,
        copy_mode=args.copy_mode,
        max_shard_size_bytes=int(args.max_shard_size_gb * 1024**3),
    )
    print(json.dumps({k: v for k, v in result.items() if k != "manifest"}, indent=2, sort_keys=True))


def export_command(args: argparse.Namespace) -> str:
    if not args.checkpoint_path or not args.config_file:
        raise ValueError("Quant export requires --checkpoint-path and --config-file")
    strategy = _strategy(args.strategy)
    plan_path = Path(args.plan_dir).expanduser() / f"{strategy.name}.json"
    parts = [
        args.python,
        "-m",
        "cosmos_framework.scripts.action_policy_server_robocasa365_quant",
        "--checkpoint-path",
        args.checkpoint_path,
        "--config-file",
        args.config_file,
        "--output-dir",
        args.output_dir,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-action-steps",
        str(args.served_action_steps),
        "--no-guardrails",
        "--deterministic-seed",
        "--no-torch-compile",
        "--quant-plan-file",
        str(plan_path),
        "--quant-calib-capture-dir",
        args.calib_capture_dir,
        "--quant-calib-limit",
        str(args.calib_limit),
        "--quant-calib-alpha",
        str(args.calib_alpha),
        "--quant-export-dir",
        args.quant_export_dir,
    ]
    return _quote_parts(parts)


def serve_command(args: argparse.Namespace) -> str:
    validation = validate_quant_artifact(args.quant_import_dir)
    parts = [
        args.python,
        "-m",
        "cosmos_framework.scripts.action_policy_server_robocasa365_quant",
        "--output-dir",
        args.output_dir,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-action-steps",
        str(args.served_action_steps),
        "--no-guardrails",
        "--no-torch-compile",
        "--quant-import-dir",
        args.quant_import_dir,
    ]
    if validation["self_contained"]:
        if args.checkpoint_path or args.config_file:
            raise ValueError("Self-contained quant serving must not use external checkpoint/config paths")
    else:
        if not args.checkpoint_path or not args.config_file:
            raise ValueError("Legacy quant serving requires --checkpoint-path and --config-file")
        parts.extend(
            [
                "--checkpoint-path",
                args.checkpoint_path,
                "--config-file",
                args.config_file,
                "--allow-legacy-quant-artifact",
            ]
        )
    return _quote_parts(parts)


def replay_command(args: argparse.Namespace) -> str:
    parts = [
        args.python,
        str(Path(__file__).with_name("replay_policy_requests.py")),
        "--capture-dir",
        args.capture_dir,
        "--output-dir",
        args.output_dir,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--limit",
        str(args.limit),
    ]
    return _quote_parts(parts)


def rollout_command(args: argparse.Namespace) -> str:
    parts = [
        args.python,
        args.rollout_script,
        "--n_episodes",
        str(args.n_episodes),
        "--policy_client_host",
        args.host,
        "--policy_client_port",
        str(args.port),
        "--max_episode_steps",
        str(args.max_episode_steps),
        "--env_name",
        args.env_name,
        "--n_action_steps",
        str(args.served_action_steps),
        "--n_envs",
        str(args.n_envs),
        "--robocasa_split",
        args.split,
    ]
    if args.disable_video:
        parts.append("--disable-video")
    return _quote_parts(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_parser = sub.add_parser("list-strategies")
    list_parser.set_defaults(func=lambda args: print(json.dumps({
        name: {
            "description": strategy.description,
            "expected_peak_alloc_gb": strategy.expected_peak_alloc_gb,
            "m13_success_rate": strategy.m13_success_rate,
        }
        for name, strategy in STRATEGIES.items()
    }, indent=2, sort_keys=True)))

    configs = sub.add_parser("write-strategy-configs")
    configs.add_argument("--output-dir", required=True)
    configs.set_defaults(func=lambda args: write_strategy_configs(Path(args.output_dir)))

    artifact = sub.add_parser("write-artifact-metadata")
    artifact.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    artifact.add_argument("--quant-artifact-dir", required=True)
    artifact.add_argument("--checkpoint-path", required=True)
    artifact.add_argument("--config-file", required=True)
    artifact.add_argument("--calib-capture-dir", required=True)
    artifact.add_argument("--calib-limit", type=int, default=128)
    artifact.add_argument("--calib-alpha", type=float, default=0.5)
    artifact.set_defaults(func=write_artifact_manifest)

    validate = sub.add_parser("validate-artifact")
    validate.add_argument("--quant-artifact-dir", required=True)
    validate.add_argument("--strategy", choices=sorted(STRATEGIES))
    validate.add_argument("--check-tensors", action="store_true")
    validate.add_argument("--require-self-contained", action="store_true")
    validate.set_defaults(func=validate_artifact_command)

    bundle = sub.add_parser("build-self-contained-bundle")
    bundle.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    bundle.add_argument("--quant-artifact-dir", required=True)
    bundle.add_argument("--checkpoint-path", required=True)
    bundle.add_argument("--config-file", required=True)
    bundle.add_argument("--tokenizer-dir", required=True)
    bundle.add_argument("--vae-path", required=True)
    bundle.add_argument("--output-dir", required=True)
    bundle.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    bundle.add_argument("--max-shard-size-gb", type=float, default=2.0)
    bundle.set_defaults(func=build_bundle_command)

    common_model = argparse.ArgumentParser(add_help=False)
    common_model.add_argument("--python", default="python")
    common_model.add_argument("--checkpoint-path", default="")
    common_model.add_argument("--config-file", default="")
    common_model.add_argument("--output-dir", required=True)
    common_model.add_argument("--host", default="127.0.0.1")
    common_model.add_argument("--port", type=int, default=5577)
    common_model.add_argument("--served-action-steps", type=int, default=8)

    export = sub.add_parser("print-export-command", parents=[common_model])
    export.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    export.add_argument("--plan-dir", required=True)
    export.add_argument("--calib-capture-dir", required=True)
    export.add_argument("--calib-limit", type=int, default=128)
    export.add_argument("--calib-alpha", type=float, default=0.5)
    export.add_argument("--quant-export-dir", required=True)
    export.set_defaults(func=lambda args: print(export_command(args)))

    serve = sub.add_parser("print-serve-command", parents=[common_model])
    serve.add_argument("--quant-import-dir", required=True)
    serve.set_defaults(func=lambda args: print(serve_command(args)))

    replay = sub.add_parser("print-replay-command")
    replay.add_argument("--python", default="python")
    replay.add_argument("--capture-dir", required=True)
    replay.add_argument("--output-dir", required=True)
    replay.add_argument("--host", default="127.0.0.1")
    replay.add_argument("--port", type=int, default=5577)
    replay.add_argument("--limit", type=int, default=32)
    replay.set_defaults(func=lambda args: print(replay_command(args)))

    rollout = sub.add_parser("print-rollout-command")
    rollout.add_argument("--python", required=True)
    rollout.add_argument("--rollout-script", required=True)
    rollout.add_argument("--host", default="127.0.0.1")
    rollout.add_argument("--port", type=int, default=5577)
    rollout.add_argument("--n-episodes", type=int, default=50)
    rollout.add_argument("--n-envs", type=int, default=5)
    rollout.add_argument("--max-episode-steps", type=int, default=1200)
    rollout.add_argument("--served-action-steps", type=int, default=8)
    rollout.add_argument("--env-name", default="robocasa/CloseFridge")
    rollout.add_argument("--split", default="target")
    rollout.add_argument("--disable-video", action=argparse.BooleanOptionalAction, default=True)
    rollout.set_defaults(func=lambda args: print(rollout_command(args)))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
