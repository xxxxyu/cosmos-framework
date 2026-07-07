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
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def export_command(args: argparse.Namespace) -> str:
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
        "--no-torch-compile",
        "--quant-import-dir",
        args.quant_import_dir,
    ]
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

    common_model = argparse.ArgumentParser(add_help=False)
    common_model.add_argument("--python", default="python")
    common_model.add_argument("--checkpoint-path", required=True)
    common_model.add_argument("--config-file", required=True)
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
