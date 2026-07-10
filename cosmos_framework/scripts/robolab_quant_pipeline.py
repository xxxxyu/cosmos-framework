# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CLI for exporting and validating self-contained Cosmos3 RoboLab bundles."""

from __future__ import annotations

import argparse
import json
import logging

from cosmos_framework.scripts.robolab_quant_bundle import (
    build_robolab_quant_bundle,
    discover_quant_targets,
    validate_robolab_quant_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect the DROID quantization map")
    inspect_parser.add_argument("--source-checkpoint", required=True)
    inspect_parser.add_argument("--strategy", required=True)

    build_parser = subparsers.add_parser("build-bundle", help="Stream-pack one self-contained bundle")
    build_parser.add_argument("--strategy", required=True)
    build_parser.add_argument("--source-checkpoint", required=True)
    build_parser.add_argument("--tokenizer-dir", required=True)
    build_parser.add_argument("--vae-path", required=True)
    build_parser.add_argument("--output-dir", required=True)
    build_parser.add_argument("--device", default="cuda:0")
    build_parser.add_argument("--calibration-stats")
    build_parser.add_argument("--calibration-alpha", type=float, default=0.5)
    build_parser.add_argument("--allow-uncalibrated-w4", action="store_true")
    build_parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    build_parser.add_argument("--max-residual-shard-size", type=int, default=2 * 1024**3)

    validate_parser = subparsers.add_parser("validate", help="Validate a deployment bundle")
    validate_parser.add_argument("--bundle-dir", required=True)
    validate_parser.add_argument("--expected-strategy")
    validate_parser.add_argument("--check-hashes", action="store_true")
    validate_parser.add_argument("--check-tensors", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args()
    if args.command == "inspect":
        targets = discover_quant_targets(args.source_checkpoint, args.strategy)
        result = {
            "strategy": args.strategy,
            "modules": len(targets),
            "w4_modules": sum(target.num_bits == 4 for target in targets),
            "w8_modules": sum(target.num_bits == 8 for target in targets),
        }
    elif args.command == "build-bundle":
        result = build_robolab_quant_bundle(
            strategy=args.strategy,
            source_checkpoint=args.source_checkpoint,
            tokenizer_dir=args.tokenizer_dir,
            vae_path=args.vae_path,
            output_dir=args.output_dir,
            device=args.device,
            calibration_stats=args.calibration_stats,
            calibration_alpha=args.calibration_alpha,
            allow_uncalibrated_w4=args.allow_uncalibrated_w4,
            copy_mode=args.copy_mode,
            max_residual_shard_size=args.max_residual_shard_size,
        )
    elif args.command == "validate":
        result = validate_robolab_quant_bundle(
            args.bundle_dir,
            expected_strategy=args.expected_strategy,
            check_hashes=args.check_hashes,
            check_tensors=args.check_tensors,
        )
        result.pop("manifest", None)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
