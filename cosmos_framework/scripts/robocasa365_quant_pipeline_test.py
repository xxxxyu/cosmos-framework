# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import pytest

from cosmos_framework.scripts.robocasa365_quant_pipeline import (
    STRATEGIES,
    validate_quant_artifact,
    write_strategy_configs,
)


def _write_minimal_artifact(root: Path, *, backend_class: str = "VllmGptqMarlinW4A16Linear") -> None:
    (root / "tensors").mkdir(parents=True)
    tensor_file = "tensors/net.language_model.model.layers.0.self_attn.q_proj.pt"
    (root / tensor_file).write_bytes(b"placeholder")
    num_bits = 4 if backend_class.endswith("W4A16Linear") else 8
    manifest = {
        "schema_version": 1,
        "checkpoint_path": "/ckpt",
        "config_file": "/config.yaml",
        "modules": [
            {
                "name": "net.language_model.model.layers.0.self_attn.q_proj",
                "backend_class": backend_class,
                "format": "vllm_marlin_wna16",
                "num_bits": num_bits,
                "group_size": 128,
                "size_k": 4096,
                "size_n": 4096,
                "tensor_file": tensor_file,
                "wtype_id": 1,
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_write_strategy_configs_contains_retained_candidates(tmp_path: Path) -> None:
    write_strategy_configs(tmp_path)

    manifest = json.loads((tmp_path / "strategies_manifest.json").read_text())
    assert set(manifest) == set(STRATEGIES)
    assert json.loads((tmp_path / "attention_w8.json").read_text())[1]["backend"] == "vllm_gptq_marlin_w8a16"


def test_validate_quant_artifact_accepts_minimal_manifest(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)

    result = validate_quant_artifact(tmp_path)

    assert result["modules"] == 1
    assert result["counts"] == {"VllmGptqMarlinW4A16Linear": 1}


def test_validate_quant_artifact_rejects_wrong_bits(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["modules"][0]["num_bits"] = 8
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="num_bits"):
        validate_quant_artifact(tmp_path)


def test_validate_quant_artifact_rejects_strategy_mismatch(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)

    with pytest.raises(ValueError, match="expects"):
        validate_quant_artifact(tmp_path, expected_strategy="full_w8")


def test_validate_quant_artifact_rejects_escaping_tensor_file(tmp_path: Path) -> None:
    _write_minimal_artifact(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["modules"][0]["tensor_file"] = "../outside.pt"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="artifact root"):
        validate_quant_artifact(tmp_path)
