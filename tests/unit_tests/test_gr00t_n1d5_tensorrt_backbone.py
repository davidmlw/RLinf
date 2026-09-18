# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from rlinf.hybrid_engines.tensorrt.persistent_engine import PersistentEngine
from rlinf.models.embodiment.gr00t.gr00t_n1d5 import tensorrt_backbone
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualified_bundle(root: Path) -> tuple[dict, Path]:
    files = {}
    bindings = {
        "vit_bf16.engine": [
            {
                "index": 0,
                "name": "pixel_values",
                "mode": "input",
                "dtype": "DataType.BF16",
                "shape": [16, 3, 224, 224],
                "profile": None,
            }
        ],
        "llm_bf16.engine": [
            {
                "index": 0,
                "name": "inputs_embeds",
                "mode": "input",
                "dtype": "DataType.BF16",
                "shape": [8, 570, 2048],
                "profile": None,
            }
        ],
    }
    for name in tensorrt_backbone._ENGINE_FILES:
        path = root / name
        path.write_bytes(name.encode())
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "bindings": bindings[name],
        }
    metadata = root / "export_metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "schema": "rlinf.gr00t-n1d5-stack-cube-true-b8-onnx.v1",
                "status": "passed",
                "model_version": "n1d5",
                "batch_size": 8,
                "image_views": 2,
                "image_batch": 16,
                "sequence_length": 570,
                "precision": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    receipt = root / "rlinf-engine-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": tensorrt_backbone._RECEIPT_SCHEMA,
                "status": "passed",
                "static_batch": 8,
                "image_views": 2,
                "image_batch": 16,
                "sequence_length": 570,
                "precision": "bfloat16",
                "silent_fallback": False,
                "export_metadata_sha256": _sha256(metadata),
                "engines": files,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "engine_dir": str(root),
        "receipt_path": str(receipt),
        "receipt_sha256": _sha256(receipt),
        "static_batch_size": 8,
        "image_views": 2,
        "image_batch_size": 16,
        "sequence_length": 570,
    }
    return config, receipt


def test_n1d5_artifact_contract_accepts_only_exact_true_b8_bundle(
    tmp_path: Path,
) -> None:
    config, _ = _qualified_bundle(tmp_path)

    result = tensorrt_backbone._validate_artifacts(config)

    assert set(result["files"]) == set(tensorrt_backbone._ENGINE_FILES)
    assert result["receipt"]["precision"] == "bfloat16"


def test_n1d5_artifact_contract_rejects_shape_drift(tmp_path: Path) -> None:
    config, receipt = _qualified_bundle(tmp_path)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["image_batch"] = 8
    receipt.write_text(json.dumps(value), encoding="utf-8")
    config["receipt_sha256"] = _sha256(receipt)

    with pytest.raises(RuntimeError, match="shape/precision mismatch"):
        tensorrt_backbone._validate_artifacts(config)


def test_n1d5_artifact_contract_rejects_export_metadata_drift(
    tmp_path: Path,
) -> None:
    config, receipt = _qualified_bundle(tmp_path)
    metadata = tmp_path / "export_metadata.json"
    value = json.loads(metadata.read_text(encoding="utf-8"))
    value["sequence_length"] = 568
    metadata.write_text(json.dumps(value), encoding="utf-8")
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_value["export_metadata_sha256"] = _sha256(metadata)
    receipt.write_text(json.dumps(receipt_value), encoding="utf-8")
    config["receipt_sha256"] = _sha256(receipt)

    with pytest.raises(RuntimeError, match="export metadata shape/precision mismatch"):
        tensorrt_backbone._validate_artifacts(config)


def test_persistent_engine_hot_path_has_no_copy_or_host_sync() -> None:
    source = inspect.getsource(PersistentEngine.__call__)

    assert ".contiguous()" not in source
    assert "is_contiguous()" in source
    assert "execute_async_v3" in source
    assert ".synchronize()" not in source


def test_n1d5_backend_hot_path_stays_cuda_resident() -> None:
    source = inspect.getsource(tensorrt_backbone.TensorRTFrozenEagleBackbone.__call__)

    assert ".cpu()" not in source
    assert ".numpy()" not in source
    assert ".item()" not in source
    assert "inputs_embeds[selected] = flattened" in source


def test_rollout_checkpoint_load_precedes_tensorrt_backbone_replacement() -> None:
    source = inspect.getsource(MultiStepRolloutWorker.init_worker)

    assert source.index("self.hf_model.load_state_dict") < source.index(
        "enable_tensorrt_backbone(tensorrt_config)"
    )
