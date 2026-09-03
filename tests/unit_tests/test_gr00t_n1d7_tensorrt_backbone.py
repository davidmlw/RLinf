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
import json
from pathlib import Path

import pytest

from rlinf.models.embodiment.gr00t.gr00t_n1d7.tensorrt_backbone import (
    _validate_artifacts,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifacts(root: Path) -> dict[str, object]:
    root.mkdir()
    engines = {}
    for name in ("llm_bf16.engine", "vit.engine"):
        path = root / name
        path.write_bytes(f"qualified-{name}".encode())
        engines[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "bindings": [],
        }
    metadata = {
        "schema_version": 1,
        "model_version": "n1d7",
        "batch_size": 8,
        "llm_seq_len": 208,
        "vit_grid_thw": [[1, 16, 16]] * 3,
    }
    metadata_path = root / "export_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    receipt = {
        "schema": "rlinf.gr00t-n1d7-trocar-true-b8-engines.v1",
        "status": "passed",
        "static_batch": 8,
        "sequence_opt": 208,
        "silent_fallback": False,
        "export_metadata_sha256": _sha256(metadata_path),
        "engines": engines,
    }
    receipt_path = root / "rlinf-engine-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return {
        "engine_dir": str(root),
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "static_batch_size": 8,
        "sequence_opt": 208,
    }


def test_validate_artifacts_accepts_exact_two_engine_bundle(tmp_path: Path) -> None:
    config = _write_artifacts(tmp_path / "engines")

    result = _validate_artifacts(config)

    assert result["receipt_sha256"] == config["receipt_sha256"]
    assert set(result["files"]) == {"llm_bf16.engine", "vit.engine"}
    assert result["metadata"]["batch_size"] == 8


def test_validate_artifacts_rejects_plan_tampering(tmp_path: Path) -> None:
    config = _write_artifacts(tmp_path / "engines")
    (tmp_path / "engines" / "vit.engine").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="plan does not match"):
        _validate_artifacts(config)


def test_validate_artifacts_rejects_unexpected_engine(tmp_path: Path) -> None:
    config = _write_artifacts(tmp_path / "engines")
    (tmp_path / "engines" / "dit_bf16.engine").write_bytes(b"unexpected")

    with pytest.raises(RuntimeError, match="exactly the qualified plans"):
        _validate_artifacts(config)


def test_validate_artifacts_rejects_static_shape_drift(tmp_path: Path) -> None:
    config = _write_artifacts(tmp_path / "engines")
    config["sequence_opt"] = 156

    with pytest.raises(RuntimeError, match="batch/profile contract mismatch"):
        _validate_artifacts(config)


def test_validate_artifacts_rejects_receipt_tampering_first(tmp_path: Path) -> None:
    config = _write_artifacts(tmp_path / "engines")
    receipt = Path(str(config["receipt_path"]))
    receipt.write_text(receipt.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="receipt SHA-256 mismatch"):
        _validate_artifacts(config)
