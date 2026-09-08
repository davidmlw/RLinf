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
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "toolkits/eos/gr00t_trocar/tensorrt/verify_refittable_dit_bundle.py"
)
SPEC = importlib.util.spec_from_file_location("verify_refittable_dit_bundle", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "build"
    engine_dir = root / "engine"
    engine_dir.mkdir(parents=True)
    plan = engine_dir / "dit_bf16_refit.engine"
    parameter_map = root / "refittable-dit-parameter-map.json"
    plan.write_bytes(b"plan")
    parameter_map.write_text('{"weights": []}\n', encoding="utf-8")
    engine_receipt = engine_dir / "rlinf-refittable-dit-engine-receipt.json"
    engine_receipt.write_text(
        json.dumps(
            {
                "schema": MODULE.ENGINE_RECEIPT_SCHEMA,
                "status": "passed",
                "engine": {"sha256": _sha256(plan)},
                "parameter_map": {"sha256": _sha256(parameter_map)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    qualification = tmp_path / "refit-lifecycle-receipt.json"
    qualification.write_text(
        json.dumps(
            {
                "schema": MODULE.QUALIFICATION_SCHEMA,
                "status": "passed",
                "provenance": {
                    "engine": {"sha256": _sha256(plan)},
                    "engine_receipt": _sha256(engine_receipt),
                    "parameter_map": _sha256(parameter_map),
                },
                "device_weights": {"source_digest_revision_0": "a" * 64},
                "fixed_probe": {
                    "thresholds": {"cosine_min": 0.999, "relative_l2_max": 0.05}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root, qualification


def test_verify_accepts_one_hash_bound_qualified_bundle(tmp_path: Path) -> None:
    root, qualification = _bundle(tmp_path)

    receipt = MODULE.verify(root, qualification)

    assert receipt["status"] == "passed"
    assert receipt["scope"] == "experimental_approximate_behavior_only"
    assert receipt["source_digest_revision_0"] == "a" * 64
    assert receipt["ppo_authority"] == "failed_ratio_kl_approximate_behavior_only"


def test_verify_rejects_a_tampered_engine_plan(tmp_path: Path) -> None:
    root, qualification = _bundle(tmp_path)
    (root / "engine/dit_bf16_refit.engine").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="build receipt"):
        MODULE.verify(root, qualification)


def test_verify_rejects_weakened_probe_thresholds(tmp_path: Path) -> None:
    root, qualification = _bundle(tmp_path)
    value = json.loads(qualification.read_text(encoding="utf-8"))
    value["fixed_probe"]["thresholds"]["cosine_min"] = 0.99
    qualification.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="weakened numerical thresholds"):
        MODULE.verify(root, qualification)


@pytest.mark.parametrize(
    ("name", "value"),
    [("cosine_min", float("nan")), ("relative_l2_max", float("inf"))],
)
def test_verify_rejects_nonfinite_probe_thresholds(
    tmp_path: Path, name: str, value: float
) -> None:
    root, qualification = _bundle(tmp_path)
    receipt = json.loads(qualification.read_text(encoding="utf-8"))
    receipt["fixed_probe"]["thresholds"][name] = value
    qualification.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="weakened numerical thresholds"):
        MODULE.verify(root, qualification)
