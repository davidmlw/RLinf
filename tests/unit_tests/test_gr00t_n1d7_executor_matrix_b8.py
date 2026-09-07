# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "toolkits/eos/gr00t_trocar/tensorrt"
CONTRACT = TOOLS / "contract-n1d7-b8-executor-matrix.json"


def _load_module():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "executor_matrix_b8_test_module", TOOLS / "executor_matrix_b8.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_balanced_three_arm_schedule() -> None:
    module = _load_module()
    names = ("eager", "pt2", "tensorrt")
    schedule = module.balanced_orders(names)
    assert len(schedule) == 6
    assert len(set(schedule)) == 6
    for position in range(3):
        assert sorted(order[position] for order in schedule) == [
            "eager",
            "eager",
            "pt2",
            "pt2",
            "tensorrt",
            "tensorrt",
        ]


def test_balanced_schedule_rejects_duplicate_arms() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="distinct arms"):
        module.balanced_orders(("eager", "eager"))


def test_stage_matrix_retains_samples_and_relative_results(monkeypatch) -> None:
    module = _load_module()

    class FakeOutput:
        def detach(self):
            return self

        def clone(self):
            return self

    monkeypatch.setattr(
        module,
        "_compare",
        lambda _reference, _candidate: {"finite": True, "bitwise_equal": True},
    )

    def arm(backbone: float, head: float):
        return lambda: (
            FakeOutput(),
            {
                "backbone_ms": backbone,
                "action_head_ms": head,
                "total_ms": backbone + head,
            },
        )

    result = module.measure_stage_matrix(
        {
            "eager": arm(30.0, 40.0),
            "pt2": arm(20.0, 20.0),
            "tensorrt": arm(10.0, 10.0),
        },
        reference="eager",
        warmup=2,
        measured=6,
        boundary="fixture",
    )
    assert result["arms"]["eager"]["total_ms"]["count"] == 6
    assert result["relative_to_reference"]["pt2"]["total_ms"][
        "speedup_vs_reference"
    ] == pytest.approx(1.75)
    assert result["relative_to_reference"]["tensorrt"]["total_ms"][
        "speedup_vs_reference"
    ] == pytest.approx(3.5)
    assert result["max_abs_stage_closure_error_ms"] == 0.0


def test_contract_freezes_refittable_action_head() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == "rlinf.gr00t-n1d7-b8-executor-matrix-contract.v1"
    action_head = contract["partition"]["refittable_action_head"]
    assert (
        action_head["ppo_lifecycle"] == "hot-updateable at a fenced revision boundary"
    )
    assert action_head["tensorrt_refit"]["online_refit"] is True
    assert action_head["tensorrt_refit"]["weight_count"] == 456
    assert action_head["tensorrt_refit"]["engine_slots"] == 2


def test_standalone_entry_requires_complete_refit_bundle() -> None:
    source = (TOOLS / "standalone_true_b8.py").read_text(encoding="utf-8")
    required = (
        "--refittable-dit-engine",
        "--refittable-dit-receipt",
        "--refittable-dit-receipt-sha256",
        "--refittable-dit-parameter-map",
        "--refittable-dit-parameter-map-sha256",
        "--refittable-dit-source-digest",
    )
    assert all(option in source for option in required)
    assert "W84 refittable DiT arguments must be supplied together" in source
