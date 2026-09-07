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


def test_stage_matrix_clears_warmup_before_measurement(monkeypatch) -> None:
    module = _load_module()

    class FakeOutput:
        def detach(self):
            return self

        def clone(self):
            return self

    calls = []
    monkeypatch.setattr(
        module,
        "_compare",
        lambda _reference, _candidate: {"finite": True, "bitwise_equal": True},
    )
    timing = {"backbone_ms": 1.0, "action_head_ms": 2.0, "total_ms": 3.0}
    module.measure_stage_matrix(
        {"a": lambda: (FakeOutput(), timing), "b": lambda: (FakeOutput(), timing)},
        reference="a",
        warmup=3,
        measured=2,
        boundary="fixture",
        before_measurement=lambda: calls.append("cleared"),
    )
    assert calls == ["cleared"]


def test_contract_freezes_refittable_action_head() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == "rlinf.gr00t-n1d7-b8-executor-matrix-contract.v2"
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


def test_executor_uses_narrow_refit_runtime_import() -> None:
    source = (TOOLS / "executor_matrix_b8.py").read_text(encoding="utf-8")
    assert "def _load_refittable_tensorrt_dit" in source
    assert "from rlinf.models.embodiment" not in source


def test_pt2_backbone_freezes_static_flash_attention_adapter() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    description = contract["partition"]["frozen_backbone"]["executors"]["pt2"]
    adapter = contract["measurement"]["pt2_static_vision_adapter"]
    assert "FakeTensor scalar" in description
    assert adapter["required_grid_rows"] == 24
    assert adapter["required_sequence_length"] == 256
    assert adapter["attention_backend"] == "flash_attention_2"
    assert adapter["attention_backend_change_allowed"] is False
    assert adapter["prior_failed_jobs"] == [5986289, 5986310]


def test_uniform_vision_sequence_length() -> None:
    module = _load_module()
    rows = [[1, 16, 16] for _ in range(24)]
    assert module._uniform_vision_sequence_length(rows) == (256, 24)
    assert module._uniform_vision_sequence_length([[2, 16, 16]]) == (256, 2)


def test_uniform_vision_sequence_length_rejects_dynamic_geometry() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="requires one sequence length"):
        module._uniform_vision_sequence_length([[1, 16, 16], [1, 8, 16]])
    with pytest.raises(ValueError, match="three values"):
        module._uniform_vision_sequence_length([[1, 16]])
    with pytest.raises(ValueError, match="positive"):
        module._uniform_vision_sequence_length([[0, 16, 16]])


def test_launcher_enables_pt2_backbone() -> None:
    source = (TOOLS / "start_executor_matrix_b8.py").read_text(encoding="utf-8")
    assert '"--pt2-backbone-unavailable-reason"' not in source


def test_pt2_backbone_code_checks_exact_b8_geometry() -> None:
    module = _load_module()
    assert module.EXPECTED_VISION_SEGMENTS == 24
    assert module.EXPECTED_VISION_SEQUENCE_LENGTH == 256
