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

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT
    / "toolkits"
    / "eos"
    / "gr00t_trocar"
    / "tensorrt"
    / "contract-n1d7-trocar-true-b8-standalone.json"
)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_w80_contract_uses_the_executed_rlinf_processor_order() -> None:
    processor = _contract()["processor"]

    assert processor["camera_order"] == [
        "left_wrist_view",
        "right_wrist_view",
        "room_view",
    ]
    assert processor["state_action_order"] == [
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    ]
    assert processor["action_delta_indices"] == list(range(16))


def test_w80_contract_requires_one_genuine_static_b8_call() -> None:
    fixture = _contract()["fixture"]

    assert fixture["batch_size"] == 8
    assert fixture["pixel_values_shape"] == [6144, 1536]
    assert fixture["image_grid_thw_shape"] == [24, 3]
    assert fixture["generated_action_shape"] == [8, 40, 132]
    assert fixture["public_action_shape"] == [8, 16, 28]
    assert fixture["b1x8_allowed"] is False
    assert fixture["b_squared_capture_allowed"] is False


def test_w80_contract_keeps_only_frozen_backbone_in_tensorrt() -> None:
    hybrid = _contract()["hybrid"]

    assert hybrid["tensorrt_engines"] == ["vit.engine", "llm_bf16.engine"]
    assert hybrid["apply_final_llm_norm"] is False
    assert hybrid["gpu_only_handoff"] is True
    assert hybrid["input_layout"] == "contiguous_fail_closed"
    assert hybrid["stream_protocol"] == "single_torch_current_stream_fail_closed"
    assert hybrid["execution_event_recorded_per_call"] is True
    assert hybrid["close_waits_for_last_execution_event"] is True
    assert hybrid["silent_eager_fallback"] is False
    assert hybrid["feature_reuse_enabled"] is False


def test_w80_contract_is_python_owned() -> None:
    implementation = _contract()["implementation"]

    assert implementation["owner"] == "rlinf"
    assert implementation["language"] == "python"
    assert "praxis_runtime" in implementation["forbidden_dependencies"]
    assert "poiesis_runtime" in implementation["forbidden_dependencies"]


def test_w80_contract_requires_paired_timing_and_hash_chain() -> None:
    contract = _contract()

    assert contract["statistics"]["paired_order"] == "alternating_ab_ba"
    assert contract["statistics"]["paired_measured"] == 30
    assert (
        contract["artifact"]["required_hash_chain"]
        == "fixture_to_export_to_engine_to_standalone_run"
    )


def test_w80_contract_freezes_the_cuda_resident_common_boundary() -> None:
    boundary = _contract()["common_boundary"]

    assert boundary["input"] == "preloaded_contiguous_cuda_tensors"
    assert boundary["stochastic_input"] == "explicit_initial_noise_tensor"
    assert boundary["output"] == "normalized_deployment_action"
    assert boundary["timing"] == (
        "one_natural_call_cuda_events_without_intermediate_host_sync"
    )
    assert boundary["arms"] == [
        "eager_backbone_eager_head",
        "tensorrt_backbone_eager_head",
        "eager_backbone_torch_compile_dit_head",
        "tensorrt_backbone_torch_compile_dit_head",
    ]
    assert boundary["required_direct_pairs"] == [
        ["eager_backbone_eager_head", "tensorrt_backbone_eager_head"],
        [
            "eager_backbone_eager_head",
            "eager_backbone_torch_compile_dit_head",
        ],
        [
            "tensorrt_backbone_eager_head",
            "tensorrt_backbone_torch_compile_dit_head",
        ],
        [
            "eager_backbone_eager_head",
            "tensorrt_backbone_torch_compile_dit_head",
        ],
        [
            "eager_backbone_torch_compile_dit_head",
            "tensorrt_backbone_torch_compile_dit_head",
        ],
    ]
    assert boundary["official_deployment_equivalence_required"] is True
    assert boundary["explicit_path_must_not_advance_rng"] is True
    assert boundary["compile_rebuild_during_measurement_allowed"] is False
