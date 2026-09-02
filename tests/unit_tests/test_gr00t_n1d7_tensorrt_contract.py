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

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT
    / "toolkits"
    / "eos"
    / "gr00t_trocar"
    / "tensorrt"
    / "contract-n1d7-trocar-b8.json"
)


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


def test_contract_keeps_rlinf_python_owned():
    contract = _load_contract()

    assert contract["implementation"] == {
        "owner": "rlinf",
        "language": "python",
        "runtime": ["tensorrt_python", "pytorch"],
        "forbidden_dependencies": [
            "praxis_runtime",
            "poiesis_runtime",
            "rust_executor",
            "aoti_executor",
        ],
    }


def test_contract_defines_true_b8_trocar_shapes():
    production = _load_contract()["production_hybrid"]

    assert production["static_batch"] == 8
    assert production["camera_order"] == [
        "room_view",
        "left_wrist_view",
        "right_wrist_view",
    ]
    assert production["vit_pixel_values_shape"] == [6144, 1536]
    assert production["deepstack_count"] == 3
    assert production["deepstack_shape"] == [1536, 2048]
    assert production["generated_action_shape"] == [8, 40, 132]
    assert production["executed_action_shape"] == [8, 16, 28]


def test_contract_isolates_the_backbone_executor():
    matched_ab = _load_contract()["matched_ab"]

    assert matched_ab["common"]["skip_unused_lm_head"] is True
    assert matched_ab["common"]["rollout_backbone_feature_transport"] is None
    assert matched_ab["common"]["action_head_executor"] == "pytorch_eager"
    assert matched_ab["control_backbone_executor"] == "pytorch"
    assert matched_ab["candidate_backbone_executor"] == "tensorrt"
    assert matched_ab["only_allowed_difference"] == (
        "rollout.model.frozen_backbone_backend"
    )


def test_contract_keeps_learning_claims_out_of_scope():
    gates = _load_contract()["runtime_gates"]

    assert gates["feature_reuse_enabled"] is False
    assert gates["ppo_authority"] == "exact_revision_eager_recompute"
    assert gates["learning_status"] == "structural_frozen_learning_unproven"
    assert gates["tensorrt_reload_rebuild_refit_after_head_update"] == 0


def test_contract_pins_torchcodec_compatibility_overlay():
    builder = _load_contract()["builder"]

    assert builder["torch"] == "2.9.0+cu128"
    assert builder["torchcodec"] == "0.8.1"
    assert "hard NVDEC dependency" in builder["torchcodec_reason"]
