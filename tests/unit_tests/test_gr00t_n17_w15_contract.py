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

import yaml

from toolkits.gr00t_trocar.w15.configs import render, validate

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "toolkits/gr00t_trocar/w15/contract.json"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _base() -> dict:
    return yaml.safe_load(
        """
runner:
  max_epochs: 2
  val_check_interval: -1
  save_interval: 2
  logger: {experiment_name: base, log_path: output}
env:
  train:
    total_num_envs: 64
    video_cfg: {video_base_dir: train}
  eval:
    video_cfg: {video_base_dir: eval}
rollout:
  enable_torch_compile: false
  torch_compile_mode: max-autotune-no-cudagraphs
  model: {skip_unused_lm_head: true}
actor:
  pre_update_same_revision_gate: {enabled: true}
  model:
    skip_unused_lm_head: true
    rollout_backbone_feature_transport: borrowed_ipc_pinned
    num_action_chunks: 16
"""
    )


def test_w15_contract_has_exact_five_arms() -> None:
    assert set(_contract()["arms"]) == {
        "eager-eager",
        "pt2-pt2",
        "trt-eager",
        "trt-pt2",
        "trt-refit-trt",
    }


def test_pt2_arm_uses_static_true_b8_backbone_and_compiled_dit() -> None:
    config = render(_base(), _contract(), "pt2-pt2", max_epochs=5, identity_gate=True)
    backbone = config["rollout"]["model"]["torch_compile_backbone"]
    assert backbone["static_batch_size"] == 8
    assert backbone["sequence_length"] == 208
    assert backbone["vision_segments"] == 24
    assert config["rollout"]["enable_torch_compile"] is True
    assert validate(
        config, _contract(), "pt2-pt2", max_epochs=5, identity_gate=True
    ) == []


def test_refit_arm_has_no_shadow_or_compile() -> None:
    config = render(
        _base(), _contract(), "trt-refit-trt", max_epochs=1, identity_gate=True
    )
    dit = config["rollout"]["model"]["tensorrt_dit"]
    assert config["rollout"]["enable_torch_compile"] is False
    assert dit["online_refit"] is True
    assert dit["lineage_receipt_mode"] == "gpu_transform_validation"
    assert dit["shadow_eager"] is False
    assert dit["compute_capability"] == [12, 0]


def test_performance_config_disables_identity_and_revision_probes() -> None:
    gated = render(_base(), _contract(), "trt-pt2", max_epochs=5, identity_gate=True)
    clean = render(_base(), _contract(), "trt-pt2", max_epochs=5, identity_gate=False)
    gated["actor"]["pre_update_same_revision_gate"]["enabled"] = False
    assert gated == clean

    gated = render(
        _base(), _contract(), "trt-refit-trt", max_epochs=5, identity_gate=True
    )
    clean = render(
        _base(), _contract(), "trt-refit-trt", max_epochs=5, identity_gate=False
    )
    assert gated["rollout"]["model"]["tensorrt_dit"]["probe_each_revision"] is True
    assert clean["rollout"]["model"]["tensorrt_dit"]["probe_each_revision"] is False
