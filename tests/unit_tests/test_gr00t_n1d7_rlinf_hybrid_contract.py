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

import yaml

ROOT = Path(__file__).resolve().parents[2]
TOOLKIT = ROOT / "toolkits/eos/gr00t_trocar"
CONTRACT = TOOLKIT / "tensorrt/contract-n1d7-rlinf-hybrid-integration.json"
CONFIGS = {
    "eager": TOOLKIT / "config-n1d7-hybrid-eager-chunk16.yaml",
    "trt_eager": TOOLKIT / "config-n1d7-hybrid-trt-eager-chunk16.yaml",
    "trt_compile": TOOLKIT / "config-n1d7-hybrid-trt-compile-chunk16.yaml",
}


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _config(name: str) -> dict:
    return yaml.safe_load(CONFIGS[name].read_text(encoding="utf-8"))


def _different_paths(left: object, right: object, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths = set()
        for key in set(left) | set(right):
            child = f"{prefix}.{key}" if prefix else key
            paths.update(_different_paths(left.get(key), right.get(key), child))
        return paths
    return set() if left == right else {prefix}


def test_w81_contract_keeps_execution_in_python() -> None:
    implementation = _contract()["implementation"]

    assert implementation["owner"] == "rlinf"
    assert implementation["language"] == "python"
    assert implementation["tensorrt_components"] == [
        "vit.engine",
        "llm_bf16.engine",
    ]
    assert implementation["compiled_component"] == "action_head.model.forward"
    assert implementation["apply_final_llm_norm"] is False


def test_w81_configs_have_only_preregistered_arm_differences() -> None:
    eager = _config("eager")
    allowed = set(_contract()["matched_ab"]["only_allowed_config_differences"])

    assert _different_paths(eager, _config("trt_eager")) <= allowed
    assert _different_paths(eager, _config("trt_compile")) <= allowed


def test_w81_configs_freeze_common_lifecycle_and_workload() -> None:
    expected_arms = _contract()["matched_ab"]["arms"]
    for name in CONFIGS:
        config = _config(name)
        model = config["rollout"]["model"]
        assert config["rollout"]["enable_offload"] is False
        assert model["skip_unused_lm_head"] is True
        assert config["actor"]["model"]["skip_unused_lm_head"] is True
        assert config["weight_syncer"]["state_dict_prefixes"] == ["action_head"]
        assert (
            model["tensorrt_backbone"]["enabled"]
            is expected_arms[name]["tensorrt_backbone"]
        )
        assert (
            config["rollout"]["enable_torch_compile"]
            is expected_arms[name]["compile_dit"]
        )
        assert config["actor"]["model"]["num_action_chunks"] == 16
        assert config["actor"]["global_batch_size"] == 2048
        assert config["actor"]["micro_batch_size"] == 128
        assert config["algorithm"]["update_epoch"] == 4


def test_w81_numerical_thresholds_are_frozen() -> None:
    gates = _contract()["numerical_gates"]

    assert gates["final_action"] == {
        "cosine_min": 0.999,
        "mean_abs_max": 0.005,
        "max_abs_max": 0.05,
        "finite": True,
    }
    assert gates["pre_update_same_revision"]["ratio_max_abs_from_one_max"] == 0.001
    assert gates["pre_update_same_revision"]["kl_max_abs_max"] == 0.001
    assert gates["one_update_cross_arm"]["gradient_relative_l2_max"] == 0.02
    assert gates["one_update_cross_arm"]["parameter_delta_relative_l2_max"] == 0.02
