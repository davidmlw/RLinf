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

import ast
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
REUSE_CONFIGS = {
    "eager_reuse": TOOLKIT / "config-n1d7-hybrid-eager-reuse-chunk16.yaml",
    "eager_compile_reuse": (
        TOOLKIT / "config-n1d7-hybrid-eager-compile-reuse-chunk16.yaml"
    ),
    "trt_eager_reuse": (TOOLKIT / "config-n1d7-hybrid-trt-eager-reuse-chunk16.yaml"),
    "trt_compile_reuse": (
        TOOLKIT / "config-n1d7-hybrid-trt-compile-reuse-chunk16.yaml"
    ),
}


def _contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _config(name: str) -> dict:
    return yaml.safe_load((CONFIGS | REUSE_CONFIGS)[name].read_text(encoding="utf-8"))


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


def test_w81_feature_reuse_resolution_has_one_executor_difference() -> None:
    control = _config("eager_reuse")
    candidate = _config("trt_eager_reuse")
    contract = _contract()["matched_feature_reuse_ab"]

    assert _different_paths(control, candidate) == set(
        contract["only_allowed_config_differences"]
    )
    for config in (control, candidate):
        assert config["actor"]["model"]["rollout_backbone_feature_transport"] == (
            "borrowed_ipc_pinned"
        )
        assert config["rollout"]["pinned_feature_verify_trajectory"] is False
        assert config["rollout"]["enable_torch_compile"] is False


def test_w81_compiled_dit_reuse_is_an_isolated_diagnostic() -> None:
    eager_head = _config("trt_eager_reuse")
    compiled_dit = _config("trt_compile_reuse")
    diagnostic = _contract()["matched_feature_reuse_ab"][
        "compiled_dit_diagnostic"
    ]

    assert diagnostic["status"] == "rejected_for_rlinf_ppo_candidate"
    assert diagnostic["evidence"]["slurm_job_id"] == "5968817"
    assert _different_paths(eager_head, compiled_dit) == set(
        diagnostic["only_allowed_differences_from_trt_eager_reuse"]
    )
    assert eager_head["rollout"]["enable_torch_compile"] is False
    assert compiled_dit["rollout"]["enable_torch_compile"] is True
    assert (
        compiled_dit["actor"]["model"]["rollout_backbone_feature_transport"]
        == "borrowed_ipc_pinned"
    )


def test_w81_compile_only_diagnostic_uses_exact_eager_features() -> None:
    eager = _config("eager_reuse")
    compiled = _config("eager_compile_reuse")
    diagnostic = _contract()["matched_feature_reuse_ab"]["compile_only_diagnostic"]

    assert _different_paths(eager, compiled) == set(
        diagnostic["only_allowed_differences_from_eager_reuse"]
    )
    assert compiled["rollout"]["enable_torch_compile"] is True
    assert compiled["rollout"]["model"]["tensorrt_backbone"]["enabled"] is False
    assert compiled["actor"]["model"]["rollout_backbone_feature_transport"] == (
        "borrowed_ipc_pinned"
    )


def test_w81_feature_reuse_sites_reference_the_matched_configs() -> None:
    matched = _contract()["matched_feature_reuse_ab"]
    expected = {
        "control": REUSE_CONFIGS["eager_reuse"],
        "candidate": REUSE_CONFIGS["trt_eager_reuse"],
    }

    for arm, expected_config in expected.items():
        site_path = TOOLKIT / matched["site_templates"][arm]
        site = json.loads(site_path.read_text(encoding="utf-8"))
        assert Path(site["experiment"]["config"]).name == expected_config.name

    diagnostic = matched["compile_only_diagnostic"]
    site = json.loads(
        (TOOLKIT / diagnostic["site_template"]).read_text(encoding="utf-8")
    )
    assert Path(site["experiment"]["config"]).name == REUSE_CONFIGS[
        "eager_compile_reuse"
    ].name


def test_w81_configs_freeze_common_lifecycle_and_workload() -> None:
    contract = _contract()
    expected_arms = contract["matched_ab"]["arms"]
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
        assert (
            config["env"]["eval"]["total_num_envs"]
            == contract["workload"]["evaluation_global_envs"]
            == contract["workload"]["rollout_lanes"]
            * contract["workload"]["evaluation_static_batch_per_lane"]
        )
        assert (
            config["actor"]["model"]["rl_head_config"]["padding_value"]
            == contract["workload"]["text_padding_length"]
            == contract["artifacts"]["sequence_length"]
        )


def test_w81_numerical_thresholds_are_frozen() -> None:
    gates = _contract()["numerical_gates"]

    assert gates["final_action"] == {
        "cosine_min": 0.999,
        "mean_abs_max": 0.005,
        "max_abs_max": 0.05,
        "finite": True,
    }
    assert gates["pre_update_same_revision"]["ratio_max_abs_from_one_max"] == 0.1
    assert gates["pre_update_same_revision"]["kl_max_abs_max"] == 0.1
    assert gates["pre_update_same_revision"]["ratio_abs_gt_1e-3_fraction_max"] == 0.001
    assert gates["pre_update_same_revision"]["value_max_abs_max"] == 0.01
    assert gates["one_update_cross_arm"]["gradient_relative_l2_max"] == 0.02
    assert gates["one_update_cross_arm"]["parameter_delta_relative_l2_max"] == 0.02

    for name in CONFIGS | REUSE_CONFIGS:
        configured = _config(name)["actor"]["pre_update_same_revision_gate"]
        assert configured == {
            "enabled": True,
            "steps": [0],
            "thresholds": gates["pre_update_same_revision"],
        }


def test_w81_standalone_ablation_separates_trt_and_compile() -> None:
    ablation = _contract()["performance"]["standalone_factorial_ablation"]

    assert set(ablation["arms"]) == {"E/E", "T/E", "E/C", "T/C"}
    assert ablation["direct_effects"] == {
        "tensorrt": "T/E versus E/E and T/C versus E/C",
        "compile": "E/C versus E/E and T/C versus T/E",
        "combined": "T/C versus E/E",
    }
    assert ablation["compiled_arms_are_deployable"] is False
    assert "feature reuse disabled" in ablation["ppo_authority_diagnostics"][
        "tensorrt_only"
    ]


def test_w81_shutdown_records_state_before_and_after_close() -> None:
    source = (ROOT / "rlinf/workers/rollout/hf/huggingface_worker.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    shutdown = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "shutdown_hybrid_runtime"
    )
    stage_calls = [
        node.args[0].value
        for node in ast.walk(shutdown)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_log_hybrid_runtime_telemetry"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]

    assert stage_calls == ["closing", "closed"]


def test_pre_update_gate_consumes_the_same_pinned_backbone_as_training() -> None:
    source = (ROOT / "rlinf/workers/actor/fsdp_actor_worker.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    gate = functions["_run_pre_update_same_revision_gate"]
    training = functions["_run_training_impl"]

    def called_methods(node: ast.AST) -> list[str]:
        return [
            call.func.attr
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        ]

    assert "_load_pinned_backbone_for_batch" in called_methods(gate)
    assert "_load_pinned_backbone_for_batch" in called_methods(training)
    assert any(
        keyword.arg == "precomputed_backbone"
        for call in ast.walk(gate)
        if isinstance(call, ast.Call)
        for keyword in call.keywords
    )
    assert "requires feature reuse to be disabled" not in ast.get_source_segment(
        source, gate
    )
