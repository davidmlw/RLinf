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
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "toolkits/eos/gr00t_trocar/tensorrt"
sys.path.insert(0, str(TOOLS))

import refittable_dit_contract as contract  # noqa: E402


def _checkpoint() -> dict:
    return {
        "action_head.model.linear.weight": {
            "shape": [3, 2],
            "dtype": "BF16",
        },
        "action_head.model.linear.bias": {
            "shape": [3],
            "dtype": "BF16",
        },
        "action_head.action_decoder.weight": {
            "shape": [4, 3],
            "dtype": "BF16",
        },
    }


def _onnx() -> dict:
    return {
        "dit.linear.bias": {
            "shape": [3],
            "dtype": "BF16",
            "consumers": [],
        },
        "onnx::MatMul_1": {
            "shape": [2, 3],
            "dtype": "BF16",
            "consumers": [{"name": "/dit/linear/MatMul", "op_type": "MatMul"}],
        },
    }


def test_build_refit_manifest_resolves_named_and_transposed_weights() -> None:
    value = contract.build_refit_manifest(_checkpoint(), _onnx())

    assert value["status"] == "passed"
    assert value["initializer_count"] == 2
    assert value["source_tensor_count"] == 2
    assert value["parameter_count"] == 9
    assert value["byte_count"] == 18
    assert value["transforms"] == {"identity": 1, "transpose_2d": 1}
    entries = {item["source_fqn"]: item for item in value["entries"]}
    assert entries["action_head.model.linear.weight"]["initializer_shape"] == [2, 3]
    assert entries["action_head.model.linear.weight"]["transform"] == "transpose_2d"


def test_build_refit_manifest_rejects_missing_initializer() -> None:
    onnx = _onnx()
    del onnx["dit.linear.bias"]

    with pytest.raises(ValueError, match="mapping is incomplete"):
        contract.build_refit_manifest(_checkpoint(), onnx)


def test_build_refit_manifest_rejects_shape_mismatch() -> None:
    onnx = _onnx()
    onnx["onnx::MatMul_1"]["shape"] = [3, 2]

    with pytest.raises(ValueError, match="shape mismatch"):
        contract.build_refit_manifest(_checkpoint(), onnx)


def test_build_refit_manifest_rejects_ambiguous_anonymous_initializer() -> None:
    onnx = _onnx()
    onnx["onnx::MatMul_1"]["consumers"].append(
        {"name": "/dit/other/MatMul", "op_type": "MatMul"}
    )

    with pytest.raises(ValueError, match="must have one consumer"):
        contract.build_refit_manifest(_checkpoint(), onnx)


def test_action_head_inventory_separates_dit_from_other_components() -> None:
    value = contract.action_head_inventory(_checkpoint())

    assert value["tensor_count"] == 3
    assert value["parameter_count"] == 21
    assert value["byte_count"] == 42
    assert value["components"]["model"]["parameter_count"] == 9
    assert value["components"]["action_decoder"]["parameter_count"] == 12


def test_model_config_gate_freezes_trainable_authority(tmp_path: Path) -> None:
    config = dict(contract.REQUIRED_MODEL_CONFIG)
    config["diffusion_model_cfg"] = {
        "attention_head_dim": 48,
        "interleave_self_attention": True,
        "num_attention_heads": 32,
        "num_layers": 32,
        "output_dim": 1024,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    value = contract.validate_model_config(path)

    assert value["fields"]["tune_projector"] is True
    assert value["fields"]["tune_diffusion_model"] is True
    assert value["fields"]["tune_vlln"] is True


def test_model_config_gate_rejects_frozen_dit(tmp_path: Path) -> None:
    config = dict(contract.REQUIRED_MODEL_CONFIG)
    config["tune_diffusion_model"] = False
    config["diffusion_model_cfg"] = {
        "attention_head_dim": 48,
        "interleave_self_attention": True,
        "num_attention_heads": 32,
        "num_layers": 32,
        "output_dim": 1024,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="tune_diffusion_model"):
        contract.validate_model_config(path)


def test_revision_fence_rejects_inference_during_refit() -> None:
    fence = contract.RefitRevisionFence()
    assert (
        fence.begin_refit(1, "source-1", "staging-1", "inventory-1", "prototype-1")
        == "b"
    )

    with pytest.raises(RuntimeError, match="cannot infer"):
        fence.begin_inference()


def test_revision_fence_commits_only_a_fully_verified_revision() -> None:
    fence = contract.RefitRevisionFence(active_probe_output_digest="output-0")
    candidate = fence.begin_refit(
        1, "source-1", "staging-1", "inventory-1", "prototype-1"
    )
    fence.mark_pytorch_applied(1, "head-1")
    fence.mark_staging_verified("staging-1")
    fence.mark_refit_complete("inventory-1", "prototype-1")
    fence.mark_probe_verified(
        probe_input_digest="probe-input",
        pytorch_output_digest="pytorch-output-1",
        trt_output_digest="trt-output-1",
        metrics_digest="probe-metrics-1",
        numerics_passed=True,
    )
    fence.commit()

    assert candidate == "b"
    assert fence.active_slot == "b"
    assert fence.active_pytorch_revision == 1
    assert fence.active_trt_revision == 1
    assert fence.active_probe_output_digest == "trt-output-1"
    assert fence.phase == "idle"


def test_revision_fence_staging_failure_after_apply_is_fail_stop() -> None:
    fence = contract.RefitRevisionFence(
        active_pytorch_revision=4,
        active_trt_revision=4,
        active_slot="b",
    )
    assert (
        fence.begin_refit(5, "source-5", "staging-5", "inventory-5", "prototype-5")
        == "a"
    )
    fence.mark_pytorch_applied(5, "head-5")

    with pytest.raises(ValueError, match="staging digest"):
        fence.mark_staging_verified("wrong")

    assert fence.active_pytorch_revision == 5
    assert fence.active_trt_revision == 4
    assert fence.active_slot == "b"
    assert fence.phase == "failed_stopped"
    with pytest.raises(RuntimeError, match="cannot infer"):
        fence.begin_inference()


def test_revision_fence_probe_failure_is_fail_stop() -> None:
    fence = contract.RefitRevisionFence(active_probe_output_digest="same-output")
    fence.begin_refit(1, "source-1", "staging-1", "inventory-1", "prototype-1")
    fence.mark_pytorch_applied(1, "head-1")
    fence.mark_staging_verified("staging-1")
    fence.mark_refit_complete("inventory-1", "prototype-1")

    with pytest.raises(ValueError, match="did not change"):
        fence.mark_probe_verified(
            probe_input_digest="probe-input",
            pytorch_output_digest="pytorch-output-1",
            trt_output_digest="same-output",
            metrics_digest="probe-metrics-1",
            numerics_passed=True,
        )

    assert fence.phase == "failed_stopped"


def test_revision_fence_rejects_mixed_active_revisions() -> None:
    fence = contract.RefitRevisionFence(
        active_pytorch_revision=5,
        active_trt_revision=4,
        phase="idle",
    )

    with pytest.raises(RuntimeError, match="mixed revisions"):
        fence.begin_inference()


def test_revision_fence_inventory_failure_after_apply_is_fail_stop() -> None:
    fence = contract.RefitRevisionFence()
    fence.begin_refit(1, "source-1", "staging-1", "inventory-1", "prototype-1")
    fence.mark_pytorch_applied(1, "head-1")
    fence.mark_staging_verified("staging-1")

    with pytest.raises(ValueError, match="refitter inventory"):
        fence.mark_refit_complete("wrong-inventory", "prototype-1")

    assert fence.phase == "failed_stopped"
    assert fence.active_pytorch_revision == 1
    assert fence.active_trt_revision == 0


def test_revision_fence_prototype_failure_after_apply_is_fail_stop() -> None:
    fence = contract.RefitRevisionFence()
    fence.begin_refit(1, "source-1", "staging-1", "inventory-1", "prototype-1")
    fence.mark_pytorch_applied(1, "head-1")
    fence.mark_staging_verified("staging-1")

    with pytest.raises(ValueError, match="prototype"):
        fence.mark_refit_complete("inventory-1", "wrong-prototype")

    assert fence.phase == "failed_stopped"


def test_revision_fence_can_abort_only_before_pytorch_apply() -> None:
    fence = contract.RefitRevisionFence()
    fence.begin_refit(1, "source-1", "staging-1", "inventory-1", "prototype-1")
    fence.abort_before_pytorch_apply()

    assert fence.phase == "idle"
    assert fence.active_pytorch_revision == 0
    assert fence.active_trt_revision == 0


def test_revision_fence_requires_monotonic_revision() -> None:
    fence = contract.RefitRevisionFence(
        active_pytorch_revision=4,
        active_trt_revision=4,
    )

    with pytest.raises(ValueError, match="must advance"):
        fence.begin_refit(4, "source", "staging", "inventory", "prototype")


def test_double_buffer_memory_preflight_preserves_headroom() -> None:
    value = contract.double_buffer_memory_preflight(
        free_before_bytes=16 << 30,
        total_device_bytes=80 << 30,
        candidate_engine_device_bytes=2 << 30,
        candidate_context_bytes=256 << 20,
        refit_workspace_bytes=512 << 20,
        transformed_staging_bytes=2 << 30,
        verification_io_bytes=256 << 20,
    )

    assert value["status"] == "qualified_double_buffer"
    assert value["double_buffer_allowed"] is True
    assert value["remaining_after_request_bytes"] >= 8 << 30


def test_double_buffer_memory_preflight_degrades_to_inventory_only() -> None:
    value = contract.double_buffer_memory_preflight(
        free_before_bytes=10 << 30,
        total_device_bytes=80 << 30,
        candidate_engine_device_bytes=2 << 30,
        candidate_context_bytes=256 << 20,
        refit_workspace_bytes=512 << 20,
        transformed_staging_bytes=2 << 30,
        verification_io_bytes=256 << 20,
    )

    assert value["status"] == "single_engine_inventory_only"
    assert value["double_buffer_allowed"] is False
    assert value["double_buffer_lifecycle_gate_allowed"] is False
    assert value["single_engine_latency_or_authority_claim_forbidden"] is True


def test_machine_contract_keeps_eager_actor_as_gradient_authority() -> None:
    path = TOOLS / "contract-n1d7-refittable-dit.json"
    value = json.loads(path.read_text(encoding="utf-8"))

    assert value["scope"]["status"] == "independent_feasibility_only"
    assert value["parameter_authority"]["tensorrt_refit_prefix"] == (
        "action_head.model"
    )
    assert value["ppo_authority"]["current_policy_and_gradient_executor"] == (
        "eager_FSDP_DiT_on_Actor"
    )
    assert value["ppo_authority"]["TensorRT_vs_TensorRT_is_not_gradient_authority"]
    assert value["refit_engine"]["refit_identical_allowed"] is False
    assert value["refit_engine"]["adoption"].startswith("double_buffered")
    assert value["refit_engine"]["failure_policy"].startswith("fail_stop")
    assert value["revision_gate"]["inference_requires_equal_revisions"] is True
    assert value["revision_gate"]["verification_digests_are_distinct"] is True
    assert value["memory_preflight"]["safety_headroom_bytes"] == 8 << 30
    assert value["memory_preflight"]["insufficient_memory_disposition"] == (
        "single_engine_inventory_only"
    )


def test_machine_contract_freezes_observed_parameter_counts() -> None:
    path = TOOLS / "contract-n1d7-refittable-dit.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    gate = value["offline_parameter_gate"]

    assert gate["expected_dit_tensors"] == 456
    assert gate["expected_dit_parameters"] == 1_091_722_240
    assert gate["expected_dit_bytes_bf16"] == 2_183_444_480
    assert gate["expected_initializer_transforms"] == {
        "identity": 263,
        "transpose_2d": 193,
    }


def test_offline_receipt_requires_explicit_generation_environment() -> None:
    source = (TOOLS / "refittable_dit_contract.py").read_text(encoding="utf-8")

    assert '"checkpoint_header_sha256"' in source
    assert '"onnx_version"' in source
    assert '"python_version"' in source
