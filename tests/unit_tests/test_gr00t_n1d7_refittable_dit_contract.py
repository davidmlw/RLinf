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

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "toolkits/eos/gr00t_trocar/tensorrt"
sys.path.insert(0, str(TOOLS))

import build_refittable_dit_b8 as build_gate  # noqa: E402
import export_refittable_dit_b8 as export_gate  # noqa: E402
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


def _admit(fence: contract.RefitRevisionFence, revision: int = 1) -> str:
    return fence.admit_patch(
        revision,
        f"patch-{revision}",
        f"source-{revision}",
        f"inventory-{revision}",
        f"prototype-{revision}",
    )


def _mark_applied(fence: contract.RefitRevisionFence, revision: int = 1) -> None:
    fence.mark_pytorch_applied(
        revision,
        observed_patch_digest=f"patch-{revision}",
        head_digest=f"head-{revision}",
        observed_source_weight_digest=f"source-{revision}",
    )


def _verify_staging(fence: contract.RefitRevisionFence, revision: int = 1) -> None:
    fence.register_staging_reference(f"source-{revision}", f"staging-{revision}")
    fence.mark_staging_verified(f"staging-{revision}")


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
    assert _admit(fence) == "b"

    with pytest.raises(RuntimeError, match="cannot infer"):
        fence.begin_inference()


def test_revision_fence_commits_only_a_fully_verified_revision() -> None:
    fence = contract.RefitRevisionFence(active_probe_output_digest="output-0")
    candidate = _admit(fence)
    _mark_applied(fence)
    _verify_staging(fence)
    fence.mark_refit_complete("inventory-1", "prototype-1")
    fence.mark_probe_verified(
        probe_input_digest="probe-input",
        pytorch_output_digest="pytorch-output-1",
        trt_output_digest="trt-output-1",
        metrics_digest="probe-metrics-1",
        numerics_passed=True,
        require_output_change=True,
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
    assert _admit(fence, 5) == "a"
    _mark_applied(fence, 5)
    fence.register_staging_reference("source-5", "staging-5")

    with pytest.raises(ValueError, match="staging digest"):
        fence.mark_staging_verified("wrong")

    assert fence.active_pytorch_revision == 5
    assert fence.active_trt_revision == 4
    assert fence.active_slot == "b"
    assert fence.phase == "failed_stopped"
    with pytest.raises(RuntimeError, match="cannot infer"):
        fence.begin_inference()


def test_revision_fence_rejects_post_patch_source_lineage_mismatch() -> None:
    fence = contract.RefitRevisionFence()
    _admit(fence)

    with pytest.raises(ValueError, match="source_weight"):
        fence.mark_pytorch_applied(
            1,
            observed_patch_digest="patch-1",
            head_digest="head-1",
            observed_source_weight_digest="unrelated-source",
        )

    assert fence.active_pytorch_revision == 1
    assert fence.active_trt_revision == 0
    assert fence.observed_source_weight_digest == "unrelated-source"
    assert fence.phase == "failed_stopped"


def test_revision_fence_rejects_unrelated_staging_reference() -> None:
    fence = contract.RefitRevisionFence()
    _admit(fence)
    _mark_applied(fence)

    with pytest.raises(ValueError, match="does not match observed bytes"):
        fence.register_staging_reference("unrelated-source", "staging-1")

    assert fence.phase == "failed_stopped"


def test_revision_fence_probe_failure_is_fail_stop() -> None:
    fence = contract.RefitRevisionFence(active_probe_output_digest="same-output")
    _admit(fence)
    _mark_applied(fence)
    _verify_staging(fence)
    fence.mark_refit_complete("inventory-1", "prototype-1")

    with pytest.raises(ValueError, match="did not change"):
        fence.mark_probe_verified(
            probe_input_digest="probe-input",
            pytorch_output_digest="pytorch-output-1",
            trt_output_digest="same-output",
            metrics_digest="probe-metrics-1",
            numerics_passed=True,
            require_output_change=True,
        )

    assert fence.phase == "failed_stopped"


def test_revision_fence_allows_explicit_no_dit_change_probe() -> None:
    fence = contract.RefitRevisionFence(active_probe_output_digest="same-output")
    _admit(fence)
    _mark_applied(fence)
    _verify_staging(fence)
    fence.mark_refit_complete("inventory-1", "prototype-1")

    fence.mark_probe_verified(
        probe_input_digest="probe-input",
        pytorch_output_digest="pytorch-output-1",
        trt_output_digest="same-output",
        metrics_digest="probe-metrics-1",
        numerics_passed=True,
        require_output_change=False,
    )

    assert fence.phase == "verified"


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
    _admit(fence)
    _mark_applied(fence)
    _verify_staging(fence)

    with pytest.raises(ValueError, match="refitter inventory"):
        fence.mark_refit_complete("wrong-inventory", "prototype-1")

    assert fence.phase == "failed_stopped"
    assert fence.active_pytorch_revision == 1
    assert fence.active_trt_revision == 0


def test_revision_fence_prototype_failure_after_apply_is_fail_stop() -> None:
    fence = contract.RefitRevisionFence()
    _admit(fence)
    _mark_applied(fence)
    _verify_staging(fence)

    with pytest.raises(ValueError, match="prototype"):
        fence.mark_refit_complete("inventory-1", "wrong-prototype")

    assert fence.phase == "failed_stopped"


def test_revision_fence_can_abort_only_before_pytorch_apply() -> None:
    fence = contract.RefitRevisionFence()
    _admit(fence)
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
        _admit(fence, 4)


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


class _FakeTensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def contiguous(self):
        return self

    def numpy(self):
        return np.zeros(self.shape, dtype=np.uint8)


class _Capture:
    def __init__(self):
        self.sa_embs = _FakeTensor((8, 41, 1536), "torch.bfloat16")
        self.vl_embs = _FakeTensor((8, 208, 2048), "torch.bfloat16")
        self.timestep = _FakeTensor((8,), "torch.int64")
        self.image_mask = _FakeTensor((8, 208), "torch.bool")
        self.backbone_attention_mask = _FakeTensor((8, 208), "torch.bool")


def test_refittable_dit_export_capture_contract() -> None:
    value = export_gate.validate_capture(_Capture())

    assert value["sa_embs"]["shape"] == [8, 41, 1536]
    assert value["vl_embs"]["shape"] == [8, 208, 2048]
    assert value["image_mask"]["dtype"] == "torch.bool"


def test_refittable_dit_export_rejects_b1_capture() -> None:
    capture = _Capture()
    capture.timestep = _FakeTensor((1,), "torch.int64")

    with pytest.raises(RuntimeError, match="captured DiT ABI mismatch"):
        export_gate.validate_capture(capture)


def test_refittable_dit_profile_preserves_static_batch_and_action_sequence() -> None:
    assert build_gate._profile_shape((8, -1, 2048), 208) == (8, 208, 2048)
    assert build_gate._profile_shape((8, 41, 1536), 208) == (8, 41, 1536)


def test_refittable_dit_binding_gate_rejects_dynamic_batch() -> None:
    bindings = [
        {"name": "sa_embs", "mode": "input", "shape": [-1, 41, 1536]},
        {"name": "vl_embs", "mode": "input", "shape": [-1, -1, 2048]},
        {"name": "timestep", "mode": "input", "shape": [-1]},
        {"name": "image_mask", "mode": "input", "shape": [-1, -1]},
        {
            "name": "backbone_attention_mask",
            "mode": "input",
            "shape": [-1, -1],
        },
    ]

    with pytest.raises(RuntimeError, match="DiT binding mismatch"):
        build_gate._validate_bindings(bindings)


def test_refitter_inventory_allows_recorded_graph_constants(monkeypatch) -> None:
    mapping = {
        "dit_refit": {
            "entries": [
                {
                    "initializer": "dit.linear.bias",
                    "parameter_count": 3,
                }
            ]
        }
    }
    inventory = {
        "named_weights": [
            {"name": "dit.linear.bias", "count": 3, "dtype": "BF16"},
            {"name": "/dit/Constant_7_output_0", "count": 1, "dtype": "INT64"},
        ]
    }
    monkeypatch.setattr(build_gate, "EXPECTED_REFIT_WEIGHTS", 1)
    value = build_gate._validate_refitter_against_map(inventory, mapping)

    assert value["mapped_trainable_count"] == 1
    assert value["derived_constant_count"] == 1
    assert value["derived_constants_policy"] == "retain_plan_value_not_updated"


def test_refitter_inventory_rejects_missing_trainable_weight(monkeypatch) -> None:
    mapping = {
        "dit_refit": {
            "entries": [{"initializer": "dit.linear.bias", "parameter_count": 3}]
        }
    }

    monkeypatch.setattr(build_gate, "EXPECTED_REFIT_WEIGHTS", 1)
    with pytest.raises(RuntimeError, match="does not expose every trainable"):
        build_gate._validate_refitter_against_map({"named_weights": []}, mapping)


def test_real_revision_probe_records_both_transforms_and_two_slots() -> None:
    source = (TOOLS / "refit_dit_real_revision_probe.py").read_text(encoding="utf-8")

    assert 'for transform in ("identity", "transpose_2d")' in source
    assert '"staging_digest_revision_0"' in source
    assert '"staging_digest_revision_1"' in source
    assert '"engine_instances": 2' in source
    assert '"old_slot_unchanged_before_switch"' in source
    assert "DEFAULT_DOUBLE_BUFFER_HEADROOM_BYTES" in source


def test_revision_zero_diagnostic_is_wired_to_existing_ppo_gate() -> None:
    model_source = (
        ROOT / "rlinf/models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py"
    ).read_text(encoding="utf-8")
    worker_source = (ROOT / "rlinf/workers/rollout/hf/huggingface_worker.py").read_text(
        encoding="utf-8"
    )
    runner_source = (ROOT / "toolkits/eos/gr00t_trocar/run_n1d7_hybrid.sh").read_text(
        encoding="utf-8"
    )
    runtime_source = (
        ROOT / "rlinf/models/embodiment/gr00t/gr00t_n1d7/tensorrt_dit.py"
    ).read_text(encoding="utf-8")
    verify_source = model_source.split("def verify_online_update_contract", 1)[1].split(
        "def hybrid_runtime_telemetry", 1
    )[0]

    assert "enable_tensorrt_dit_diagnostic" in model_source
    assert "tensorrt_dit.verify_revision(revision)" in model_source
    assert "if contract is not None:" in verify_source
    assert "if contract is None:" not in verify_source
    assert "verify_online_update(applied_version)" in worker_source
    assert "W83_TRT_DIT_DIAGNOSTIC" in runner_source
    assert "W83_TRT_DIT_ONLINE" in runner_source
    assert "++rollout.model.tensorrt_dit.online_refit=true" in runner_source
    assert "++rollout.model.tensorrt_dit.lineage_receipt_mode=" in runner_source
    assert "++rollout.model.enable_eager_dit_timing=true" in runner_source
    assert "revision_zero_PPO_identity_diagnostic_no_online_refit" in runtime_source
    assert "online_double_slot_refittable_tensorrt_dit" in runtime_source
    assert "refuses online updates" in runtime_source


def test_revision_zero_diagnostic_matches_alternate_vl_dit_keywords() -> None:
    runtime_source = (
        ROOT / "rlinf/models/embodiment/gr00t/gr00t_n1d7/tensorrt_dit.py"
    ).read_text(encoding="utf-8")
    executor_source = runtime_source.split("class RefittableTensorRTDiT", 1)[1]
    signature = executor_source.split("def __call__(", 1)[1].split(
        ") -> torch.Tensor:", 1
    )[0]

    assert "hidden_states: torch.Tensor" in signature
    assert "encoder_hidden_states: torch.Tensor" in signature
    assert '"sa_embs": hidden_states' in runtime_source
    assert '"vl_embs": encoder_hidden_states' in runtime_source


def test_revision_zero_diagnostic_eager_shadow_preserves_tensorrt_output() -> None:
    import torch

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.tensorrt_dit import (
        TensorRTDiTRevisionZeroDiagnostic,
    )

    class FakeEngine:
        def set_runtime_tensor_shape(self, _name, _shape):
            return None

        def __call__(self, **_inputs):
            return {"output": torch.full((8, 41, 1024), 1.125, dtype=torch.bfloat16)}

    diagnostic = TensorRTDiTRevisionZeroDiagnostic.__new__(
        TensorRTDiTRevisionZeroDiagnostic
    )
    diagnostic.closed = False
    diagnostic.active_revision = 0
    diagnostic.expected_revision = 0
    diagnostic.engine = FakeEngine()
    diagnostic.shadow_eager = True
    diagnostic.eager_forward = lambda **_inputs: torch.ones(
        (8, 41, 1024), dtype=torch.bfloat16
    )
    diagnostic.shadow_calls = 0
    diagnostic.shadow_by_timestep = {}
    diagnostic._phase = "idle"
    diagnostic._timing_events = []

    output = diagnostic(
        hidden_states=torch.zeros((8, 41, 1536), dtype=torch.bfloat16),
        encoder_hidden_states=torch.zeros((8, 208, 2048), dtype=torch.bfloat16),
        timestep=torch.full((8,), 250, dtype=torch.int64),
        image_mask=torch.ones((8, 208), dtype=torch.bool),
        backbone_attention_mask=torch.ones((8, 208), dtype=torch.bool),
    )
    summary = diagnostic._shadow_summary()

    torch.testing.assert_close(
        output, torch.full((8, 41, 1024), 1.125, dtype=torch.bfloat16)
    )
    assert summary["trajectory_executor"] == "tensorrt"
    assert summary["calls"] == 1
    assert summary["per_timestep"][0]["timestep_bucket"] == 250
    assert summary["per_timestep"][0]["mean_abs"] == pytest.approx(0.125)
    assert summary["per_timestep"][0]["max_abs"] == pytest.approx(0.125)


def test_online_refit_adopts_only_the_verified_inactive_slot(monkeypatch) -> None:
    import torch

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.tensorrt_dit import (
        RefittableTensorRTDiT,
    )

    executor = RefittableTensorRTDiT.__new__(RefittableTensorRTDiT)
    executor.active_revision = 0
    executor.active_slot = 0
    executor.engine = "slot-0"
    executor.engines = ["slot-0", "slot-1"]
    executor._phase = "idle"
    executor._probe = {"frozen": object()}
    executor._initial_probe_pending = False
    executor._memory = {"minimum_free_device_bytes": 1}
    executor.refit_records = []
    executor.probe_each_revision = True
    executor.lineage_receipt_mode = "qualification_sha256"
    executor._source_state = lambda: {"weight": object()}
    executor._stage_weights = lambda _source: (
        {"weight": object()},
        {"staging_device_ms": 2.0, "staging_wall_ms": 2.5},
    )
    executor._validate_staging = lambda _source, _staged: {
        "staging_validation_device_ms": 1.0,
        "staging_validation_wall_ms": 1.5,
    }
    executor._lineage_digests = lambda _staged: ("source-1", "staging-1", 6.0)
    executor._refit_slot = lambda slot, _staged: {
        "slot": slot,
        "weight_count": 1,
        "set_weights_wall_ms": 3.0,
        "refit_wall_ms": 4.0,
        "refit_device_ms": 4.0,
    }
    executor._verify_probe = lambda engine: {"engine": engine, "cosine": 1.0}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (100, 200))

    executor._adopt_online_revision(1)

    assert executor.active_revision == 1
    assert executor.active_slot == 1
    assert executor.engine == "slot-1"
    assert executor._phase == "idle"
    assert executor.observed_source_digest == "source-1"
    assert executor.observed_staging_digest == "staging-1"
    assert executor.refit_records[-1]["probe"]["engine"] == "slot-1"


def test_online_refit_probe_failure_keeps_old_slot_and_fail_stops(monkeypatch) -> None:
    import torch

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.tensorrt_dit import (
        RefittableTensorRTDiT,
    )

    executor = RefittableTensorRTDiT.__new__(RefittableTensorRTDiT)
    executor.active_revision = 0
    executor.active_slot = 0
    executor.engine = "slot-0"
    executor.engines = ["slot-0", "slot-1"]
    executor._phase = "idle"
    executor._probe = {"frozen": object()}
    executor._initial_probe_pending = False
    executor._memory = {"minimum_free_device_bytes": 1}
    executor.refit_records = []
    executor.probe_each_revision = True
    executor._source_state = lambda: {}
    executor._stage_weights = lambda _source: (
        {},
        {"staging_device_ms": 0.0, "staging_wall_ms": 0.0},
    )
    executor._validate_staging = lambda _source, _staged: {
        "staging_validation_device_ms": 0.0,
        "staging_validation_wall_ms": 0.0,
    }
    executor._lineage_digests = lambda _staged: ("source-1", "staging-1", 0.0)
    executor._refit_slot = lambda _slot, _staged: {}
    executor._verify_probe = lambda _engine: (_ for _ in ()).throw(
        RuntimeError("probe failed")
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (100, 200))

    with pytest.raises(RuntimeError, match="probe failed"):
        executor._adopt_online_revision(1)

    assert executor.active_revision == 0
    assert executor.active_slot == 0
    assert executor.engine == "slot-0"
    assert executor._phase == "failed_stopped"


def test_online_refit_freezes_first_live_input_as_revision_probe() -> None:
    import torch

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.tensorrt_dit import (
        RefittableTensorRTDiT,
    )

    output = torch.full((8, 41, 1024), 1.125, dtype=torch.bfloat16)

    class FakeEngine:
        def set_runtime_tensor_shape(self, _name, _shape):
            return None

        def __call__(self, **_inputs):
            return {"output": output}

    executor = RefittableTensorRTDiT.__new__(RefittableTensorRTDiT)
    executor.closed = False
    executor.active_revision = 0
    executor.active_slot = 0
    executor.engine = FakeEngine()
    executor.engines = [executor.engine]
    executor.eager_forward = lambda **_inputs: output.clone()
    executor.minimum_probe_cosine = 0.999
    executor.maximum_probe_relative_l2 = 0.05
    executor._phase = "idle"
    executor._probe = None
    executor._initial_probe_pending = True
    executor._probe_input_digest = None
    executor.refit_records = []
    executor._timing_events = []
    executor.shadow_eager = False

    result = executor(
        hidden_states=torch.zeros((8, 41, 1536), dtype=torch.float32),
        encoder_hidden_states=torch.zeros((8, 208, 2048), dtype=torch.float32),
        timestep=torch.full((8,), 250, dtype=torch.int64),
        image_mask=torch.ones((8, 208), dtype=torch.bool),
        backbone_attention_mask=torch.ones((8, 208), dtype=torch.bool),
    )

    assert result is output
    assert executor._initial_probe_pending is False
    assert executor._probe_input_digest is not None
    assert executor._probe["sa_embs"].dtype == torch.bfloat16
    assert executor._probe["vl_embs"].dtype == torch.bfloat16
    assert executor.refit_records[-1]["initial_live_probe"]["cosine"] == pytest.approx(
        1.0, abs=2e-7
    )
    assert executor.refit_records[-1]["probe_input_digest"]


def test_online_refit_live_probe_failure_fail_stops() -> None:
    import torch

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.tensorrt_dit import (
        RefittableTensorRTDiT,
    )

    executor = RefittableTensorRTDiT.__new__(RefittableTensorRTDiT)
    executor._initial_probe_pending = True
    executor._probe = None
    executor.active_revision = 0
    executor.active_slot = 0
    executor.engines = []
    executor.eager_forward = lambda **_inputs: torch.zeros(
        (8, 41, 1024), dtype=torch.bfloat16
    )
    executor.minimum_probe_cosine = 0.999
    executor.maximum_probe_relative_l2 = 0.05
    executor.refit_records = []
    executor._phase = "idle"

    inputs = {
        "sa_embs": torch.zeros((8, 41, 1536), dtype=torch.bfloat16),
        "vl_embs": torch.zeros((8, 208, 2048), dtype=torch.bfloat16),
        "timestep": torch.full((8,), 250, dtype=torch.int64),
        "image_mask": torch.ones((8, 208), dtype=torch.bool),
        "backbone_attention_mask": torch.ones((8, 208), dtype=torch.bool),
    }
    with pytest.raises(RuntimeError, match="revision probe failed"):
        executor._freeze_initial_live_probe(
            inputs, torch.ones((8, 41, 1024), dtype=torch.bfloat16)
        )

    assert executor._phase == "failed_stopped"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"probe_each_revision": False}, "probe every revision"),
        ({"minimum_free_device_bytes": (8 << 30) - 1}, "at least 8 GiB"),
        ({"ppo_authority_status": "passed"}, "failed PPO authority"),
        ({"lineage_receipt_mode": "disabled"}, "lineage_receipt_mode"),
    ],
)
def test_online_refit_rejects_weakened_safety_config(override, message) -> None:
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.tensorrt_dit import (
        RefittableTensorRTDiT,
    )

    config = {
        "online_refit": True,
        "probe_each_revision": True,
        "minimum_free_device_bytes": 8 << 30,
        "ppo_authority_status": "failed_ratio_kl_approximate_behavior_only",
        **override,
    }

    with pytest.raises(ValueError, match=message):
        RefittableTensorRTDiT(None, config)


def test_online_refit_gpu_validation_mode_skips_host_sha_receipt() -> None:
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.tensorrt_dit import (
        RefittableTensorRTDiT,
    )

    executor = RefittableTensorRTDiT.__new__(RefittableTensorRTDiT)
    executor.lineage_receipt_mode = "gpu_transform_validation"

    assert executor._lineage_digests({}) == (None, None, 0.0)


def test_eager_timing_waits_for_end_event(monkeypatch) -> None:
    import torch

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.tensorrt_dit import (
        CudaTimedDiTForward,
    )

    events = []

    class FakeEvent:
        def __init__(self, **_kwargs):
            self.synchronized = False
            events.append(self)

        def record(self, _stream):
            return None

        def synchronize(self):
            self.synchronized = True

        def elapsed_time(self, end):
            assert end.synchronized
            return 7.0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    timed = CudaTimedDiTForward(lambda value: value + 1)
    timed.set_revision(4)

    assert timed(2) == 3
    telemetry = timed.telemetry()

    assert events[0].synchronized is False
    assert events[1].synchronized is True
    assert telemetry["by_revision"]["4"]["mean_ms"] == 7.0
