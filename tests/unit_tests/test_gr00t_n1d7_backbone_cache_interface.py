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

from types import SimpleNamespace

import pytest
import torch


def _forward_inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "state": torch.arange(batch_size * 2, dtype=torch.float32).reshape(
            batch_size, 1, 2
        ),
        "state_mask": torch.ones(batch_size, 1, dtype=torch.bool),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.int64),
        "input_ids": torch.ones(batch_size, 3, dtype=torch.int64),
        "attention_mask": torch.ones(batch_size, 3, dtype=torch.int64),
        "pixel_values": torch.ones(batch_size, 1, 2),
        "image_grid_thw": torch.ones(batch_size, 1, 3, dtype=torch.int64),
        "chains": torch.zeros(batch_size, 2, 1, 2),
        "denoise_inds": torch.zeros(batch_size, 1, dtype=torch.int64),
    }


def test_n1d7_fresh_and_precomputed_backbone_outputs_match():
    pytest.importorskip("gr00t")
    from transformers.feature_extraction_utils import BatchFeature

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )

    class FakeActionHead:
        action_chunk = 1
        dtype = torch.float32
        rl_config = {"joint_logprob": False}

        @staticmethod
        def prepare_input(batch):
            return BatchFeature(
                data={
                    key: batch[key] for key in ("state", "state_mask", "embodiment_id")
                }
            )

        def __call__(
            self,
            *,
            backbone_output,
            action_input,
            chains,
            denoise_inds,
            compute_values,
        ):
            del action_input, chains, denoise_inds, compute_values
            features = backbone_output["backbone_features"]
            log_probs = features[:, None, :1, :2]
            values = features.mean(dim=(1, 2), keepdim=False)[:, None]
            return log_probs, values

    class FakePolicy:
        valid_action_dim = 2
        action_dim = 2
        padding_value = 0
        device = torch.device("cpu")
        _prepare_action_head_input = (
            GR00T_N1_7_ForRLActionPrediction._prepare_action_head_input
        )

        def __init__(self):
            self.action_head = FakeActionHead()
            self.backbone_calls = 0

        def prepare_input(self, normalized_input):
            return BatchFeature(data=normalized_input), self.action_head.prepare_input(
                normalized_input
            )

        def _forward_backbone(self, backbone_inputs):
            self.backbone_calls += 1
            state = backbone_inputs["state"]
            features = state.reshape(state.shape[0], 1, -1) * 2
            return BatchFeature(
                data={
                    "backbone_features": features,
                    "backbone_attention_mask": torch.ones(
                        features.shape[:2], dtype=torch.bool
                    ),
                    "image_mask": torch.zeros(features.shape[:2], dtype=torch.bool),
                }
            )

    forward_inputs = _forward_inputs()
    prev_logprobs = torch.zeros(2, 1, 1, 2)
    fresh_policy = FakePolicy()
    fresh = GR00T_N1_7_ForRLActionPrediction.default_forward(
        fresh_policy,
        forward_inputs=forward_inputs,
        prev_logprobs=prev_logprobs,
    )

    state = forward_inputs["state"]
    precomputed = {
        "backbone_features": state.reshape(state.shape[0], 1, -1) * 2,
        "backbone_attention_mask": torch.ones(2, 1, dtype=torch.bool),
        "image_mask": torch.zeros(2, 1, dtype=torch.bool),
    }
    cached_policy = FakePolicy()
    cached = GR00T_N1_7_ForRLActionPrediction.default_forward(
        cached_policy,
        forward_inputs={
            key: value
            for key, value in forward_inputs.items()
            if key
            not in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        },
        prev_logprobs=prev_logprobs,
        precomputed_backbone=precomputed,
    )

    assert fresh_policy.backbone_calls == 1
    assert cached_policy.backbone_calls == 0
    for key in ("logprobs", "prev_logprobs", "values"):
        torch.testing.assert_close(cached[key], fresh[key])


def test_n1d7_rollout_captures_all_raw_backbone_outputs():
    pytest.importorskip("gr00t")
    from transformers.feature_extraction_utils import BatchFeature

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )
    from rlinf.utils.backbone_cache import (
        ROLLOUT_BACKBONE_FEATURE_KEY,
        ROLLOUT_BACKBONE_IMAGE_MASK_KEY,
        ROLLOUT_BACKBONE_MASK_KEY,
    )

    raw_features = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    attention_mask = torch.ones(2, 2, dtype=torch.bool)
    image_mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)

    class MutatingActionHead:
        @staticmethod
        def get_rl_action(backbone_outputs, action_inputs, mode):
            del action_inputs, mode
            backbone_outputs["backbone_features"] = (
                backbone_outputs["backbone_features"] + 100
            )
            return {}, {
                "actions": torch.zeros(2, 1, 2),
                "chains": torch.zeros(2, 2, 1, 2),
                "denoise_inds": torch.zeros(2, 1, dtype=torch.int64),
                "prev_logprobs": torch.zeros(2, 1, 1, 2),
                "prev_values": torch.zeros(2, 1),
            }

    class FakePolicy:
        capture_rollout_backbone_output = True
        action_head = MutatingActionHead()
        _finalize_rollout_forward_inputs = (
            GR00T_N1_7_ForRLActionPrediction._finalize_rollout_forward_inputs
        )

        @staticmethod
        def prepare_input(normalized_input):
            del normalized_input
            return BatchFeature(data={}), BatchFeature(data={})

        @staticmethod
        def _forward_backbone(backbone_inputs):
            del backbone_inputs
            return BatchFeature(
                data={
                    "backbone_features": raw_features,
                    "backbone_attention_mask": attention_mask,
                    "image_mask": image_mask,
                }
            )

    _, result = GR00T_N1_7_ForRLActionPrediction._get_rl_action(
        FakePolicy(), _forward_inputs()
    )
    captured = result["forward_inputs"]

    torch.testing.assert_close(captured[ROLLOUT_BACKBONE_FEATURE_KEY], raw_features)
    torch.testing.assert_close(captured[ROLLOUT_BACKBONE_MASK_KEY], attention_mask)
    torch.testing.assert_close(captured[ROLLOUT_BACKBONE_IMAGE_MASK_KEY], image_mask)


def test_n1d7_rollout_snapshots_reused_backend_outputs():
    pytest.importorskip("gr00t")
    from transformers.feature_extraction_utils import BatchFeature

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )
    from rlinf.utils.backbone_cache import ROLLOUT_BACKBONE_FEATURE_KEY

    persistent_features = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)

    class FakeBackend:
        reuses_output_buffers = True

    class FakeActionHead:
        @staticmethod
        def get_rl_action(backbone_outputs, action_inputs, mode):
            del backbone_outputs, action_inputs, mode
            return {}, {
                "actions": torch.zeros(2, 1, 2),
                "chains": torch.zeros(2, 2, 1, 2),
                "denoise_inds": torch.zeros(2, 1, dtype=torch.int64),
                "prev_logprobs": torch.zeros(2, 1, 1, 2),
                "prev_values": torch.zeros(2, 1),
            }

    class FakePolicy:
        capture_rollout_backbone_output = True
        action_head = FakeActionHead()
        _tensorrt_backbone = FakeBackend()
        _finalize_rollout_forward_inputs = (
            GR00T_N1_7_ForRLActionPrediction._finalize_rollout_forward_inputs
        )

        @staticmethod
        def prepare_input(normalized_input):
            del normalized_input
            return BatchFeature(data={}), BatchFeature(data={})

        @staticmethod
        def _forward_backbone(backbone_inputs):
            del backbone_inputs
            return BatchFeature(
                data={
                    "backbone_features": persistent_features,
                    "backbone_attention_mask": torch.ones(2, 2, dtype=torch.bool),
                    "image_mask": torch.zeros(2, 2, dtype=torch.bool),
                }
            )

    expected = persistent_features.clone()
    _, result = GR00T_N1_7_ForRLActionPrediction._get_rl_action(
        FakePolicy(), _forward_inputs()
    )
    captured = result["forward_inputs"][ROLLOUT_BACKBONE_FEATURE_KEY]
    persistent_features.add_(100)

    torch.testing.assert_close(captured, expected)
    assert captured.data_ptr() != persistent_features.data_ptr()


def test_n1d7_forward_backbone_prefers_installed_tensorrt_backend():
    pytest.importorskip("gr00t")
    from transformers.feature_extraction_utils import BatchFeature

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )

    class FakeBackend:
        def __init__(self):
            self.calls = 0

        def __call__(self, inputs):
            self.calls += 1
            return {
                "backbone_features": inputs["features"] + 1,
                "backbone_attention_mask": torch.ones(1, 1, dtype=torch.bool),
                "image_mask": torch.zeros(1, 1, dtype=torch.bool),
            }

    backend = FakeBackend()
    policy = SimpleNamespace(_tensorrt_backbone=backend)
    result = GR00T_N1_7_ForRLActionPrediction._forward_backbone(
        policy, BatchFeature(data={"features": torch.zeros(1, 1, 2)})
    )

    assert backend.calls == 1
    torch.testing.assert_close(result["backbone_features"], torch.ones(1, 1, 2))


def test_n1d7_eval_action_uses_backend_neutral_backbone():
    pytest.importorskip("gr00t")
    from transformers.feature_extraction_utils import BatchFeature

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )

    class FakeActionHead:
        def __init__(self):
            self.calls = 0

        def get_action(self, backbone_outputs, action_inputs):
            self.calls += 1
            torch.testing.assert_close(action_inputs["state"], torch.ones(1, 1, 2))
            return BatchFeature(
                data={"action_pred": backbone_outputs["backbone_features"] + 1}
            )

    class FakePolicy:
        device = torch.device("cpu")
        compute_dtype = torch.float32

        def __init__(self):
            self.action_head = FakeActionHead()
            self.backbone_calls = 0

        @staticmethod
        def prepare_input(normalized_input):
            return (
                BatchFeature(data={"features": normalized_input["features"]}),
                BatchFeature(data={"state": normalized_input["state"]}),
            )

        def _forward_backbone(self, backbone_inputs):
            self.backbone_calls += 1
            return BatchFeature(
                data={"backbone_features": backbone_inputs["features"] * 2}
            )

    policy = FakePolicy()
    result = GR00T_N1_7_ForRLActionPrediction._get_action_from_normalized_input(
        policy,
        {
            "features": torch.ones(1, 1, 2),
            "state": torch.ones(1, 1, 2),
        },
    )

    assert policy.backbone_calls == 1
    assert policy.action_head.calls == 1
    torch.testing.assert_close(result, torch.full((1, 1, 2), 3.0))


def test_n1d7_compiled_dit_allows_in_place_weight_update(monkeypatch):
    pytest.importorskip("gr00t")
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )

    model = torch.nn.Linear(2, 2)
    policy = SimpleNamespace(action_head=SimpleNamespace(model=model))
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)

    GR00T_N1_7_ForRLActionPrediction.enable_torch_compile(policy)
    parameter_id = id(model.weight)
    parameter_pointer = model.weight.data_ptr()
    with torch.no_grad():
        model.weight.add_(1)
    GR00T_N1_7_ForRLActionPrediction.verify_online_update_contract(policy)

    assert id(model.weight) == parameter_id
    assert model.weight.data_ptr() == parameter_pointer


def test_n1d7_compiled_dit_rejects_parameter_replacement(monkeypatch):
    pytest.importorskip("gr00t")
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )

    model = torch.nn.Linear(2, 2)
    policy = SimpleNamespace(action_head=SimpleNamespace(model=model))
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    GR00T_N1_7_ForRLActionPrediction.enable_torch_compile(policy)
    model.weight = torch.nn.Parameter(model.weight.detach().clone())

    with pytest.raises(RuntimeError, match="replaced compiled DiT"):
        GR00T_N1_7_ForRLActionPrediction.verify_online_update_contract(policy)


def test_n1d7_tensorrt_dit_revision_check_without_compiled_contract():
    pytest.importorskip("gr00t")
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )

    class FakeTensorRTDiT:
        def __init__(self):
            self.revisions = []

        def verify_revision(self, revision):
            self.revisions.append(revision)

    tensorrt_dit = FakeTensorRTDiT()
    policy = SimpleNamespace(_tensorrt_dit_diagnostic=tensorrt_dit)

    GR00T_N1_7_ForRLActionPrediction.verify_online_update_contract(policy, revision=0)

    assert tensorrt_dit.revisions == [0]


def test_n1d7_online_tensorrt_dit_revision_check_without_compiled_contract():
    pytest.importorskip("gr00t")
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        GR00T_N1_7_ForRLActionPrediction,
    )

    class FakeTensorRTDiT:
        def __init__(self):
            self.revisions = []

        def verify_revision(self, revision):
            self.revisions.append(revision)

    tensorrt_dit = FakeTensorRTDiT()
    policy = SimpleNamespace(_tensorrt_dit=tensorrt_dit)

    GR00T_N1_7_ForRLActionPrediction.verify_online_update_contract(policy, revision=3)

    assert tensorrt_dit.revisions == [3]
