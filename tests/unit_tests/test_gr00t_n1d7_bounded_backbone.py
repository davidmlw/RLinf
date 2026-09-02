# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch

from rlinf.models.embodiment.gr00t.gr00t_n1d7.bounded_backbone import (
    run_bounded_frozen_qwen3_backbone,
)


class _BaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kwargs = None
        self.language_model = torch.nn.Module()
        self.language_model.norm = _OffsetNorm()

    def forward(self, **kwargs):
        self.kwargs = kwargs
        input_ids = kwargs["input_ids"]
        hidden = torch.arange(
            input_ids.numel() * 4,
            dtype=torch.float32,
        ).reshape(*input_ids.shape, 4)
        return SimpleNamespace(last_hidden_state=self.language_model.norm(hidden))


class _OffsetNorm(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 100


class _CountingHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(7, 4), requires_grad=False)
        self.rows = []
        self.inputs = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.rows.append(int(hidden_states.shape[0]))
        self.inputs.append(hidden_states.detach().clone())
        return torch.nn.functional.linear(hidden_states, self.weight)


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        conditional = torch.nn.Module()
        conditional.model = _BaseModel()
        conditional.lm_head = _CountingHead()
        conditional.config = SimpleNamespace(image_token_id=9)
        self.model = conditional
        self.eval_calls = 0

    def set_frozen_modules_to_eval_mode(self) -> None:
        self.eval_calls += 1


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 9, 2], [9, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        "pixel_values": torch.ones(2, 3),
        "image_grid_thw": torch.ones(2, 3, dtype=torch.int64),
        "ignored": torch.tensor(1),
    }


def test_bounded_backbone_keeps_last_hidden_state_and_chunks_logits() -> None:
    backbone = _Backbone()
    output = run_bounded_frozen_qwen3_backbone(
        backbone,
        _inputs(),
        compute_unused_logits=True,
        logits_chunk_rows=3,
    )

    assert output["backbone_features"].shape == (2, 3, 4)
    assert output["backbone_features"][0, 0, 0].item() == 0
    assert backbone.model.lm_head.inputs[0][0, 0].item() == 100
    assert torch.equal(
        output["image_mask"],
        torch.tensor([[False, True, False], [True, False, False]]),
    )
    assert torch.equal(
        output["backbone_attention_mask"],
        torch.tensor([[True, True, False], [True, False, False]]),
    )
    assert backbone.model.lm_head.rows == [3, 3]
    assert backbone.eval_calls == 1
    assert backbone.model.model.kwargs["output_hidden_states"] is False
    assert backbone.model.model.kwargs["return_dict"] is True
    assert "ignored" not in backbone.model.model.kwargs


def test_bounded_backbone_can_skip_unused_logits() -> None:
    backbone = _Backbone()
    run_bounded_frozen_qwen3_backbone(
        backbone,
        _inputs(),
        compute_unused_logits=False,
        logits_chunk_rows=3,
    )
    assert backbone.model.lm_head.rows == []


def test_bounded_backbone_rejects_trainable_parameters() -> None:
    backbone = _Backbone()
    backbone.model.lm_head.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="fully frozen"):
        run_bounded_frozen_qwen3_backbone(
            backbone,
            _inputs(),
            compute_unused_logits=False,
            logits_chunk_rows=3,
        )
