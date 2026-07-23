# Copyright 2025 The RLinf Authors.
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

import torch

from rlinf.models.embodiment.gr00t.gr00t_n1d5.unused_logits import (
    _NoOpLmHead,
    replace_unused_lm_head,
)


def _model(lm_head: torch.nn.Module) -> torch.nn.Module:
    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.eagle_model = torch.nn.Module()
    model.backbone.eagle_model.language_model = torch.nn.Module()
    model.backbone.eagle_model.language_model.lm_head = lm_head
    model.backbone.eagle_model.language_model.model = SimpleNamespace()
    return model


def test_replace_unused_lm_head_preserves_parameters_and_state_dict() -> None:
    model = _model(torch.nn.Linear(4, 7, bias=False))
    original_head = model.backbone.eagle_model.language_model.lm_head
    original_state = model.state_dict()
    original_parameters = {id(parameter) for parameter in model.parameters()}

    assert replace_unused_lm_head(model) is True

    replacement = model.backbone.eagle_model.language_model.lm_head
    assert isinstance(replacement, _NoOpLmHead)
    assert replacement.weight is original_head.weight
    assert {id(parameter) for parameter in model.parameters()} == original_parameters
    assert model.state_dict().keys() == original_state.keys()
    assert replacement(torch.randn(2, 3, 4)) is None


def test_replace_unused_lm_head_is_idempotent() -> None:
    model = _model(torch.nn.Linear(4, 7, bias=True))

    assert replace_unused_lm_head(model) is True
    replacement = model.backbone.eagle_model.language_model.lm_head
    assert replace_unused_lm_head(model) is False
    assert model.backbone.eagle_model.language_model.lm_head is replacement
