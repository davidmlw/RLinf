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

"""Skip the unused Eagle vocabulary projection in the GR00T action caller."""

from __future__ import annotations

import torch


class _NoOpLmHead(torch.nn.Module):
    """Preserve the original lm_head parameters while skipping its projection."""

    def __init__(self, lm_head: torch.nn.Module) -> None:
        super().__init__()
        if not hasattr(lm_head, "weight"):
            raise TypeError("Eagle lm_head must expose a weight parameter")

        self.in_features = getattr(lm_head, "in_features", None)
        self.out_features = getattr(lm_head, "out_features", None)
        self.weight = lm_head.weight
        self.bias = getattr(lm_head, "bias", None)

    def forward(self, hidden_states: torch.Tensor) -> None:
        del hidden_states
        return None


def replace_unused_lm_head(model: torch.nn.Module) -> bool:
    """Disable logits for GR00T's hidden-state-only Eagle backbone caller.

    The replacement happens after from_pretrained so checkpoint loading is
    unaffected. It keeps the original parameters registered under the same
    state-dict keys, preserving weight tying and FSDP structure. This is only
    valid for the GR00T action path, which consumes Eagle hidden states and
    never consumes language-model logits.

    Returns True when a replacement was made and False when the model was
    already patched.
    """

    language_model = model.backbone.eagle_model.language_model
    if isinstance(language_model.lm_head, _NoOpLmHead):
        return False

    language_model.lm_head = _NoOpLmHead(language_model.lm_head)
    return True
