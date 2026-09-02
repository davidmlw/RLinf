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

"""Memory-bounded execution of the frozen GR00T N1.7 Qwen3-VL backbone."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


_QWEN3_BACKBONE_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
)


def run_bounded_frozen_qwen3_backbone(
    backbone: torch.nn.Module,
    vl_input: Mapping[str, torch.Tensor],
    *,
    compute_unused_logits: bool,
    logits_chunk_rows: int,
) -> dict[str, torch.Tensor]:
    """Return the final hidden state without retaining every decoder layer.

    The upstream GR00T N1.7 backbone asks the Hugging Face causal-LM wrapper for
    all hidden states and then keeps only the last one. At large PPO
    microbatches, the returned tuple alone exceeds an H100's memory. A pre-hook
    on the final language-model norm captures the exact pre-norm final state used
    by GR00T without retaining that tuple.

    Baseline runs can still execute the otherwise-unused vocabulary projection.
    It is split by token rows only to bound the discarded output allocation; the
    complete projection is evaluated. Optimized runs disable it explicitly.
    """

    if logits_chunk_rows <= 0:
        raise ValueError("Qwen3-VL logits_chunk_rows must be positive")
    trainable = [
        name
        for name, parameter in backbone.named_parameters()
        if parameter.requires_grad
    ]
    if trainable:
        raise RuntimeError(
            "bounded Qwen3-VL execution requires a fully frozen backbone; "
            f"found trainable parameters: {trainable[:4]}"
        )

    missing = [key for key in _QWEN3_BACKBONE_INPUT_KEYS if key not in vl_input]
    if missing:
        raise KeyError(f"Qwen3-VL backbone inputs are incomplete: missing={missing}")

    conditional_model = getattr(backbone, "model", None)
    base_model = getattr(conditional_model, "model", None)
    lm_head = getattr(conditional_model, "lm_head", None)
    config = getattr(conditional_model, "config", None)
    if base_model is None or lm_head is None or config is None:
        raise RuntimeError("unsupported Qwen3-VL backbone model layout")

    set_frozen_eval = getattr(backbone, "set_frozen_modules_to_eval_mode", None)
    if not callable(set_frozen_eval):
        raise RuntimeError("Qwen3-VL backbone lacks frozen-module eval control")
    set_frozen_eval()

    selected_input = {key: vl_input[key] for key in _QWEN3_BACKBONE_INPUT_KEYS}
    language_model = getattr(base_model, "language_model", None)
    final_norm = getattr(language_model, "norm", None)
    if not isinstance(final_norm, torch.nn.Module):
        raise RuntimeError("Qwen3-VL final language-model norm was not found")

    pre_norm_hidden_states: list[torch.Tensor] = []

    def _capture_pre_norm_hidden_states(
        _module: torch.nn.Module,
        inputs: tuple[Any, ...],
    ) -> None:
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("Qwen3-VL final norm received invalid inputs")
        pre_norm_hidden_states.append(inputs[0])

    hook = final_norm.register_forward_pre_hook(_capture_pre_norm_hidden_states)
    with torch.no_grad():
        try:
            outputs: Any = base_model(
                **selected_input,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            hook.remove()
        if len(pre_norm_hidden_states) != 1:
            raise RuntimeError(
                "Qwen3-VL final norm must execute exactly once per backbone forward"
            )
        hidden_states = pre_norm_hidden_states[0]
        post_norm_hidden_states = getattr(outputs, "last_hidden_state", None)
        if post_norm_hidden_states is None:
            post_norm_hidden_states = outputs[0]
        if (
            not isinstance(hidden_states, torch.Tensor)
            or hidden_states.ndim != 3
            or not isinstance(post_norm_hidden_states, torch.Tensor)
            or post_norm_hidden_states.shape != hidden_states.shape
        ):
            raise RuntimeError(
                "Qwen3-VL base model returned invalid pre/post-norm hidden states"
            )

        if compute_unused_logits:
            flat_hidden = post_norm_hidden_states.reshape(
                -1, post_norm_hidden_states.shape[-1]
            )
            for hidden_chunk in flat_hidden.split(logits_chunk_rows, dim=0):
                logits = lm_head(hidden_chunk)
                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.shape[0] != hidden_chunk.shape[0]
                ):
                    raise RuntimeError(
                        "Qwen3-VL lm_head returned an invalid logits tensor"
                    )
                del logits

        image_mask = selected_input["input_ids"] == config.image_token_id
        attention_mask = selected_input["attention_mask"] == 1

    return {
        "backbone_features": hidden_states,
        "backbone_attention_mask": attention_mask,
        "image_mask": image_mask,
    }
