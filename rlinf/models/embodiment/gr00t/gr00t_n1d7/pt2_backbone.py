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

"""Static true-B8 PT2 executor for the frozen GR00T N1.7 backbone."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

_VISION_ATTENTION_CLASS = "Qwen3VLVisionAttention"


def _uniform_vision_sequence_length(
    grid_rows: Sequence[Sequence[int]],
) -> tuple[int, int]:
    """Validate Qwen vision geometry and return per-segment token count."""

    lengths = []
    for row in grid_rows:
        if len(row) != 3:
            raise ValueError(f"image_grid_thw row must have three values: {row!r}")
        temporal, height, width = (int(value) for value in row)
        if temporal < 1 or height < 1 or width < 1:
            raise ValueError(f"image_grid_thw values must be positive: {row!r}")
        lengths.extend([height * width] * temporal)
    if not lengths:
        raise ValueError("image_grid_thw must contain at least one vision segment")
    unique_lengths = set(lengths)
    if len(unique_lengths) != 1:
        raise ValueError(
            "PT2 static vision adapter requires one sequence length: "
            f"{sorted(unique_lengths)}"
        )
    return lengths[0], len(lengths)


def _install_static_vision_flash_attention(
    backbone: torch.nn.Module,
    *,
    sequence_length: int,
) -> Callable[[], None]:
    """Install the fixture-qualified static max length used by compiled ViT."""

    vision_modules = sum(
        type(module).__name__ == _VISION_ATTENTION_CLASS
        for module in backbone.modules()
    )
    if vision_modules < 1:
        raise RuntimeError("PT2 Backbone found no Qwen vision attention modules")

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    local_mapping = ALL_ATTENTION_FUNCTIONS._local_mapping
    had_local_override = "flash_attention_2" in local_mapping
    previous_local = local_mapping.get("flash_attention_2")
    original = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

    def static_vision_flash_attention(
        module: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if type(module).__name__ == _VISION_ATTENTION_CLASS:
            kwargs["max_length_q"] = sequence_length
            kwargs["max_length_k"] = sequence_length
        return original(module, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = static_vision_flash_attention

    def restore() -> None:
        if had_local_override:
            ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = previous_local
        else:
            del ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

    return restore


class PT2FrozenBackbone:
    """Compile a frozen ViT+LLM for one fail-closed static rollout shape."""

    reuses_output_buffers = False

    def __init__(self, backbone: torch.nn.Module, config: Mapping[str, Any]):
        self.backbone = backbone
        self.config = dict(config)
        self.mode = str(config.get("mode", "max-autotune-no-cudagraphs"))
        self.expected_batch_size = int(config.get("static_batch_size", 8))
        self.expected_sequence_length = int(config.get("sequence_length", 208))
        self.expected_vision_segments = int(config.get("vision_segments", 24))
        self.expected_vision_sequence_length = int(
            config.get("vision_sequence_length", 256)
        )
        if self.expected_batch_size < 1:
            raise ValueError("PT2 Backbone static_batch_size must be positive")
        if self.expected_sequence_length < 1:
            raise ValueError("PT2 Backbone sequence_length must be positive")

        self._restore_attention = _install_static_vision_flash_attention(
            backbone,
            sequence_length=self.expected_vision_sequence_length,
        )
        self._eager_forward = backbone.forward
        try:
            self._compiled_forward = torch.compile(
                self._eager_forward,
                mode=self.mode,
                dynamic=False,
            )
        except Exception:
            self._restore_attention()
            raise
        self.call_count = 0
        self.first_call_wall_ms: float | None = None
        self.closed = False

    def _validate_inputs(self, inputs: Mapping[str, Any]) -> None:
        input_ids = inputs.get("input_ids")
        image_grid_thw = inputs.get("image_grid_thw")
        if not torch.is_tensor(input_ids) or tuple(input_ids.shape) != (
            self.expected_batch_size,
            self.expected_sequence_length,
        ):
            actual = None if input_ids is None else tuple(input_ids.shape)
            raise RuntimeError(
                "PT2 Backbone text geometry changed: "
                f"{actual} != "
                f"({self.expected_batch_size}, {self.expected_sequence_length})"
            )
        if not torch.is_tensor(image_grid_thw):
            raise RuntimeError("PT2 Backbone requires tensor image_grid_thw")
        grid_rows = image_grid_thw.detach().cpu().tolist()
        sequence_length, segment_count = _uniform_vision_sequence_length(grid_rows)
        if (
            sequence_length != self.expected_vision_sequence_length
            or segment_count != self.expected_vision_segments
        ):
            raise RuntimeError(
                "PT2 Backbone visual geometry changed: "
                f"segments={segment_count}, sequence_length={sequence_length}"
            )

    def __call__(self, inputs: Mapping[str, Any]) -> Any:
        if self.closed:
            raise RuntimeError("PT2 Backbone is closed")
        self._validate_inputs(inputs)
        started = time.perf_counter() if self.call_count == 0 else None
        output = self._compiled_forward(inputs)
        if started is not None:
            torch.cuda.synchronize()
            self.first_call_wall_ms = (time.perf_counter() - started) * 1000.0
        self.call_count += 1
        return output

    def telemetry(self) -> dict[str, Any]:
        counters = torch._dynamo.utils.counters
        return {
            "enabled": True,
            "mode": self.mode,
            "static_batch_size": self.expected_batch_size,
            "sequence_length": self.expected_sequence_length,
            "vision_segments": self.expected_vision_segments,
            "vision_sequence_length": self.expected_vision_sequence_length,
            "call_count": self.call_count,
            "first_call_wall_ms": self.first_call_wall_ms,
            "unique_graphs": int(counters["stats"]["unique_graphs"]),
            "graph_breaks": int(sum(counters["graph_break"].values())),
            "closed": self.closed,
        }

    def close(self) -> None:
        if self.closed:
            return
        self._restore_attention()
        self.closed = True
