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

"""Common CUDA-resident timing boundary for the N1.7 true-B8 policy."""

from __future__ import annotations

import hashlib
from typing import Any


def make_explicit_noise_head(action_head: Any) -> Any:
    """Wrap the deployment action head with an explicit initial-noise input."""

    import torch

    class ExplicitNoiseActionHead(torch.nn.Module):
        def __init__(self, head: Any) -> None:
            super().__init__()
            self.head = head

        @torch.no_grad()
        def forward(
            self,
            backbone_features: Any,
            backbone_attention_mask: Any,
            image_mask: Any,
            state: Any,
            embodiment_id: Any,
            initial_actions: Any,
        ) -> Any:
            head = self.head
            processed = head.process_backbone_output(
                {
                    "backbone_features": backbone_features,
                    "backbone_attention_mask": backbone_attention_mask,
                    "image_mask": image_mask,
                }
            )
            state_features = head.state_encoder(
                state.view(state.shape[0], 1, -1), embodiment_id
            )
            actions = initial_actions
            step_size = 1.0 / head.num_inference_timesteps

            for inference_step in range(head.num_inference_timesteps):
                timestep = int(
                    inference_step
                    / head.num_inference_timesteps
                    * head.num_timestep_buckets
                )
                timesteps = torch.full(
                    (actions.shape[0],), timestep, device=actions.device
                )
                action_features = head.action_encoder(actions, timesteps, embodiment_id)
                if head.config.add_pos_embed:
                    positions = torch.arange(
                        action_features.shape[1],
                        dtype=torch.long,
                        device=actions.device,
                    )
                    action_features = action_features + head.position_embedding(
                        positions
                    ).unsqueeze(0)
                model_arguments = {
                    "hidden_states": torch.cat(
                        (state_features, action_features), dim=1
                    ),
                    "encoder_hidden_states": processed["backbone_features"],
                    "timestep": timesteps,
                }
                if head.config.use_alternate_vl_dit:
                    model_arguments.update(
                        image_mask=processed["image_mask"],
                        backbone_attention_mask=processed["backbone_attention_mask"],
                    )
                model_output = head.model(**model_arguments)
                prediction = head.action_decoder(model_output, embodiment_id)
                actions = actions + step_size * prediction[:, -head.action_horizon :]
            return actions

    return ExplicitNoiseActionHead(action_head).eval()


def prepare_cuda_inputs(model: Any, collated: dict[str, Any]) -> tuple[Any, Any]:
    """Move model inputs to CUDA once and fail closed on layout or device drift."""

    import torch

    backbone_inputs, action_inputs = model.prepare_input(collated)
    for group_name, group in (
        ("backbone", backbone_inputs),
        ("action", action_inputs),
    ):
        for name, value in tuple(group.items()):
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"common-boundary {group_name}.{name} is not a tensor"
                )
            value = value.contiguous()
            if value.device.type != "cuda" or not value.is_contiguous():
                raise RuntimeError(
                    f"common-boundary {group_name}.{name} is not contiguous CUDA"
                )
            group[name] = value
    return backbone_inputs, action_inputs


def tensor_manifest(value: Any) -> dict[str, Any]:
    """Describe and hash one CUDA tensor without depending on NumPy BF16 support."""

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor, got {type(value)!r}")
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": value.device.type,
        "contiguous": value.is_contiguous(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def prepared_manifest(backbone_inputs: Any, action_inputs: Any) -> dict[str, Any]:
    """Create a stable manifest for both halves of the prepared model input."""

    return {
        "backbone": {
            name: tensor_manifest(value)
            for name, value in sorted(backbone_inputs.items())
        },
        "action": {
            name: tensor_manifest(value)
            for name, value in sorted(action_inputs.items())
        },
    }


def call_with_explicit_noise(
    model: Any,
    explicit_head: Any,
    prepared: tuple[Any, Any],
    initial_actions: Any,
) -> Any:
    """Run the CUDA-resident backbone and explicit-noise deployment head."""

    import torch

    backbone_inputs, action_inputs = prepared
    with torch.inference_mode():
        backbone_output = model.backbone(backbone_inputs)
        return explicit_head(
            backbone_output["backbone_features"],
            backbone_output["backbone_attention_mask"],
            backbone_output["image_mask"],
            action_inputs["state"],
            action_inputs["embodiment_id"],
            initial_actions,
        )


def cuda_event_call(
    model: Any,
    explicit_head: Any,
    prepared: tuple[Any, Any],
    initial_actions: Any,
) -> tuple[Any, dict[str, float]]:
    """Measure one natural call with CUDA events and no intermediate host wait."""

    import torch

    start = torch.cuda.Event(enable_timing=True)
    backbone_done = torch.cuda.Event(enable_timing=True)
    action_done = torch.cuda.Event(enable_timing=True)
    backbone_inputs, action_inputs = prepared

    with torch.inference_mode():
        start.record()
        backbone_output = model.backbone(backbone_inputs)
        backbone_done.record()
        actions = explicit_head(
            backbone_output["backbone_features"],
            backbone_output["backbone_attention_mask"],
            backbone_output["image_mask"],
            action_inputs["state"],
            action_inputs["embodiment_id"],
            initial_actions,
        )
        action_done.record()
    action_done.synchronize()
    return actions, {
        "backbone_ms": start.elapsed_time(backbone_done),
        "action_head_ms": backbone_done.elapsed_time(action_done),
        "total_ms": start.elapsed_time(action_done),
    }
