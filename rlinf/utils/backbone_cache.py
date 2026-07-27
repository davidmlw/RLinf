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

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

BACKBONE_CACHE_OUTPUT_KEY = "_rlinf_backbone_cache"
BACKBONE_CACHE_SAMPLE_IDS_KEY = "_rlinf_backbone_cache_sample_ids"

# W62: keys carrying the Rollout-computed frozen-backbone feature through the
# trajectory (piggybacked on forward_inputs so it rides the existing per-sample
# merge/split/shuffle machinery). Actor reuses this as the detached backbone
# conditioning instead of recomputing self.backbone(...).
ROLLOUT_BACKBONE_FEATURE_KEY = "rollout_backbone_features"
ROLLOUT_BACKBONE_MASK_KEY = "rollout_backbone_attention_mask"
ROLLOUT_BACKBONE_SAMPLE_IDS_KEY = "_rlinf_rollout_backbone_sample_ids"
ROLLOUT_BACKBONE_TRANSPORT_KEY = "rollout_backbone_feature_transport"
ROLLOUT_BACKBONE_BORROWED_IPC = "borrowed_ipc"
ROLLOUT_BACKBONE_INPUT_KEYS = (
    "eagle_input_ids",
    "eagle_attention_mask",
    "eagle_pixel_values",
    "eagle_image_sizes",
)

BackboneCacheKey = tuple[int, ...]
BackboneOutput = dict[str, torch.Tensor]


def filter_rollout_backbone_transport(
    forward_inputs: dict[str, torch.Tensor],
    *,
    reuse_enabled: bool,
) -> bool:
    """Keep either raw Eagle inputs or a complete reusable backbone output."""
    if not reuse_enabled:
        forward_inputs.pop(ROLLOUT_BACKBONE_FEATURE_KEY, None)
        forward_inputs.pop(ROLLOUT_BACKBONE_MASK_KEY, None)
        return False

    has_complete_feature = all(
        key in forward_inputs
        for key in (ROLLOUT_BACKBONE_FEATURE_KEY, ROLLOUT_BACKBONE_MASK_KEY)
    )
    if not has_complete_feature:
        forward_inputs.pop(ROLLOUT_BACKBONE_FEATURE_KEY, None)
        forward_inputs.pop(ROLLOUT_BACKBONE_MASK_KEY, None)
        return False

    for key in ROLLOUT_BACKBONE_INPUT_KEYS:
        forward_inputs.pop(key, None)
    return True


def make_backbone_cache_key(sample_ids: torch.Tensor) -> BackboneCacheKey:
    if sample_ids.device.type != "cpu":
        raise ValueError("backbone cache sample IDs must stay on CPU")
    if sample_ids.ndim != 1:
        raise ValueError(
            f"backbone cache sample IDs must be 1D, got shape={tuple(sample_ids.shape)}"
        )
    if sample_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"backbone cache sample IDs must be integers, got dtype={sample_ids.dtype}"
        )
    key = tuple(int(value) for value in sample_ids.tolist())
    if not key or len(set(key)) != len(key):
        raise ValueError("backbone cache sample IDs must be non-empty and unique")
    return key


@dataclass(frozen=True)
class BorrowedBackboneCacheStats:
    bytes: int
    samples: int
    loads: int
    view_samples: int
    gathered_samples: int


class BorrowedBackboneCache:
    """Actor-side views of Rollout-owned CUDA IPC feature blocks."""

    def __init__(
        self,
        tensors: list[torch.Tensor],
        block_sizes: list[int],
        lease_id: str,
    ) -> None:
        if not lease_id:
            raise ValueError("borrowed backbone lease ID must be non-empty")
        if not block_sizes or len(tensors) != 2 * len(block_sizes):
            raise ValueError(
                "borrowed backbone cache requires one feature/mask pair per block"
            )

        self.lease_id = lease_id
        self._features = tensors[0::2]
        self._masks = tensors[1::2]
        self._block_sizes = [int(size) for size in block_sizes]
        self._starts: list[int] = []
        total = 0
        byte_count = 0
        for block_idx, (feature, mask, size) in enumerate(
            zip(self._features, self._masks, self._block_sizes)
        ):
            if size <= 0:
                raise ValueError(f"borrowed backbone block {block_idx} is empty")
            if feature.device.type != "cuda" or mask.device.type != "cuda":
                raise ValueError("borrowed backbone tensors must stay on CUDA")
            if int(feature.shape[0]) != size or int(mask.shape[0]) != size:
                raise ValueError(
                    f"borrowed backbone block {block_idx} sample count mismatch"
                )
            if feature.device != mask.device:
                raise ValueError(
                    f"borrowed backbone block {block_idx} spans multiple devices"
                )
            self._starts.append(total)
            total += size
            byte_count += sum(
                tensor.numel() * tensor.element_size() for tensor in (feature, mask)
            )

        self._samples = total
        self._bytes = byte_count
        self._loads = 0
        self._view_samples = 0
        self._gathered_samples = 0

    @property
    def samples(self) -> int:
        return self._samples

    def _validate_sample_ids(self, sample_ids: torch.Tensor) -> torch.Tensor:
        if sample_ids.device.type != "cpu":
            sample_ids = sample_ids.cpu()
        sample_ids = sample_ids.to(dtype=torch.int64).reshape(-1)
        if sample_ids.numel() == 0:
            raise ValueError("borrowed backbone sample IDs must be non-empty")
        min_id = int(sample_ids.min().item())
        max_id = int(sample_ids.max().item())
        if min_id < 0 or max_id >= self._samples:
            raise IndexError(
                f"borrowed backbone sample IDs [{min_id}, {max_id}] exceed "
                f"lease size {self._samples}"
            )
        return sample_ids

    def load(self, sample_ids: torch.Tensor) -> BackboneOutput:
        sample_ids = self._validate_sample_ids(sample_ids)
        self._loads += 1

        first = int(sample_ids[0].item())
        block_idx = max(
            idx for idx, start in enumerate(self._starts) if start <= first
        )
        local_first = first - self._starts[block_idx]
        expected = torch.arange(first, first + sample_ids.numel(), dtype=torch.int64)
        if (
            torch.equal(sample_ids, expected)
            and local_first + sample_ids.numel() <= self._block_sizes[block_idx]
        ):
            self._view_samples += sample_ids.numel()
            selection = slice(local_first, local_first + sample_ids.numel())
            return {
                "backbone_features": self._features[block_idx][selection],
                "backbone_attention_mask": self._masks[block_idx][selection],
            }

        boundaries = torch.tensor(
            self._starts[1:], dtype=torch.int64, device="cpu"
        )
        block_ids = torch.bucketize(sample_ids, boundaries, right=True)
        device = self._features[0].device
        feature_out = torch.empty(
            (sample_ids.numel(), *self._features[0].shape[1:]),
            dtype=self._features[0].dtype,
            device=device,
        )
        mask_out = torch.empty(
            (sample_ids.numel(), *self._masks[0].shape[1:]),
            dtype=self._masks[0].dtype,
            device=device,
        )
        for idx in block_ids.unique(sorted=True).tolist():
            positions = torch.nonzero(block_ids == idx, as_tuple=False).reshape(-1)
            local_ids = sample_ids[positions] - self._starts[idx]
            positions_device = positions.to(device=device, non_blocking=True)
            local_ids_device = local_ids.to(device=device, non_blocking=True)
            feature_out.index_copy_(
                0,
                positions_device,
                self._features[idx].index_select(0, local_ids_device),
            )
            mask_out.index_copy_(
                0,
                positions_device,
                self._masks[idx].index_select(0, local_ids_device),
            )

        self._gathered_samples += sample_ids.numel()
        return {
            "backbone_features": feature_out,
            "backbone_attention_mask": mask_out,
        }

    def stats(self) -> BorrowedBackboneCacheStats:
        return BorrowedBackboneCacheStats(
            bytes=self._bytes,
            samples=self._samples,
            loads=self._loads,
            view_samples=self._view_samples,
            gathered_samples=self._gathered_samples,
        )

    def clear(self) -> None:
        self._features.clear()
        self._masks.clear()
        self._block_sizes.clear()
        self._starts.clear()


def validate_frozen_backbone(backbone: nn.Module) -> None:
    parameters = list(backbone.parameters())
    if not parameters:
        raise ValueError("backbone cache requires a backbone with parameters")
    trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    if trainable:
        raise ValueError(
            "backbone cache requires every backbone parameter to be frozen; "
            f"found {trainable} trainable parameters"
        )
    if backbone.training:
        raise ValueError("backbone cache requires the backbone to be in eval mode")


@dataclass(frozen=True)
class BackboneCacheStats:
    entries: int
    bytes: int
    hits: int
    misses: int


class PinnedBackboneCache:
    def __init__(
        self,
        device: torch.device | str | int,
        *,
        pin_memory: bool = True,
    ) -> None:
        self._device = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        self._pin_memory = pin_memory
        self._entries: dict[BackboneCacheKey, BackboneOutput] = {}
        self._bytes = 0
        self._hits = 0
        self._misses = 0

    def __contains__(self, key: BackboneCacheKey) -> bool:
        return key in self._entries

    def store(
        self,
        key: BackboneCacheKey,
        output: Mapping[str, torch.Tensor],
    ) -> None:
        if key in self._entries:
            raise KeyError(f"duplicate backbone cache key: {key}")
        if not output:
            raise ValueError("cannot cache an empty backbone output")

        host_output: BackboneOutput = {}
        for name, value in output.items():
            if not isinstance(name, str) or not isinstance(value, torch.Tensor):
                raise TypeError(
                    "backbone cache only supports string-to-tensor mappings"
                )
            host_value = torch.empty_like(
                value,
                device="cpu",
                pin_memory=self._pin_memory,
            )
            host_value.copy_(value.detach(), non_blocking=self._pin_memory)
            host_output[name] = host_value
            self._bytes += host_value.numel() * host_value.element_size()

        self._entries[key] = host_output
        self._misses += 1

    def load(self, key: BackboneCacheKey) -> BackboneOutput:
        try:
            host_output = self._entries[key]
        except KeyError as error:
            raise KeyError(
                "missing frozen-backbone cache entry; sample mapping changed "
                f"within one PPO update: {key}"
            ) from error

        self._hits += 1
        return {
            name: value.to(
                device=self._device,
                non_blocking=self._pin_memory,
            )
            for name, value in host_output.items()
        }

    def synchronize(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.current_stream(self._device).synchronize()

    def stats(self) -> BackboneCacheStats:
        return BackboneCacheStats(
            entries=len(self._entries),
            bytes=self._bytes,
            hits=self._hits,
            misses=self._misses,
        )

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0
