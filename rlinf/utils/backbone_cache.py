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

BackboneCacheKey = tuple[int, ...]
BackboneOutput = dict[str, torch.Tensor]


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
