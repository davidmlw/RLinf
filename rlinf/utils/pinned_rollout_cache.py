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

import time
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from rlinf.utils.backbone_cache import BackboneOutput


@dataclass(frozen=True)
class PinnedRolloutBackboneCacheStats:
    bytes: int
    samples: int
    stored_samples: int
    loads: int
    loaded_samples: int
    allocation_seconds: float
    d2h_seconds: float
    cpu_gather_seconds: float
    h2d_seconds: float
    staging_bytes: int


class PinnedRolloutBackboneCache:
    """Actor-owned pinned cache populated from short-lived CUDA IPC views."""

    def __init__(
        self,
        *,
        total_samples: int,
        feature_example: torch.Tensor | None = None,
        mask_example: torch.Tensor | None = None,
        tensor_examples: Mapping[str, torch.Tensor] | None = None,
        device: torch.device | str | int,
    ) -> None:
        if total_samples <= 0:
            raise ValueError("pinned rollout backbone cache requires positive samples")
        if tensor_examples is None:
            if feature_example is None or mask_example is None:
                raise ValueError("pinned rollout backbone examples are incomplete")
            examples = {
                "backbone_features": feature_example,
                "backbone_attention_mask": mask_example,
            }
        else:
            if feature_example is not None or mask_example is not None:
                raise ValueError("use either named or legacy backbone examples")
            examples = dict(tensor_examples)
        if not examples:
            raise ValueError("pinned rollout backbone examples must not be empty")
        batch_sizes = {int(tensor.shape[0]) for tensor in examples.values()}
        if len(batch_sizes) != 1:
            raise ValueError("pinned rollout backbone tensor batch mismatch")
        if any(tensor.device.type != "cuda" for tensor in examples.values()):
            raise ValueError(
                "pinned rollout backbone source tensors must be CUDA tensors"
            )

        self._device = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        if self._device.type != "cuda":
            raise ValueError(
                "pinned rollout backbone destination must be a CUDA device"
            )
        self._samples = int(total_samples)
        allocation_start = time.perf_counter()
        self._tensors: dict[str, torch.Tensor] | None = {
            name: torch.empty(
                (self._samples, *example.shape[1:]),
                dtype=example.dtype,
                device="cpu",
                pin_memory=True,
            )
            for name, example in examples.items()
        }
        self._allocation_seconds = time.perf_counter() - allocation_start
        self._bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in self._tensors.values()
        )
        self._copy_stream = torch.cuda.Stream(device=self._device)
        self._stored_samples = 0
        self._loads = 0
        self._loaded_samples = 0
        self._d2h_seconds = 0.0
        self._cpu_gather_seconds = 0.0
        self._h2d_seconds = 0.0
        self._staging: list[dict[str, object]] = [{}, {}]

    @property
    def samples(self) -> int:
        return self._samples

    def store_block(
        self,
        *,
        offset: int,
        feature: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        self.store_blocks(offset=offset, blocks=[(feature, mask)])

    def store_blocks(
        self,
        *,
        offset: int,
        blocks: list[tuple[torch.Tensor, torch.Tensor] | Mapping[str, torch.Tensor]],
    ) -> None:
        if self._tensors is None:
            raise RuntimeError("pinned rollout backbone cache was cleared")
        if offset != self._stored_samples:
            raise ValueError(
                "pinned rollout backbone blocks must arrive in order: "
                f"expected offset {self._stored_samples}, got {offset}"
            )
        if not blocks:
            raise ValueError("pinned rollout backbone batch must not be empty")

        validated_blocks: list[tuple[dict[str, torch.Tensor], int, int]] = []
        end = offset
        for block in blocks:
            if isinstance(block, Mapping):
                tensors = dict(block)
            else:
                feature, mask = block
                tensors = {
                    "backbone_features": feature,
                    "backbone_attention_mask": mask,
                }
            if set(tensors) != set(self._tensors):
                raise ValueError("pinned rollout backbone tensor fields changed")
            if any(tensor.device != self._device for tensor in tensors.values()):
                raise ValueError(
                    "pinned rollout backbone block is on the wrong CUDA device"
                )
            batch_sizes = {int(tensor.shape[0]) for tensor in tensors.values()}
            if len(batch_sizes) != 1:
                raise ValueError("pinned rollout backbone tensor block mismatch")
            block_size = batch_sizes.pop()
            block_end = end + block_size
            if block_end > self._samples:
                raise ValueError("pinned rollout backbone block exceeds cache capacity")
            for name, tensor in tensors.items():
                target = self._tensors[name]
                if tuple(tensor.shape[1:]) != tuple(target.shape[1:]):
                    raise ValueError(f"pinned rollout backbone {name} shape changed")
                if tensor.dtype != target.dtype:
                    raise ValueError(f"pinned rollout backbone {name} dtype changed")
            validated_blocks.append((tensors, end, block_end))
            end = block_end

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self._copy_stream):
            start_event.record(self._copy_stream)
            for tensors, block_offset, block_end in validated_blocks:
                for name, tensor in tensors.items():
                    self._tensors[name][block_offset:block_end].copy_(
                        tensor.detach(), non_blocking=True
                    )
            end_event.record(self._copy_stream)
        end_event.synchronize()
        self._d2h_seconds += start_event.elapsed_time(end_event) / 1000.0
        self._stored_samples = end

    def finalize(self) -> None:
        if self._stored_samples != self._samples:
            raise RuntimeError(
                "pinned rollout backbone stream ended early: "
                f"stored {self._stored_samples} of {self._samples} samples"
            )

    def _complete_staging_slot(self, slot: dict[str, object]) -> None:
        end_event = slot.pop("end_event", None)
        start_event = slot.pop("start_event", None)
        if end_event is not None:
            assert isinstance(end_event, torch.cuda.Event)
            assert isinstance(start_event, torch.cuda.Event)
            end_event.synchronize()
            self._h2d_seconds += start_event.elapsed_time(end_event) / 1000.0

    def _get_staging_slot(self, sample_count: int) -> dict[str, object]:
        if self._tensors is None:
            raise RuntimeError("pinned rollout backbone cache was cleared")
        slot = self._staging[self._loads % len(self._staging)]
        self._complete_staging_slot(slot)
        staging_tensors = slot.setdefault("tensors", {})
        assert isinstance(staging_tensors, dict)
        for name, source in self._tensors.items():
            shape = (sample_count, *source.shape[1:])
            staging = staging_tensors.get(name)
            if not isinstance(staging, torch.Tensor) or tuple(staging.shape) != shape:
                staging_tensors[name] = torch.empty(
                    shape,
                    dtype=source.dtype,
                    device="cpu",
                    pin_memory=True,
                )
        return slot

    def load(self, sample_ids: torch.Tensor) -> BackboneOutput:
        if self._tensors is None:
            raise RuntimeError("pinned rollout backbone cache was cleared")
        if self._stored_samples != self._samples:
            raise RuntimeError("pinned rollout backbone cache is incomplete")
        sample_ids = sample_ids.reshape(-1).to(dtype=torch.int64, device="cpu")
        if sample_ids.numel() == 0:
            raise ValueError("pinned rollout backbone sample IDs must be non-empty")
        min_id = int(sample_ids.min().item())
        max_id = int(sample_ids.max().item())
        if min_id < 0 or max_id >= self._samples:
            raise IndexError(
                f"pinned rollout backbone sample IDs [{min_id}, {max_id}] exceed "
                f"cache size {self._samples}"
            )

        slot = self._get_staging_slot(sample_ids.numel())
        staging_tensors = slot["tensors"]
        assert isinstance(staging_tensors, dict)
        gather_start = time.perf_counter()
        for name, source in self._tensors.items():
            torch.index_select(source, 0, sample_ids, out=staging_tensors[name])
        self._cpu_gather_seconds += time.perf_counter() - gather_start

        outputs = {
            name: torch.empty_like(staging, device=self._device)
            for name, staging in staging_tensors.items()
        }
        stream = torch.cuda.current_stream(self._device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        for name, staging in staging_tensors.items():
            outputs[name].copy_(staging, non_blocking=True)
        end_event.record(stream)
        slot["start_event"] = start_event
        slot["end_event"] = end_event
        self._loads += 1
        self._loaded_samples += sample_ids.numel()
        return outputs

    def stats(self) -> PinnedRolloutBackboneCacheStats:
        for slot in self._staging:
            self._complete_staging_slot(slot)
        staging_bytes = sum(
            value.numel() * value.element_size()
            for slot in self._staging
            for value in (
                list(slot.get("tensors", {}).values())
                if isinstance(slot.get("tensors"), dict)
                else []
            )
            if isinstance(value, torch.Tensor)
        )
        return PinnedRolloutBackboneCacheStats(
            bytes=self._bytes,
            samples=self._samples,
            stored_samples=self._stored_samples,
            loads=self._loads,
            loaded_samples=self._loaded_samples,
            allocation_seconds=self._allocation_seconds,
            d2h_seconds=self._d2h_seconds,
            cpu_gather_seconds=self._cpu_gather_seconds,
            h2d_seconds=self._h2d_seconds,
            staging_bytes=staging_bytes,
        )

    def clear(self) -> None:
        for slot in self._staging:
            self._complete_staging_slot(slot)
            slot.clear()
        self._tensors = None
