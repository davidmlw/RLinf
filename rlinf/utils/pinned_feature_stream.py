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

from typing import Any

from rlinf.utils.backbone_cache import ROLLOUT_BACKBONE_SAMPLE_ID_STRIDE
from rlinf.utils.pinned_rollout_cache import PinnedRolloutBackboneCache


async def receive_pinned_rollout_backbone_stream(
    worker: Any,
    *,
    source_rank: int,
    consumer_rank: int,
    model_version: int,
    expected_blocks: int,
    expected_samples: int,
    rollout_group_name: str,
) -> tuple[PinnedRolloutBackboneCache, dict[str, Any], dict[str, float]]:
    if expected_blocks <= 0 or expected_samples <= 0:
        raise ValueError("pinned backbone stream requires positive expected counts")

    cache: PinnedRolloutBackboneCache | None = None
    stream_metadata: dict[str, Any] | None = None
    received_bytes = 0
    for expected_block_index in range(expected_blocks):
        received = await worker.recv(
            src_group_name=rollout_group_name,
            src_rank=source_rank,
            async_op=True,
            borrowed_ipc=True,
        ).async_wait()
        if not isinstance(received, tuple) or len(received) != 2:
            raise RuntimeError("pinned backbone IPC requires tensors and metadata")
        tensors, metadata = received
        if not isinstance(metadata, dict) or metadata.get("schema") != 2:
            raise RuntimeError(f"unsupported pinned backbone metadata: {metadata}")
        if not isinstance(tensors, list) or len(tensors) != 2:
            raise RuntimeError("pinned backbone block requires one feature/mask pair")
        if int(metadata.get("block_index", -1)) != expected_block_index:
            raise RuntimeError("pinned backbone blocks arrived out of order")
        if int(metadata.get("total_blocks", -1)) != expected_blocks:
            raise RuntimeError("pinned backbone total block count changed")
        if int(metadata.get("total_samples", -1)) != expected_samples:
            raise RuntimeError("pinned backbone total sample count changed")
        if int(metadata.get("producer_rank", -1)) != source_rank:
            raise RuntimeError("pinned backbone producer rank does not match route")
        if int(metadata.get("consumer_rank", -1)) != consumer_rank:
            raise RuntimeError("pinned backbone consumer rank does not match Actor")
        if int(metadata.get("model_version", -1)) != model_version:
            raise RuntimeError("pinned backbone model version does not match Actor")
        expected_base = source_rank * ROLLOUT_BACKBONE_SAMPLE_ID_STRIDE
        if int(metadata.get("sample_id_base", -1)) != expected_base:
            raise RuntimeError("pinned backbone sample-ID namespace is invalid")

        feature, mask = tensors
        block_size = int(metadata.get("block_size", -1))
        offset = int(metadata.get("offset", -1))
        if block_size <= 0 or feature.shape[0] != block_size:
            raise RuntimeError("pinned backbone feature block size mismatch")
        if mask.shape[0] != block_size:
            raise RuntimeError("pinned backbone mask block size mismatch")
        block_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in (feature, mask)
        )
        if block_bytes != int(metadata.get("bytes", -1)):
            raise RuntimeError("pinned backbone block byte count mismatch")

        if cache is None:
            cache = PinnedRolloutBackboneCache(
                total_samples=expected_samples,
                feature_example=feature,
                mask_example=mask,
                device=worker.torch_platform.current_device(),
            )
            stream_metadata = dict(metadata)
        elif stream_metadata is None or metadata.get("lease_id") != stream_metadata.get(
            "lease_id"
        ):
            raise RuntimeError("pinned backbone lease changed within one stream")

        cache.store_block(offset=offset, feature=feature, mask=mask)
        received_bytes += block_bytes
        lease_id = str(metadata.get("lease_id", ""))
        tensors.clear()
        del feature, mask, received
        worker.torch_platform.ipc_collect()
        worker.send(
            {
                "schema": 2,
                "lease_id": lease_id,
                "block_index": expected_block_index,
            },
            dst_group_name=rollout_group_name,
            dst_rank=source_rank,
        )

        completed = expected_block_index + 1
        log_stride = max(1, expected_blocks // 4)
        if expected_block_index == 0 or completed % log_stride == 0:
            worker.log_info(
                "W63_PINNED_IPC_BLOCK_STORED "
                f"rank={consumer_rank} lease={lease_id} "
                f"blocks={completed}/{expected_blocks} bytes={received_bytes}"
            )

    if cache is None or stream_metadata is None:
        raise RuntimeError("pinned backbone stream produced no cache")
    cache.finalize()
    stats = cache.stats()
    worker.log_info(
        "W63_PINNED_IPC_CACHE_READY "
        f"rank={consumer_rank} lease={stream_metadata['lease_id']} "
        f"samples={stats.samples} bytes={stats.bytes} "
        f"alloc_s={stats.allocation_seconds:.6f} d2h_s={stats.d2h_seconds:.6f}"
    )
    return (
        cache,
        stream_metadata,
        {
            "actor/pinned_feature_bytes": float(stats.bytes),
            "actor/pinned_feature_samples": float(stats.samples),
            "actor/pinned_feature_allocation_seconds": stats.allocation_seconds,
            "actor/pinned_feature_d2h_seconds": stats.d2h_seconds,
        },
    )
