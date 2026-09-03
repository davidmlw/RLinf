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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Awaitable, Callable, Optional, TypeVar

import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.tensor import DTensor

from rlinf.scheduler import CollectiveGroupOptions, Worker

SendFn = Callable[[Any], Awaitable[None]]
RecvFn = Callable[[], Awaitable[Any]]
TensorValue = TypeVar("TensorValue")


class WeightSyncer(ABC):
    def __init__(self):
        self._sender_initialized: bool = False
        self._receiver_initialized: bool = False
        self._comm_options: Optional[CollectiveGroupOptions] = None
        self._state_dict_prefixes: tuple[str, ...] | None = None

    @property
    def comm_options(self) -> Optional[CollectiveGroupOptions]:
        """``CollectiveGroupOptions`` to pass to broadcast/send/recv calls
        performed during weight sync. Populated by :meth:`create` from the
        ``use_ring_sync``, ``nccl_max_ctas`` and ``nccl_min_ctas`` keys on
        the weight syncer config; ``None`` if every option is at its default
        (matching the legacy behavior where no options were supplied)."""
        return self._comm_options

    @property
    def state_dict_prefixes(self) -> tuple[str, ...] | None:
        """Prefixes defining the model state visible to this syncer."""

        return self._state_dict_prefixes

    @staticmethod
    def _matches_prefix(name: str, prefix: str) -> bool:
        return name == prefix or name.startswith(f"{prefix}.")

    def select_state_dict(
        self, state_dict: Mapping[str, TensorValue]
    ) -> dict[str, TensorValue]:
        """Return the configured synchronization view of ``state_dict``.

        A restricted view lets inference replace immutable model components
        without requiring their state keys to remain present solely for the
        synchronization handshake. Prefixes are fail-closed: every configured
        prefix must match at least one key.
        """

        if self._state_dict_prefixes is None:
            return dict(state_dict)

        matched = dict.fromkeys(self._state_dict_prefixes, False)
        selected: dict[str, TensorValue] = {}
        for name, value in state_dict.items():
            for prefix in self._state_dict_prefixes:
                if self._matches_prefix(name, prefix):
                    matched[prefix] = True
                    selected[name] = value
                    break
        missing = [prefix for prefix, found in matched.items() if not found]
        if missing:
            raise ValueError(
                "Weight sync state_dict prefixes did not match any keys: "
                f"{missing}"
            )
        return selected

    def select_param_names(
        self,
        names: list[str],
        *,
        required_names: list[str] | None = None,
    ) -> list[str]:
        """Filter synchronized names and reject excluded trainable parameters."""

        if self._state_dict_prefixes is None:
            return list(names)
        selected = [
            name
            for name in names
            if any(
                self._matches_prefix(name, prefix)
                for prefix in self._state_dict_prefixes
            )
        ]
        if not selected:
            raise ValueError("Weight sync state_dict view selected no parameters")
        required = set(required_names or ())
        missing_required = sorted(required - set(selected))
        if missing_required:
            raise ValueError(
                "Weight sync state_dict view excludes trainable parameters: "
                f"{missing_required[:8]}"
            )
        return selected

    @abstractmethod
    async def sync(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        send: SendFn,
        version: int | torch.Tensor,
    ) -> None: ...

    @abstractmethod
    async def apply(self, model: torch.nn.Module, recv: RecvFn) -> int: ...

    async def init_sender(
        self,
        state_dict: dict[str, torch.Tensor | DTensor],
        param_names_need_sync: list[str],
        send: SendFn,
        recv: RecvFn | None = None,
        is_sender: bool = True,
    ) -> None:
        del state_dict, send, recv, param_names_need_sync, is_sender
        self._sender_initialized = True

    async def init_receiver(
        self,
        state_dict: dict[str, torch.Tensor | DTensor] | None,
        recv: RecvFn,
        send: SendFn | None = None,
    ) -> None:
        del state_dict, recv, send
        self._receiver_initialized = True

    @classmethod
    def create(cls, config: DictConfig) -> "WeightSyncer":
        assert config is not None, "Weight syncer config must be provided"
        syncer_type = OmegaConf.select(config, "type")
        if syncer_type == "bucket":
            from .bucket_syncer import BucketWeightSyncer

            bucket_config = OmegaConf.select(config, "bucket")
            assert bucket_config is not None, (
                "Bucket config must be provided for bucket weight syncer"
            )
            syncer: "WeightSyncer" = BucketWeightSyncer(
                bucket_size=OmegaConf.select(bucket_config, "bucket_size"),
                bucket_dtype=OmegaConf.select(bucket_config, "bucket_dtype"),
                bucket_device=OmegaConf.select(
                    bucket_config, "bucket_device", default=Worker.torch_device_type
                ),
                is_agent=OmegaConf.select(bucket_config, "is_agent", default=False),
                load_instant=OmegaConf.select(
                    bucket_config, "load_instant", default=True
                ),
            )
        elif syncer_type == "patch":
            from .patch_syncer import PatchWeightSyncer

            patch_config = OmegaConf.select(config, "patch")
            assert patch_config is not None, (
                "Patch config must be provided for patch weight syncer"
            )
            syncer = PatchWeightSyncer(
                snapshot_device=OmegaConf.select(
                    patch_config, "snapshot_device", default="cpu"
                ),
                delta_encoding=OmegaConf.select(
                    patch_config, "delta_encoding", default=True
                ),
                compression_algorithm=OmegaConf.select(
                    patch_config,
                    "compression_algorithm",
                    default=OmegaConf.select(
                        patch_config, "compression", default="none"
                    ),
                ),
                transport_device=OmegaConf.select(
                    patch_config, "transport_device", default=Worker.torch_device_type
                ),
                init_sync_enabled=OmegaConf.select(
                    patch_config, "init_sync.enabled", default=False
                ),
                init_sync_prefixes=OmegaConf.select(
                    patch_config, "init_sync.prefixes", default=None
                ),
                init_sync_bucket_size=OmegaConf.select(
                    patch_config,
                    "init_sync.bucket_size",
                    default=OmegaConf.select(
                        patch_config,
                        "init_sync.buckets_size",
                        default=128 * 1024 * 1024,
                    ),
                ),
            )
        else:
            raise ValueError(f"Unsupported weight syncer type: {syncer_type}")

        prefixes = OmegaConf.select(config, "state_dict_prefixes", default=None)
        if prefixes is not None:
            prefixes = tuple(str(prefix) for prefix in prefixes)
            if not prefixes or any(not prefix for prefix in prefixes):
                raise ValueError(
                    "weight_syncer.state_dict_prefixes must contain non-empty prefixes"
                )
            if len(set(prefixes)) != len(prefixes):
                raise ValueError(
                    "weight_syncer.state_dict_prefixes must not contain duplicates"
                )
            syncer._state_dict_prefixes = prefixes

        syncer._comm_options = cls._build_comm_options(config)
        return syncer

    @staticmethod
    def _build_comm_options(
        config: DictConfig,
    ) -> Optional[CollectiveGroupOptions]:
        """Build ``CollectiveGroupOptions`` from the weight syncer config.

        Reads three top-level keys (all optional, all default to the equivalent
        of the underlying ``CollectiveGroupOptions`` default):

        - ``use_ring_sync`` (bool): route the broadcast through the ring
          algorithm (one cross-group hop + parallel fan-out from the first
          receiver) by setting ``CollectiveGroupOptions.use_ring_broadcast``.
        - ``nccl_max_ctas`` / ``nccl_min_ctas`` (int): forwarded to
          ``CollectiveGroupOptions.accel_max_ctas`` / ``accel_min_ctas`` to
          cap how much GPU SM resource NCCL consumes during weight sync.
        """
        use_ring = OmegaConf.select(config, "use_ring_sync", default=False)
        max_ctas = OmegaConf.select(config, "nccl_max_ctas", default=None)
        min_ctas = OmegaConf.select(config, "nccl_min_ctas", default=None)
        if not use_ring and max_ctas is None and min_ctas is None:
            return None
        return CollectiveGroupOptions(
            use_ring_broadcast=bool(use_ring),
            accel_max_ctas=max_ctas,
            accel_min_ctas=min_ctas,
        )

    def sender_initialized(self) -> bool:
        return self._sender_initialized

    def receiver_initialized(self) -> bool:
        return self._receiver_initialized
