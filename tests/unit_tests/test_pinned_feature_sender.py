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

import asyncio

import pytest
import torch

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class _ImmediateWork:
    def __init__(self, value) -> None:
        self.value = value

    async def async_wait(self):
        return self.value


class _NeverWork:
    async def async_wait(self):
        await asyncio.Event().wait()


class _FakePlatform:
    def __init__(self) -> None:
        self.collects = 0

    def ipc_collect(self) -> None:
        self.collects += 1


class _FakeSender:
    _abort_pinned_feature_stream = (
        MultiStepRolloutWorker._abort_pinned_feature_stream
    )

    def __init__(self, ack) -> None:
        self._rank = 2
        self.version = 7
        self.actor_group_name = "actor"
        self.torch_platform = _FakePlatform()
        self.sent = []
        self.logs = []
        self.ack = ack
        self._pinned_feature_ipc_timeout_seconds = 0.05
        self._borrowed_feature_tensors = [
            torch.ones(2, 3),
            torch.ones(2, 3, dtype=torch.bool),
        ]
        self._borrowed_feature_block_sizes = [2]
        self._borrowed_feature_samples = 2
        self._borrowed_feature_active_lease = "lease-1"
        self._borrowed_feature_consumer_rank = 2
        self._borrowed_stream_expected_blocks = 1
        self._borrowed_stream_expected_samples = 2
        self._borrowed_stream_expected_batches = 1
        self._borrowed_stream_batches = 0
        self._borrowed_stream_blocks = 0
        self._borrowed_stream_bytes = 0
        self._borrowed_stream_wait_seconds = 0.0

    def send(self, tensors, **kwargs) -> None:
        self.sent.append((tensors, kwargs))

    def recv(self, **kwargs):
        assert kwargs["async_op"] is True
        if self.ack is None:
            return _NeverWork()
        return _ImmediateWork(self.ack)

    def log_info(self, message) -> None:
        self.logs.append(message)


def _flush(sender):
    return asyncio.run(
        MultiStepRolloutWorker._flush_borrowed_feature_batch(sender)
    )


def _ok_ack():
    return {
        "schema": 3,
        "status": "ok",
        "lease_id": "lease-1",
        "batch_index": 0,
        "completed_blocks": 1,
    }


def test_sender_releases_batch_only_after_valid_ack():
    sender = _FakeSender(_ok_ack())

    _flush(sender)

    assert sender._borrowed_stream_batches == 1
    assert sender._borrowed_stream_blocks == 1
    assert sender._borrowed_feature_tensors == []
    assert sender._borrowed_feature_block_sizes == []
    assert sender._borrowed_feature_active_lease == "lease-1"
    assert sender.torch_platform.collects == 1


def test_sender_aborts_lease_when_actor_nacks_batch():
    ack = _ok_ack()
    ack.update(status="error", error="RuntimeError: wrong model version")
    sender = _FakeSender(ack)

    with pytest.raises(RuntimeError, match="wrong model version"):
        _flush(sender)

    assert sender._borrowed_feature_tensors == []
    assert sender._borrowed_feature_block_sizes == []
    assert sender._borrowed_feature_active_lease is None
    assert sender._borrowed_feature_consumer_rank is None
    assert sender.torch_platform.collects == 1


def test_sender_times_out_and_aborts_lease_when_actor_does_not_ack():
    sender = _FakeSender(None)

    with pytest.raises(TimeoutError, match="lease lease-1"):
        _flush(sender)

    assert sender._borrowed_feature_tensors == []
    assert sender._borrowed_feature_block_sizes == []
    assert sender._borrowed_feature_active_lease is None
    assert sender._borrowed_feature_consumer_rank is None
    assert sender.torch_platform.collects == 1
