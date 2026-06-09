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

"""Tests for rlinf.utils.nsight_profiler."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from rlinf.utils import nsight_profiler
from rlinf.utils.nsight_profiler import NsightProfiler


@pytest.fixture(autouse=True)
def _reset_module_flag():
    """Reset the module-level flag between tests so they stay isolated."""
    nsight_profiler._profiling_active = False
    yield
    nsight_profiler._profiling_active = False


class TestStartStop:
    """Module-level start_profile / stop_profile drive torch.cuda.profiler."""

    def test_start_sets_active_and_calls_cuda_profiler_start(self):
        with patch("torch.cuda.profiler.start") as mock_start:
            nsight_profiler.start_profile(step_idx=5)
        assert nsight_profiler.is_profiling_active() is True
        mock_start.assert_called_once_with()

    def test_stop_clears_active_and_calls_cuda_profiler_stop(self):
        with patch("torch.cuda.profiler.start"):
            nsight_profiler.start_profile()
        with patch("torch.cuda.profiler.stop") as mock_stop:
            nsight_profiler.stop_profile()
        assert nsight_profiler.is_profiling_active() is False
        mock_stop.assert_called_once_with()

    def test_double_start_is_idempotent(self):
        with patch("torch.cuda.profiler.start") as mock_start:
            nsight_profiler.start_profile()
            nsight_profiler.start_profile()
        # Second call must not invoke cuda profiler start again.
        assert mock_start.call_count == 1

    def test_stop_without_start_is_noop(self):
        with patch("torch.cuda.profiler.stop") as mock_stop:
            nsight_profiler.stop_profile()
        mock_stop.assert_not_called()


class TestAnnotateSync:
    """@NsightProfiler.annotate on a sync function is transparent off / wraps on."""

    def test_returns_value_when_inactive(self):
        @NsightProfiler.annotate("test/op")
        def doubled(x):
            return x * 2

        assert doubled(5) == 10
        assert nsight_profiler.is_profiling_active() is False

    def test_propagates_exception_when_inactive(self):
        @NsightProfiler.annotate("test/op")
        def boom():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError, match="nope"):
            boom()

    def test_emits_torch_range_when_active(self):
        @NsightProfiler.annotate("test/op", color="green")
        def doubled(x):
            return x * 2

        with patch("torch.cuda.profiler.start"):
            nsight_profiler.start_profile()

        with (
            patch.object(nsight_profiler, "_TORCH_NVTX_AVAILABLE", True),
            patch("torch.cuda.nvtx.range_push") as mock_push,
            patch("torch.cuda.nvtx.range_pop") as mock_pop,
        ):
            result = doubled(7)

        assert result == 14
        mock_push.assert_called_once_with("test/op")
        mock_pop.assert_called_once_with()

    def test_falls_back_to_python_nvtx_when_torch_nvtx_unavailable(self):
        @NsightProfiler.annotate("test/op", color="green")
        def doubled(x):
            return x * 2

        with patch("torch.cuda.profiler.start"):
            nsight_profiler.start_profile()

        with (
            patch.object(nsight_profiler, "_TORCH_NVTX_AVAILABLE", False),
            patch.object(nsight_profiler, "_nvtx") as mock_nvtx,
            patch.object(nsight_profiler, "_NVTX_AVAILABLE", True),
        ):
            mock_nvtx.start_range.return_value = (1, 2)
            result = doubled(7)

        assert result == 14
        mock_nvtx.start_range.assert_called_once_with(
            message="test/op", color="green", domain=None
        )
        mock_nvtx.end_range.assert_called_once_with((1, 2))

    def test_ends_range_on_exception(self):
        @NsightProfiler.annotate("test/op")
        def boom():
            raise ValueError("oops")

        with patch("torch.cuda.profiler.start"):
            nsight_profiler.start_profile()

        with (
            patch.object(nsight_profiler, "_TORCH_NVTX_AVAILABLE", True),
            patch("torch.cuda.nvtx.range_push") as mock_push,
            patch("torch.cuda.nvtx.range_pop") as mock_pop,
        ):
            with pytest.raises(ValueError, match="oops"):
                boom()

        mock_push.assert_called_once_with("test/op")
        mock_pop.assert_called_once_with()

    def test_uses_function_name_when_message_is_none(self):
        @NsightProfiler.annotate()
        def my_specific_function(x):
            return x

        with patch("torch.cuda.profiler.start"):
            nsight_profiler.start_profile()

        with (
            patch.object(nsight_profiler, "_TORCH_NVTX_AVAILABLE", True),
            patch("torch.cuda.nvtx.range_push") as mock_push,
            patch("torch.cuda.nvtx.range_pop"),
        ):
            my_specific_function(0)

        mock_push.assert_called_once_with("my_specific_function")


class TestAnnotateAsync:
    """@NsightProfiler.annotate on an async function works the same way."""

    def test_returns_value_when_inactive(self):
        @NsightProfiler.annotate("test/async_op")
        async def add(a, b):
            return a + b

        assert asyncio.get_event_loop().run_until_complete(add(2, 3)) == 5

    def test_emits_torch_range_when_active(self):
        @NsightProfiler.annotate("test/async_op")
        async def add(a, b):
            return a + b

        with patch("torch.cuda.profiler.start"):
            nsight_profiler.start_profile()

        with (
            patch.object(nsight_profiler, "_TORCH_NVTX_AVAILABLE", True),
            patch("torch.cuda.nvtx.range_push") as mock_push,
            patch("torch.cuda.nvtx.range_pop") as mock_pop,
        ):
            result = asyncio.get_event_loop().run_until_complete(add(2, 3))

        assert result == 5
        mock_push.assert_called_once_with("test/async_op")
        mock_pop.assert_called_once_with()


class TestAnnotateNoNvtxPackage:
    """When the nvtx package is unavailable, the decorator must be a pass-through."""

    def test_sync_function_runs_normally(self):
        @NsightProfiler.annotate("test/op")
        def doubled(x):
            return x * 2

        with patch("torch.cuda.profiler.start"):
            nsight_profiler.start_profile()
        with (
            patch.object(nsight_profiler, "_TORCH_NVTX_AVAILABLE", False),
            patch.object(nsight_profiler, "_NVTX_AVAILABLE", False),
        ):
            assert doubled(4) == 8
