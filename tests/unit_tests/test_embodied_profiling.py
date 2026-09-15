# Copyright 2025 The RLinf Authors.
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

"""Focused tests for embodied-runner profiling-window decisions."""

from rlinf.runners.embodied_runner import EmbodiedRunner


def _runner(*, steps: set[int] | None, continuous: bool) -> EmbodiedRunner:
    runner = object.__new__(EmbodiedRunner)
    runner._profile_all_steps = steps is None
    runner._profile_steps = steps
    runner._profile_continuous = continuous
    runner._profile_start_step = min(steps) if continuous and steps else None
    runner._profile_end_step = max(steps) if continuous and steps else None
    return runner


def test_continuous_profile_window_opens_and_closes_only_once():
    runner = _runner(steps={1, 2}, continuous=True)

    assert [runner._should_open_profiling_window(step) for step in range(4)] == [
        False,
        True,
        False,
        False,
    ]
    assert [runner._should_close_profiling_window(step) for step in range(4)] == [
        False,
        False,
        True,
        False,
    ]


def test_noncontinuous_profile_window_preserves_per_step_behavior():
    runner = _runner(steps={1, 3}, continuous=False)

    assert [runner._should_open_profiling_window(step) for step in range(4)] == [
        False,
        True,
        False,
        True,
    ]
    assert [runner._should_close_profiling_window(step) for step in range(4)] == [
        False,
        True,
        False,
        True,
    ]
