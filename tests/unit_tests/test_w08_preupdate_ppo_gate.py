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

import inspect
from pathlib import Path

import pytest
import torch

from rlinf.workers.actor.fsdp_actor_worker import (
    EmbodiedFSDPActor,
    compute_preupdate_ppo_gate_stats,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def test_chunk_level_preupdate_gate_uses_the_ppo_ratio_boundary() -> None:
    old = torch.zeros((2, 1, 2), dtype=torch.float32)
    current = torch.tensor(
        [[[0.0, 0.0]], [[0.1, 0.1]]], dtype=torch.float32
    )

    stats = compute_preupdate_ppo_gate_stats(
        current,
        old,
        torch.ones((2, 1), dtype=torch.bool),
        logprob_type="chunk_level",
        single_action_dim=2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )

    assert stats["count"].item() == 2
    assert stats["finite_count"].item() == 2
    assert stats["outside_clip_count"].item() == 1
    assert stats["log_ratio_sum"].item() == pytest.approx(0.2)


def test_preupdate_gate_marks_nonfinite_ratios() -> None:
    stats = compute_preupdate_ppo_gate_stats(
        torch.tensor([[[float("nan")]]]),
        torch.zeros((1, 1, 1)),
        None,
        logprob_type="chunk_level",
        single_action_dim=1,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )

    assert stats["count"].item() == 1
    assert stats["finite_count"].item() == 0


def test_w08_gate_runs_before_the_first_optimizer_step() -> None:
    source = inspect.getsource(EmbodiedFSDPActor._run_training_impl)

    assert source.index("_measure_preupdate_ppo_gate") < source.index(
        "self.optimizer_step()"
    )


def test_rollout_retains_hybrid_telemetry_before_close() -> None:
    source = inspect.getsource(MultiStepRolloutWorker.close_hybrid_runtime_worker)

    assert source.index("hybrid_runtime_telemetry()") < source.index(
        "close_hybrid_runtime()"
    )
    assert "HYBRID_RUNTIME_TELEMETRY" in source


def test_rollout_hybrid_cleanup_does_not_override_worker_group_close() -> None:
    assert "_close" not in MultiStepRolloutWorker.__dict__


def test_w08_launcher_disables_static_b8_incompatible_final_eval() -> None:
    launcher = (
        Path(__file__).parents[2]
        / "toolkits/l20/gr00t_stack_cube/run_w08.sh"
    ).read_text(encoding="utf-8")

    assert "export SAVE_INTERVAL=-1" in launcher
    assert "export VAL_CHECK_INTERVAL=-1" in launcher
