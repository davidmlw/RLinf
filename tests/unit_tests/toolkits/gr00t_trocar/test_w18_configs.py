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

import copy

from toolkits.gr00t_trocar.w18.configs import (
    PROFILE,
    render_variant,
    validate_variant,
)


def _base() -> dict:
    return {
        "cluster": {"profiling": {"enabled": True}},
        "runner": {"logger": {"experiment_name": "old", "log_path": "/old"}},
        "algorithm": {"update_epoch": 4},
        "env": {
            "train": {"video_cfg": {"video_base_dir": "/old/train"}},
            "eval": {"video_cfg": {"video_base_dir": "/old/eval"}},
        },
        "actor": {
            "micro_batch_size": 8,
            "global_batch_size": 2048,
            "fsdp_config": {"sharding_strategy": "full_shard"},
        },
    }


def test_performance_variant_only_changes_allowed_fields():
    base = _base()
    candidate = render_variant(base, micro_batch_size=32, diagnostic=False)

    assert validate_variant(
        base,
        candidate,
        spec={"micro_batch_size": 32, "diagnostic": False},
    ) == []
    assert "profiling" not in candidate["cluster"]
    assert candidate["actor"]["micro_batch_size"] == 32


def test_diagnostic_selects_all_actor_ranks_and_one_step():
    base = _base()
    candidate = render_variant(base, micro_batch_size=8, diagnostic=True)

    assert validate_variant(
        base,
        candidate,
        spec={"micro_batch_size": 8, "diagnostic": True},
    ) == []
    assert candidate["cluster"]["profiling"] == PROFILE
    assert PROFILE["worker_groups"] == ["ActorGroup"]
    assert PROFILE["ranks"] == list(range(8))
    assert PROFILE["steps"] == [1]


def test_shard_grad_op_variant_only_changes_allowed_fields():
    base = _base()
    candidate = render_variant(
        base,
        micro_batch_size=64,
        diagnostic=False,
        sharding_strategy="shard_grad_op",
    )

    assert validate_variant(
        base,
        candidate,
        spec={
            "micro_batch_size": 64,
            "diagnostic": False,
            "sharding_strategy": "shard_grad_op",
        },
    ) == []
    assert candidate["actor"]["fsdp_config"]["sharding_strategy"] == (
        "shard_grad_op"
    )


def test_rank0_shard_grad_op_diagnostic_is_deterministic():
    base = _base()
    candidate = render_variant(
        base,
        micro_batch_size=64,
        diagnostic=True,
        sharding_strategy="shard_grad_op",
        diagnostic_ranks=[0],
    )

    assert validate_variant(
        base,
        candidate,
        spec={
            "micro_batch_size": 64,
            "diagnostic": True,
            "sharding_strategy": "shard_grad_op",
            "ranks": [0],
        },
    ) == []
    assert candidate["cluster"]["profiling"]["ranks"] == [0]


def test_validation_rejects_workload_drift():
    base = _base()
    candidate = render_variant(base, micro_batch_size=16, diagnostic=False)
    candidate["algorithm"]["update_epoch"] = 3

    errors = validate_variant(
        base,
        candidate,
        spec={"micro_batch_size": 16, "diagnostic": False},
    )

    assert "candidate does not match deterministic rendering" in errors
    assert "update_epoch must remain 4" in errors
    assert any("unexpected diff paths" in error for error in errors)


def test_validation_rejects_non_divisible_microbatch():
    base = _base()
    candidate = copy.deepcopy(base)
    candidate = render_variant(candidate, micro_batch_size=48, diagnostic=False)

    errors = validate_variant(
        base,
        candidate,
        spec={"micro_batch_size": 48, "diagnostic": False},
    )

    assert "micro_batch_size must divide the per-rank batch of 256" in errors
