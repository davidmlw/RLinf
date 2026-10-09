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
from pathlib import Path

from toolkits.gr00t_trocar.w16.configs import PROFILE, render, validate


def _base(*, optimized: bool) -> dict:
    return {
        "cluster": {"num_nodes": 1},
        "runner": {
            "max_epochs": 5,
            "val_check_interval": -1,
            "save_interval": 5,
            "logger": {"experiment_name": "base", "log_path": "/old"},
        },
        "env": {
            "train": {"video_cfg": {"video_base_dir": "/old/train"}},
            "eval": {"video_cfg": {"video_base_dir": "/old/eval"}},
        },
        "rollout": {
            "model": {
                "skip_unused_lm_head": True,
                "tensorrt_backbone": {
                    "enabled": optimized,
                    **({"engine_dir": "/engines"} if optimized else {}),
                },
                "tensorrt_dit": {
                    "enabled": optimized,
                    **(
                        {
                            "online_refit": True,
                            "probe_each_revision": False,
                            "shadow_eager": False,
                        }
                        if optimized
                        else {}
                    ),
                },
            }
        },
        "actor": {
            "pre_update_same_revision_gate": {"enabled": False},
            "model": {
                "skip_unused_lm_head": True,
                "rollout_backbone_feature_transport": "borrowed_ipc_pinned",
            },
        },
    }


def test_render_freezes_two_endpoint_profile_contract():
    endpoints = render(_base(optimized=False), _base(optimized=True))

    assert validate(endpoints) == []
    assert endpoints["original"]["cluster"]["profiling"] == PROFILE
    assert endpoints["optimized"]["cluster"]["profiling"] == PROFILE
    assert endpoints["original"]["runner"]["max_epochs"] == 4
    assert endpoints["original"]["actor"]["model"][
        "rollout_backbone_feature_transport"
    ] is None
    assert endpoints["optimized"]["rollout"]["model"]["tensorrt_dit"][
        "online_refit"
    ]


def test_validate_rejects_unexpected_endpoint_drift():
    endpoints = render(_base(optimized=False), _base(optimized=True))
    broken = copy.deepcopy(endpoints)
    broken["optimized"]["runner"]["weight_sync_interval"] = 2

    errors = validate(broken)

    assert any("unexpected endpoint diff paths" in error for error in errors)


def test_validate_rejects_expensive_refit_probe():
    endpoints = render(_base(optimized=False), _base(optimized=True))
    endpoints["optimized"]["rollout"]["model"]["tensorrt_dit"][
        "probe_each_revision"
    ] = True

    errors = validate(endpoints)

    assert any("optimized endpoint mismatch" in error for error in errors)


def test_profile_uses_bounded_timeline_collection():
    assert PROFILE["options"]["trace"] == "cuda,nvtx,osrt,vulkan"
    assert "cuda-memory-usage" not in PROFILE["options"]


def test_launcher_keeps_nsys_temporary_streams_out_of_shared_memory():
    launcher = (
        Path(__file__).parents[4]
        / "toolkits/gr00t_trocar/w16/run_rtx6000_nsys.sh"
    ).read_text(encoding="utf-8")

    assert "NSYS_TMPDIR=/w16-run/scratch/nsys-tmp" in launcher
    assert "NSYS_TMPDIR=/dev/shm" not in launcher
