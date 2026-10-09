#!/usr/bin/env python3
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

"""Render the two retained W16 Nsight endpoint configurations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROFILE = {
    "backend": "nsight",
    "enabled": True,
    "worker_groups": ["ActorGroup", "RolloutGroup", "EnvGroup"],
    "ranks": [0],
    "steps": [1, 2],
    "continuous": True,
    "output_dir": "/w16-run/nsights",
    "options": {
        "trace": "cuda,nvtx,osrt,vulkan",
        "sample": "none",
        "cpuctxsw": "none",
        "osrt-threshold": 1000,
        "cuda-memory-usage": True,
        "force-overwrite": True,
    },
    "flags": [],
}

EXPECTED_DIFF_PREFIXES = (
    "actor.model.rollout_backbone_feature_transport",
    "actor.model.skip_unused_lm_head",
    "rollout.model.skip_unused_lm_head",
    "rollout.model.tensorrt_backbone",
    "rollout.model.tensorrt_dit",
    "runner.logger.experiment_name",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_diff_paths(left[key], right[key], path))
        return paths
    if isinstance(left, list):
        return [] if left == right else [prefix]
    return [] if left == right else [prefix]


def _set_common(config: dict[str, Any], endpoint: str) -> None:
    config["cluster"]["profiling"] = copy.deepcopy(PROFILE)
    config["runner"]["max_epochs"] = 4
    config["runner"]["val_check_interval"] = -1
    config["runner"]["save_interval"] = 4
    config["runner"]["logger"]["experiment_name"] = f"w16_{endpoint}"
    config["runner"]["logger"]["log_path"] = "/w16-run/output"
    config["env"]["train"]["video_cfg"]["video_base_dir"] = (
        "/w16-run/output/video/train"
    )
    config["env"]["eval"]["video_cfg"]["video_base_dir"] = (
        "/w16-run/output/video/eval"
    )
    config["actor"]["pre_update_same_revision_gate"]["enabled"] = False


def render(
    eager_base: dict[str, Any], optimized_base: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    original = copy.deepcopy(eager_base)
    optimized = copy.deepcopy(optimized_base)
    _set_common(original, "original")
    _set_common(optimized, "optimized")

    original["rollout"]["model"]["skip_unused_lm_head"] = False
    original["actor"]["model"]["skip_unused_lm_head"] = False
    original["actor"]["model"]["rollout_backbone_feature_transport"] = None

    return {"original": original, "optimized": optimized}


def validate(endpoints: dict[str, dict[str, Any]]) -> list[str]:
    errors = []
    original = endpoints["original"]
    optimized = endpoints["optimized"]
    for endpoint, config in endpoints.items():
        if config["cluster"].get("profiling") != PROFILE:
            errors.append(f"{endpoint}: profiling contract mismatch")
        if config["runner"]["max_epochs"] != 4:
            errors.append(f"{endpoint}: max_epochs must be 4")
        if config["actor"]["pre_update_same_revision_gate"]["enabled"]:
            errors.append(f"{endpoint}: PPO identity recompute must be disabled")

    original_contract = {
        "feature_transport": original["actor"]["model"][
            "rollout_backbone_feature_transport"
        ],
        "rollout_skip_logits": original["rollout"]["model"][
            "skip_unused_lm_head"
        ],
        "actor_skip_logits": original["actor"]["model"]["skip_unused_lm_head"],
        "trt_backbone": original["rollout"]["model"]["tensorrt_backbone"].get(
            "enabled", False
        ),
        "trt_dit": original["rollout"]["model"]["tensorrt_dit"].get(
            "enabled", False
        ),
    }
    if original_contract != {
        "feature_transport": None,
        "rollout_skip_logits": False,
        "actor_skip_logits": False,
        "trt_backbone": False,
        "trt_dit": False,
    }:
        errors.append(f"original endpoint mismatch: {original_contract!r}")

    optimized_contract = {
        "feature_transport": optimized["actor"]["model"][
            "rollout_backbone_feature_transport"
        ],
        "rollout_skip_logits": optimized["rollout"]["model"][
            "skip_unused_lm_head"
        ],
        "actor_skip_logits": optimized["actor"]["model"]["skip_unused_lm_head"],
        "trt_backbone": optimized["rollout"]["model"]["tensorrt_backbone"].get(
            "enabled", False
        ),
        "trt_dit": optimized["rollout"]["model"]["tensorrt_dit"].get(
            "enabled", False
        ),
        "trt_dit_online_refit": optimized["rollout"]["model"][
            "tensorrt_dit"
        ].get("online_refit", False),
        "trt_dit_probe_each_revision": optimized["rollout"]["model"][
            "tensorrt_dit"
        ].get("probe_each_revision"),
        "trt_dit_shadow_eager": optimized["rollout"]["model"][
            "tensorrt_dit"
        ].get("shadow_eager"),
    }
    if optimized_contract != {
        "feature_transport": "borrowed_ipc_pinned",
        "rollout_skip_logits": True,
        "actor_skip_logits": True,
        "trt_backbone": True,
        "trt_dit": True,
        "trt_dit_online_refit": True,
        "trt_dit_probe_each_revision": False,
        "trt_dit_shadow_eager": False,
    }:
        errors.append(f"optimized endpoint mismatch: {optimized_contract!r}")

    diff_paths = _diff_paths(original, optimized)
    unexpected = [
        path
        for path in diff_paths
        if not any(
            path == allowed or path.startswith(f"{allowed}.")
            for allowed in EXPECTED_DIFF_PREFIXES
        )
    ]
    if unexpected:
        errors.append(f"unexpected endpoint diff paths: {unexpected!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager-base", type=Path, required=True)
    parser.add_argument("--optimized-base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()

    endpoints = render(
        _load_yaml(args.eager_base), _load_yaml(args.optimized_base)
    )
    errors = validate(endpoints)
    if errors:
        raise ValueError("\n".join(errors))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for endpoint, config in endpoints.items():
        path = args.output_dir / f"{endpoint}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths[endpoint] = path

    manifest = {
        "schema": "rlinf.w16.rtx6000-nsys-configs/v1",
        "source_revision": args.source_revision,
        "capture": {
            "worker_groups": PROFILE["worker_groups"],
            "ranks": PROFILE["ranks"],
            "steps": PROFILE["steps"],
            "continuous": PROFILE["continuous"],
            "expected_reports_per_endpoint": 3,
        },
        "endpoints": {
            endpoint: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for endpoint, path in paths.items()
        },
        "diff_paths": _diff_paths(endpoints["original"], endpoints["optimized"]),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
