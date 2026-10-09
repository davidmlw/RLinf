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

"""Render and validate the five W15 full-RL executor arms."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONTRACT = ROOT / "contract.json"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _backbone_config(contract: dict[str, Any], backend: str) -> dict[str, Any]:
    workload = contract["workload"]
    runtime = contract["runtime"]
    if backend == "eager":
        return {
            "torch_compile_backbone": {"enabled": False},
            "tensorrt_backbone": {"enabled": False},
        }
    if backend == "pt2":
        return {
            "torch_compile_backbone": {
                "enabled": True,
                "mode": "max-autotune-no-cudagraphs",
                "static_batch_size": workload["policy_batch_per_rank"],
                "sequence_length": workload["text_sequence_length"],
                "vision_segments": workload["vision_segments"],
                "vision_sequence_length": workload["vision_sequence_length"],
            },
            "tensorrt_backbone": {"enabled": False},
        }
    if backend == "tensorrt":
        artifact = contract["artifacts"]["backbone"]
        return {
            "torch_compile_backbone": {"enabled": False},
            "tensorrt_backbone": {
                "enabled": True,
                **artifact,
                "static_batch_size": workload["policy_batch_per_rank"],
                "sequence_opt": workload["text_sequence_length"],
                "runtime_version": runtime["tensorrt_version"],
                "runtime_distribution": runtime["tensorrt_distribution"],
                "compute_capability": runtime["compute_capability"],
            },
        }
    raise ValueError(f"unsupported backbone backend: {backend}")


def _dit_config(
    contract: dict[str, Any], backend: str, *, identity_gate: bool
) -> dict[str, Any]:
    runtime = contract["runtime"]
    if backend == "eager":
        return {"enable_torch_compile": False, "tensorrt_dit": {"enabled": False}}
    if backend == "pt2":
        return {"enable_torch_compile": True, "tensorrt_dit": {"enabled": False}}
    if backend == "refittable_tensorrt":
        artifact = contract["artifacts"]["dit"]
        return {
            "enable_torch_compile": False,
            "tensorrt_dit": {
                "enabled": True,
                **artifact,
                "revision": 0,
                "runtime_version": runtime["tensorrt_version"],
                "runtime_distribution": runtime["tensorrt_distribution"],
                "compute_capability": runtime["compute_capability"],
                "online_refit": True,
                "lineage_receipt_mode": "gpu_transform_validation",
                "probe_each_revision": identity_gate,
                "minimum_probe_cosine": 0.999,
                "maximum_probe_relative_l2": 0.05,
                "minimum_free_device_bytes": 8589934592,
                "ppo_authority_status": (
                    "failed_ratio_kl_approximate_behavior_only"
                ),
                "shadow_eager": False,
            },
        }
    raise ValueError(f"unsupported DiT backend: {backend}")


def render(
    base: dict[str, Any],
    contract: dict[str, Any],
    arm: str,
    *,
    max_epochs: int,
    identity_gate: bool,
) -> dict[str, Any]:
    if arm not in contract["arms"]:
        raise ValueError(f"unknown W15 arm: {arm}")
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")

    result = copy.deepcopy(base)
    arm_contract = contract["arms"][arm]
    rollout = result["rollout"]
    rollout_model = rollout["model"]
    rollout_model.update(_backbone_config(contract, arm_contract["backbone"]))
    dit = _dit_config(
        contract, arm_contract["dit"], identity_gate=identity_gate
    )
    rollout["enable_torch_compile"] = dit.pop("enable_torch_compile")
    rollout_model.update(dit)
    rollout["torch_compile_mode"] = "max-autotune-no-cudagraphs"

    result["runner"]["max_epochs"] = max_epochs
    result["runner"]["val_check_interval"] = -1
    result["runner"]["save_interval"] = max_epochs
    result["runner"]["logger"]["experiment_name"] = f"w15_{arm.replace('-', '_')}"
    result["runner"]["logger"]["log_path"] = "/w15-run/output"
    result["env"]["train"]["video_cfg"]["video_base_dir"] = (
        "/w15-run/output/video/train"
    )
    result["env"]["eval"]["video_cfg"]["video_base_dir"] = (
        "/w15-run/output/video/eval"
    )
    result["actor"]["pre_update_same_revision_gate"]["enabled"] = identity_gate
    return result


def validate(
    config: dict[str, Any],
    contract: dict[str, Any],
    arm: str,
    *,
    max_epochs: int,
    identity_gate: bool,
) -> list[str]:
    expected = render(
        config,
        contract,
        arm,
        max_epochs=max_epochs,
        identity_gate=identity_gate,
    )
    errors = []
    for path in (
        ("runner", "max_epochs"),
        ("runner", "val_check_interval"),
        ("runner", "save_interval"),
        ("runner", "logger", "experiment_name"),
        ("rollout", "enable_torch_compile"),
        ("rollout", "torch_compile_mode"),
        ("rollout", "model", "torch_compile_backbone"),
        ("rollout", "model", "tensorrt_backbone"),
        ("rollout", "model", "tensorrt_dit"),
        ("actor", "pre_update_same_revision_gate", "enabled"),
    ):
        actual_value: Any = config
        expected_value: Any = expected
        for key in path:
            actual_value = actual_value[key]
            expected_value = expected_value[key]
        if actual_value != expected_value:
            errors.append(
                f"{'.'.join(path)}: expected {expected_value!r}, got {actual_value!r}"
            )

    common = {
        "total_num_envs": config["env"]["train"]["total_num_envs"],
        "num_action_chunks": config["actor"]["model"]["num_action_chunks"],
        "feature_transport": config["actor"]["model"][
            "rollout_backbone_feature_transport"
        ],
        "rollout_skip_logits": config["rollout"]["model"]["skip_unused_lm_head"],
        "actor_skip_logits": config["actor"]["model"]["skip_unused_lm_head"],
    }
    wanted_common = {
        "total_num_envs": contract["workload"]["global_envs"],
        "num_action_chunks": contract["workload"]["num_action_chunks"],
        "feature_transport": "borrowed_ipc_pinned",
        "rollout_skip_logits": True,
        "actor_skip_logits": True,
    }
    if common != wanted_common:
        errors.append(f"common workload mismatch: {common!r} != {wanted_common!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--identity-gate", choices=("on", "off"), default="on")
    args = parser.parse_args()

    base = _load_yaml(args.base)
    contract = _load_json(args.contract)
    identity_gate = args.identity_gate == "on"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": contract["schema"], "configs": {}}
    for arm in contract["arms"]:
        config = render(
            base,
            contract,
            arm,
            max_epochs=args.max_epochs,
            identity_gate=identity_gate,
        )
        errors = validate(
            config,
            contract,
            arm,
            max_epochs=args.max_epochs,
            identity_gate=identity_gate,
        )
        if errors:
            raise ValueError("\n".join(errors))
        path = args.output_dir / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        manifest["configs"][arm] = str(path)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
