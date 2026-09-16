#!/usr/bin/env python3
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

"""Run a finite native Isaac/Vulkan Assemble Trocar environment gate."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from offline_asset_mirror import install_offline_asset_mirror

TASK_ID = "Isaac-Assemble-Trocar-G129-Dex3-RLinf-v0"
CAMERAS = ("front_camera", "left_wrist_camera", "right_wrist_camera")


def _tensor_inventory(value: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    import torch

    if isinstance(value, dict):
        inventory = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            inventory.update(_tensor_inventory(child, child_prefix))
        return inventory
    if isinstance(value, torch.Tensor):
        return {
            prefix: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "finite": bool(torch.isfinite(value).all().item())
                if value.is_floating_point()
                else True,
            }
        }
    return {}


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("isaacsim", "isaaclab", "isaaclab_tasks", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run(num_envs: int, steps: int, asset_mirror: Path) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    receipt: dict[str, Any] = {
        "schema": "rlinf.w04.horde-native-env-smoke/v1",
        "status": "failed",
        "task_id": TASK_ID,
        "num_envs": num_envs,
        "steps": steps,
        "asset_mirror": os.fspath(asset_mirror.resolve()),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "vk_driver_files": os.environ.get("VK_DRIVER_FILES"),
        },
        "started_ns": started_ns,
    }
    simulation_app = None
    env = None
    cleanup_errors = []
    try:
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True, enable_cameras=True)
        simulation_app = launcher.app

        import gymnasium as gym
        import torch

        install_offline_asset_mirror(asset_mirror)
        import isaaclab_tasks  # noqa: F401, PLC0415
        from isaaclab_tasks.utils import parse_env_cfg

        torch.cuda.reset_peak_memory_stats()
        cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=num_envs)
        env = gym.make(TASK_ID, cfg=cfg, render_mode="rgb_array").unwrapped

        torch.cuda.synchronize()
        reset_start = time.monotonic_ns()
        observation, _ = env.reset()
        torch.cuda.synchronize()
        reset_end = time.monotonic_ns()
        reset_tensors = _tensor_inventory(observation)

        action = torch.zeros(env.action_space.shape, device=env.device)
        step_receipts = []
        latest_observation = observation
        for index in range(steps):
            torch.cuda.synchronize()
            start = time.monotonic_ns()
            latest_observation, reward, terminated, truncated, _ = env.step(action)
            torch.cuda.synchronize()
            end = time.monotonic_ns()
            step_receipts.append(
                {
                    "index": index,
                    "start_ns": start,
                    "end_ns": end,
                    "wall_ms": (end - start) / 1e6,
                    "reward_mean": float(reward.float().mean().item()),
                    "terminated": int(terminated.sum().item()),
                    "truncated": int(truncated.sum().item()),
                }
            )

        step_tensors = _tensor_inventory(latest_observation)
        camera_paths = {
            camera: [path for path in step_tensors if path.endswith(camera)]
            for camera in CAMERAS
        }
        if any(len(paths) != 1 for paths in camera_paths.values()):
            raise RuntimeError(f"incomplete camera outputs: {camera_paths}")
        if not all(item["finite"] for item in step_tensors.values()):
            raise RuntimeError("environment produced a non-finite tensor")

        receipt.update(
            {
                "status": "passed",
                "packages": _package_versions(),
                "reset": {
                    "start_ns": reset_start,
                    "end_ns": reset_end,
                    "wall_ms": (reset_end - reset_start) / 1e6,
                    "tensors": reset_tensors,
                },
                "step_receipts": step_receipts,
                "step_tensors": step_tensors,
                "camera_paths": camera_paths,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
    except BaseException as exc:  # Retain diagnostics even when Kit raises during startup.
        receipt["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as exc:
                cleanup_errors.append(
                    {"component": "env", "type": type(exc).__name__, "message": str(exc)}
                )
        if simulation_app is not None:
            try:
                simulation_app.close()
            except BaseException as exc:
                cleanup_errors.append(
                    {
                        "component": "simulation_app",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
    receipt["cleanup"] = {
        "status": "passed" if not cleanup_errors else "failed",
        "errors": cleanup_errors,
    }
    receipt["ended_ns"] = time.monotonic_ns()
    if cleanup_errors:
        receipt["status"] = "failed"
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--asset-mirror", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num-envs and --steps must be positive")
    receipt = run(args.num_envs, args.steps, args.asset_mirror)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
