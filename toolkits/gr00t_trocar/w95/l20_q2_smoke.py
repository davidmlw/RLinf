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

"""Finite W96 Q2 smoke for eager true-B8 inference and one Vulkan Env turn."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

CAMERAS = ("front_camera", "left_wrist_camera", "right_wrist_camera")
TASK_ID = "IsaacContrib-Assemble-Trocar-G129-Dex3"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_receipt(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "min": float(array.min()),
        "max": float(array.max()),
        "sha256": _sha256_bytes(array.tobytes()),
    }


def _action_tensor(action: dict[str, Any]) -> Any:
    import numpy as np

    ordered = []
    for suffix in ("left_arm", "right_arm", "left_hand", "right_hand"):
        matches = [value for key, value in action.items() if key.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"action output does not uniquely contain {suffix}")
        ordered.append(np.asarray(matches[0]))
    result = np.concatenate(ordered, axis=-1)
    if result.shape != (8, 16, 28):
        raise RuntimeError(
            f"public action shape is not true-B8 chunk16: {result.shape}"
        )
    if not np.isfinite(result).all():
        raise RuntimeError("public action contains non-finite values")
    return result


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    if any(name.startswith("RLINF_GROOT_TRT") for name in os.environ):
        raise RuntimeError("Q2 eager smoke forbids TensorRT engine environment")
    source = Path("/workspace/rlinf-src/toolkits/eos/gr00t_trocar/tensorrt")
    sys.path.insert(0, str(source))
    from resident_b1 import _load_api  # noqa: PLC0415
    from trocar_b8_fixture import (  # noqa: PLC0415
        _assert_collated_contract,
        raw_observation,
    )
    from trocar_b8_model_view import (  # noqa: PLC0415
        materialize_trocar_model_view,
    )

    model = args.model.resolve(strict=True)
    backbone = args.backbone_model.resolve(strict=True)
    metadata = args.metadata.resolve(strict=True)
    model_view = Path("/w96-run/scratch/model-view")
    if model_view.exists():
        raise RuntimeError(f"model view already exists: {model_view}")
    view_receipt = materialize_trocar_model_view(model, backbone, metadata, model_view)

    api = _load_api(Path("/workspace/gr00t-n17"))
    api["set_seed"](args.seed)
    policy = api["Gr00tPolicy"](
        model_path=str(model_view),
        embodiment_tag=api["EmbodimentTag"].NEW_EMBODIMENT,
        device="cuda",
        strict=True,
    )
    metadata_value = json.loads(metadata.read_text(encoding="utf-8"))
    observation = raw_observation(metadata_value, args.prompt)
    collated = api["prepare_model_inputs"](policy, observation)
    if isinstance(collated, tuple):
        collated = collated[0]
    collated_shapes = _assert_collated_contract(collated)

    import torch

    api["set_seed"](args.seed)
    torch.cuda.synchronize()
    action, _ = policy.get_action(observation)
    torch.cuda.synchronize()
    public_action = _action_tensor(action)
    return {
        "schema": "rlinf.w96.q2-true-b8-eager/v1",
        "status": "passed",
        "backend": "pytorch_eager",
        "batch_size": 8,
        "executed_action_chunks": 16,
        "single_model_call": True,
        "b1x8": False,
        "tensorrt_engine_used": False,
        "seed": args.seed,
        "collated_shapes": collated_shapes,
        "public_action": _array_receipt(public_action),
        "model_view": view_receipt,
    }


def _tensor_tree(value: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    import torch

    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_tensor_tree(child, path))
    elif isinstance(value, torch.Tensor):
        finite = bool(torch.isfinite(value).all().item())
        result[prefix] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "finite": finite,
        }
    return result


def run_env(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.pop("DISPLAY", None)
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    simulation_app = None
    env = None
    try:
        from isaaclab.app import AppLauncher  # noqa: PLC0415

        simulation_app = AppLauncher(headless=True, enable_cameras=True).app

        import gymnasium as gym  # noqa: PLC0415
        import isaaclab_tasks  # noqa: F401, PLC0415
        import torch  # noqa: PLC0415
        from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: PLC0415

        env_cfg = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
        env_cfg.seed = args.seed
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = "cuda:0"
        env = gym.make(TASK_ID, cfg=env_cfg, render_mode="rgb_array").unwrapped
        observation, reset_info = env.reset()
        action = torch.zeros(env.action_space.shape, device=env.device)
        next_observation, reward, terminated, truncated, step_info = env.step(action)
        torch.cuda.synchronize()

        reset_tensors = _tensor_tree(observation)
        step_tensors = _tensor_tree(next_observation)
        camera_paths = {
            camera: [path for path in step_tensors if path.endswith(camera)]
            for camera in CAMERAS
        }
        if any(len(paths) != 1 for paths in camera_paths.values()):
            raise RuntimeError(f"Vulkan camera outputs are incomplete: {camera_paths}")
        if not all(item["finite"] for item in step_tensors.values()):
            raise RuntimeError("environment observation contains non-finite tensors")
        scalar_outputs = (reward, terminated, truncated)
        if not all(torch.isfinite(value).all().item() for value in scalar_outputs):
            raise RuntimeError("environment step output contains non-finite values")
        return {
            "schema": "rlinf.w96.q2-vulkan-env-turn/v1",
            "status": "passed",
            "task_id": TASK_ID,
            "num_envs": 1,
            "steps": 1,
            "renderer": "Vulkan/RTX",
            "physics": "PhysX",
            "ray_started": False,
            "reset_tensor_shapes": reset_tensors,
            "step_tensor_shapes": step_tensors,
            "camera_paths": camera_paths,
            "reward": _array_receipt(reward.detach().cpu().numpy()),
            "terminated": terminated.detach().cpu().tolist(),
            "truncated": truncated.detach().cpu().tolist(),
            "reset_info_type": type(reset_info).__name__,
            "step_info_type": type(step_info).__name__,
        }
    finally:
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()
        sys.argv = original_argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    model = subparsers.add_parser("model")
    model.add_argument("--model", type=Path, required=True)
    model.add_argument("--backbone-model", type=Path, required=True)
    model.add_argument("--metadata", type=Path, required=True)
    model.add_argument("--seed", type=int, default=47)
    model.add_argument("--prompt", default="assemble trocar from tray")
    model.add_argument("--output", type=Path, required=True)
    env = subparsers.add_parser("env")
    env.add_argument("--seed", type=int, default=64201)
    env.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run_model(args) if args.command == "model" else run_env(args)
    except Exception as error:
        traceback.print_exc()
        receipt = {
            "schema": "rlinf.w96.q2-smoke-failure/v1",
            "status": "failed",
            "phase": args.command,
            "error": str(error),
        }
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
