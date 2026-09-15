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

"""Register the W43 kitless Stack Cube task with RLInf.

Only the environment adapter is new.  GR00T preprocessing/action conversion,
Ray worker ownership, rollout, FSDP PPO, and evaluation remain the native
RLInf W85 implementations.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

_registered = False
RLINF_TASK_ID = "RLInf-W43-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0"
TASK_DESCRIPTION = "Stack the red block on the blue block, then stack the green block on the red block."


def _prepend_isaaclab_sources(source_root: Path) -> tuple[Path, ...]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Isaac Lab source root does not exist: {source_root}")
    projects = tuple(
        path.resolve()
        for path in sorted(source_root.iterdir())
        if path.is_dir() and (path / "pyproject.toml").is_file()
    )
    if not projects:
        raise RuntimeError(
            f"Isaac Lab source root contains no extension projects: {source_root}"
        )
    project_strings = [str(path) for path in projects]
    sys.path[:] = project_strings + [
        path for path in sys.path if path not in project_strings
    ]
    return projects


def _quaternion_xyzw_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Match W85 while canonicalizing the backend-dependent quaternion sign."""

    from rlinf.envs.isaaclab.utils import quat2axisangle_torch

    canonical = torch.where(quaternion[:, 3:] < 0, -quaternion, quaternion)
    return quat2axisangle_torch(canonical)


def _extract_stack_cube_observation(
    observation: dict[str, Any],
    *,
    num_envs: int,
) -> dict[str, Any]:
    try:
        policy = observation["policy"]
        cameras = observation.get("rgb_camera", policy)
        table_image = cameras["table_cam"]
        wrist_image = cameras["wrist_cam"]
        eef_pos = policy["eef_pos"]
        eef_quat_wxyz = policy["eef_quat"]
        gripper_pos = policy["gripper_pos"]
    except KeyError as error:
        raise ValueError(
            f"W43 Stack Cube observation lacks required field {error}"
        ) from error

    expected_image_shape = (num_envs, 256, 256, 3)
    if tuple(table_image.shape) != expected_image_shape:
        raise ValueError(
            f"table image must be {expected_image_shape}, got {tuple(table_image.shape)}"
        )
    if tuple(wrist_image.shape) != expected_image_shape:
        raise ValueError(
            f"wrist image must be {expected_image_shape}, got {tuple(wrist_image.shape)}"
        )
    if tuple(eef_pos.shape) != (num_envs, 3):
        raise ValueError(f"eef_pos must be [{num_envs},3]")
    if tuple(eef_quat_wxyz.shape) != (num_envs, 4):
        raise ValueError(f"eef_quat must be [{num_envs},4]")
    if tuple(gripper_pos.shape) != (num_envs, 2):
        raise ValueError(f"gripper_pos must be [{num_envs},2]")

    quaternion_xyzw = eef_quat_wxyz[:, [1, 2, 3, 0]]
    states = torch.cat(
        [
            eef_pos,
            _quaternion_xyzw_to_axis_angle(quaternion_xyzw),
            gripper_pos,
        ],
        dim=1,
    )
    if tuple(states.shape) != (num_envs, 8):
        raise ValueError(f"public state must be [{num_envs},8]")
    return {
        "main_images": table_image,
        "wrist_images": wrist_image,
        "states": states,
        "task_descriptions": [TASK_DESCRIPTION] * num_envs,
    }


def _create_env_wrapper() -> type:
    from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

    class W43StackCubeEnv(IsaaclabBaseEnv):
        """RLInf environment wrapper around the W42-equivalent task config."""

        def _make_env_function(self):
            seed = self.seed
            num_envs = self.cfg.init_params.num_envs

            def make_env_isaaclab():
                import gymnasium as gym
                from isaaclab_tasks.utils import launch_simulation
                from w43_newton_stack_cube import GYM_TASK_ID, build_env_cfg

                source_root = Path(os.environ["W43_ISAACLAB_SOURCE_ROOT"])
                _prepend_isaaclab_sources(source_root)
                env_cfg = build_env_cfg(seed=seed, num_envs=num_envs)
                context = launch_simulation(
                    env_cfg,
                    {"visualizer": None, "visualizer_explicit": True},
                )
                context.__enter__()

                class SimulationContextOwner:
                    def __init__(self, simulation_context):
                        self._simulation_context = simulation_context
                        self._closed = False

                    def close(self) -> None:
                        if not self._closed:
                            self._closed = True
                            self._simulation_context.__exit__(None, None, None)

                owner = SimulationContextOwner(context)
                try:
                    env = gym.make(GYM_TASK_ID, cfg=env_cfg).unwrapped
                except Exception:
                    owner.close()
                    raise
                return env, owner

            return make_env_isaaclab

        def _wrap_obs(self, obs):
            return _extract_stack_cube_observation(obs, num_envs=self.num_envs)

    return W43StackCubeEnv


def _install_initial_evaluation_mode() -> None:
    """Replace the training loop with one native RLInf r0 evaluation.

    Worker construction, initial Actor-to-Rollout synchronization, EnvWorker
    evaluation, rollout inference, and metric aggregation remain unchanged.
    The isolated process exits immediately afterwards, so the evaluation does
    not perturb the subsequent training-reset distribution.
    """

    if os.environ.get("W43_INITIAL_EVAL_ONLY") != "true":
        return
    from rlinf.runners.embodied_runner import EmbodiedRunner

    if getattr(EmbodiedRunner.run, "_w43_initial_eval", False):
        return

    def run_initial_evaluation(self) -> None:
        self.update_rollout_weights()
        metrics = self.evaluate()
        output = Path(os.environ["W43_ATTEMPT_ROOT"]) / "results"
        output.mkdir(parents=True, exist_ok=True)
        receipt = output / "r0-evaluation.json"
        payload = {
            "schema": "rlinf.w43.initial-evaluation.v1",
            "policy_revision": "r0",
            "episodes": 96,
            "metrics": metrics,
        }
        temporary = receipt.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(receipt)
        print(f"RLINF_W43_R0_RECEIPT={receipt}", flush=True)
        print(f"RLINF_W43_R0_METRICS={json.dumps(metrics, sort_keys=True)}", flush=True)
        self._finish_run()

    run_initial_evaluation._w43_initial_eval = True
    EmbodiedRunner.run = run_initial_evaluation


def register() -> None:
    """Install exactly one task ID into RLInf's IsaacLab registry."""

    global _registered
    if _registered:
        return
    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

    configured_ids: set[str] = set()
    import yaml

    config_path = os.environ.get("RLINF_CONFIG_FILE")
    if not config_path:
        raise ValueError("RLINF_CONFIG_FILE not set")
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for section in ("train", "eval"):
        task_id = (
            config.get("env", {}).get(section, {}).get("init_params", {}).get("id")
        )
        if task_id:
            configured_ids.add(task_id)
    if configured_ids != {RLINF_TASK_ID}:
        raise RuntimeError(
            f"W43 requires only task {RLINF_TASK_ID}, got {sorted(configured_ids)}"
        )
    REGISTER_ISAACLAB_ENVS[RLINF_TASK_ID] = _create_env_wrapper()
    _install_initial_evaluation_mode()
    _registered = True
