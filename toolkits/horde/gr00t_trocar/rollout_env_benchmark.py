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

"""Benchmark a true-B8 GR00T policy interleaved with the native Trocar Env."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .offline_asset_mirror import install_offline_asset_mirror
except ImportError:  # Direct script execution.
    from offline_asset_mirror import install_offline_asset_mirror

TASK_ID = "IsaacContrib-Assemble-Trocar-G129-Dex3"
BATCH_SIZE = 8
ACTION_CHUNKS = 16
PUBLIC_ACTION_DIM = 28
ENV_ACTION_DIM = 43
ACTION_PREFIX_PAD = 15
PROMPT = "assemble trocar from tray"
CAMERA_ORDER = ("left_wrist_view", "right_wrist_view", "room_view")
ACTION_ORDER = ("left_arm", "right_arm", "left_hand", "right_hand")
BACKENDS = ("eager-eager", "trt-eager", "trt-pt2", "trt-refit-trt")


def _write_receipt(output: Path, receipt: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(f".{output.name}.pending")
    pending.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(pending, output)


def _statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "samples_ms": values,
        "mean_ms": mean,
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "sample_std_ms": sample_std,
        "cv": sample_std / mean if mean else None,
    }


def _package_versions() -> dict[str, str | None]:
    result = {}
    for name in (
        "isaacsim",
        "isaaclab",
        "isaaclab_tasks",
        "torch",
        "torchvision",
        "tensorrt-cu12",
        "transformers",
        "flash-attn",
        "torchcodec",
        "gr00t",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _memory_snapshot() -> dict[str, Any]:
    import torch

    free, total = torch.cuda.mem_get_info()
    return {
        "torch_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
        "device_free_bytes": free,
        "device_total_bytes": total,
        "device_used_bytes": total - free,
        "host_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
    }


def observation_to_policy(observation: dict[str, Any], prompt: str = PROMPT) -> dict:
    """Map native Isaac tensors to the exact W03 N1.7 policy contract."""

    cameras = observation["camera_images"]
    policy = observation["policy"]
    body = policy["robot_joint_state"][:, 15:29]
    hands = policy["robot_dex3_joint_state"]
    if tuple(body.shape) != (BATCH_SIZE, 14):
        raise RuntimeError(f"unexpected body state shape: {tuple(body.shape)}")
    if tuple(hands.shape) != (BATCH_SIZE, 14):
        raise RuntimeError(f"unexpected Dex3 state shape: {tuple(hands.shape)}")

    def image(name: str) -> np.ndarray:
        value = cameras[name].detach().cpu().numpy()
        return np.expand_dims(value.astype(np.uint8, copy=False), axis=1)

    def state(value: Any) -> np.ndarray:
        array = value.detach().cpu().float().numpy()
        return np.expand_dims(array.astype(np.float32, copy=False), axis=1)

    return {
        "video": {
            "left_wrist_view": image("left_wrist_camera"),
            "right_wrist_view": image("right_wrist_camera"),
            "room_view": image("front_camera"),
        },
        "state": {
            "left_arm": state(body[:, 0:7]),
            "right_arm": state(body[:, 7:14]),
            "left_hand": state(hands[:, 0:7]),
            "right_hand": state(hands[:, 7:14]),
        },
        "language": {
            "annotation.human.action.task_description": [
                [prompt] for _ in range(BATCH_SIZE)
            ]
        },
    }


def actions_to_env(actions: dict[str, np.ndarray], device: Any) -> Any:
    """Concatenate four 7-D chunks and apply the Trocar 15-D prefix."""

    import torch

    components = []
    for name in ACTION_ORDER:
        value = actions.get(name)
        if value is None:
            value = actions.get(f"action.{name}")
        if value is None:
            raise KeyError(f"policy action omits {name}: {sorted(actions)}")
        components.append(value[:, :ACTION_CHUNKS, :])
    public = np.concatenate(components, axis=-1).astype(np.float32, copy=False)
    if public.shape != (BATCH_SIZE, ACTION_CHUNKS, PUBLIC_ACTION_DIM):
        raise RuntimeError(f"unexpected public action shape: {public.shape}")
    padded = np.pad(
        public,
        ((0, 0), (0, 0), (ACTION_PREFIX_PAD, 0)),
        mode="constant",
    )
    if padded.shape != (BATCH_SIZE, ACTION_CHUNKS, ENV_ACTION_DIM):
        raise RuntimeError(f"unexpected Env action shape: {padded.shape}")
    return torch.from_numpy(padded).to(device=device)


def _prepared_shape_contract(prepared: tuple[Any, Any]) -> dict[str, list[int]]:
    backbone, action = prepared
    shapes = {
        "input_ids": list(backbone["input_ids"].shape),
        "pixel_values": list(backbone["pixel_values"].shape),
        "image_grid_thw": list(backbone["image_grid_thw"].shape),
        "state": list(action["state"].shape),
        "embodiment_id": list(action["embodiment_id"].shape),
    }
    expected = {
        "input_ids": [8, 208],
        "pixel_values": [6144, 1536],
        "image_grid_thw": [24, 3],
        "state": [8, 1, 132],
        "embodiment_id": [8],
    }
    if shapes != expected:
        raise RuntimeError(f"live true-B8 contract changed: {shapes} != {expected}")
    return shapes


class _CudaStageTimer:
    def __init__(self, target: Any) -> None:
        self.target = target
        self.events: list[tuple[Any, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = self.target(*args, **kwargs)
        end.record()
        self.events.append((start, end))
        return result

    def take_one(self) -> float:
        if len(self.events) != 1:
            raise RuntimeError(f"expected one stage event, found {len(self.events)}")
        start, end = self.events.pop()
        end.synchronize()
        return float(start.elapsed_time(end))


def _capture_first_policy_call(policy: Any, observation: dict) -> tuple[dict, dict]:
    original = policy.model.prepare_input
    captured: dict[str, Any] = {}

    def capture(*args: Any, **kwargs: Any) -> Any:
        prepared = original(*args, **kwargs)
        captured["shapes"] = _prepared_shape_contract(prepared)
        return prepared

    policy.model.prepare_input = capture
    try:
        policy.check_observation(observation)
        actions, _ = policy._get_action(observation)
    finally:
        policy.model.prepare_input = original
    return actions, captured


def _load_tools(rlinf_source: Path) -> dict[str, Any]:
    path = rlinf_source / "toolkits/eos/gr00t_trocar/tensorrt"
    sys.path.insert(0, os.fspath(path))
    from benchmark_precision_b8 import (  # noqa: PLC0415
        _load_policy,
        _load_refittable_runtime,
        _setup_trt,
        _sha256,
    )

    return {
        "load_policy": _load_policy,
        "load_refittable_runtime": _load_refittable_runtime,
        "setup_trt": _setup_trt,
        "sha256": _sha256,
    }


def _configure_backend(
    args: argparse.Namespace,
    policy: Any,
    tools: dict[str, Any],
) -> dict[str, Any]:
    import torch

    lifecycle: dict[str, Any] = {"backend": args.backend}
    trt_api = vit_engine = llm_engine = None
    if args.backend != "eager-eager":
        started = time.monotonic_ns()
        trt_api, vit_engine, llm_engine = tools["setup_trt"](
            policy, args.gr00t_source, args.engines
        )
        lifecycle["trt_backbone_setup_ms"] = (time.monotonic_ns() - started) / 1e6
        lifecycle["trt_backbone_precision"] = {
            "vit": str(vit_engine.dtype_of("pixel_values")),
            "llm": str(llm_engine.dtype_of("inputs_embeds")),
        }

    refittable = None
    if args.backend == "trt-pt2":
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        action_model = policy.model.action_head.model
        action_model.forward = torch.compile(
            action_model.forward,
            mode=args.compile_mode,
            dynamic=False,
        )
        lifecycle["pt2"] = {"mode": args.compile_mode, "dynamic": False}
    elif args.backend == "trt-refit-trt":
        runtime = tools["load_refittable_runtime"](args.refittable_runtime_source)
        parameter_map = json.loads(args.refittable_dit_parameter_map.read_text())
        action_model = policy.model.action_head.model
        source_digest = runtime._ordered_source_digest(
            action_model, parameter_map["dit_refit"]["entries"]
        )
        import tensorrt as trt  # noqa: PLC0415

        started = time.monotonic_ns()
        refittable = runtime.RefittableTensorRTDiT(
            action_model,
            {
                "engine_path": os.fspath(args.refittable_dit_engine),
                "receipt_path": os.fspath(args.refittable_dit_receipt),
                "receipt_sha256": tools["sha256"](args.refittable_dit_receipt),
                "parameter_map_path": os.fspath(args.refittable_dit_parameter_map),
                "parameter_map_sha256": tools["sha256"](
                    args.refittable_dit_parameter_map
                ),
                "source_digest_revision_0": source_digest,
                "revision": 0,
                "runtime_version": trt.__version__,
                "runtime_distribution": "tensorrt-cu12",
                "compute_capability": list(torch.cuda.get_device_capability()),
                "online_refit": True,
                "probe_each_revision": True,
                "minimum_free_device_bytes": args.refittable_minimum_free_bytes,
                "ppo_authority_status": "failed_ratio_kl_approximate_behavior_only",
                "lineage_receipt_mode": "gpu_transform_validation",
                "shadow_eager": False,
            },
        )
        refittable.verify_revision(0)
        action_model.forward = refittable
        lifecycle["refittable_dit"] = {
            "source_digest_revision_0": source_digest,
            "setup_before_live_probe_ms": (time.monotonic_ns() - started) / 1e6,
        }
    return {
        "lifecycle": lifecycle,
        "trt_api": trt_api,
        "vit_engine": vit_engine,
        "llm_engine": llm_engine,
        "refittable": refittable,
    }


def _execute_env_chunk(env: Any, actions: Any) -> tuple[Any, Any, Any, Any, dict]:
    latest = None
    dispatch_ms = []
    for index in range(ACTION_CHUNKS):
        started = time.monotonic_ns()
        latest = env.step(actions[:, index])
        dispatch_ms.append((time.monotonic_ns() - started) / 1e6)
    assert latest is not None
    observation, reward, terminated, truncated, info = latest
    return observation, reward, terminated, truncated, {
        "physical_step_host_dispatch_ms": dispatch_ms,
        "info_keys": sorted(info),
    }


def _post_measurement_summary(
    decisions: list[dict[str, Any]],
    measured_seconds: float,
) -> dict[str, Any]:
    timing_fields = (
        "rollout_wall_ms",
        "backbone_cuda_ms",
        "action_head_cuda_ms",
        "model_cuda_ms",
        "rollout_non_model_wall_ms",
        "env_wall_ms",
        "rollout_env_union_ms",
    )
    measured = len(decisions)
    return {
        "timing": {
            field: _statistics([float(item[field]) for item in decisions])
            for field in timing_fields
        },
        "throughput": {
            "measured_wall_seconds": measured_seconds,
            "policy_calls_per_second": measured / measured_seconds,
            "policy_decisions_per_second": measured * BATCH_SIZE / measured_seconds,
            "physical_actions_per_second": (
                measured * BATCH_SIZE * ACTION_CHUNKS / measured_seconds
            ),
            "policy_decisions": measured * BATCH_SIZE,
            "physical_actions": measured * BATCH_SIZE * ACTION_CHUNKS,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    receipt: dict[str, Any] = {
        "schema": "rlinf.w04.horde-rollout-env/v1",
        "status": "pending",
        "execution_status": "pending",
        "stage": "pre_app_launch",
        "backend": args.backend,
        "task_id": TASK_ID,
        "contract": {
            "num_envs": BATCH_SIZE,
            "true_policy_batch": BATCH_SIZE,
            "cameras": list(CAMERA_ORDER),
            "language_sequence_length": 208,
            "generated_action_horizon": 40,
            "executed_action_horizon": ACTION_CHUNKS,
            "public_action_dim": PUBLIC_ACTION_DIM,
            "env_action_dim": ENV_ACTION_DIM,
            "warmup_decisions": args.warmup,
            "measured_decisions": args.measured,
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "vk_driver_files": os.environ.get("VK_DRIVER_FILES"),
        },
        "started_ns": started_ns,
    }
    _write_receipt(args.output, receipt)
    simulation_app = None
    env = None
    policy = None
    backend_state: dict[str, Any] = {}
    cleanup_errors = []
    try:
        from isaaclab.app import AppLauncher

        app_started = time.monotonic_ns()
        simulation_app = AppLauncher(headless=True, enable_cameras=True).app
        receipt["lifecycle"] = {
            "app_launch_ms": (time.monotonic_ns() - app_started) / 1e6
        }
        receipt["stage"] = "app_launched"
        receipt["packages"] = _package_versions()
        _write_receipt(args.output, receipt)

        import gymnasium as gym
        import torch

        install_offline_asset_mirror(args.asset_mirror)
        import isaaclab_tasks  # noqa: F401, PLC0415
        from isaaclab_tasks.utils import parse_env_cfg

        env_started = time.monotonic_ns()
        cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=BATCH_SIZE)
        env = gym.make(TASK_ID, cfg=cfg, render_mode="rgb_array").unwrapped
        receipt["lifecycle"]["env_create_ms"] = (
            time.monotonic_ns() - env_started
        ) / 1e6
        torch.cuda.synchronize()
        reset_started = time.monotonic_ns()
        observation, _ = env.reset()
        torch.cuda.synchronize()
        receipt["lifecycle"]["env_reset_ms"] = (
            time.monotonic_ns() - reset_started
        ) / 1e6
        receipt["memory_after_env"] = _memory_snapshot()
        receipt["stage"] = "env_ready"
        _write_receipt(args.output, receipt)

        tools = _load_tools(args.rlinf_source)
        load_started = time.monotonic_ns()
        policy = tools["load_policy"](args.gr00t_source, args.model)
        torch.cuda.synchronize()
        receipt["lifecycle"]["policy_load_ms"] = (
            time.monotonic_ns() - load_started
        ) / 1e6
        backend_state = _configure_backend(args, policy, tools)
        receipt["lifecycle"].update(backend_state["lifecycle"])
        receipt["memory_after_backend_setup"] = _memory_snapshot()
        receipt["stage"] = "backend_ready"
        _write_receipt(args.output, receipt)

        policy_observation = observation_to_policy(observation)
        first_call_started = time.monotonic_ns()
        first_actions, shape_receipt = _capture_first_policy_call(
            policy, policy_observation
        )
        torch.cuda.synchronize()
        receipt["lifecycle"]["first_policy_call_ms"] = (
            time.monotonic_ns() - first_call_started
        ) / 1e6
        receipt["live_shape_contract"] = shape_receipt["shapes"]

        refittable = backend_state.get("refittable")
        if refittable is not None:
            refit_started = time.monotonic_ns()
            refittable.verify_revision(1)
            torch.cuda.synchronize()
            receipt["lifecycle"]["refit_adopt_revision_1_ms"] = (
                time.monotonic_ns() - refit_started
            ) / 1e6
            post_refit_started = time.monotonic_ns()
            first_actions, _ = policy._get_action(policy_observation)
            torch.cuda.synchronize()
            receipt["lifecycle"]["post_refit_first_call_ms"] = (
                time.monotonic_ns() - post_refit_started
            ) / 1e6
            receipt["lifecycle"]["refittable_dit"]["before_measurement"] = (
                refittable.telemetry()
            )
        elif args.backend == "trt-pt2":
            receipt["lifecycle"]["pt2"]["unique_graphs_after_first_call"] = int(
                torch._dynamo.utils.counters["stats"]["unique_graphs"]
            )

        warmup_actions = actions_to_env(first_actions, env.device)
        for warmup_index in range(args.warmup):
            observation, *_ = _execute_env_chunk(env, warmup_actions)
            torch.cuda.synchronize()
            policy_observation = observation_to_policy(observation)
            torch.manual_seed(args.seed + warmup_index)
            torch.cuda.manual_seed_all(args.seed + warmup_index)
            warmup_actions_dict, _ = policy._get_action(policy_observation)
            warmup_actions = actions_to_env(warmup_actions_dict, env.device)
        torch.cuda.synchronize()

        backbone_timer = _CudaStageTimer(policy.model.backbone.forward)
        head_timer = _CudaStageTimer(policy.model.action_head.get_action)
        policy.model.backbone.forward = backbone_timer
        policy.model.action_head.get_action = head_timer

        initial_camera = observation["camera_images"]["front_camera"].detach().clone()
        torch.cuda.reset_peak_memory_stats()
        decisions = []
        latest_reward = latest_terminated = latest_truncated = None
        latest_actions = None
        measured_total_ms = 0.0
        for index in range(args.measured):
            decision_started = time.monotonic_ns()
            rollout_started = time.monotonic_ns()
            policy_observation = observation_to_policy(observation)
            torch.manual_seed(args.seed + index)
            torch.cuda.manual_seed_all(args.seed + index)
            action_dict, _ = policy._get_action(policy_observation)
            env_actions = actions_to_env(action_dict, env.device)
            torch.cuda.synchronize()
            rollout_ended = time.monotonic_ns()
            backbone_ms = backbone_timer.take_one()
            head_ms = head_timer.take_one()

            env_started = time.monotonic_ns()
            (
                observation,
                latest_reward,
                latest_terminated,
                latest_truncated,
                env_detail,
            ) = _execute_env_chunk(env, env_actions)
            torch.cuda.synchronize()
            env_ended = time.monotonic_ns()
            decision_ended = time.monotonic_ns()
            rollout_ms = (rollout_ended - rollout_started) / 1e6
            env_ms = (env_ended - env_started) / 1e6
            union_ms = (env_ended - rollout_started) / 1e6
            measured_total_ms += union_ms
            decisions.append(
                {
                    "index": index,
                    "decision_start_ns": decision_started,
                    "decision_end_ns": decision_ended,
                    "rollout_start_ns": rollout_started,
                    "rollout_end_ns": rollout_ended,
                    "env_start_ns": env_started,
                    "env_end_ns": env_ended,
                    "rollout_wall_ms": rollout_ms,
                    "backbone_cuda_ms": backbone_ms,
                    "action_head_cuda_ms": head_ms,
                    "model_cuda_ms": backbone_ms + head_ms,
                    "rollout_non_model_wall_ms": rollout_ms
                    - backbone_ms
                    - head_ms,
                    "env_wall_ms": env_ms,
                    "rollout_env_union_ms": union_ms,
                    "rollout_env_overlap_ms": 0.0,
                    "rollout_env_uncovered_ms": max(
                        0.0, union_ms - rollout_ms - env_ms
                    ),
                    **env_detail,
                }
            )
            latest_actions = env_actions

        # Sanity checks deliberately run after the retained interval.
        final_camera = observation["camera_images"]["front_camera"]
        sanity = {
            "actions_finite": bool(torch.isfinite(latest_actions).all().item()),
            "reward_finite": bool(torch.isfinite(latest_reward).all().item()),
            "terminated_count": int(latest_terminated.sum().item()),
            "truncated_count": int(latest_truncated.sum().item()),
            "camera_changed": bool(torch.ne(initial_camera, final_camera).any().item()),
            "final_action_shape": list(latest_actions.shape),
        }
        if not sanity["actions_finite"] or not sanity["reward_finite"]:
            raise RuntimeError(f"post-run finite check failed: {sanity}")
        if not sanity["camera_changed"]:
            raise RuntimeError("post-run environment-progress check found no camera change")

        receipt.update(
            {
                "execution_status": "passed",
                "status": "passed",
                "stage": "execution_completed",
                "decisions": decisions,
                **_post_measurement_summary(decisions, measured_total_ms / 1000),
                "sanity": sanity,
                "peak_memory": {
                    **_memory_snapshot(),
                    "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
            }
        )
        if backend_state.get("vit_engine") is not None:
            receipt["trt_backbone_telemetry"] = {
                "vit": backend_state["vit_engine"].telemetry(),
                "llm": backend_state["llm_engine"].telemetry(),
            }
        if refittable is not None:
            receipt["refittable_dit_telemetry"] = refittable.telemetry()
        if args.backend == "trt-pt2":
            receipt["lifecycle"]["pt2"]["unique_graphs_after_measurement"] = int(
                torch._dynamo.utils.counters["stats"]["unique_graphs"]
            )
        _write_receipt(args.output, receipt)
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["execution_status"] = "failed"
        receipt["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_receipt(args.output, receipt)
    finally:
        receipt["stage"] = "cleanup_started"
        _write_receipt(args.output, receipt)
        refittable = backend_state.get("refittable")
        if refittable is not None:
            try:
                refittable.close()
            except BaseException as exc:
                cleanup_errors.append(
                    {
                        "component": "refittable_dit",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        if backend_state.get("trt_api") is not None and policy is not None:
            try:
                backend_state["trt_api"].close_tensorrt_engines(policy)
            except BaseException as exc:
                cleanup_errors.append(
                    {
                        "component": "trt_backbone",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        policy = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except BaseException:
            pass
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
    if receipt.get("execution_status") == "passed" and not cleanup_errors:
        receipt["status"] = "passed"
        receipt["stage"] = "completed"
    else:
        receipt["status"] = "failed"
    _write_receipt(args.output, receipt)
    return receipt


def _path(value: str) -> Path:
    return Path(value).resolve(strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--rlinf-source", type=_path, required=True)
    parser.add_argument("--gr00t-source", type=_path, required=True)
    parser.add_argument("--model", type=_path, required=True)
    parser.add_argument("--asset-mirror", type=_path, required=True)
    parser.add_argument("--engines", type=_path)
    parser.add_argument("--refittable-runtime-source", type=_path)
    parser.add_argument("--refittable-dit-engine", type=_path)
    parser.add_argument("--refittable-dit-receipt", type=_path)
    parser.add_argument("--refittable-dit-parameter-map", type=_path)
    parser.add_argument("--refittable-minimum-free-bytes", type=int, default=4 << 30)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.measured < 1:
        parser.error("--warmup and --measured must be positive")
    if args.backend != "eager-eager" and args.engines is None:
        parser.error("TensorRT Backbone arms require --engines")
    refit_paths = (
        args.refittable_runtime_source,
        args.refittable_dit_engine,
        args.refittable_dit_receipt,
        args.refittable_dit_parameter_map,
    )
    if args.backend == "trt-refit-trt" and not all(refit_paths):
        parser.error("trt-refit-trt requires all refittable DiT artifact arguments")
    receipt = run(args)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
