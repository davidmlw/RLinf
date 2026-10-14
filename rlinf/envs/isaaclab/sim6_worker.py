# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Sim6-only process entrypoint; imports neither RLInf models nor Poiesis."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import traceback
from multiprocessing.connection import Client
from pathlib import Path

from portable_arrays import decode, encode


def main() -> None:
    """Serve the original physical-step/reset interface over private CPU IPC."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    runtime = Path(spec["runtime"])
    connection = Client(args.socket, family="AF_UNIX")
    env = simulation = None
    try:
        for name in ("home", "cache", "kit-root", "kit-data", "kit-logs"):
            (runtime / name).mkdir()
        os.environ.update(
            HOME=str(runtime / "home"), XDG_CACHE_HOME=str(runtime / "cache")
        )
        os.environ.pop("DISPLAY", None)
        lab = Path(spec["isaaclab_source"])
        sys.path[:0] = [
            str(p)
            for p in sorted(lab.iterdir())
            if (p / "pyproject.toml").exists() or (p / "setup.py").exists()
        ]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        if visible.startswith("GPU-"):
            import subprocess

            visible = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={visible}",
                    "--query-gpu=index",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
        if not visible.isdecimal():
            raise ValueError("Sim6 sidecar requires exactly one visible physical GPU")
        sys.argv = [
            sys.argv[0],
            "--portable-root",
            str(runtime / "kit-root"),
            f"--/log/file={runtime / 'kit-logs/isaacsim.log'}",
            f"--/app/tokens/logs={runtime / 'kit-logs'}",
            f"--/app/tokens/data={runtime / 'kit-data'}",
        ]
        # Serialize cold construction only. All physical stepping is parallel.
        with Path(spec["startup_lock"]).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            import isaacsim  # noqa: F401
            from isaaclab_tasks.utils import launch_simulation
            from sim6_task import resolve_task

            cfg = resolve_task(seed=spec["seed"], num_envs=spec["num_envs"])
            simulation = launch_simulation(
                cfg,
                {
                    "headless": True,
                    "multi_gpu": False,
                    "visualizer": None,
                    "visualizer_explicit": True,
                    "kit_args": f"--/renderer/activeGpu={int(visible)}",
                },
            )
            simulation.__enter__()
            import gymnasium as gym

            env = gym.make(
                "RLInf-Stack-Cube-Franka-Sim6-v1", cfg=cfg, render_mode="rgb_array"
            ).unwrapped
        connection.send({"result": {"ready": True, "pid": os.getpid()}})
        while True:
            request = connection.recv()
            method = request["method"]
            values = decode(request["kwargs"], device="cuda:0")
            if method == "reset":
                if values["env_ids"] is None:
                    result = env.reset(seed=values["seed"])
                else:
                    result = env.reset(seed=values["seed"], env_ids=values["env_ids"])
            elif method == "step":
                result = env.step(values["actions"])
            elif method == "close":
                env.close()
                env = None
                connection.send({"result": None})
                break
            else:
                raise ValueError(f"Unknown simulator command: {method}")
            # Lab3 emits xyzw. Preserve RLInf's unchanged wxyz wrapper ABI.
            observation = result[0]
            observation["policy"]["eef_quat"] = observation["policy"]["eef_quat"][
                :, [3, 0, 1, 2]
            ]
            connection.send({"result": encode(result)})
    except BaseException:
        try:
            connection.send({"error": traceback.format_exc()})
        except (OSError, EOFError):
            pass
        raise
    finally:
        connection.close()
        if env is not None:
            env.close()
        if simulation is not None:
            simulation.__exit__(None, None, None)


if __name__ == "__main__":
    main()
