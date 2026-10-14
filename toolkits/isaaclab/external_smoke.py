# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""One real task/camera boundary check without Policy, Trainer or PPO."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from rlinf.envs.isaaclab.external_process import ExternalIsaacLabEnv


def main() -> None:
    """Run eight environments and sixteen neutral physical steps."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-python", required=True)
    parser.add_argument("--isaaclab-source", required=True)
    args = parser.parse_args()
    root = Path(os.environ["WORKSPACE_ATTEMPT_ROOT"])
    env = ExternalIsaacLabEnv(
        {
            "python": args.sim_python,
            "isaaclab_source": args.isaaclab_source,
            "startup_lock": str(root / "control/sim-startup.lock"),
        },
        num_envs=8,
        seed=0,
    )
    try:
        observation, _ = env.reset(seed=0)
        actions = torch.zeros((8, 7), device="cuda:0")
        actions[:, -1] = 1
        for _ in range(16):
            observation, reward, terminated, truncated, _ = env.step(actions)
        policy = observation["policy"]
        finite = all(
            bool(torch.isfinite(policy[name]).all())
            for name in ("eef_pos", "eef_quat", "gripper_pos")
        )
        finite = finite and bool(torch.isfinite(reward).all())
        cameras = {}
        for name in ("table_cam", "wrist_cam"):
            pixels = policy[name]
            cameras[name] = {
                "shape": list(pixels.shape),
                "min": float(pixels.min()),
                "max": float(pixels.max()),
                "std": float(pixels.float().std()),
            }
            from PIL import Image

            Image.fromarray(pixels[0].cpu().numpy()).save(
                root / "results" / f"{name}.png"
            )
        passed = finite and all(
            x["shape"] == [8, 256, 256, 3] and x["std"] > 0 for x in cameras.values()
        )
        receipt = {
            "passed": passed,
            "physical_steps": 16,
            "environments": 8,
            "torch": torch.__version__,
            "cameras": cameras,
            "state_finite": finite,
        }
        (root / "results/task-camera.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        if not passed:
            raise RuntimeError("The external simulator task/camera gate failed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
