# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Lab3/PhysX task migration, independent of either framework's PPO code."""

from __future__ import annotations

import sys
from typing import Any

TASK_ID = "RLInf-Stack-Cube-Franka-Sim6-v1"


def resolve_task(*, seed: int, num_envs: int) -> Any:
    """Select the same upstream task/assets/renderer/reward as the RTX control."""
    import gymnasium as gym
    from isaaclab.managers import RewardTermCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from isaaclab_tasks.manager_based.manipulation.stack import mdp
    from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_env_cfg import (
        FrankaCubeStackVisuomotorEnvCfg,
    )
    from isaaclab_tasks.utils import resolve_task_config

    @configclass
    class RewardsCfg:
        success = RewardTermCfg(func=mdp.cubes_stacked, weight=20.0)

    @configclass
    class StackCubeSim6Cfg(FrankaCubeStackVisuomotorEnvCfg):
        rewards: Any = RewardsCfg()

        def __post_init__(self) -> None:
            super().__post_init__()
            self.sim.render_interval = self.decimation
            for name in ("table_cam", "wrist_cam"):
                camera = getattr(self.scene, name)
                camera.height = camera.width = 256
                camera.renderer_cfg = IsaacRtxRendererCfg()

    gym.register(
        TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": StackCubeSim6Cfg},
        disable_env_checker=True,
    )
    original = sys.argv
    try:
        sys.argv = [sys.argv[0], "presets=physx"]
        cfg, _ = resolve_task_config(TASK_ID, "")
    finally:
        sys.argv = original
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.sim.device = "cuda:0"
    return cfg
