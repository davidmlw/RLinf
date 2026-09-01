"""W68 Newton/MJWarp adaptation for the Isaac Lab Assemble Trocar task."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_physx.physics import PhysxCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.simulation_cfg import RenderCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.manager_based.manipulation.assemble_trocar import mdp
from isaaclab_tasks.manager_based.manipulation.assemble_trocar.g129_dex3_env_cfg import (
    G1AssembleTrocarEnvCfg,
)
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


TASK_ID = "W68-Assemble-Trocar-G129-Dex3-Newton-v0"
TIP_OFFSET_LOCAL = {
    "trocar_1": (0.0, 0.0, 0.064),
    "trocar_2": (0.0, 0.0, -0.0963866084642796),
}


def _num_substeps() -> int:
    raw = os.environ.get("W68_NEWTON_NUM_SUBSTEPS", "2")
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError("W68_NEWTON_NUM_SUBSTEPS must be an integer") from error
    if value <= 0:
        raise RuntimeError("W68_NEWTON_NUM_SUBSTEPS must be positive")
    return value


@configclass
class W68PhysicsCfg(PresetCfg):
    """Backend choices with the proven Poiesis MJWarp settings."""

    default = PhysxCfg(bounce_threshold_velocity=0.01)
    physx = default
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            # The live GR00T policy drives substantially more contacts than
            # the zero-action capacity smoke. Keep enough solver workspace to
            # avoid dropping constraints in the 8-env-per-rank workload.
            njmax=2048,
            nconmax=1024,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            ls_parallel=False,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=_num_substeps(),
        debug_mode=False,
    )


def get_trocar_tip_position(
    env: Any,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("trocar_1"),
) -> torch.Tensor:
    """Compute the tip position without requiring a live USD/Kit stage.

    The offsets are immutable asset geometry. They were extracted from the two
    frozen Healthcare USD files with USD's local-to-world transform oracle.
    """

    try:
        offset = TIP_OFFSET_LOCAL[asset_cfg.name]
    except KeyError as error:
        raise ValueError(f"invalid trocar asset: {asset_cfg.name}") from error

    obj = env.scene[asset_cfg.name]
    root_pos_w = obj.data.root_pos_w.torch
    root_quat_w = obj.data.root_quat_w.torch
    local = torch.tensor(offset, dtype=root_pos_w.dtype, device=env.device)
    local = local.unsqueeze(0).expand(env.num_envs, -1)
    return root_pos_w + quat_apply(root_quat_w, local)


def _freeze_newton_actuator_defaults(cfg: Any) -> None:
    """Remove backend-dependent USD actuator fallback values."""

    for actuator in cfg.scene.robot.actuators.values():
        if actuator.effort_limit_sim is None:
            actuator.effort_limit_sim = actuator.effort_limit
        if actuator.velocity_limit_sim is None:
            actuator.velocity_limit_sim = actuator.velocity_limit
        if actuator.armature is None:
            actuator.armature = 0.0
        if actuator.friction is None:
            actuator.friction = 0.0
        if actuator.dynamic_friction is None:
            actuator.dynamic_friction = 0.0
        if actuator.viscous_friction is None:
            actuator.viscous_friction = 0.0


@configclass
class W68NewtonTrocarEnvCfg(G1AssembleTrocarEnvCfg):
    """Kitless Newton task preserving the public Trocar policy contract."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.sim.physics = W68PhysicsCfg()
        self.sim.render = RenderCfg()

        tray_usd = os.environ.get("W68_SANITIZED_TRAY_USD")
        if not tray_usd or not Path(tray_usd).is_file():
            raise RuntimeError("W68_SANITIZED_TRAY_USD must name the frozen sanitized tray USD")
        self.scene.tray.spawn.usd_path = tray_usd

        for camera_name in (
            "front_camera",
            "left_wrist_camera",
            "right_wrist_camera",
        ):
            camera = getattr(self.scene, camera_name)
            camera.renderer_cfg = MultiBackendRendererCfg()
            camera.data_types = ["rgb"]

        # DomeLightCfg and the RTX render settings are Kit-only. The Newton
        # renderer owns its lighting path and resolves each camera via presets.
        self.scene.light = None
        self.scene.replicate_physics = True

        # RewardManager omits zero-weight terms, but update_task_stage is a
        # zero-return state transition that must run before sparse rewards.
        self.rewards.update_stage.weight = 1.0
        _freeze_newton_actuator_defaults(self)


def register_task() -> None:
    """Register the W68 task without changing Isaac Lab's upstream task ID."""

    # The original reward helper imports pxr and reads env.scene.stage. A
    # kitless Newton process has no live USD stage, so use the frozen offsets.
    mdp.rewards.get_trocar_tip_position = get_trocar_tip_position
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": f"{__name__}:W68NewtonTrocarEnvCfg"},
            disable_env_checker=True,
        )
