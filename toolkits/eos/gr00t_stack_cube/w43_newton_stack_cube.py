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

"""Isaac Lab 3 OvPhysX/Newton-renderer Stack Cube control task for W43.

The environment configuration is a standalone RLInf adapter derived from the
qualified Poiesis W42 task at Poiesis ``b846e4eb``.  It deliberately contains
no Poiesis runtime imports: RLInf owns process management, rollout, PPO, and
evaluation.  Keeping the environment mutations equivalent supplies the
missing RLInf/EOS/Newton quadrant without coupling the two frameworks.
"""

from __future__ import annotations

from typing import Any

import isaaclab.sim as sim_utils
import warp as wp
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.shapes.shapes import spawn_cuboid
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ovphysx import tensor_types as ovphysx_tensor_types
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_env_cfg import (
    FrankaCubeStackVisuomotorEnvCfg,
)
from pxr import Gf, Sdf, UsdGeom

RLINF_TASK_ID = "RLInf-W43-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0"
GYM_TASK_ID = "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0"
TASK_DESCRIPTION = "Stack the red block on the blue block, then stack the green block on the red block."
CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256
SUCCESS_REWARD_WEIGHT = 20.0

_END_EFFECTOR_BODY = "panda_hand"
_END_EFFECTOR_OFFSET = (0.0, 0.0, 0.1034)
_NATIVE_TABLE_SIZE = (1.0, 1.0, 0.1)
_NATIVE_TABLE_POSITION = (0.5, 0.0, -0.05)
_NATIVE_TABLE_LINEAR_COLOR = (0.005, 0.005, 0.005)
_FRANKA_ACTUATOR_VELOCITY_LIMITS = {
    "panda_shoulder": 2.175,
    "panda_forearm": 2.61,
    "panda_hand": 0.2,
}


def _spawn_native_table(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: Any,
) -> Any:
    """Spawn the Newton-visible kinematic table qualified by Poiesis W42."""

    root = spawn_cuboid(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    mesh = root.GetStage().GetPrimAtPath(f"{root.GetPath().pathString}/geometry/mesh")
    display_color = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.constant,
    )
    display_color.Set([Gf.Vec3f(*_NATIVE_TABLE_LINEAR_COLOR)])
    return root


def _native_table_cfg() -> RigidObjectCfg:
    spawn = sim_utils.CuboidCfg(
        size=_NATIVE_TABLE_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        semantic_tags=[("class", "table")],
    )
    spawn.func = _spawn_native_table
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_NATIVE_TABLE_POSITION,
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=spawn,
    )


def _reshape_ovphysx_jacobian(
    raw: Any,
    *,
    num_envs: int,
    num_bodies: int,
    num_joints: int,
) -> Any:
    expected = (num_envs, (num_bodies - 1) * 6, num_joints)
    if tuple(raw.shape) != expected:
        raise RuntimeError(
            "unexpected fixed-base OVPhysX Jacobian shape: "
            f"expected {expected}, got {tuple(raw.shape)}"
        )
    return raw.view(num_envs, num_bodies - 1, 6, num_joints)


def _shift_com_jacobian_to_link_origin(
    jacobian: Any,
    body_quat_w: Any,
    body_com_pos_b: Any,
) -> Any:
    import torch

    shifted = jacobian.clone()
    com_offset_w = math_utils.quat_apply(body_quat_w, body_com_pos_b)
    shifted[:, :3] += torch.cross(
        com_offset_w.unsqueeze(-1).expand_as(shifted[:, :3]),
        shifted[:, 3:],
        dim=1,
    )
    return shifted


class OvPhysxDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Feed OVPhysX Jacobians into Isaac Lab's existing DLS controller."""

    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(cfg, env)
        if not self._asset.is_fixed_base:
            raise RuntimeError("the W43 OVPhysX adapter requires a fixed base")
        binding = self._asset._get_binding(ovphysx_tensor_types.JACOBIAN)
        if binding is None:
            raise RuntimeError("OVPhysX ARTICULATION_JACOBIAN is unavailable")
        self._ovphysx_jacobian_binding = binding
        self._ovphysx_jacobian_buffer = wp.zeros(
            binding.shape, dtype=wp.float32, device=self.device
        )
        self._ovphysx_jacobian_tensor = wp.to_torch(self._ovphysx_jacobian_buffer)

    @property
    def jacobian_w(self) -> Any:
        self._ovphysx_jacobian_binding.read(self._ovphysx_jacobian_buffer)
        full = _reshape_ovphysx_jacobian(
            self._ovphysx_jacobian_tensor,
            num_envs=self.num_envs,
            num_bodies=self._asset.num_bodies,
            num_joints=self._asset.num_joints,
        )
        body_id = self._body_idx
        body_quat_w = self._asset.data.body_quat_w.torch[:, body_id]
        body_com_pos_b = self._asset.data.body_com_pos_b.torch[:, body_id]
        jacobian = _shift_com_jacobian_to_link_origin(
            full[:, self._jacobi_body_idx], body_quat_w, body_com_pos_b
        )
        return jacobian[:, :, self._jacobi_joint_ids]


def _end_effector_pose_world(
    env: Any,
    robot_cfg: SceneEntityCfg,
) -> tuple[Any, Any]:
    import torch

    robot = env.scene[robot_cfg.name]
    body_id = robot_cfg.body_ids[0]
    body_pos_w = robot.data.body_pos_w.torch[:, body_id]
    body_quat_xyzw = robot.data.body_quat_w.torch[:, body_id]
    offset = body_pos_w.new_tensor(_END_EFFECTOR_OFFSET).expand_as(body_pos_w)
    position = body_pos_w + math_utils.quat_apply(body_quat_xyzw, offset)
    if not bool(torch.isfinite(position).all()) or not bool(
        torch.isfinite(body_quat_xyzw).all()
    ):
        raise RuntimeError("OvPhysX end-effector pose is non-finite")
    return position, body_quat_xyzw


def ovphysx_ee_frame_pos(
    env: Any,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=_END_EFFECTOR_BODY),
) -> Any:
    position, _ = _end_effector_pose_world(env, robot_cfg)
    return position - env.scene.env_origins


def ovphysx_ee_frame_quat_wxyz(
    env: Any,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=_END_EFFECTOR_BODY),
) -> Any:
    _, quaternion_xyzw = _end_effector_pose_world(env, robot_cfg)
    return quaternion_xyzw[:, [3, 0, 1, 2]]


def ovphysx_object_grasped(
    env: Any,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
) -> Any:
    import torch

    robot = env.scene[robot_cfg.name]
    object_asset = env.scene[object_cfg.name]
    end_effector_pos, _ = _end_effector_pose_world(env, robot_cfg)
    pose_diff = torch.linalg.vector_norm(
        object_asset.data.root_pos_w.torch - end_effector_pos, dim=1
    )
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    if len(gripper_joint_ids) != 2:
        raise RuntimeError("Stack Cube requires two Franka gripper joints")
    closed = pose_diff < diff_threshold
    open_value = robot.data.joint_pos.torch.new_tensor(env.cfg.gripper_open_val)
    for joint_id in gripper_joint_ids:
        closed = torch.logical_and(
            closed,
            torch.abs(robot.data.joint_pos.torch[:, joint_id] - open_value)
            > env.cfg.gripper_threshold,
        )
    return closed


def ovphysx_object_obs(
    env: Any,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=_END_EFFECTOR_BODY),
) -> Any:
    import torch

    cube_1 = env.scene[cube_1_cfg.name]
    cube_2 = env.scene[cube_2_cfg.name]
    cube_3 = env.scene[cube_3_cfg.name]
    cube_1_pos_w = cube_1.data.root_pos_w.torch
    cube_2_pos_w = cube_2.data.root_pos_w.torch
    cube_3_pos_w = cube_3.data.root_pos_w.torch
    end_effector_pos, _ = _end_effector_pose_world(env, robot_cfg)
    return torch.cat(
        (
            cube_1_pos_w - env.scene.env_origins,
            cube_1.data.root_quat_w.torch,
            cube_2_pos_w - env.scene.env_origins,
            cube_2.data.root_quat_w.torch,
            cube_3_pos_w - env.scene.env_origins,
            cube_3.data.root_quat_w.torch,
            cube_1_pos_w - end_effector_pos,
            cube_2_pos_w - end_effector_pos,
            cube_3_pos_w - end_effector_pos,
            cube_1_pos_w - cube_2_pos_w,
            cube_2_pos_w - cube_3_pos_w,
            cube_1_pos_w - cube_3_pos_w,
        ),
        dim=1,
    )


def _configure_frame_adapter(env_cfg: Any) -> None:
    env_cfg.actions.arm_action.class_type = OvPhysxDifferentialInverseKinematicsAction
    env_cfg.scene.ee_frame = None
    env_cfg.observations.policy.eef_pos.func = ovphysx_ee_frame_pos
    env_cfg.observations.policy.eef_pos.params = {
        "robot_cfg": SceneEntityCfg("robot", body_names=_END_EFFECTOR_BODY)
    }
    env_cfg.observations.policy.eef_quat.func = ovphysx_ee_frame_quat_wxyz
    env_cfg.observations.policy.eef_quat.params = {
        "robot_cfg": SceneEntityCfg("robot", body_names=_END_EFFECTOR_BODY)
    }
    env_cfg.observations.policy.object.func = ovphysx_object_obs
    env_cfg.observations.policy.object.params = {
        "cube_1_cfg": SceneEntityCfg("cube_1"),
        "cube_2_cfg": SceneEntityCfg("cube_2"),
        "cube_3_cfg": SceneEntityCfg("cube_3"),
        "robot_cfg": SceneEntityCfg("robot", body_names=_END_EFFECTOR_BODY),
    }
    for term, object_name in (
        (env_cfg.observations.subtask_terms.grasp_1, "cube_2"),
        (env_cfg.observations.subtask_terms.grasp_2, "cube_3"),
    ):
        term.func = ovphysx_object_grasped
        term.params = {
            "robot_cfg": SceneEntityCfg("robot", body_names=_END_EFFECTOR_BODY),
            "object_cfg": SceneEntityCfg(object_name),
        }


@configclass
class StackCubeRewardsCfg:
    """Sparse reward contract inherited from RLInf W85."""

    success = RewTerm(func=mdp.cubes_stacked, weight=SUCCESS_REWARD_WEIGHT)


def build_env_cfg(*, seed: int, num_envs: int) -> Any:
    """Build the exact W42 OvPhysX/Newton-renderer environment contract."""

    env_cfg = FrankaCubeStackVisuomotorEnvCfg()
    env_cfg.seed = seed
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = "cuda:0"
    env_cfg.rewards = StackCubeRewardsCfg()
    env_cfg.scene.plane = None
    env_cfg.scene.table = _native_table_cfg()
    env_cfg.sim.physics = OvPhysxCfg()
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.scene.table_cam.height = CAMERA_HEIGHT
    env_cfg.scene.table_cam.width = CAMERA_WIDTH
    env_cfg.scene.wrist_cam.height = CAMERA_HEIGHT
    env_cfg.scene.wrist_cam.width = CAMERA_WIDTH
    env_cfg.scene.table_cam.renderer_cfg = NewtonWarpRendererCfg()
    env_cfg.scene.wrist_cam.renderer_cfg = NewtonWarpRendererCfg()
    env_cfg.scene.table_cam.data_types = ["rgb"]
    env_cfg.scene.wrist_cam.data_types = ["rgb"]
    env_cfg.scene.wrist_cam.update_latest_camera_pose = True
    env_cfg.events.randomize_light = None
    env_cfg.events.randomize_table_visual_material = None
    env_cfg.events.randomize_robot_arm_visual_texture = None
    env_cfg.scene.replicate_physics = True
    for name, velocity_limit in _FRANKA_ACTUATOR_VELOCITY_LIMITS.items():
        actuator = env_cfg.scene.robot.actuators[name]
        actuator.velocity_limit_sim = velocity_limit
        actuator.velocity_limit = velocity_limit
        actuator.effort_limit = actuator.effort_limit_sim
        if actuator.armature is None:
            actuator.armature = 0.0
        actuator.friction = 0.0
        actuator.dynamic_friction = 0.0
        actuator.viscous_friction = 0.0
    _configure_frame_adapter(env_cfg)
    return env_cfg
