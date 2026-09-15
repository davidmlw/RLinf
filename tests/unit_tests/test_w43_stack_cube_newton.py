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

import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
W43 = ROOT / "toolkits/eos/gr00t_stack_cube"


def test_w43_contract_derives_original_stack_cube_counts() -> None:
    contract = json.loads((W43 / "control-contract.json").read_text())
    workload = contract["workload"]

    assert (
        workload["global_environments"]
        * workload["rollout_epochs"]
        * workload["physical_horizon_per_epoch"]
        == workload["physical_actions_per_outer_step"]
        == 57_344
    )
    assert (
        workload["physical_actions_per_outer_step"] // workload["action_chunk"]
        == workload["policy_decisions_per_outer_step"]
        == 3_584
    )
    assert (
        workload["policy_decisions_per_outer_step"]
        // workload["global_batch"]
        * workload["ppo_epochs"]
        == workload["optimizer_updates_per_outer_step"]
        == 21
    )
    assert (
        workload["fixed_evaluation_environments"]
        * workload["fixed_evaluation_rollout_epochs"]
        == workload["fixed_evaluation_episodes"]
        == 96
    )


def test_w43_config_keeps_w85_algorithm_and_disables_optimizations() -> None:
    config = yaml.safe_load((W43 / "config-newton-control.yaml").read_text())

    assert config["cluster"]["component_placement"]["env"] == "0-7"
    assert config["env"]["train"]["total_num_envs"] == 64
    assert config["env"]["train"]["rollout_epoch"] == 2
    assert config["env"]["train"]["max_steps_per_rollout_epoch"] == 448
    assert config["env"]["train"]["auto_reset"] is False
    assert config["env"]["eval"]["auto_reset"] is False
    assert config["env"]["eval"]["total_num_envs"] == 32
    assert config["env"]["eval"]["rollout_epoch"] == 3
    assert config["env"]["train"]["seed"] == 0
    assert config["env"]["eval"]["seed"] == 42
    assert config["actor"]["global_batch_size"] == 512
    assert config["actor"]["micro_batch_size"] == 32
    assert config["actor"]["model"]["num_action_chunks"] == 16
    assert config["algorithm"]["update_epoch"] == 3
    assert config["algorithm"]["logprob_type"] == "chunk_level"
    assert config["algorithm"]["reward_type"] == "chunk_level"
    assert config["actor"]["optim"]["lr"] == 1.0e-5
    assert config["actor"]["optim"]["value_lr"] == 2.0e-5
    text = (W43 / "config-newton-control.yaml").read_text()
    for forbidden in ("feature_reuse", "tensorrt", "torch_compile", "async"):
        assert forbidden not in text.lower()


def test_w43_adapter_freezes_the_qualified_newton_camera_contract() -> None:
    task = (W43 / "w43_newton_stack_cube.py").read_text()
    extension = (W43 / "w43_rlinf_extension.py").read_text()

    assert "OvPhysxCfg()" in task
    assert "NewtonWarpRendererCfg()" in task
    assert "wrist_cam.update_latest_camera_pose = True" in task
    assert "scene.replicate_physics = True" in task
    assert "randomize_light = None" in task
    assert "import poiesis" not in task.lower()
    assert "AppLauncher" not in task
    assert "from w43_newton_stack_cube import RLINF_TASK_ID" not in extension
    assert "quaternion[:, 3:] < 0" in extension
    assert '"policy_revision": "r0"' in extension
    assert "self.update_rollout_weights()" in extension
    assert "metrics = self.evaluate()" in extension
    assert "self.rollout.set_global_step(0)" in extension
    assert "partial auto-reset" in extension
    assert "neutral_action[:, -1] = 1.0" in extension


def test_w43_shell_launcher_is_syntactically_valid() -> None:
    launcher = (W43 / "run_control.sh").read_text()
    assert "rollout_seed=64101" in launcher
    assert "rollout_seed=864101" in launcher
    assert "actor.model.value_head_init_seed=1234" in launcher
    assert 'short_tmp="/tmp/kiln/$allocation_id/' in launcher
    assert "from ray.scripts.scripts import main" in launcher
    assert "w43_rlinf_extension.register()" in launcher
    assert "runpy.run_path(entrypoint" in launcher
    assert "val_interval=1" in launcher
    subprocess.run(
        ["bash", "-n", str(W43 / "run_control.sh")],
        check=True,
    )
