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

import pytest

from toolkits.gr00t_trocar.w12 import rtx6000_qualification_launcher as launcher
from toolkits.gr00t_trocar.w12 import rtx6000_runtime_probe as probe

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "toolkits/gr00t_trocar/w12/contract-rtx6000-n1d7.json"
Q2_SMOKE = ROOT / "toolkits/gr00t_trocar/w95/l20_q2_smoke.py"


def test_contract_freezes_rtx6000_true_b8_workload() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["hardware"]["gpu_count"] == 8
    assert contract["hardware"]["gpu_name"] == probe.EXPECTED_GPU_NAME
    assert contract["hardware"]["compute_capability"] == [12, 0]
    assert contract["workload"]["profile"] == "absolute_correctness_b8"
    assert contract["workload"]["policy_batch_per_rank"] == 8
    assert contract["workload"]["num_action_chunks"] == 16
    assert contract["measurement"]["retained_steps"] == [1, 2, 3, 4]


def _gpu(index: int) -> dict:
    return {
        "index": index,
        "name": probe.EXPECTED_GPU_NAME,
        "compute_capability": probe.EXPECTED_COMPUTE_CAPABILITY,
    }


def test_gpu_inventory_accepts_exact_eight_rtx6000_devices() -> None:
    assert probe._gpu_inventory_matches([_gpu(index) for index in range(8)])


def test_gpu_inventory_rejects_wrong_compute_capability() -> None:
    gpus = [_gpu(index) for index in range(8)]
    gpus[3]["compute_capability"] = [8, 9]
    assert not probe._gpu_inventory_matches(gpus)


def test_driver_library_authorities_are_split_by_capability() -> None:
    assert probe._driver_library_path_allowed(
        "nvidia_vulkan_icd_library",
        "/w12-driver/libGLX_nvidia.so.595.58.03",
    )
    assert not probe._driver_library_path_allowed(
        "nvidia_vulkan_icd_library",
        "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.595.58.03",
    )


def test_vulkan_receipt_reinterprets_exact_rtx6000_inventory() -> None:
    receipt = {
        "status": "failed",
        "error": "Vulkan physical-device inventory does not match 8 L20s",
        "driver_files_matches": True,
        "loader_matches": True,
        "calls": {"destroy_instance": {"status": "passed"}},
        "devices": [
            {
                "index": index,
                "vendor_id": 0x10DE,
                "device_type": probe.common.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                "device_name": probe.EXPECTED_GPU_NAME,
            }
            for index in range(8)
        ],
    }
    result = probe._reinterpret_vulkan_receipt(receipt)
    assert result["status"] == "passed"
    assert "error" not in result


def test_vulkan_receipt_rejects_missing_device() -> None:
    receipt = {
        "status": "failed",
        "driver_files_matches": True,
        "loader_matches": True,
        "calls": {"destroy_instance": {"status": "passed"}},
        "devices": [],
    }
    result = probe._reinterpret_vulkan_receipt(receipt)
    assert result["status"] == "failed"
    assert result["inventory_matches"] is False


def test_q2_command_qualifies_true_b8_and_both_env_sizes() -> None:
    command = launcher._command("q2")
    assert "rtx6000_runtime_probe.py" in command
    assert " model " in command
    assert command.count(" env --num-envs ") == 2
    assert "env --num-envs 1" in command
    assert "env --num-envs 8" in command
    smoke_source = Q2_SMOKE.read_text(encoding="utf-8")
    assert "env_cfg.scene.num_envs = num_envs" in smoke_source


def test_container_args_freeze_network_image_and_gpu_contract(tmp_path: Path) -> None:
    names = (
        "source",
        "gr00t",
        "overlay",
        "tensorrt",
        "graphics",
        "model",
        "backbone",
        "config",
        "extension",
        "assets_override",
        "metadata",
    )
    inputs = {}
    for name in names:
        path = tmp_path / name
        path.mkdir()
        inputs[name] = path
    run_root = tmp_path / "run"
    (run_root / "scratch/assets-cache").mkdir(parents=True)
    argv = launcher._container_args(
        Path("/usr/bin/docker"), "fixture", run_root, inputs, "true"
    )
    assert argv[:4] == ["/usr/bin/docker", "run", "--name", "fixture"]
    assert (
        argv[argv.index("--user") + 1]
        == f"{launcher.os.getuid()}:{launcher.os.getgid()}"
    )
    assert argv[argv.index("--gpus") + 1] == "all"
    assert argv[argv.index("--network") + 1] == "none"
    assert launcher.IMAGE in argv
    assert f"PYTHONPATH={launcher.PYTHONPATH}" in argv
    assert "USER=liweim" in argv
    assert "HOME=/w12-run/scratch/home" in argv
    assert "TORCHINDUCTOR_CACHE_DIR=/w12-run/scratch/torchinductor" in argv
    assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics" in argv
    assert any(value.endswith(":/w12-run:rw") for value in argv)
    assert any(value.endswith(":/w96-run:rw") for value in argv)
    assert any(value.endswith(":/isaac-sim/kit/cache:rw") for value in argv)
    assert any(value.endswith(":/isaac-sim/kit/data:rw") for value in argv)


def test_container_absence_rejects_daemon_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(_argv, *, check=True):
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="daemon unavailable"
        )

    monkeypatch.setattr(launcher, "_run", fake_run)
    with pytest.raises(launcher.QualificationError, match="absence check failed"):
        launcher._ensure_absent(Path("/usr/bin/docker"), "fixture")


def test_require_inputs_rejects_unqualified_assets_override(tmp_path: Path) -> None:
    source = tmp_path / "source"
    bundle = tmp_path / "bundle"
    source.mkdir()
    required = (
        "sources/isaac-gr00t",
        "python/w96-overlay",
        "runtime/tensorrt-10.15.1.29",
        "model/GR00T-N1.7-3B",
        "model/Cosmos-Reason2-2B",
        "config/absolute-correctness-b8-all-off.yaml",
        "overrides/extension.py",
        "overrides/trocar-metadata.json",
        "assets/assets-cache",
    )
    for relative in required:
        path = bundle / relative
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)
    assets_override = tmp_path / "assets.py"
    assets_override.write_text("unqualified", encoding="utf-8")

    with pytest.raises(
        launcher.QualificationError, match="offline assets override SHA256"
    ):
        launcher._require_inputs(source, bundle, assets_override)


def _env_receipt(num_envs: int, status: str = "pending_cleanup") -> dict:
    tensors = {
        f"camera_images.{camera}": {
            "shape": [num_envs, 224, 224, 3],
            "dtype": "torch.uint8",
            "device": "cuda:0",
            "finite": True,
        }
        for camera in launcher.EXPECTED_CAMERAS
    }
    return {
        "schema": launcher.EXPECTED_ENV_SCHEMA,
        "status": status,
        "phase": "env",
        "task_id": launcher.EXPECTED_ENV_TASK,
        "num_envs": num_envs,
        "steps": 1,
        "renderer": "Vulkan/RTX",
        "physics": "PhysX",
        "ray_started": False,
        "isaac_launcher_contract": {"status": "passed"},
        "reset_tensor_shapes": tensors,
        "step_tensor_shapes": tensors,
        "camera_paths": {
            camera: [f"camera_images.{camera}"] for camera in launcher.EXPECTED_CAMERAS
        },
        "reward": {"finite": True, "shape": [num_envs]},
        "terminated": [False] * num_envs,
        "truncated": [False] * num_envs,
        "cleanup": {"status": "pending" if status == "pending_cleanup" else "passed"},
    }


@pytest.mark.parametrize("num_envs", [1, 8])
def test_env_receipt_accepts_scoped_native_close_observability(num_envs: int) -> None:
    result = launcher._env_receipt_gate(_env_receipt(num_envs), num_envs)
    assert result["status"] == "passed"
    assert result["cleanup_observability"] == "native_close_pending_outer_cleanup"


def test_env_receipt_rejects_missing_camera() -> None:
    receipt = _env_receipt(8)
    del receipt["step_tensor_shapes"]["camera_images.front_camera"]
    assert launcher._env_receipt_gate(receipt, 8)["status"] == "failed"


def test_env_receipt_rejects_primary_error() -> None:
    receipt = _env_receipt(1)
    receipt["primary_error"] = {"message": "fixture"}
    assert launcher._env_receipt_gate(receipt, 1)["status"] == "failed"
