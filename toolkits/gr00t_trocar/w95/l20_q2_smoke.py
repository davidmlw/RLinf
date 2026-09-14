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
import importlib
import importlib.metadata
import importlib.util
import json
import os
import site
import sys
import traceback
from pathlib import Path
from typing import Any

CAMERAS = ("front_camera", "left_wrist_camera", "right_wrist_camera")
TASK_ID = "IsaacContrib-Assemble-Trocar-G129-Dex3"
ISAAC_LAUNCHER_ENVIRONMENT = {
    "ISAAC_PATH": "/isaac-sim",
    "EXP_PATH": "/isaac-sim/apps",
    "CARB_APP_PATH": "/isaac-sim/kit",
}
ISAAC_RENDERING_EXPERIENCE = Path(
    "/workspace/isaaclab/apps/isaaclab.python.headless.rendering.kit"
)
ISAAC_PYTHON_WRAPPER = Path("/isaac-sim/python.sh")
ISAAC_SETUP_SCRIPT = Path("/isaac-sim/setup_python_env.sh")
ISAAC_VERSION_AUTHORITY = Path("/isaac-sim/VERSION")
EXPECTED_KIT_PYTHON = Path("/isaac-sim/kit/python/bin/python3")
EXPECTED_TORCH_ORIGIN = Path(
    "/isaac-sim/kit/python/lib/python3.12/site-packages/torch/__init__.py"
)
EXPECTED_TORCH_ORIGIN_SHA256 = (
    "51c90fe34a7cf869517d1bab4cd114498790e169ec668f1a154c35ec64117a5e"
)
EXPECTED_TORCH_VERSION = "2.10.0+cu128"
FROZEN_W96_PYTHON_PREFIXES = (
    "/w96-overlay",
    "/w96-trt-runtime",
    "/workspace/gr00t-n17",
    "/workspace/rlinf-src",
)
MODULE_AUTHORITIES = {
    "isaacsim": ("/isaac-sim",),
    "isaacsim.simulation_app": ("/isaac-sim",),
    "isaaclab": ("/workspace/isaaclab/source/isaaclab",),
    "isaaclab_tasks": ("/workspace/isaaclab/source/isaaclab_tasks",),
}


class _EnvSmokeError(RuntimeError):
    def __init__(self, message: str, launcher_contract: dict[str, Any]) -> None:
        super().__init__(message)
        self.launcher_contract = launcher_contract


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_receipt(path: Path) -> dict[str, Any]:
    receipt: dict[str, Any] = {"literal": str(path)}
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        receipt.update(status="failed", resolved=None, error=str(error))
        return receipt
    is_file = resolved.is_file()
    readable = os.access(resolved, os.R_OK)
    receipt.update(
        status="passed" if is_file and readable else "failed",
        resolved=str(resolved),
        is_file=is_file,
        readable=readable,
        size=resolved.stat().st_size if is_file else None,
        sha256=(_sha256_bytes(resolved.read_bytes()) if is_file and readable else None),
    )
    return receipt


def _under_authority(path: Path, authorities: tuple[str, ...]) -> bool:
    return any(path == Path(root) or path.is_relative_to(root) for root in authorities)


def _origin_receipt(
    name: str, literal: str | None, *, resolution: str
) -> dict[str, Any]:
    exact_origin = EXPECTED_TORCH_ORIGIN if name == "torch" else None
    authorities = MODULE_AUTHORITIES.get(name, ())
    receipt: dict[str, Any] = {
        "module": name,
        "literal": literal,
        "resolution": resolution,
        "expected_authorities": list(authorities),
        "expected_exact_origin": str(exact_origin) if exact_origin else None,
    }
    if not literal:
        receipt.update(status="failed", resolved=None, authority_matches=False)
        return receipt
    try:
        resolved = Path(literal).resolve(strict=True)
    except OSError as error:
        receipt.update(
            status="failed",
            resolved=None,
            authority_matches=False,
            error=str(error),
        )
        return receipt
    authority_matches = (
        resolved == exact_origin.resolve(strict=True)
        if exact_origin is not None
        else _under_authority(resolved, authorities)
    )
    origin_sha256 = _sha256_bytes(resolved.read_bytes())
    expected_sha256 = EXPECTED_TORCH_ORIGIN_SHA256 if name == "torch" else None
    hash_matches = expected_sha256 is None or origin_sha256 == expected_sha256
    receipt.update(
        status="passed" if authority_matches and hash_matches else "failed",
        resolved=str(resolved),
        authority_matches=authority_matches,
        origin_sha256=origin_sha256,
        expected_sha256=expected_sha256,
        hash_matches=hash_matches,
    )
    return receipt


def _module_spec_receipt(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    return _origin_receipt(
        name, spec.origin if spec is not None else None, resolution="find_spec"
    )


def _imported_module_receipt(name: str) -> dict[str, Any]:
    module = importlib.import_module(name)
    return _origin_receipt(name, getattr(module, "__file__", None), resolution="import")


def _python_paths_receipt() -> dict[str, Any]:
    pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_entries = [entry for entry in pythonpath.split(os.pathsep) if entry]
    missing_frozen_prefixes = [
        prefix
        for prefix in FROZEN_W96_PYTHON_PREFIXES
        if prefix not in pythonpath_entries
    ]
    foreign_pythonpath_entries = [
        entry
        for entry in pythonpath_entries
        if not _under_authority(
            Path(entry), (*FROZEN_W96_PYTHON_PREFIXES, "/isaac-sim")
        )
    ]
    user_site = site.getusersitepackages()
    forbidden_sys_path_entries = [
        entry
        for entry in sys.path
        if entry
        and (
            Path(entry) == Path("/tmp")
            or Path(entry).is_relative_to("/tmp")
            or Path(entry) == Path(user_site)
            or Path(entry).is_relative_to(user_site)
        )
    ]
    executable = Path(sys.executable)
    expected_executable = EXPECTED_KIT_PYTHON.resolve(strict=True)
    executable_resolved = executable.resolve(strict=True)
    same_executable = executable.samefile(EXPECTED_KIT_PYTHON)
    return {
        "status": "passed"
        if (
            same_executable
            and not missing_frozen_prefixes
            and not foreign_pythonpath_entries
            and not forbidden_sys_path_entries
            and site.ENABLE_USER_SITE is False
        )
        else "failed",
        "executable_literal": sys.executable,
        "executable_resolved": str(executable_resolved),
        "expected_executable_literal": str(EXPECTED_KIT_PYTHON),
        "expected_executable_resolved": str(expected_executable),
        "same_executable": same_executable,
        "pythonpath": pythonpath,
        "pythonpath_entries": pythonpath_entries,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "sys_path": list(sys.path),
        "user_site": user_site,
        "user_site_enabled": site.ENABLE_USER_SITE,
        "missing_frozen_prefixes": missing_frozen_prefixes,
        "foreign_pythonpath_entries": foreign_pythonpath_entries,
        "forbidden_sys_path_entries": forbidden_sys_path_entries,
    }


def _torch_distribution_receipt() -> dict[str, Any]:
    version = importlib.metadata.version("torch")
    return {
        "name": "torch",
        "version": version,
        "expected_version": EXPECTED_TORCH_VERSION,
        "version_matches": version == EXPECTED_TORCH_VERSION,
        "status": "passed" if version == EXPECTED_TORCH_VERSION else "failed",
    }


def _bootstrap_failure(
    receipt: dict[str, Any], stage: str, error: Exception | str
) -> dict[str, Any]:
    receipt.update(
        status="failed",
        error_stage=stage,
        error=str(error),
        error_type=type(error).__name__ if isinstance(error, Exception) else None,
    )
    return receipt


def run_bootstrap(_args: argparse.Namespace) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": "rlinf.w96.q2-isaac-bootstrap/v1",
        "status": "failed",
        "app_created": False,
    }
    try:
        receipt["launcher_contract"] = _isaac_launcher_contract_receipt()
        receipt["files"] = {
            "python_wrapper": _file_receipt(ISAAC_PYTHON_WRAPPER),
            "setup_script": _file_receipt(ISAAC_SETUP_SCRIPT),
            "version_authority": _file_receipt(ISAAC_VERSION_AUTHORITY),
        }
        receipt["paths"] = _python_paths_receipt()
    except Exception as error:
        return _bootstrap_failure(receipt, "static_contract", error)

    specs: dict[str, Any] = {}
    receipt["module_specs"] = specs
    for name in ("isaacsim", "isaaclab", "isaaclab_tasks", "torch"):
        try:
            specs[name] = _module_spec_receipt(name)
        except Exception as error:
            specs[name] = {"module": name, "status": "failed", "error": str(error)}
            return _bootstrap_failure(receipt, f"module_spec:{name}", error)
    try:
        receipt["torch_distribution"] = _torch_distribution_receipt()
    except Exception as error:
        return _bootstrap_failure(receipt, "torch_distribution", error)

    static_gate = (
        receipt["launcher_contract"]["status"] == "passed"
        and all(item["status"] == "passed" for item in receipt["files"].values())
        and receipt["paths"]["status"] == "passed"
        and all(item["status"] == "passed" for item in specs.values())
        and receipt["torch_distribution"]["status"] == "passed"
    )
    if not static_gate:
        return _bootstrap_failure(receipt, "static_gate", "static gate did not pass")

    imported: dict[str, Any] = {}
    receipt["imported_modules"] = imported
    for name in ("isaacsim", "torch"):
        try:
            imported[name] = _imported_module_receipt(name)
        except Exception as error:
            imported[name] = {"module": name, "status": "failed", "error": str(error)}
            return _bootstrap_failure(receipt, f"module_import:{name}", error)
    imported_torch_version = getattr(
        importlib.import_module("torch"), "__version__", None
    )
    receipt["torch_imported_version"] = {
        "version": imported_torch_version,
        "expected_version": EXPECTED_TORCH_VERSION,
        "version_matches": imported_torch_version == EXPECTED_TORCH_VERSION,
    }
    try:
        specs["isaacsim.simulation_app"] = _module_spec_receipt(
            "isaacsim.simulation_app"
        )
        imported["isaacsim.simulation_app"] = _imported_module_receipt(
            "isaacsim.simulation_app"
        )
        simulation_app = importlib.import_module(
            "isaacsim.simulation_app"
        ).SimulationApp
        receipt["simulation_app_class_module"] = simulation_app.__module__
    except Exception as error:
        return _bootstrap_failure(receipt, "simulation_app_import", error)

    imported_gate = all(
        item["status"] == "passed" for item in imported.values()
    ) and all(
        imported[name].get("resolved") == specs[name].get("resolved")
        for name in imported
    )
    gate = (
        imported_gate
        and receipt["torch_imported_version"]["version_matches"]
        and receipt["simulation_app_class_module"].startswith("isaacsim.")
    )
    receipt["status"] = "passed" if gate else "failed"
    if not gate:
        receipt["error_stage"] = "imported_gate"
        receipt["error"] = "imported module gate did not pass"
    return receipt


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


def _isaac_launcher_contract_receipt() -> dict[str, Any]:
    environment = {}
    for name, expected in ISAAC_LAUNCHER_ENVIRONMENT.items():
        observed = os.environ.get(name)
        entry: dict[str, Any] = {
            "expected_literal": expected,
            "observed_literal": observed,
            "literal_matches": observed == expected,
        }
        try:
            resolved = Path(observed).resolve(strict=True) if observed else None
        except OSError as error:
            entry.update(
                status="failed",
                resolved_path=None,
                is_directory=False,
                readable_and_searchable=False,
                error=str(error),
            )
        else:
            is_directory = resolved is not None and resolved.is_dir()
            readable = resolved is not None and os.access(resolved, os.R_OK | os.X_OK)
            passed = entry["literal_matches"] and is_directory and readable
            entry.update(
                status="passed" if passed else "failed",
                resolved_path=str(resolved) if resolved is not None else None,
                is_directory=is_directory,
                readable_and_searchable=readable,
            )
        environment[name] = entry

    experience: dict[str, Any] = {"expected_literal": str(ISAAC_RENDERING_EXPERIENCE)}
    try:
        resolved_experience = ISAAC_RENDERING_EXPERIENCE.resolve(strict=True)
    except OSError as error:
        experience.update(
            status="failed",
            resolved_path=None,
            is_file=False,
            readable=False,
            error=str(error),
        )
    else:
        is_file = resolved_experience.is_file()
        readable = os.access(resolved_experience, os.R_OK)
        experience.update(
            status="passed" if is_file and readable else "failed",
            resolved_path=str(resolved_experience),
            is_file=is_file,
            readable=readable,
            size=resolved_experience.stat().st_size if is_file else None,
            sha256=(
                _sha256_bytes(resolved_experience.read_bytes())
                if is_file and readable
                else None
            ),
        )

    gate = all(item["status"] == "passed" for item in environment.values()) and (
        experience["status"] == "passed"
    )
    return {
        "status": "passed" if gate else "failed",
        "environment": environment,
        "rendering_experience": experience,
        "setup_conda_env_sourced": False,
    }


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
    launcher_contract = _isaac_launcher_contract_receipt()
    if launcher_contract["status"] != "passed":
        return {
            "schema": "rlinf.w96.q2-smoke-failure/v1",
            "status": "failed",
            "phase": "env",
            "error": "Isaac launcher environment contract did not pass",
            "isaac_launcher_contract": launcher_contract,
        }
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
            "isaac_launcher_contract": launcher_contract,
            "reset_tensor_shapes": reset_tensors,
            "step_tensor_shapes": step_tensors,
            "camera_paths": camera_paths,
            "reward": _array_receipt(reward.detach().cpu().numpy()),
            "terminated": terminated.detach().cpu().tolist(),
            "truncated": truncated.detach().cpu().tolist(),
            "reset_info_type": type(reset_info).__name__,
            "step_info_type": type(step_info).__name__,
        }
    except Exception as error:
        raise _EnvSmokeError(str(error), launcher_contract) from error
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
    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "model":
            receipt = run_model(args)
        elif args.command == "bootstrap":
            receipt = run_bootstrap(args)
        else:
            receipt = run_env(args)
    except Exception as error:
        traceback.print_exc()
        receipt = {
            "schema": "rlinf.w96.q2-smoke-failure/v1",
            "status": "failed",
            "phase": args.command,
            "error": str(error),
        }
        if isinstance(error, _EnvSmokeError):
            receipt["isaac_launcher_contract"] = error.launcher_contract
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
