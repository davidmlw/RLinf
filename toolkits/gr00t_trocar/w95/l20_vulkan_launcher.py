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

"""Fail-closed W96 launcher for the qualified L20 Vulkan runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .contract import validate as validate_contract
    from .git_tree_attestation import verify_attestation
    from .tree_manifest import materialize_manifest, verify_manifest
except ImportError:
    from contract import validate as validate_contract
    from git_tree_attestation import verify_attestation
    from tree_manifest import materialize_manifest, verify_manifest

import yaml

SITE_SCHEMA = "rlinf.w96.l20-vulkan-site/v1"
RUNTIME_SCHEMA = "rlinf.w96.n1d7-l20-image-runtime/v1"
Q1_SCHEMA = "rlinf.w96.l20-vulkan-q1/v1"
Q2_SCHEMA = "rlinf.w96.l20-vulkan-q2/v1"
REQUIRED_MANIFESTS = {
    "assets-final-readonly.json",
    "config.json",
    "models-GR00T-N1.7-Cosmos.json",
    "overrides.json",
    "python-w96-overlay.json",
    "python-wheelhouse.json",
    "runtime-tensorrt-10.15.1.29.json",
    "sources-isaac-gr00t.json",
    "sources-rlinf.json",
}
EXPECTED_PYTHONPATH = (
    "/w96-overlay:/w96-trt-runtime:/workspace/gr00t-n17:/workspace/rlinf-src"
)
EXPECTED_PYTHON_EXECUTABLE = "/isaac-sim/kit/python/bin/python3"
EXPECTED_DRIVER_LIBRARY_ROOTS = [
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/lib/x86_64-linux-gnu",
    "/lib64",
    "/usr/local/nvidia/lib",
    "/usr/local/nvidia/lib64",
]
FORBIDDEN_DURABLE_PREFIXES = ("/tmp", "/dev/shm")
MINIMUM_HEADROOM_BYTES = 80 * 1024**3


class LaunchError(RuntimeError):
    """Raised before unqualified state can be launched."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaunchError(f"cannot load JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise LaunchError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    os.replace(temporary, path)


def _absolute(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise LaunchError(f"{label} must be an absolute path: {value}")
    if any(
        path == Path(prefix) or Path(prefix) in path.parents
        for prefix in FORBIDDEN_DURABLE_PREFIXES
    ):
        raise LaunchError(f"{label} uses forbidden durable storage: {value}")
    return path


def _file(value: str, label: str) -> Path:
    path = _absolute(value, label)
    if not path.is_file():
        raise LaunchError(f"{label} is not a file: {path}")
    return path


def _directory(value: str, label: str) -> Path:
    path = _absolute(value, label)
    if not path.is_dir():
        raise LaunchError(f"{label} is not a directory: {path}")
    return path


def _require_hash(path: Path, expected: str, label: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise LaunchError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )


def _manifest_set(path: Path, expected_sha256: str) -> dict[str, str]:
    _require_hash(path, expected_sha256, "manifest set")
    result = {}
    for line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (/.+)", line)
        if match is None:
            raise LaunchError(f"invalid manifest-set line: {line!r}")
        digest, raw_path = match.groups()
        manifest = _file(raw_path, "manifest-set member")
        name = manifest.name
        if name in result:
            raise LaunchError(f"duplicate manifest-set member: {name}")
        _require_hash(manifest, digest, f"manifest {name}")
        result[name] = digest
    if set(result) != REQUIRED_MANIFESTS:
        raise LaunchError(
            "manifest-set membership mismatch: "
            f"expected={sorted(REQUIRED_MANIFESTS)} observed={sorted(result)}"
        )
    return result


def _verify_image_receipt(receipt: dict[str, Any], runtime: dict[str, Any]) -> None:
    image = runtime["container_image"]
    for key in ("repo_digest", "image_id"):
        if receipt.get(key) != image[key]:
            raise LaunchError(f"canonical image receipt differs at {key}")
    hashes = receipt.get("hashes", {})
    if hashes.get("canonical_identity_sha256") != image["canonical_identity_sha256"]:
        raise LaunchError("canonical image identity hash mismatch")
    if receipt.get("architecture") != "amd64" or receipt.get("os") != "linux":
        raise LaunchError("canonical image platform must be linux/amd64")


def _verify_overlay_absence(overlay: Path, runtime: dict[str, Any]) -> None:
    config = runtime["python_overlay"]
    normalized = {
        child.name.split(".dist-info", 1)[0].lower().replace("_", "-")
        for child in overlay.iterdir()
        if child.name.endswith(".dist-info")
    }
    forbidden_distributions = {
        value.lower().replace("_", "-") for value in config["forbidden_distributions"]
    }
    present = sorted(normalized & forbidden_distributions)
    forbidden_paths = sorted(
        name
        for name in config["forbidden_top_level_paths"]
        if (overlay / name).exists()
    )
    if present or forbidden_paths:
        raise LaunchError(
            "Python overlay shadows runtime authority: "
            f"distributions={present} paths={forbidden_paths}"
        )


def validate_site(site_path: Path) -> dict[str, Any]:
    """Validate immutable inputs without starting a container or GPU work."""
    site_path = site_path.resolve(strict=True)
    site = _load(site_path)
    if site.get("schema") != SITE_SCHEMA:
        raise LaunchError(f"unsupported site schema: {site.get('schema')}")

    runtime_path = _file(site["runtime_spec"]["path"], "runtime spec")
    _require_hash(runtime_path, site["runtime_spec"]["sha256"], "runtime spec")
    runtime = _load(runtime_path)
    if runtime.get("schema") != RUNTIME_SCHEMA:
        raise LaunchError(f"unsupported runtime schema: {runtime.get('schema')}")
    if runtime.get("scope") != "w95_l20_vulkan_w88_reproduction":
        raise LaunchError("runtime spec is not the W96 L20 authority")
    if ":".join(runtime["pythonpath"]) != EXPECTED_PYTHONPATH:
        raise LaunchError("runtime spec PYTHONPATH differs from W96 contract")
    if runtime["python"].get("expected_executable") != EXPECTED_PYTHON_EXECUTABLE:
        raise LaunchError("runtime spec Python executable differs from W96 contract")
    nvidia_runtime = runtime.get("nvidia_runtime", {})
    if nvidia_runtime.get("vulkan_icd") != ("/etc/vulkan/icd.d/nvidia_icd.json"):
        raise LaunchError("runtime spec does not pin the NVIDIA Vulkan ICD")
    if nvidia_runtime.get("vulkan_implementation_soname") != ("libGLX_nvidia.so.0"):
        raise LaunchError("runtime spec does not pin the NVIDIA Vulkan SONAME")
    if nvidia_runtime.get("allowed_driver_library_roots") != (
        EXPECTED_DRIVER_LIBRARY_ROOTS
    ):
        raise LaunchError("runtime spec driver library roots differ from launcher")
    required_forbidden = {"numpy", "pandas"}
    if not required_forbidden <= set(
        runtime["python_overlay"]["forbidden_distributions"]
    ):
        raise LaunchError("runtime spec must forbid numpy and pandas distributions")
    if not required_forbidden <= set(
        runtime["python_overlay"]["forbidden_top_level_paths"]
    ):
        raise LaunchError("runtime spec must forbid numpy and pandas top-level paths")

    image_receipt_path = _file(
        site["image"]["canonical_receipt"], "canonical image receipt"
    )
    _require_hash(
        image_receipt_path,
        site["image"]["canonical_receipt_sha256"],
        "canonical image receipt",
    )
    _verify_image_receipt(_load(image_receipt_path), runtime)
    if site["image"]["reference"] != runtime["container_image"]["repo_digest"]:
        raise LaunchError("site image reference differs from runtime spec")

    manifest_set_path = _file(site["manifest_set"]["path"], "manifest set")
    manifest_hashes = _manifest_set(manifest_set_path, site["manifest_set"]["sha256"])
    root_map = site["manifest_set"]["roots"]
    if set(root_map) != REQUIRED_MANIFESTS:
        raise LaunchError("manifest root map does not cover exactly nine manifests")
    tree_receipts = {}
    for name in sorted(REQUIRED_MANIFESTS):
        manifest_path = manifest_set_path.parent / name
        root = _directory(root_map[name], f"tree root for {name}")
        errors = verify_manifest(root, _load(manifest_path))
        if errors:
            raise LaunchError(f"tree verification failed for {name}: {errors}")
        tree_receipts[name] = {
            "manifest_sha256": manifest_hashes[name],
            "root": str(root),
        }

    source = site["source"]
    source_root = _directory(source["root"], "RLinf source root")
    attestation_path = _file(source["attestation"], "Git-tree attestation")
    _require_hash(
        attestation_path, source["attestation_sha256"], "Git-tree attestation"
    )
    attestation = _load(attestation_path)
    if attestation.get("revision") != source["revision"]:
        raise LaunchError("site source revision differs from Git-tree attestation")
    source_verification = verify_attestation(source_root, attestation)
    if source_verification["status"] != "passed":
        raise LaunchError(f"source Git-tree verification failed: {source_verification}")

    inputs = site["inputs"]
    for label in (
        "gr00t_source",
        "model",
        "backbone_model",
        "python_overlay",
        "tensorrt_runtime",
        "config_root",
        "overrides_root",
        "asset_seed",
    ):
        _directory(inputs[label], f"input {label}")
    _file(inputs["resolved_config"], "resolved config")
    _file(inputs["trocar_metadata"], "Trocar metadata")
    _file(inputs["extension"], "RLinf extension")
    _file(inputs["assets_override"], "IsaacLab assets override")
    docker = _file(site["docker"]["path"], "Docker client")
    _require_hash(docker, site["docker"]["sha256"], "Docker client")
    _verify_overlay_absence(Path(inputs["python_overlay"]), runtime)

    workload = site.get("workload", {})
    if workload != {
        "profile": "absolute_correctness_b8",
        "arm": "all_off",
    }:
        raise LaunchError("Q1/Q2 site must use absolute_correctness_b8/all_off")
    resolved_config_path = Path(inputs["resolved_config"])
    resolved_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    contract_path = source_root / "toolkits/gr00t_trocar/w95/contract-v1.json"
    contract_errors = validate_contract(
        resolved_config,
        _load(contract_path),
        workload["profile"],
        workload["arm"],
    )
    if contract_errors:
        raise LaunchError(f"resolved workload config is invalid: {contract_errors}")

    exact_roots = {
        "rlinf_source": "sources-rlinf.json",
        "gr00t_source": "sources-isaac-gr00t.json",
        "python_overlay": "python-w96-overlay.json",
        "tensorrt_runtime": "runtime-tensorrt-10.15.1.29.json",
        "config_root": "config.json",
        "overrides_root": "overrides.json",
        "asset_seed": "assets-final-readonly.json",
    }
    for input_name, manifest_name in exact_roots.items():
        input_root = Path(inputs[input_name]).resolve(strict=True)
        manifest_root = Path(root_map[manifest_name]).resolve(strict=True)
        if input_root != manifest_root:
            raise LaunchError(
                f"input {input_name} is not the root verified by {manifest_name}"
            )
    if Path(inputs["rlinf_source"]).resolve() != source_root:
        raise LaunchError("RLinf mount source differs from attested source root")

    model_bundle = Path(root_map["models-GR00T-N1.7-Cosmos.json"]).resolve(strict=True)
    model_root = Path(inputs["model"]).resolve(strict=True)
    backbone_root = Path(inputs["backbone_model"]).resolve(strict=True)
    for label, path in (("model", model_root), ("backbone model", backbone_root)):
        if model_bundle not in path.parents:
            raise LaunchError(f"{label} is outside the verified model bundle")
    file_authorities = {
        "resolved_config": "config_root",
        "trocar_metadata": "overrides_root",
        "extension": "overrides_root",
        "assets_override": "overrides_root",
    }
    for file_name, root_name in file_authorities.items():
        file_path = Path(inputs[file_name]).resolve(strict=True)
        authority = Path(inputs[root_name]).resolve(strict=True)
        if authority not in file_path.parents:
            raise LaunchError(f"input {file_name} is outside its verified {root_name}")

    model_config = _load(model_root / "config.json")
    if model_config.get("model_name") != "nvidia/Cosmos-Reason2-2B":
        raise LaunchError("GR00T checkpoint does not declare the expected backbone")
    if backbone_root.name != "Cosmos-Reason2-2B":
        raise LaunchError("backbone model path must end in Cosmos-Reason2-2B")

    return {
        "schema": "rlinf.w96.l20-vulkan-static-validation/v1",
        "status": "passed",
        "site": str(site_path),
        "site_sha256": _sha256(site_path),
        "runtime_spec_sha256": _sha256(runtime_path),
        "source": source_verification,
        "manifest_set_sha256": _sha256(manifest_set_path),
        "tree_receipts": tree_receipts,
        "image_reference": site["image"]["reference"],
        "canonical_image_identity": runtime["container_image"],
        "pythonpath": EXPECTED_PYTHONPATH,
    }


def _parse_quota_line(output: str, mount: str) -> dict[str, int]:
    line = next(
        (line for line in output.splitlines() if line.strip().startswith(mount)),
        None,
    )
    if line is None:
        raise LaunchError(f"cannot parse lfs quota output for {mount}")
    fields = line.split()
    if len(fields) < 4:
        raise LaunchError(f"incomplete lfs quota line: {line}")
    try:
        used_kib, quota_kib, limit_kib = map(int, fields[1:4])
    except ValueError as error:
        raise LaunchError(f"invalid lfs quota values: {line}") from error
    return {
        "used_kib": used_kib,
        "soft_quota_kib": quota_kib,
        "hard_limit_kib": limit_kib,
    }


def _quota_receipt(mount: str) -> dict[str, Any]:
    result = subprocess.run(
        ["lfs", "quota", "-u", str(os.getuid()), mount],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    values = _parse_quota_line(result.stdout, mount)
    used_kib = values["used_kib"]
    quota_kib = values["soft_quota_kib"]
    limit_kib = values["hard_limit_kib"]
    headroom_bytes = (limit_kib - used_kib) * 1024
    receipt = {
        "mount": mount,
        "used_kib": used_kib,
        "soft_quota_kib": quota_kib,
        "hard_limit_kib": limit_kib,
        "headroom_bytes": headroom_bytes,
        "required_headroom_bytes": MINIMUM_HEADROOM_BYTES,
        "raw_stdout": result.stdout,
    }
    if headroom_bytes < MINIMUM_HEADROOM_BYTES:
        raise LaunchError(f"HOME quota headroom is below 80 GiB: {receipt}")
    return receipt


def _gpu_receipt() -> dict[str, Any]:
    inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    gpus = []
    for line in inventory.stdout.splitlines():
        index, name, compute_capability, uuid = [
            item.strip() for item in line.split(",")
        ]
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "compute_capability": compute_capability,
                "uuid": uuid,
            }
        )
    if len(gpus) != 8 or [gpu["index"] for gpu in gpus] != list(range(8)):
        raise LaunchError(f"expected exactly GPU indices 0..7: {gpus}")
    if any(gpu["name"] != "NVIDIA L20" for gpu in gpus):
        raise LaunchError(f"all GPUs must be NVIDIA L20: {gpus}")
    if any(gpu["compute_capability"] != "8.9" for gpu in gpus):
        raise LaunchError(f"all GPUs must have compute capability 8.9: {gpus}")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    active = [line for line in processes.stdout.splitlines() if line.strip()]
    if active:
        raise LaunchError(f"L20 GPUs are not idle: {active}")
    return {"gpus": gpus, "active_compute_processes": active}


def _canonical_image_matches(docker: Path, site: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [str(docker), "image", "inspect", site["image"]["reference"]],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    inspect = json.loads(result.stdout)
    if not isinstance(inspect, list) or len(inspect) != 1:
        raise LaunchError("docker image inspect must return exactly one image")
    image = inspect[0]
    canonical = _load(Path(site["image"]["canonical_receipt"]))
    checks = {
        "image_id": image.get("Id") == canonical["image_id"],
        "repo_digest": canonical["repo_digest"] in image.get("RepoDigests", []),
        "architecture": image.get("Architecture") == canonical["architecture"],
        "os": image.get("Os") == canonical["os"],
        "rootfs_layers": image.get("RootFS") == canonical["rootfs"],
        "immutable_config": image.get("Config") == canonical["config"],
    }
    if not all(checks.values()):
        raise LaunchError(f"Docker image differs from canonical identity: {checks}")
    return {"checks": checks, "raw_inspect": image}


def _copy_retained_inputs(
    site_path: Path,
    run_root: Path,
    static: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    site = _load(site_path)
    receipts = run_root / "receipts"
    receipts.mkdir(parents=True)
    shutil.copy2(site_path, receipts / "site.json")
    shutil.copy2(site["inputs"]["resolved_config"], receipts / "config.yaml")
    shutil.copy2(Path(__file__), receipts / "launcher.py")
    environment = {
        "declared_container_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": EXPECTED_PYTHONPATH,
            "NVIDIA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
            "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
        },
        "host": {
            "uid": os.getuid(),
            "gid": os.getgid(),
            "hostname": os.uname().nodename,
        },
    }
    _write(receipts / "environment.json", environment)
    inputs = {
        "schema": "rlinf.w96.l20-vulkan-inputs/v1",
        "phase": phase,
        "static_validation": static,
        "retained_files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in {
                "site": receipts / "site.json",
                "config": receipts / "config.yaml",
                "launcher": receipts / "launcher.py",
                "environment": receipts / "environment.json",
            }.items()
        },
    }
    _write(receipts / "inputs.json", inputs)
    return inputs


def _prepare_run(
    site_path: Path, run_root: Path, phase: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    static = validate_site(site_path)
    site = _load(site_path)
    run_base = _directory(site["run_root_base"], "run root base")
    run_root = _absolute(str(run_root), "run root")
    if run_base not in run_root.parents:
        raise LaunchError(f"run root must be below {run_base}: {run_root}")
    if run_root.exists():
        raise LaunchError(f"run root already exists: {run_root}")
    run_root.mkdir(parents=True)
    _copy_retained_inputs(site_path, run_root, static, phase)
    try:
        quota = _quota_receipt(site["preflight"]["quota_mount"])
        gpus = _gpu_receipt()
        image = _canonical_image_matches(Path(site["docker"]["path"]), site)
        preflight = {
            "schema": "rlinf.w96.l20-vulkan-preflight/v1",
            "status": "passed",
            "quota": quota,
            "gpu": gpus,
            "image": image["checks"],
        }
        _write(run_root / "receipts/preflight.json", preflight)
        _write(run_root / "receipts/image-inspect.json", image["raw_inspect"])
    except Exception as error:
        _write(
            run_root / "receipts/preflight.json",
            {
                "schema": "rlinf.w96.l20-vulkan-preflight/v1",
                "status": "failed",
                "error": str(error),
            },
        )
        raise

    asset_seed = Path(site["inputs"]["asset_seed"])
    manifest = _load(
        Path(site["manifest_set"]["path"]).parent / "assets-final-readonly.json"
    )
    asset_cache = run_root / "scratch/assets-cache"
    materialize_manifest(asset_seed, asset_cache, manifest)
    if verify_manifest(asset_cache, manifest):
        raise LaunchError("per-run asset-cache seed verification failed")
    for path in [asset_cache, *asset_cache.rglob("*")]:
        if not path.is_symlink():
            writable_bits = 0o700 if path.is_dir() else 0o600
            path.chmod(path.stat().st_mode | writable_bits)
    return site, static, preflight


def _container_name(run_root: Path, phase: str) -> str:
    slug = re.sub(r"[^a-z0-9.-]+", "-", run_root.name.lower()).strip("-")
    return f"w96-{phase}-{slug}"[:63]


def _inspect_confirms_absent(
    result: subprocess.CompletedProcess[str], container: str
) -> bool:
    stderr = result.stderr.lower()
    return (
        result.returncode != 0
        and container.lower() in stderr
        and ("no such object" in stderr or "no such container" in stderr)
    )


def _common_docker_args(
    site: dict[str, Any],
    run_root: Path,
    container: str,
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    inputs = site["inputs"]
    args = [
        site["docker"]["path"],
        "run",
        "--name",
        container,
        "--gpus",
        "all",
        "--network",
        "none",
        "--entrypoint",
        "/bin/bash",
        "--shm-size=64g",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--cap-add=SYS_ADMIN",
        "--cap-add=SYS_PTRACE",
        "--security-opt",
        "seccomp=unconfined",
        "-e",
        "HOME=/tmp",
        "-e",
        "PYTHONNOUSERSITE=1",
        "-e",
        f"PYTHONPATH={EXPECTED_PYTHONPATH}",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics",
        "-e",
        "VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json",
        "-v",
        f"{inputs['rlinf_source']}:/workspace/rlinf-src:ro",
        "-v",
        f"{inputs['gr00t_source']}:/workspace/gr00t-n17:ro",
        "-v",
        f"{inputs['python_overlay']}:/w96-overlay:ro",
        "-v",
        f"{inputs['tensorrt_runtime']}:/w96-trt-runtime:ro",
        "-v",
        f"{inputs['model']}:/models/GR00T-N1.7-3B:ro",
        "-v",
        f"{inputs['backbone_model']}:/w96-model-inputs/Cosmos-Reason2-2B:ro",
        "-v",
        f"{inputs['resolved_config']}:/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config/isaaclab_ppo_gr00t_assemble_trocar_prod.yaml:ro",
        "-v",
        f"{inputs['extension']}:/workspace/isaaclab/source/isaaclab_contrib/isaaclab_contrib/rl/rlinf/extension.py:ro",
        "-v",
        f"{inputs['assets_override']}:/workspace/isaaclab/source/isaaclab/isaaclab/utils/assets.py:ro",
        "-v",
        f"{run_root / 'scratch/assets-cache'}:/tmp/Assets",
        "-v",
        f"{run_root}:/w96-run",
    ]
    args.extend(extra_args)
    args.append(site["image"]["reference"])
    return args


def _run_container(
    site: dict[str, Any],
    run_root: Path,
    phase: str,
    command: str,
    extra_args: tuple[str, ...] = (),
) -> dict[str, Any]:
    docker = Path(site["docker"]["path"])
    container = _container_name(run_root, phase)
    inspect_existing = subprocess.run(
        [str(docker), "inspect", container],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if inspect_existing.returncode == 0:
        raise LaunchError(f"container name already exists: {container}")
    if not _inspect_confirms_absent(inspect_existing, container):
        raise LaunchError(
            "cannot confirm initial container-name availability: "
            f"exit={inspect_existing.returncode} stderr={inspect_existing.stderr!r}"
        )
    args = _common_docker_args(site, run_root, container, extra_args=extra_args) + [
        "-c",
        command,
    ]
    _write(
        run_root / "receipts/container-command.json",
        {"argv": args, "container": container, "phase": phase},
    )
    ownership_args = [
        str(docker),
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "/bin/chown",
        "-v",
        f"{run_root}:/w96-run",
        site["image"]["reference"],
        "-hR",
        f"{os.getuid()}:{os.getgid()}",
        "/w96-run",
    ]
    _write(
        run_root / "receipts/ownership-normalization-command.json",
        {"argv": ownership_args, "requires_gpu": False},
    )
    result = None
    run_error = None
    remove = None
    post_remove_inspect = None
    ownership = None
    ownership_audit = None
    try:
        result = subprocess.run(
            args,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        (run_root / "container.stdout").write_text(result.stdout, encoding="utf-8")
        (run_root / "container.stderr").write_text(result.stderr, encoding="utf-8")
    except BaseException as error:
        run_error = error
    finally:
        inspect_result = subprocess.run(
            [str(docker), "inspect", container],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if inspect_result.returncode == 0:
            (run_root / "receipts/container-inspect.json").write_text(
                inspect_result.stdout, encoding="utf-8"
            )
        remove = subprocess.run(
            [str(docker), "rm", "-f", container],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        post_remove_inspect = subprocess.run(
            [str(docker), "inspect", container],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        container_absent = _inspect_confirms_absent(post_remove_inspect, container)
        safe_to_normalize = remove.returncode == 0 and container_absent
        if safe_to_normalize:
            ownership = subprocess.run(
                ownership_args,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            (run_root / "ownership.stdout").write_text(
                ownership.stdout, encoding="utf-8"
            )
            (run_root / "ownership.stderr").write_text(
                ownership.stderr, encoding="utf-8"
            )
            ownership_audit = _audit_run_ownership(run_root)
        _write(
            run_root / "receipts/cleanup.json",
            {
                "container_remove_exit_code": remove.returncode,
                "container_remove_stdout": remove.stdout,
                "container_remove_stderr": remove.stderr,
                "post_remove_inspect_exit_code": post_remove_inspect.returncode,
                "post_remove_inspect_stdout": post_remove_inspect.stdout,
                "post_remove_inspect_stderr": post_remove_inspect.stderr,
                "container_absent_confirmed": container_absent,
                "ownership_normalization_attempted": safe_to_normalize,
                "ownership_exit_code": (
                    ownership.returncode if ownership is not None else None
                ),
                "ownership": ownership_audit,
            },
        )
    if remove is None or remove.returncode != 0:
        raise LaunchError(
            "workload container removal failed; ownership normalization was not run"
        )
    if post_remove_inspect is None or not _inspect_confirms_absent(
        post_remove_inspect, container
    ):
        raise LaunchError(
            "workload container absence was not confirmed; "
            "ownership normalization was not run"
        )
    if ownership is None or ownership_audit is None:
        raise LaunchError("output ownership normalization was not run")
    if ownership.returncode != 0:
        raise LaunchError(
            "container output ownership normalization failed; "
            "see retained ownership logs"
        )
    if ownership_audit["status"] != "passed":
        raise LaunchError(f"run output ownership audit failed: {ownership_audit}")
    if run_error is not None:
        raise run_error
    if result is None:
        raise LaunchError("container execution produced no result")
    if result.returncode != 0:
        raise LaunchError(
            f"{phase} container failed with {result.returncode}; see retained logs"
        )
    return {
        "container": container,
        "exit_code": result.returncode,
        "output_owner_normalized": True,
    }


def _audit_run_ownership(run_root: Path) -> dict[str, Any]:
    expected_uid = os.getuid()
    expected_gid = os.getgid()
    wrong_owner = []
    unreadable = []
    for path in [run_root, *run_root.rglob("*")]:
        metadata = path.lstat()
        relative = "." if path == run_root else path.relative_to(run_root).as_posix()
        if metadata.st_uid != expected_uid or metadata.st_gid != expected_gid:
            wrong_owner.append(relative)
        if not path.is_symlink() and not os.access(path, os.R_OK):
            unreadable.append(relative)
    return {
        "status": "passed" if not (wrong_owner or unreadable) else "failed",
        "expected_uid": expected_uid,
        "expected_gid": expected_gid,
        "wrong_owner": wrong_owner,
        "unreadable": unreadable,
    }


def run_q1(site_path: Path, run_root: Path) -> dict[str, Any]:
    """Qualify NVIDIA runtime injection before importing Isaac or Ray."""
    site, static, preflight = _prepare_run(site_path, run_root, "q1")
    command = (
        'set -euo pipefail; export LD_LIBRARY_PATH="/w96-trt-runtime/'
        'tensorrt_libs:${LD_LIBRARY_PATH:-}"; exec '
        f"{EXPECTED_PYTHON_EXECUTABLE} "
        "/workspace/rlinf-src/toolkits/gr00t_trocar/w95/l20_runtime_probe.py "
        "--output /w96-run/q1-runtime.json"
    )
    container = _run_container(site, run_root, "q1", command)
    runtime_receipt = _load(run_root / "q1-runtime.json")
    if runtime_receipt.get("status") != "passed":
        raise LaunchError("Q1 runtime probe did not pass")
    result = {
        "schema": Q1_SCHEMA,
        "status": "passed",
        "site_sha256": static["site_sha256"],
        "source_revision": static["source"]["revision"],
        "manifest_set_sha256": static["manifest_set_sha256"],
        "image_reference": static["image_reference"],
        "preflight": preflight,
        "runtime_receipt": {
            "path": str(run_root / "q1-runtime.json"),
            "sha256": _sha256(run_root / "q1-runtime.json"),
        },
        "container": container,
    }
    _write(run_root / "q1-receipt.json", result)
    return result


def _verify_q1(q1_path: Path, static: dict[str, Any]) -> dict[str, Any]:
    q1 = _load(q1_path)
    if q1.get("schema") != Q1_SCHEMA or q1.get("status") != "passed":
        raise LaunchError("Q1 receipt is not passed")
    expected = {
        "site_sha256": static["site_sha256"],
        "source_revision": static["source"]["revision"],
        "manifest_set_sha256": static["manifest_set_sha256"],
        "image_reference": static["image_reference"],
    }
    for key, value in expected.items():
        if q1.get(key) != value:
            raise LaunchError(f"Q1 receipt differs from Q2 static input at {key}")
    runtime_path = Path(q1["runtime_receipt"]["path"])
    _require_hash(runtime_path, q1["runtime_receipt"]["sha256"], "Q1 runtime receipt")
    return q1


def run_q2(site_path: Path, run_root: Path, q1_path: Path) -> dict[str, Any]:
    """Run one eager true-B8 call and one minimal Vulkan environment turn."""
    static = validate_site(site_path)
    _verify_q1(q1_path.resolve(strict=True), static)
    site, static, preflight = _prepare_run(site_path, run_root, "q2")
    command = (
        'set -euo pipefail; export LD_LIBRARY_PATH="/w96-trt-runtime/'
        'tensorrt_libs:${LD_LIBRARY_PATH:-}"; '
        f"{EXPECTED_PYTHON_EXECUTABLE} "
        "/workspace/rlinf-src/toolkits/gr00t_trocar/w95/l20_runtime_probe.py "
        "--output /w96-run/q2-runtime.json; "
        f"{EXPECTED_PYTHON_EXECUTABLE} "
        "/workspace/rlinf-src/toolkits/gr00t_trocar/w95/l20_q2_smoke.py "
        "model --model /models/GR00T-N1.7-3B "
        "--backbone-model /w96-model-inputs/Cosmos-Reason2-2B "
        "--metadata /w96-inputs/trocar/metadata.json "
        "--output /w96-run/q2-model.json; "
        f"{EXPECTED_PYTHON_EXECUTABLE} "
        "/workspace/rlinf-src/toolkits/gr00t_trocar/w95/l20_q2_smoke.py "
        "env --output /w96-run/q2-env.json"
    )
    inputs = site["inputs"]
    extra_args = (
        "-v",
        f"{inputs['trocar_metadata']}:/w96-inputs/trocar/metadata.json:ro",
        "-e",
        "W77_BACKBONE_MODEL_ROOT=/w96-model-inputs/Cosmos-Reason2-2B",
        "-e",
        "W77_TROCAR_METADATA=/w96-inputs/trocar/metadata.json",
        "-e",
        "RLINF_EXT_MODULE=toolkits.gr00t_trocar.vulkan_extension",
        "-e",
        "RLINF_CONFIG_FILE=/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/contrib/assemble_trocar/config/isaaclab_ppo_gr00t_assemble_trocar_prod.yaml",
    )
    container = _run_container(site, run_root, "q2", command, extra_args=extra_args)
    model = _load(run_root / "q2-model.json")
    env = _load(run_root / "q2-env.json")
    runtime = _load(run_root / "q2-runtime.json")
    if any(receipt.get("status") != "passed" for receipt in (runtime, model, env)):
        raise LaunchError("Q2 runtime, model or environment receipt did not pass")
    result = {
        "schema": Q2_SCHEMA,
        "status": "passed",
        "site_sha256": static["site_sha256"],
        "source_revision": static["source"]["revision"],
        "manifest_set_sha256": static["manifest_set_sha256"],
        "image_reference": static["image_reference"],
        "q1_receipt": {"path": str(q1_path), "sha256": _sha256(q1_path)},
        "preflight": preflight,
        "runtime": {
            "path": str(run_root / "q2-runtime.json"),
            "sha256": _sha256(run_root / "q2-runtime.json"),
        },
        "model": {
            "path": str(run_root / "q2-model.json"),
            "sha256": _sha256(run_root / "q2-model.json"),
        },
        "environment": {
            "path": str(run_root / "q2-env.json"),
            "sha256": _sha256(run_root / "q2-env.json"),
        },
        "container": container,
        "tensorrt_engine_used": False,
        "nsys_used": False,
    }
    _write(run_root / "q2-receipt.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--site", type=Path, required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--site", type=Path, required=True)
    plan.add_argument("--run-root", type=Path, required=True)
    plan.add_argument("--phase", choices=("q1", "q2"), required=True)
    q1 = subparsers.add_parser("q1")
    q1.add_argument("--site", type=Path, required=True)
    q1.add_argument("--run-root", type=Path, required=True)
    q2 = subparsers.add_parser("q2")
    q2.add_argument("--site", type=Path, required=True)
    q2.add_argument("--run-root", type=Path, required=True)
    q2.add_argument("--q1-receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            result = validate_site(args.site)
        elif args.command == "plan":
            static = validate_site(args.site)
            result = {
                "schema": "rlinf.w96.l20-vulkan-plan/v1",
                "status": "passed",
                "phase": args.phase,
                "run_root": str(args.run_root),
                "static_validation": static,
                "container_or_gpu_started": False,
            }
        elif args.command == "q1":
            result = run_q1(args.site, args.run_root)
        else:
            result = run_q2(args.site, args.run_root, args.q1_receipt)
    except Exception as error:
        print(f"W96 launcher failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
