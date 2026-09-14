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

"""Collect the W96 NVIDIA-runtime receipt before Isaac or Ray starts."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

EXPECTED_MODULES = {
    "torch": {
        "distribution": "torch",
        "version": "2.10.0+cu128",
        "origin": (
            "/isaac-sim/kit/python/lib/python3.12/site-packages/torch/__init__.py"
        ),
    },
    "transformers": {
        "distribution": "transformers",
        "version": "4.57.3",
        "origin": "/w96-overlay/transformers/__init__.py",
    },
    "rlinf": {
        "distribution": "rlinf",
        "version": "0.2",
        "origin": "/workspace/rlinf-src/rlinf/__init__.py",
    },
    "gr00t": {
        "distribution": "gr00t",
        "version": "1.1.0",
        "origin": "/workspace/gr00t-n17/gr00t/__init__.py",
    },
    "tensorrt": {
        "distribution": "tensorrt-cu12",
        "version": "10.15.1.29",
        "origin": "/w96-trt-runtime/tensorrt/__init__.py",
    },
}
EXPECTED_PYTHONPATH = (
    "/w96-overlay:/w96-trt-runtime:/workspace/gr00t-n17:/workspace/rlinf-src"
)
EXPECTED_PYTHON_EXECUTABLE = "/isaac-sim/kit/python/bin/python3"
EXPECTED_DRIVER_LIBRARY_ROOTS = (
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/lib/x86_64-linux-gnu",
    "/lib64",
    "/usr/local/nvidia/lib",
    "/usr/local/nvidia/lib64",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _soname(path: Path) -> str | None:
    result = subprocess.run(
        ["readelf", "-d", str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    match = re.search(r"\(SONAME\).*?\[(.*?)\]", result.stdout)
    return match.group(1) if match else None


def _mapped_library(basename: str) -> Path | None:
    for line in Path("/proc/self/maps").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if not fields:
            continue
        candidate = fields[-1]
        if candidate.startswith("/") and Path(candidate).name.startswith(basename):
            return Path(os.path.realpath(candidate))
    return None


def _load_library(name: str) -> dict[str, Any]:
    try:
        ctypes.CDLL(name, mode=ctypes.RTLD_LOCAL)
    except OSError as error:
        return {"requested": name, "status": "failed", "error": str(error)}
    path = _mapped_library(name.split(".so", 1)[0] + ".so")
    if path is None:
        return {"requested": name, "status": "failed", "error": "mapped path not found"}
    return {
        "requested": name,
        "status": "passed",
        "resolved_path": str(path),
        "soname": _soname(path),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _driver_library_path_allowed(path: str) -> bool:
    resolved = Path(path).resolve(strict=True)
    return any(
        resolved == Path(root) or Path(root) in resolved.parents
        for root in EXPECTED_DRIVER_LIBRARY_ROOTS
    )


def _parse_vulkan_devices(output: str) -> list[dict[str, Any]]:
    parts = re.split(r"(?m)^GPU(\d+):\s*$", output)
    devices = []
    for offset in range(1, len(parts), 2):
        index = int(parts[offset])
        block = parts[offset + 1]

        def field(name: str) -> str | None:
            match = re.search(rf"(?m)^\s*{name}\s*=\s*(.+?)\s*$", block)
            return match.group(1) if match else None

        devices.append(
            {
                "index": index,
                "vendor_id": field("vendorID"),
                "device_type": field("deviceType"),
                "device_name": field("deviceName"),
                "driver_id": field("driverID"),
            }
        )
    return devices


def _vulkan_devices_are_expected(devices: list[dict[str, Any]]) -> bool:
    return [device["index"] for device in devices] == list(range(8)) and all(
        device["vendor_id"] == "0x10de"
        and device["device_type"] == "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"
        and device["device_name"] == "NVIDIA L20"
        and device["driver_id"] == "DRIVER_ID_NVIDIA_PROPRIETARY"
        for device in devices
    )


def _vulkan_receipt() -> dict[str, Any]:
    executable = shutil.which("vulkaninfo")
    if executable is None:
        return {"status": "failed", "error": "vulkaninfo is not installed"}
    executable = os.path.realpath(executable)
    if executable != "/usr/bin/vulkaninfo":
        return {
            "status": "failed",
            "error": f"unexpected vulkaninfo origin: {executable}",
        }
    result = subprocess.run(
        [executable, "--summary"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    devices = _parse_vulkan_devices(result.stdout)
    gate = result.returncode == 0 and _vulkan_devices_are_expected(devices)
    return {
        "status": "passed" if gate else "failed",
        "executable": executable,
        "executable_sha256": _sha256(Path(executable)),
        "command": [executable, "--summary"],
        "exit_code": result.returncode,
        "devices": devices,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _module_receipt(name: str) -> dict[str, Any]:
    module = importlib.import_module(name)
    expected = EXPECTED_MODULES[name]
    try:
        distribution = metadata.distribution(expected["distribution"])
        distribution_receipt = {
            "name": distribution.metadata["Name"],
            "version": distribution.version,
            "metadata_root": str(distribution.locate_file("")),
            "found": True,
        }
        version = distribution.version
    except metadata.PackageNotFoundError:
        distribution_receipt = {
            "name": expected["distribution"],
            "found": False,
        }
        version = getattr(module, "__version__", None)
    origin = os.path.realpath(module.__file__)
    return {
        "name": name,
        "version": version,
        "origin": origin,
        "origin_sha256": _sha256(Path(origin)),
        "distribution": distribution_receipt,
        "version_matches": version == expected["version"],
        "origin_matches": origin == expected["origin"],
    }


def run() -> dict[str, Any]:
    """Collect and validate driver, GPU, module and Vulkan identities."""
    modules = [_module_receipt(name) for name in EXPECTED_MODULES]
    import torch

    device_count = torch.cuda.device_count()
    gpus = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory": properties.total_memory,
                "uuid": str(properties.uuid),
            }
        )
    sm89_gate = device_count == 8 and all(
        gpu["name"] == "NVIDIA L20" and gpu["compute_capability"] == [8, 9]
        for gpu in gpus
    )

    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    smi_rows = [line.strip() for line in smi.stdout.splitlines() if line.strip()]
    smi_fields = [[item.strip() for item in row.split(",")] for row in smi_rows]
    smi_gate = (
        len(smi_fields) == 8
        and all(len(fields) == 5 for fields in smi_fields)
        and len({fields[4] for fields in smi_fields}) == 1
    )

    expected_icd_path = Path("/etc/vulkan/icd.d/nvidia_icd.json")
    icd_path = Path(os.environ.get("VK_DRIVER_FILES", str(expected_icd_path)))
    icd = json.loads(icd_path.read_text(encoding="utf-8"))
    icd_library = icd["ICD"]["library_path"]
    libraries = {
        "libcuda": _load_library("libcuda.so.1"),
        "nvidia_vulkan_icd_library": _load_library(icd_library),
    }
    library_gate = all(
        value["status"] == "passed"
        and value["soname"] == Path(value["requested"]).name
        and Path(value["resolved_path"]).is_absolute()
        and _driver_library_path_allowed(value["resolved_path"])
        and len(value["sha256"]) == 64
        for value in libraries.values()
    )
    icd_gate = (
        icd_path == expected_icd_path
        and icd_library == "libGLX_nvidia.so.0"
        and icd.get("ICD", {}).get("api_version") is not None
    )
    vulkan = _vulkan_receipt()
    vulkan_gate = vulkan["status"] == "passed"
    path_gate = (
        os.environ.get("PYTHONPATH") == EXPECTED_PYTHONPATH
        and os.environ.get("PYTHONNOUSERSITE") == "1"
        and os.path.realpath(sys.executable) == EXPECTED_PYTHON_EXECUTABLE
    )
    module_gate = all(
        module["version_matches"] and module["origin_matches"] for module in modules
    )
    receipt = {
        "schema": "rlinf.w96.l20-nvidia-runtime-origin/v1",
        "status": "passed"
        if all(
            (
                sm89_gate,
                smi_gate,
                library_gate,
                icd_gate,
                vulkan_gate,
                path_gate,
                module_gate,
                torch.version.cuda == "12.8",
            )
        )
        else "failed",
        "pre_isaac_pre_ray": True,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "executable_matches": (
                os.path.realpath(sys.executable) == EXPECTED_PYTHON_EXECUTABLE
            ),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        },
        "modules": modules,
        "torch_cuda": {
            "version": torch.version.cuda,
            "device_count": device_count,
            "gpus": gpus,
        },
        "nvidia_smi": {
            "rows": smi_rows,
            "single_driver_version": smi_fields[0][4] if smi_gate else None,
        },
        "driver_libraries": libraries,
        "vulkan_icd": {
            "path": str(icd_path),
            "sha256": _sha256(icd_path),
            "content": icd,
            "library": icd_library,
        },
        "vulkan_physical_devices": vulkan,
        "gates": {
            "eight_l20_sm89": sm89_gate,
            "nvidia_smi_inventory": smi_gate,
            "torch_cuda_12_8": torch.version.cuda == "12.8",
            "pythonpath": path_gate,
            "module_origins": module_gate,
            "driver_library_origins": library_gate,
            "nvidia_vulkan_icd": icd_gate,
            "vulkan_physical_devices": vulkan_gate,
        },
    }
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run()
    except Exception as error:
        receipt = {
            "schema": "rlinf.w96.l20-nvidia-runtime-origin/v1",
            "status": "failed",
            "error": str(error),
        }
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
