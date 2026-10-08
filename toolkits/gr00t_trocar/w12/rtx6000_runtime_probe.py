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

"""Collect the W12 RTX PRO 6000 runtime receipt before Isaac or Ray starts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from toolkits.gr00t_trocar.w95 import l20_runtime_probe as common

EXPECTED_GPU_COUNT = 8
EXPECTED_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
EXPECTED_COMPUTE_CAPABILITY = [12, 0]


def _gpu_inventory_matches(gpus: list[dict[str, Any]]) -> bool:
    return [gpu["index"] for gpu in gpus] == list(range(EXPECTED_GPU_COUNT)) and all(
        gpu["name"] == EXPECTED_GPU_NAME
        and gpu["compute_capability"] == EXPECTED_COMPUTE_CAPABILITY
        for gpu in gpus
    )


def _reinterpret_vulkan_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    devices = receipt.get("devices", [])
    inventory_matches = [device.get("index") for device in devices] == list(
        range(EXPECTED_GPU_COUNT)
    ) and all(
        device.get("vendor_id") == 0x10DE
        and device.get("device_type")
        == common.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
        and device.get("device_name") == EXPECTED_GPU_NAME
        for device in devices
    )
    gate = (
        receipt.get("driver_files_matches") is True
        and receipt.get("loader_matches") is True
        and receipt.get("calls", {}).get("destroy_instance", {}).get("status")
        == "passed"
        and inventory_matches
    )
    result = dict(receipt)
    result["status"] = "passed" if gate else "failed"
    result["expected"] = {
        "count": EXPECTED_GPU_COUNT,
        "vendor_id": 0x10DE,
        "device_type": common.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
        "device_name": EXPECTED_GPU_NAME,
    }
    result["inventory_matches"] = inventory_matches
    if gate:
        result.pop("error", None)
    else:
        result["error"] = (
            "Vulkan physical-device inventory does not match "
            f"{EXPECTED_GPU_COUNT} RTX PRO 6000 devices"
        )
    return result


def run() -> dict[str, Any]:
    """Collect and validate driver, GPU, module and Vulkan identities."""
    modules = [common._module_receipt(name) for name in common.EXPECTED_MODULES]
    import torch

    gpus = []
    for index in range(torch.cuda.device_count()):
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
    gpu_gate = _gpu_inventory_matches(gpus)

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
        len(smi_fields) == EXPECTED_GPU_COUNT
        and all(len(fields) == 5 for fields in smi_fields)
        and len({fields[4] for fields in smi_fields}) == 1
        and all(fields[2] == EXPECTED_GPU_NAME for fields in smi_fields)
        and all(fields[3] == "12.0" for fields in smi_fields)
    )

    expected_icd_path = Path("/etc/vulkan/icd.d/nvidia_icd.json")
    icd_path = Path(os.environ.get("VK_DRIVER_FILES", str(expected_icd_path)))
    icd = json.loads(icd_path.read_text(encoding="utf-8"))
    icd_library = icd["ICD"]["library_path"]
    libraries = {
        "libcuda": common._load_library("libcuda.so.1"),
        "nvidia_vulkan_icd_library": common._load_library(icd_library),
    }
    library_gate = all(
        value["status"] == "passed"
        and value["soname"] == Path(value["requested"]).name
        and Path(value["resolved_path"]).is_absolute()
        and common._driver_library_path_allowed(value["resolved_path"])
        and len(value["sha256"]) == 64
        for value in libraries.values()
    )
    icd_gate = (
        icd_path == expected_icd_path
        and icd_library == "libGLX_nvidia.so.0"
        and icd.get("ICD", {}).get("api_version") is not None
    )
    vulkan = _reinterpret_vulkan_receipt(common._vulkan_receipt())
    python_environment_gate = (
        os.environ.get("PYTHONPATH") == common.EXPECTED_PYTHONPATH
        and os.environ.get("PYTHONNOUSERSITE") == "1"
    )
    python_executable = common._python_executable_receipt(sys.executable)
    module_gate = all(
        module["version_matches"] and module["origin_matches"] for module in modules
    )
    gates = {
        "eight_rtx_pro_6000_cc120": gpu_gate,
        "nvidia_smi_inventory": smi_gate,
        "torch_cuda_12_8": torch.version.cuda == "12.8",
        "python_environment": python_environment_gate,
        "python_executable": python_executable["status"] == "passed",
        "module_origins": module_gate,
        "driver_library_origins": library_gate,
        "nvidia_vulkan_icd": icd_gate,
        "vulkan_physical_devices": vulkan["status"] == "passed",
    }
    return {
        "schema": "rlinf.w12.rtx6000-nvidia-runtime-origin/v1",
        "status": "passed" if all(gates.values()) else "failed",
        "pre_isaac_pre_ray": True,
        "python": {
            "version": sys.version,
            "executable": python_executable,
            "pythonpath": os.environ.get("PYTHONPATH"),
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        },
        "modules": modules,
        "torch_cuda": {
            "version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "gpus": gpus,
        },
        "nvidia_smi": {
            "rows": smi_rows,
            "single_driver_version": smi_fields[0][4] if smi_gate else None,
        },
        "driver_libraries": libraries,
        "vulkan_icd": {
            "path": str(icd_path),
            "sha256": common._sha256(icd_path),
            "content": icd,
            "library": icd_library,
        },
        "vulkan_physical_devices": vulkan,
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run()
    except Exception as error:
        receipt = {
            "schema": "rlinf.w12.rtx6000-nvidia-runtime-origin/v1",
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
