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
VK_SUCCESS = 0
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
VK_API_VERSION_1_0 = 1 << 22
VK_PHYSICAL_DEVICE_PROPERTIES_BUFFER_SIZE = 4096


class _VkApplicationInfo(ctypes.Structure):
    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    )


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(_VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.POINTER(ctypes.c_char_p)),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p)),
    )


class _VkPhysicalDevicePropertiesPrefix(ctypes.Structure):
    _fields_ = (
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_uint32),
        ("deviceName", ctypes.c_char * 256),
        ("pipelineCacheUUID", ctypes.c_ubyte * 16),
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


class _CtypesVulkanApi:
    """Minimal Vulkan 1.0 loader binding used before Isaac or Ray starts."""

    def __init__(self) -> None:
        self._library = ctypes.CDLL("libvulkan.so.1", mode=ctypes.RTLD_LOCAL)
        path = _mapped_library("libvulkan.so")
        if path is None:
            raise RuntimeError("mapped libvulkan.so.1 path was not found")
        self.loader_receipt = {
            "requested": "libvulkan.so.1",
            "resolved_path": str(path),
            "soname": _soname(path),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }

        self._create_instance = self._library.vkCreateInstance
        self._create_instance.argtypes = (
            ctypes.POINTER(_VkInstanceCreateInfo),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        self._create_instance.restype = ctypes.c_int32
        self._enumerate_physical_devices = self._library.vkEnumeratePhysicalDevices
        self._enumerate_physical_devices.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_void_p),
        )
        self._enumerate_physical_devices.restype = ctypes.c_int32
        self._get_physical_device_properties = (
            self._library.vkGetPhysicalDeviceProperties
        )
        self._get_physical_device_properties.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        self._get_physical_device_properties.restype = None
        self._destroy_instance = self._library.vkDestroyInstance
        self._destroy_instance.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self._destroy_instance.restype = None

    def create_instance(self) -> tuple[int, ctypes.c_void_p]:
        application = _VkApplicationInfo(
            sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext=None,
            pApplicationName=b"rlinf-w96-q1",
            applicationVersion=1,
            pEngineName=b"rlinf-w96-q1",
            engineVersion=1,
            apiVersion=VK_API_VERSION_1_0,
        )
        create_info = _VkInstanceCreateInfo(
            sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pNext=None,
            flags=0,
            pApplicationInfo=ctypes.pointer(application),
            enabledLayerCount=0,
            ppEnabledLayerNames=None,
            enabledExtensionCount=0,
            ppEnabledExtensionNames=None,
        )
        instance = ctypes.c_void_p()
        return_code = self._create_instance(
            ctypes.byref(create_info), None, ctypes.byref(instance)
        )
        return int(return_code), instance

    def enumerate_count(self, instance: ctypes.c_void_p) -> tuple[int, int]:
        count = ctypes.c_uint32()
        return_code = self._enumerate_physical_devices(
            instance, ctypes.byref(count), None
        )
        return int(return_code), count.value

    def enumerate_devices(
        self, instance: ctypes.c_void_p, count: int
    ) -> tuple[int, list[ctypes.c_void_p], int]:
        devices = (ctypes.c_void_p * count)()
        returned_count = ctypes.c_uint32(count)
        return_code = self._enumerate_physical_devices(
            instance, ctypes.byref(returned_count), devices
        )
        bounded_count = min(returned_count.value, count)
        return (
            int(return_code),
            [ctypes.c_void_p(devices[index]) for index in range(bounded_count)],
            returned_count.value,
        )

    def get_properties(self, device: ctypes.c_void_p) -> dict[str, Any]:
        # The full Vulkan 1.0 structure has large nested limits and sparse-property
        # fields. A generously sized buffer receives it; only its stable prefix is read.
        buffer = ctypes.create_string_buffer(
            VK_PHYSICAL_DEVICE_PROPERTIES_BUFFER_SIZE
        )
        self._get_physical_device_properties(
            device, ctypes.cast(buffer, ctypes.c_void_p)
        )
        properties = _VkPhysicalDevicePropertiesPrefix.from_buffer(buffer)
        name = bytes(properties.deviceName).split(b"\0", 1)[0].decode(
            "utf-8", errors="strict"
        )
        return {
            "api_version": properties.apiVersion,
            "driver_version": properties.driverVersion,
            "vendor_id": properties.vendorID,
            "device_id": properties.deviceID,
            "device_type": properties.deviceType,
            "device_name": name,
            "pipeline_cache_uuid": bytes(properties.pipelineCacheUUID).hex(),
        }

    def destroy_instance(self, instance: ctypes.c_void_p) -> None:
        self._destroy_instance(instance, None)


def _vulkan_devices_are_expected(devices: list[dict[str, Any]]) -> bool:
    return [device["index"] for device in devices] == list(range(8)) and all(
        device["vendor_id"] == 0x10DE
        and device["device_type"] == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
        and device["device_name"] == "NVIDIA L20"
        for device in devices
    )


def _vulkan_receipt(api: Any | None = None) -> dict[str, Any]:
    """Enumerate physical devices through the Vulkan loader and always clean up."""
    calls: dict[str, Any] = {}
    devices: list[dict[str, Any]] = []
    instance: Any | None = None
    error: str | None = None
    loader: dict[str, Any] = {}
    try:
        api = api or _CtypesVulkanApi()
        loader = api.loader_receipt
        create_return, instance = api.create_instance()
        calls["create_instance"] = {
            "return_code": create_return,
            "instance_created": bool(instance),
        }
        if create_return != VK_SUCCESS or not instance:
            raise RuntimeError(
                "vkCreateInstance failed: "
                f"return_code={create_return}, instance_created={bool(instance)}"
            )

        count_return, count = api.enumerate_count(instance)
        calls["enumerate_count"] = {
            "return_code": count_return,
            "count": count,
        }
        if count_return != VK_SUCCESS or not 0 < count <= 64:
            raise RuntimeError(
                "vkEnumeratePhysicalDevices count failed: "
                f"return_code={count_return}, count={count}"
            )

        enumerate_return, handles, returned_count = api.enumerate_devices(
            instance, count
        )
        calls["enumerate_devices"] = {
            "return_code": enumerate_return,
            "requested_count": count,
            "returned_count": returned_count,
            "handle_count": len(handles),
        }
        if (
            enumerate_return != VK_SUCCESS
            or returned_count != count
            or len(handles) != count
            or not all(bool(handle) for handle in handles)
        ):
            raise RuntimeError(
                "vkEnumeratePhysicalDevices list failed: "
                f"return_code={enumerate_return}, requested_count={count}, "
                f"returned_count={returned_count}, handle_count={len(handles)}"
            )

        property_calls = []
        calls["get_properties"] = property_calls
        for index, handle in enumerate(handles):
            try:
                properties = api.get_properties(handle)
            except Exception as property_error:
                property_calls.append(
                    {
                        "index": index,
                        "status": "failed",
                        "return_code": None,
                        "api_return_type": "void",
                        "error": str(property_error),
                    }
                )
                raise RuntimeError(
                    f"vkGetPhysicalDeviceProperties failed for index {index}: "
                    f"{property_error}"
                ) from property_error
            property_calls.append(
                {
                    "index": index,
                    "status": "passed",
                    "return_code": None,
                    "api_return_type": "void",
                }
            )
            devices.append({"index": index, **properties})
    except Exception as probe_error:
        error = str(probe_error)
    finally:
        if api is not None and instance:
            try:
                api.destroy_instance(instance)
            except Exception as destroy_error:
                calls["destroy_instance"] = {
                    "attempted": True,
                    "status": "failed",
                    "return_code": None,
                    "api_return_type": "void",
                    "error": str(destroy_error),
                }
                error = f"{error}; {destroy_error}" if error else str(destroy_error)
            else:
                calls["destroy_instance"] = {
                    "attempted": True,
                    "status": "passed",
                    "return_code": None,
                    "api_return_type": "void",
                }
        else:
            calls["destroy_instance"] = {
                "attempted": False,
                "status": "not_applicable",
            }

    driver_files_gate = (
        os.environ.get("VK_DRIVER_FILES")
        == "/etc/vulkan/icd.d/nvidia_icd.json"
    )
    loader_gate = (
        loader.get("requested") == "libvulkan.so.1"
        and loader.get("soname") == "libvulkan.so.1"
        and Path(loader.get("resolved_path", "")).is_absolute()
        and _driver_library_path_allowed(loader["resolved_path"])
        and len(loader.get("sha256", "")) == 64
    )
    gate = (
        error is None
        and driver_files_gate
        and loader_gate
        and calls.get("destroy_instance", {}).get("status") == "passed"
        and _vulkan_devices_are_expected(devices)
    )
    receipt = {
        "status": "passed" if gate else "failed",
        "loader": loader,
        "vk_driver_files": os.environ.get("VK_DRIVER_FILES"),
        "driver_files_matches": driver_files_gate,
        "loader_matches": loader_gate,
        "expected": {
            "count": 8,
            "vendor_id": 0x10DE,
            "device_type": VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
            "device_name": "NVIDIA L20",
        },
        "calls": calls,
        "devices": devices,
    }
    if error is not None:
        receipt["error"] = error
    elif not gate:
        receipt["error"] = "Vulkan physical-device inventory does not match 8 L20s"
    return receipt


def _python_executable_receipt(
    observed: str, expected: str = EXPECTED_PYTHON_EXECUTABLE
) -> dict[str, Any]:
    observed_resolved = os.path.realpath(observed)
    expected_resolved = os.path.realpath(expected)
    literal_matches = observed == expected
    resolved_matches = observed_resolved == expected_resolved
    try:
        same_file = os.path.samefile(observed, expected)
    except OSError:
        same_file = False
    gate = literal_matches and (resolved_matches or same_file)
    return {
        "status": "passed" if gate else "failed",
        "observed_literal": observed,
        "expected_literal": expected,
        "literal_matches": literal_matches,
        "observed_resolved": observed_resolved,
        "expected_resolved": expected_resolved,
        "resolved_matches": resolved_matches,
        "same_file": same_file,
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
    python_environment_gate = (
        os.environ.get("PYTHONPATH") == EXPECTED_PYTHONPATH
        and os.environ.get("PYTHONNOUSERSITE") == "1"
    )
    python_executable = _python_executable_receipt(sys.executable)
    python_executable_gate = python_executable["status"] == "passed"
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
                python_environment_gate,
                python_executable_gate,
                module_gate,
                torch.version.cuda == "12.8",
            )
        )
        else "failed",
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
            "python_environment": python_environment_gate,
            "python_executable": python_executable_gate,
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
