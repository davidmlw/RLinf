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

"""Probe the Horde SM120 CUDA, TensorRT, and Vulkan runtime."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_GPU = "NVIDIA RTX PRO 4500 Blackwell Server Edition"
EXPECTED_COMPUTE_CAPABILITY = (12, 0)
EXPECTED_TENSORRT = "10.15.1.29"
EXPECTED_VULKAN_VENDOR = 0x10DE
VK_SUCCESS = 0
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
VK_API_VERSION_1_0 = 1 << 22


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
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _vulkan_probe() -> dict[str, Any]:
    driver_file = Path(
        os.environ.get("VK_DRIVER_FILES", "/etc/vulkan/icd.d/nvidia_icd.json")
    ).resolve(strict=True)
    library = ctypes.CDLL("libvulkan.so.1", mode=ctypes.RTLD_LOCAL)
    create_instance = library.vkCreateInstance
    create_instance.argtypes = (
        ctypes.POINTER(_VkInstanceCreateInfo),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    create_instance.restype = ctypes.c_int32
    enumerate_devices = library.vkEnumeratePhysicalDevices
    enumerate_devices.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_void_p),
    )
    enumerate_devices.restype = ctypes.c_int32
    get_properties = library.vkGetPhysicalDeviceProperties
    get_properties.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    get_properties.restype = None
    destroy_instance = library.vkDestroyInstance
    destroy_instance.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    destroy_instance.restype = None

    application = _VkApplicationInfo(
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pNext=None,
        pApplicationName=b"rlinf-w03-runtime-probe",
        applicationVersion=1,
        pEngineName=b"rlinf-w03-runtime-probe",
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
    create_code = int(create_instance(ctypes.byref(create_info), None, ctypes.byref(instance)))
    if create_code != VK_SUCCESS or not instance:
        raise RuntimeError(f"vkCreateInstance failed: {create_code}")
    try:
        count = ctypes.c_uint32()
        count_code = int(enumerate_devices(instance, ctypes.byref(count), None))
        if count_code != VK_SUCCESS or count.value != 1:
            raise RuntimeError(
                f"expected one Vulkan device, got code={count_code}, count={count.value}"
            )
        handles = (ctypes.c_void_p * count.value)()
        list_count = ctypes.c_uint32(count.value)
        list_code = int(enumerate_devices(instance, ctypes.byref(list_count), handles))
        if list_code != VK_SUCCESS or list_count.value != count.value:
            raise RuntimeError(
                f"Vulkan device enumeration failed: code={list_code}, count={list_count.value}"
            )
        devices = []
        for handle in handles:
            buffer = ctypes.create_string_buffer(4096)
            get_properties(handle, ctypes.cast(buffer, ctypes.c_void_p))
            props = _VkPhysicalDevicePropertiesPrefix.from_buffer(buffer)
            name = bytes(props.deviceName).split(b"\0", 1)[0].decode()
            devices.append(
                {
                    "api_version": props.apiVersion,
                    "driver_version": props.driverVersion,
                    "vendor_id": props.vendorID,
                    "device_id": props.deviceID,
                    "device_type": props.deviceType,
                    "device_name": name,
                    "pipeline_cache_uuid": bytes(props.pipelineCacheUUID).hex(),
                }
            )
    finally:
        destroy_instance(instance, None)

    device = devices[0]
    if (
        device["vendor_id"] != EXPECTED_VULKAN_VENDOR
        or device["device_type"] != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
        or device["device_name"] != EXPECTED_GPU
    ):
        raise RuntimeError(f"unexpected Vulkan device: {device}")
    return {
        "driver_file": str(driver_file),
        "driver_file_sha256": _sha256(driver_file),
        "devices": devices,
    }


def _tensorrt_probe(torch: Any) -> dict[str, Any]:
    import tensorrt as trt

    if trt.__version__ != EXPECTED_TENSORRT:
        raise RuntimeError(f"unexpected TensorRT version: {trt.__version__}")
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    input_tensor = network.add_input("input", trt.float32, (1, 4))
    output_tensor = network.add_identity(input_tensor).get_output(0)
    output_tensor.name = "output"
    network.mark_output(output_tensor)
    serialized = builder.build_serialized_network(
        network, builder.create_builder_config()
    )
    if serialized is None:
        raise RuntimeError("TensorRT failed to build an SM120 identity engine")
    plan = bytes(serialized)
    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the identity engine")
    context = engine.create_execution_context()
    input_value = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]], device="cuda", dtype=torch.float32
    )
    output_value = torch.empty_like(input_value)
    context.set_tensor_address("input", input_value.data_ptr())
    context.set_tensor_address("output", output_value.data_ptr())
    executed = context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    if not executed or not torch.equal(input_value, output_value):
        raise RuntimeError("TensorRT identity execution did not preserve its input")
    return {
        "version": trt.__version__,
        "distribution_version": importlib.metadata.version("tensorrt-cu12"),
        "module": str(Path(trt.__file__).resolve(strict=True)),
        "platform_has_fast_fp16": builder.platform_has_fast_fp16,
        "engine_bytes": len(plan),
        "engine_sha256": hashlib.sha256(plan).hexdigest(),
        "io_names": [
            engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
        ],
        "output": output_value.cpu().tolist(),
    }


def probe() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch reports no CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one CUDA device, got {torch.cuda.device_count()}")
    capability = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    if capability != EXPECTED_COMPUTE_CAPABILITY or name != EXPECTED_GPU:
        raise RuntimeError(f"unexpected CUDA device: {name}, capability={capability}")
    left = torch.randn((2048, 2048), device="cuda", dtype=torch.bfloat16)
    result = left @ left
    torch.cuda.synchronize()
    if not torch.isfinite(result).all():
        raise RuntimeError("SM120 BF16 matmul produced non-finite output")
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "schema": "rlinf.w03.horde-runtime-probe.v1",
        "status": "passed",
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "nvidia_smi": smi.stdout.strip(),
        },
        "cuda": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device_name": name,
            "compute_capability": list(capability),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "bf16_matmul_finite": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "tensorrt": _tensorrt_probe(torch),
        "vulkan": _vulkan_probe(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
