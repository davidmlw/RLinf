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

"""Build and inspect the GR00T N1.5 true-B8 BF16 ViT+LLM plans."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any


SCHEMA = "rlinf.gr00t-n1d5-stack-cube-true-b8-engines.v1"
EXPORT_SCHEMA = "rlinf.gr00t-n1d5-stack-cube-true-b8-onnx.v1"
PLAN_NAMES = {
    "vit_bf16.onnx": "vit_bf16.engine",
    "llm_bf16.onnx": "llm_bf16.engine",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_table(engine: Any) -> list[dict[str, Any]]:
    import tensorrt as trt

    result = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        shape = tuple(engine.get_tensor_shape(name))
        profile = None
        if mode == trt.TensorIOMode.INPUT and any(value < 0 for value in shape):
            minimum, optimum, maximum = engine.get_tensor_profile_shape(name, 0)
            profile = {
                "min": list(minimum),
                "opt": list(optimum),
                "max": list(maximum),
            }
        result.append(
            {
                "index": index,
                "name": name,
                "mode": (
                    "input" if mode == trt.TensorIOMode.INPUT else "output"
                ),
                "dtype": str(engine.get_tensor_dtype(name)),
                "shape": list(shape),
                "profile": profile,
            }
        )
    return result


def _validate_bindings(name: str, bindings: list[dict[str, Any]]) -> None:
    expected = {
        "vit_bf16.engine": [
            ("pixel_values", "input", "DataType.BF16", [16, 3, 224, 224]),
            ("image_features", "output", "DataType.BF16", [16, 256, 2048]),
        ],
        "llm_bf16.engine": [
            ("inputs_embeds", "input", "DataType.BF16", [8, 570, 2048]),
            ("attention_mask", "input", "DataType.INT64", [8, 570]),
            ("backbone_features", "output", "DataType.BF16", [8, 570, 2048]),
        ],
    }[name]
    actual = [
        (item["name"], item["mode"], item["dtype"], item["shape"])
        for item in bindings
    ]
    if actual != expected:
        raise RuntimeError(f"{name} bindings differ: {actual} != {expected}")
    if any(item["profile"] is not None for item in bindings):
        raise RuntimeError(f"{name} must not contain dynamic profiles")


def _precision_histogram(inspector_text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in ("BF16", "FP32", "FP16", "INT64", "INT32", "BOOL"):
        count = inspector_text.upper().count(name)
        if count:
            result[name] = count
    return result


def _build_one(
    onnx: Path,
    plan: Path,
    workspace_mib: int,
    logger: Any,
) -> dict[str, Any]:
    import tensorrt as trt

    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT ONNX parse failed for {onnx.name}: {errors}")
    for index in range(network.num_inputs):
        value = network.get_input(index)
        if any(dimension < 0 for dimension in value.shape):
            raise RuntimeError(
                f"{onnx.name} input is not true static B8: {value.name} {value.shape}"
            )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mib << 20
    )
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_wall_s = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {onnx.name}")
    plan.write_bytes(serialized)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError(f"TensorRT failed to deserialize {plan.name}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"TensorRT failed to create context for {plan.name}")
    bindings = _binding_table(engine)
    _validate_bindings(plan.name, bindings)
    inspector = engine.create_engine_inspector()
    inspector_text = inspector.get_engine_information(
        trt.LayerInformationFormat.JSON
    )
    inspector_path = plan.with_suffix(".inspector.json")
    inspector_path.write_text(inspector_text + "\n", encoding="utf-8")
    return {
        "path": str(plan),
        "bytes": plan.stat().st_size,
        "sha256": _sha256(plan),
        "build_wall_s": build_wall_s,
        "builder_flags": ["STRONGLY_TYPED"],
        "bindings": bindings,
        "num_layers": int(engine.num_layers),
        "device_memory_size": int(engine.device_memory_size),
        "device_memory_size_v2": int(engine.device_memory_size_v2),
        "num_optimization_profiles": int(engine.num_optimization_profiles),
        "inspector": {
            "path": inspector_path.name,
            "bytes": inspector_path.stat().st_size,
            "sha256": _sha256(inspector_path),
            "precision_mentions": _precision_histogram(inspector_text),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import tensorrt as trt
    import torch

    source = args.onnx.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    metadata_path = source / "export_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "schema": EXPORT_SCHEMA,
        "status": "passed",
        "model_version": "n1d5",
        "batch_size": 8,
        "image_views": 2,
        "image_batch": 16,
        "sequence_length": 570,
        "precision": "bfloat16",
    }
    actual = {name: metadata.get(name) for name in expected_metadata}
    if actual != expected_metadata:
        raise RuntimeError(f"ONNX metadata differs: {actual} != {expected_metadata}")
    actual_onnx = {path.name for path in source.glob("*.onnx")}
    if actual_onnx != set(PLAN_NAMES):
        raise RuntimeError(f"unexpected ONNX files: {sorted(actual_onnx)}")

    logger = trt.Logger(trt.Logger.INFO)
    engines = {}
    for onnx_name, plan_name in PLAN_NAMES.items():
        engines[plan_name] = _build_one(
            source / onnx_name,
            output / plan_name,
            args.workspace_mib,
            logger,
        )
    shutil.copyfile(metadata_path, output / "export_metadata.json")
    capability = list(torch.cuda.get_device_capability())
    if capability != [8, 9]:
        raise RuntimeError(f"W06 plans must be built on SM89, found {capability}")
    receipt = {
        "schema": SCHEMA,
        "status": "passed",
        "static_batch": 8,
        "image_views": 2,
        "image_batch": 16,
        "sequence_length": 570,
        "precision": "bfloat16",
        "silent_fallback": False,
        "workspace_mib": args.workspace_mib,
        "export_metadata_sha256": _sha256(metadata_path),
        "engines": engines,
        "runtime": {
            "tensorrt_module": trt.__version__,
            "tensorrt_distribution": importlib.metadata.version("tensorrt-cu12"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "driver": torch.cuda.driver_version()
            if hasattr(torch.cuda, "driver_version")
            else None,
            "device": torch.cuda.get_device_name(),
            "compute_capability": capability,
        },
    }
    receipt_path = output / "rlinf-engine-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-mib", type=int, default=16384)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W06 N1.5 TensorRT build failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
