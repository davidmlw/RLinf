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

"""Build and inventory a distinct true-B8 refittable TensorRT DiT plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

EXPECTED_BATCH = 8
EXPECTED_ACTION_SEQUENCE = 41
EXPECTED_VL_SEQUENCE = 208
EXPECTED_REFIT_WEIGHTS = 456


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _dtype_name(value: Any) -> str:
    return str(value).split(".")[-1].upper()


def _profile_shape(shape: tuple[int, ...], vl_sequence: int) -> tuple[int, ...]:
    return tuple(vl_sequence if dimension < 0 else dimension for dimension in shape)


def _binding_table(engine: Any) -> list[dict[str, Any]]:
    result = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(engine.get_tensor_shape(name))
        mode = str(engine.get_tensor_mode(name)).split(".")[-1].lower()
        profile = None
        if mode == "input" and any(dimension < 0 for dimension in shape):
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
                "mode": mode,
                "dtype": _dtype_name(engine.get_tensor_dtype(name)),
                "shape": list(shape),
                "profile": profile,
            }
        )
    return result


def _validate_bindings(bindings: list[dict[str, Any]]) -> None:
    inputs = {item["name"]: item for item in bindings if item["mode"] == "input"}
    expected = {
        "sa_embs": [EXPECTED_BATCH, EXPECTED_ACTION_SEQUENCE, 1536],
        "vl_embs": [EXPECTED_BATCH, -1, 2048],
        "timestep": [EXPECTED_BATCH],
        "image_mask": [EXPECTED_BATCH, -1],
        "backbone_attention_mask": [EXPECTED_BATCH, -1],
    }
    actual = {name: item["shape"] for name, item in inputs.items()}
    if actual != expected:
        raise RuntimeError(f"DiT binding mismatch: {actual} != {expected}")
    for name in ("vl_embs", "image_mask", "backbone_attention_mask"):
        profile = inputs[name]["profile"]
        if profile is None or profile["opt"] != [
            EXPECTED_BATCH,
            EXPECTED_VL_SEQUENCE,
        ] + ([2048] if name == "vl_embs" else []):
            raise RuntimeError(f"DiT L=208 profile mismatch for {name}: {profile}")


def _refitter_inventory(refitter: Any) -> dict[str, Any]:
    names = sorted(refitter.get_all_weights())
    entries = []
    for name in names:
        prototype = refitter.get_weights_prototype(name)
        entries.append(
            {
                "name": name,
                "dtype": _dtype_name(prototype.dtype),
                "count": int(prototype.size),
                "nbytes": int(prototype.nbytes),
            }
        )
    layer_names, roles = refitter.get_all()
    layer_role = sorted(
        (
            {
                "layer": name,
                "role": _dtype_name(role),
            }
            for name, role in zip(layer_names, roles, strict=True)
        ),
        key=lambda item: (item["layer"], item["role"]),
    )
    return {
        "named_weights": entries,
        "named_weight_count": len(entries),
        "named_weight_digest": _canonical_sha256(entries),
        "layer_roles": layer_role,
        "layer_role_count": len(layer_role),
        "layer_role_digest": _canonical_sha256(layer_role),
        "missing_weights_before_update": sorted(refitter.get_missing_weights()),
    }


def _validate_refitter_against_map(
    inventory: dict[str, Any], mapping: dict[str, Any]
) -> dict[str, Any]:
    mapped = {entry["initializer"]: entry for entry in mapping["dit_refit"]["entries"]}
    actual = {entry["name"]: entry for entry in inventory["named_weights"]}
    if len(mapped) != EXPECTED_REFIT_WEIGHTS:
        raise RuntimeError(f"unexpected mapped trainable weight count: {len(mapped)}")
    missing = sorted(set(mapped) - set(actual))
    extras = sorted(set(actual) - set(mapped))
    if missing:
        raise RuntimeError(
            "TensorRT refitter does not expose every trainable ONNX initializer: "
            f"missing={missing[:8]}"
        )
    mismatches = {}
    for name, expected in mapped.items():
        observed = actual[name]
        if (
            observed["count"] != expected["parameter_count"]
            or observed["dtype"] != "BF16"
        ):
            mismatches[name] = {"expected": expected, "observed": observed}
    if mismatches:
        raise RuntimeError(
            f"TensorRT prototype mismatch: {list(mismatches.items())[:4]}"
        )
    derived = [actual[name] for name in extras]
    invalid = [item for item in derived if not item["name"].startswith("/dit/")]
    if invalid:
        raise RuntimeError(
            "unexpected non-parameter refitter names outside the DiT graph: "
            f"{invalid[:8]}"
        )
    return {
        "mapped_trainable_count": len(mapped),
        "mapped_trainable_digest": _canonical_sha256(
            [actual[name] for name in sorted(mapped)]
        ),
        "derived_constant_count": len(extras),
        "derived_constant_names": extras,
        "derived_constant_digest": _canonical_sha256(extras),
        "derived_constants_policy": "retain_plan_value_not_updated",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import tensorrt as trt
    import torch

    onnx = args.onnx.resolve(strict=True)
    mapping_path = args.parameter_map.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.mkdir(parents=True)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if mapping["dit_refit"]["initializer_count"] != EXPECTED_REFIT_WEIGHTS:
        raise RuntimeError("offline parameter map did not pass the 456-weight gate")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT ONNX parse failed: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20)
    config.set_flag(trt.BuilderFlag.REFIT)
    if config.get_flag(trt.BuilderFlag.REFIT) is not True:
        raise RuntimeError("TensorRT builder did not retain BuilderFlag.REFIT")
    if hasattr(trt.BuilderFlag, "REFIT_IDENTICAL") and config.get_flag(
        trt.BuilderFlag.REFIT_IDENTICAL
    ):
        raise RuntimeError("REFIT_IDENTICAL is forbidden for trainable DiT weights")

    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        value = network.get_input(index)
        shape = tuple(value.shape)
        if shape[0] != EXPECTED_BATCH:
            raise RuntimeError(f"input {value.name} is not static B8: {shape}")
        if any(dimension < 0 for dimension in shape):
            minimum = _profile_shape(shape, 1)
            optimum = _profile_shape(shape, EXPECTED_VL_SEQUENCE)
            maximum = _profile_shape(shape, 416)
            # TensorRT 10.15 mutates the profile but returns None here, despite
            # newer API documentation describing a boolean return value.
            profile.set_shape(value.name, minimum, optimum, maximum)
            observed = tuple(tuple(item) for item in profile.get_shape(value.name))
            expected_profile = (minimum, optimum, maximum)
            if observed != expected_profile:
                raise RuntimeError(
                    f"failed to set TensorRT profile for {value.name}: "
                    f"{observed} != {expected_profile}"
                )
    config.add_optimization_profile(profile)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_wall_s = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the refittable DiT engine")
    plan = output / "dit_bf16_refit.engine"
    plan.write_bytes(serialized)

    free_before, total = torch.cuda.mem_get_info()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the refittable DiT engine")
    free_after_engine, _ = torch.cuda.mem_get_info()
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT failed to create a DiT execution context")
    free_after_context, _ = torch.cuda.mem_get_info()
    bindings = _binding_table(engine)
    _validate_bindings(bindings)
    refitter = trt.Refitter(engine, logger)
    refitter.weights_validation = True
    inventory = _refitter_inventory(refitter)
    classification = _validate_refitter_against_map(inventory, mapping)

    receipt = {
        "schema": "rlinf.gr00t-n1d7-trocar-true-b8-refittable-dit-engine.v1",
        "status": "passed",
        "builder_flags": ["STRONGLY_TYPED", "REFIT"],
        "refit_identical": False,
        "workspace_mib": args.workspace_mib,
        "build_wall_s": build_wall_s,
        "onnx": {
            "path": str(onnx),
            "bytes": onnx.stat().st_size,
            "sha256": _sha256(onnx),
        },
        "onnx_external_data": {
            "path": str(onnx.with_suffix(onnx.suffix + ".data")),
            "bytes": onnx.with_suffix(onnx.suffix + ".data").stat().st_size,
            "sha256": _sha256(onnx.with_suffix(onnx.suffix + ".data")),
        },
        "parameter_map": {"path": str(mapping_path), "sha256": _sha256(mapping_path)},
        "engine": {
            "path": str(plan),
            "bytes": plan.stat().st_size,
            "sha256": _sha256(plan),
            "device_memory_size": int(engine.device_memory_size),
            "device_memory_size_v2": int(engine.device_memory_size_v2),
            "num_optimization_profiles": int(engine.num_optimization_profiles),
            "bindings": bindings,
        },
        "refitter": {**inventory, "classification": classification},
        "memory": {
            "total_device_bytes": total,
            "free_before_deserialize": free_before,
            "free_after_deserialize": free_after_engine,
            "free_after_context": free_after_context,
            "deserialize_delta_bytes": free_before - free_after_engine,
            "context_delta_bytes": free_after_engine - free_after_context,
        },
        "runtime": {
            "tensorrt": trt.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "python": platform.python_version(),
        },
    }
    (output / "rlinf-refittable-dit-engine-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--parameter-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-mib", type=int, default=8192)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W83 refittable DiT build failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
