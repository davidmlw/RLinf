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

"""Build and inspect the exact two-engine true-B8 TensorRT backbone bundle."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

EXPECTED_ENGINES = frozenset({"vit.engine", "llm_bf16.engine"})
VIT_ONNX_PRECISIONS = {
    "vit_bf16.onnx": "bf16",
    "vit_fp32.onnx": "fp32",
}


def _onnx_dtype_audit(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load_model(path, load_external_data=False)

    def tensor_type(value: Any) -> str:
        return onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)

    initializer_dtypes = Counter(
        onnx.TensorProto.DataType.Name(value.data_type)
        for value in model.graph.initializer
    )
    cast_targets = Counter()
    cast_nodes = []
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        target = next(
            (attribute.i for attribute in node.attribute if attribute.name == "to"),
            None,
        )
        target_name = (
            onnx.TensorProto.DataType.Name(target) if target is not None else "UNKNOWN"
        )
        cast_targets[target_name] += 1
        cast_nodes.append(
            {
                "name": node.name,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "to": target_name,
            }
        )
    return {
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "inputs": {value.name: tensor_type(value) for value in model.graph.input},
        "outputs": {value.name: tensor_type(value) for value in model.graph.output},
        "initializer_dtype_histogram": dict(sorted(initializer_dtypes.items())),
        "cast_target_histogram": dict(sorted(cast_targets.items())),
        "cast_nodes": cast_nodes,
    }


def _precision_tokens(value: Any) -> set[str]:
    text = json.dumps(value, sort_keys=True).upper()
    result = set()
    if "BFLOAT16" in text or "BF16" in text:
        result.add("BF16")
    text_without_bfloat = text.replace("BFLOAT16", "")
    aliases = {
        "FP16": ("FLOAT16", "FP16", "HALF"),
        "FP32": ("FLOAT32", "FP32", '"FLOAT"'),
        "TF32": ("TF32",),
        "INT8": ("INT8",),
        "INT32": ("INT32",),
        "BOOL": ("BOOL",),
    }
    for canonical, spellings in aliases.items():
        if any(spelling in text_without_bfloat for spelling in spellings):
            result.add(canonical)
    return result


def _tactic_precision(tactic: str) -> str:
    lowered = tactic.lower()
    if "bf16" in lowered:
        return "bf16"
    if "tf32" in lowered:
        return "tf32"
    if re.search(r"(^|[^a-z0-9])f32([^a-z0-9]|$)", lowered):
        return "fp32"
    if "fp16" in lowered or re.search(r"(^|[^a-z0-9])f16([^a-z0-9]|$)", lowered):
        return "fp16"
    return "unclassified"


def _inspector_precision_summary(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        layers = raw.get("Layers", raw.get("layers", []))
    elif isinstance(raw, list):
        layers = raw
    else:
        raise RuntimeError(f"unexpected TensorRT inspector JSON type: {type(raw)}")
    if not isinstance(layers, list) or not layers:
        raise RuntimeError("TensorRT inspector returned no layers")

    layer_types = Counter()
    io_precisions = Counter()
    tactic_precisions = Counter()
    key_gemms = []
    fp32_islands = []
    for layer in layers:
        if not isinstance(layer, dict):
            raise RuntimeError("TensorRT inspector layer is not an object")
        name = str(layer.get("Name", layer.get("name", "")))
        layer_type = str(layer.get("LayerType", layer.get("type", "unknown")))
        tactic = str(layer.get("TacticName", layer.get("tactic", "")))
        inputs = layer.get("Inputs", layer.get("inputs", []))
        outputs = layer.get("Outputs", layer.get("outputs", []))
        tokens = _precision_tokens({"inputs": inputs, "outputs": outputs})
        tactic_precision = _tactic_precision(tactic)
        resolved_precision = tactic_precision
        precision_authority = "tactic_name"
        if tactic_precision == "unclassified":
            floating_tokens = tokens.intersection({"BF16", "FP16", "FP32", "TF32"})
            if len(floating_tokens) == 1:
                resolved_precision = {
                    "BF16": "bf16",
                    "FP16": "fp16",
                    "FP32": "fp32",
                    "TF32": "tf32",
                }[next(iter(floating_tokens))]
                precision_authority = "inspector_io_dtype"
        layer_types[layer_type] += 1
        io_precisions.update(tokens)
        tactic_precisions[resolved_precision] += 1
        record = {
            "name": name,
            "layer_type": layer_type,
            "tactic": tactic,
            "tactic_precision": tactic_precision,
            "resolved_precision": resolved_precision,
            "precision_authority": precision_authority,
            "io_precisions": sorted(tokens),
        }
        if "gemm" in tactic.lower() or "xmma" in tactic.lower():
            key_gemms.append(record)
        if "FP32" in tokens or resolved_precision in {"fp32", "tf32"}:
            fp32_islands.append(record)
    return {
        "layer_count": len(layers),
        "layer_type_histogram": dict(sorted(layer_types.items())),
        "io_precision_histogram": dict(sorted(io_precisions.items())),
        "tactic_precision_histogram": dict(sorted(tactic_precisions.items())),
        "key_gemms": key_gemms,
        "fp32_islands": fp32_islands,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _engine_audit(path: Path, inspector_output: Path) -> dict[str, Any]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT failed to deserialize {path}")
    bindings = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = list(engine.get_tensor_shape(name))
        mode = str(engine.get_tensor_mode(name)).split(".")[-1].lower()
        profile = None
        if mode == "input" and any(dimension < 0 for dimension in shape):
            minimum, optimum, maximum = engine.get_tensor_profile_shape(name, 0)
            profile = {
                "min": list(minimum),
                "opt": list(optimum),
                "max": list(maximum),
            }
        bindings.append(
            {
                "index": index,
                "name": name,
                "mode": mode,
                "dtype": str(engine.get_tensor_dtype(name)),
                "shape": shape,
                "profile": profile,
            }
        )
    inspector = engine.create_engine_inspector()
    raw_text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    raw = json.loads(raw_text)
    inspector_output.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "bindings": bindings,
        "inspector": {
            "path": inspector_output.name,
            "sha256": _sha256(inspector_output),
            "summary": _inspector_precision_summary(raw),
        },
    }


def _assert_static_b8(bindings: dict[str, list[dict[str, Any]]]) -> None:
    vit_inputs = {
        item["name"]: item for item in bindings["vit.engine"] if item["mode"] == "input"
    }
    if vit_inputs.get("pixel_values", {}).get("shape") != [6144, 1536]:
        raise RuntimeError(f"ViT is not true static B8: {vit_inputs}")
    llm_inputs = {
        item["name"]: item
        for item in bindings["llm_bf16.engine"]
        if item["mode"] == "input"
    }
    expected_static = {
        "inputs_embeds": [8, -1, 2048],
        "attention_mask": [8, -1],
        "position_ids": [3, 8, -1],
        "visual_pos_masks": [8, -1],
        "deepstack_0": [1536, 2048],
        "deepstack_1": [1536, 2048],
        "deepstack_2": [1536, 2048],
    }
    actual = {name: item["shape"] for name, item in llm_inputs.items()}
    if actual != expected_static:
        raise RuntimeError(
            f"LLM true-B8 binding mismatch: {actual} != {expected_static}"
        )
    for name in ("inputs_embeds", "attention_mask", "position_ids", "visual_pos_masks"):
        profile = llm_inputs[name]["profile"]
        if profile is None:
            raise RuntimeError(f"LLM dynamic sequence profile missing for {name}")
        if 208 not in profile["opt"]:
            raise RuntimeError(
                f"LLM opt profile does not include L=208 for {name}: {profile}"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    onnx = args.onnx.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() and not args.reuse_existing:
        raise RuntimeError(f"output already exists: {output}")
    output.mkdir(parents=True, exist_ok=args.reuse_existing)
    actual_onnx = {path.name for path in onnx.glob("*.onnx")}
    vit_onnx = actual_onnx.intersection(VIT_ONNX_PRECISIONS)
    if len(vit_onnx) != 1 or actual_onnx != vit_onnx | {"llm_bf16.onnx"}:
        raise RuntimeError(f"unexpected ONNX set: {actual_onnx}")
    vit_precision = VIT_ONNX_PRECISIONS[vit_onnx.pop()]

    from build_tensorrt_engine import build_full_pipeline  # noqa: PLC0415

    reused_llm = None
    if not args.reuse_existing:
        build_full_pipeline(
            onnx_dir=str(onnx),
            engine_dir=str(output),
            precision="bf16",
            workspace_mb=args.workspace,
            only=(
                frozenset({"ViT"})
                if args.reuse_llm_engine
                else frozenset({"ViT", "LLM"})
            ),
        )
        if args.reuse_llm_engine:
            reused_llm = args.reuse_llm_engine.resolve(strict=True)
            shutil.copyfile(reused_llm, output / "llm_bf16.engine")
    engine_paths = sorted(output.glob("*.engine"))
    if {path.name for path in engine_paths} != EXPECTED_ENGINES:
        raise RuntimeError("build did not produce the exact two-engine bundle")
    audits = {
        path.name: _engine_audit(path, output / f"{path.name}.inspector.json")
        for path in engine_paths
    }
    bindings = {name: audit["bindings"] for name, audit in audits.items()}
    _assert_static_b8(bindings)
    vit_binding_dtypes = {
        item["dtype"] for item in bindings["vit.engine"] if "dtype" in item
    }
    expected_vit_dtype = {
        "bf16": "DataType.BF16",
        "fp32": "DataType.FLOAT",
    }[vit_precision]
    if vit_binding_dtypes != {expected_vit_dtype}:
        raise RuntimeError(
            f"ViT engine dtype mismatch: {vit_binding_dtypes} != {expected_vit_dtype}"
        )
    onnx_audits = {
        path.name: _onnx_dtype_audit(path) for path in sorted(onnx.glob("*.onnx"))
    }
    vit_onnx_name = next(iter(set(onnx_audits).intersection(VIT_ONNX_PRECISIONS)))
    expected_onnx_dtype = {"bf16": "BFLOAT16", "fp32": "FLOAT"}[vit_precision]
    pixel_dtype = onnx_audits[vit_onnx_name]["inputs"].get("pixel_values")
    if pixel_dtype != expected_onnx_dtype:
        raise RuntimeError(
            "ViT ONNX pixel_values dtype mismatch: "
            f"{pixel_dtype} != {expected_onnx_dtype}"
        )
    llm_binding_dtypes = {
        item["dtype"] for item in bindings["llm_bf16.engine"] if "dtype" in item
    }
    if "DataType.FLOAT" in llm_binding_dtypes:
        raise RuntimeError(f"LLM engine exposes FP32 bindings: {llm_binding_dtypes}")
    vit_summary = audits["vit.engine"]["inspector"]["summary"]
    non_bf16_gemms = [
        item
        for item in vit_summary["key_gemms"]
        if item["resolved_precision"] != "bf16"
    ]
    if vit_precision == "bf16" and not vit_summary["key_gemms"]:
        raise RuntimeError("BF16 ViT inspector identified no GEMM/XMMA tactics")
    if vit_precision == "bf16" and non_bf16_gemms:
        raise RuntimeError(
            f"BF16 ViT retained non-BF16 key GEMMs: {non_bf16_gemms}"
        )
    metadata = onnx / "export_metadata.json"
    shutil.copyfile(metadata, output / "export_metadata.json")
    receipt = {
        "schema": "rlinf.gr00t-n1d7-trocar-true-b8-engines.v1",
        "status": "passed",
        "workspace_mib": args.workspace,
        "reused_existing_plans": args.reuse_existing,
        "reused_llm_engine": (
            {
                "source": str(reused_llm),
                "sha256": _sha256(reused_llm),
            }
            if reused_llm
            else None
        ),
        "export_metadata_sha256": _sha256(metadata),
        "engines": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "bindings": bindings[path.name],
                "inspector": audits[path.name]["inspector"],
            }
            for path in engine_paths
        },
        "onnx": onnx_audits,
        "precision_gates": {
            "vit_onnx_pixel_values": pixel_dtype,
            "vit_engine_binding_dtypes": sorted(vit_binding_dtypes),
            "llm_engine_binding_dtypes": sorted(llm_binding_dtypes),
            "vit_key_gemm_count": len(vit_summary["key_gemms"]),
            "vit_non_bf16_key_gemms": non_bf16_gemms,
            "vit_fp32_islands": vit_summary["fp32_islands"],
        },
        "static_batch": 8,
        "sequence_opt": 208,
        "vit_precision": vit_precision,
        "llm_precision": "bf16",
        "silent_fallback": False,
    }
    (output / "rlinf-engine-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=int, default=8192)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--reuse-llm-engine", type=Path)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W80 true-B8 engine build failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
