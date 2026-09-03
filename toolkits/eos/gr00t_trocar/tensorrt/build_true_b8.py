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
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

EXPECTED_ENGINES = frozenset({"vit.engine", "llm_bf16.engine"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_table(path: Path) -> list[dict[str, Any]]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT failed to deserialize {path}")
    result = []
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
        result.append(
            {
                "index": index,
                "name": name,
                "mode": mode,
                "dtype": str(engine.get_tensor_dtype(name)),
                "shape": shape,
                "profile": profile,
            }
        )
    return result


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
    if actual_onnx != {"vit_fp32.onnx", "llm_bf16.onnx"}:
        raise RuntimeError(f"unexpected ONNX set: {actual_onnx}")

    from build_tensorrt_engine import build_full_pipeline  # noqa: PLC0415

    if not args.reuse_existing:
        build_full_pipeline(
            onnx_dir=str(onnx),
            engine_dir=str(output),
            precision="bf16",
            workspace_mb=args.workspace,
            only=frozenset({"ViT", "LLM"}),
        )
    engine_paths = sorted(output.glob("*.engine"))
    if {path.name for path in engine_paths} != EXPECTED_ENGINES:
        raise RuntimeError("build did not produce the exact two-engine bundle")
    bindings = {path.name: _binding_table(path) for path in engine_paths}
    _assert_static_b8(bindings)
    metadata = onnx / "export_metadata.json"
    shutil.copyfile(metadata, output / "export_metadata.json")
    receipt = {
        "schema": "rlinf.gr00t-n1d7-trocar-true-b8-engines.v1",
        "status": "passed",
        "workspace_mib": args.workspace,
        "reused_existing_plans": args.reuse_existing,
        "export_metadata_sha256": _sha256(metadata),
        "engines": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "bindings": bindings[path.name],
            }
            for path in engine_paths
        },
        "static_batch": 8,
        "sequence_opt": 208,
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
