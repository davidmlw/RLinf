# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run and retain the official GR00T N1.7 B1 TensorRT oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from model_view import materialize_local_model_view

EXPECTED_ENGINES = (
    "vit.engine",
    "llm_bf16.engine",
    "vl_self_attention.engine",
    "state_encoder.engine",
    "action_encoder.engine",
    "dit_bf16.engine",
    "action_decoder.engine",
)
EXPECTED_ONNX = (
    "vit_fp32.onnx",
    "llm_bf16.onnx",
    "vl_self_attention.onnx",
    "state_encoder.onnx",
    "action_encoder.onnx",
    "dit_bf16.onnx",
    "action_decoder.onnx",
)
COMPARISON_RE = re.compile(
    r"\[(?P<section>6[ab])\] (?P<label>[^\n:]+).*?"
    r"Cosine Similarity:\s*(?P<cosine>[0-9.]+).*?"
    r"L1 Mean Error:\s*(?P<mean_abs>[0-9.]+).*?"
    r"L(?:∞|\\u221e) Max Error:\s*(?P<max_abs>[0-9.]+)",
    re.DOTALL,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _executable(path: Path) -> Path:
    """Validate an executable while preserving a virtualenv symlink path."""
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"builder Python must be an absolute executable: {path}")
    return path


def _artifact_inventory(root: Path, names: tuple[str, ...]) -> dict[str, Any]:
    inventory = {}
    for name in names:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"official pipeline omitted artifact: {path}")
        inventory[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return inventory


def _engine_inventory(root: Path) -> dict[str, Any]:
    import tensorrt as trt

    inventory = _artifact_inventory(root, EXPECTED_ENGINES)
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    for name, artifact in inventory.items():
        engine = runtime.deserialize_cuda_engine((root / name).read_bytes())
        if engine is None:
            raise RuntimeError(f"could not deserialize TensorRT engine: {name}")
        tensors = []
        for index in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(index)
            mode = str(engine.get_tensor_mode(tensor_name)).split(".")[-1].lower()
            tensor = {
                "name": tensor_name,
                "mode": mode,
                "dtype": str(engine.get_tensor_dtype(tensor_name)),
                "shape": list(engine.get_tensor_shape(tensor_name)),
            }
            if mode == "input" and engine.num_optimization_profiles:
                tensor["profiles"] = [
                    {
                        "min": list(shapes[0]),
                        "opt": list(shapes[1]),
                        "max": list(shapes[2]),
                    }
                    for profile in range(engine.num_optimization_profiles)
                    for shapes in [
                        engine.get_tensor_profile_shape(tensor_name, profile)
                    ]
                ]
            tensors.append(tensor)
        artifact["optimization_profiles"] = engine.num_optimization_profiles
        artifact["tensors"] = tensors
    return inventory


def _onnx_inventory(root: Path) -> dict[str, Any]:
    required = set(EXPECTED_ONNX)
    present = {path.name for path in root.iterdir() if path.is_file()}
    missing = sorted(required - present)
    if missing:
        raise RuntimeError(f"official pipeline omitted ONNX artifacts: {missing}")
    names = tuple(
        sorted(
            name for name in present if name.endswith(".onnx") or ".onnx.data" in name
        )
    )
    return _artifact_inventory(root, names)


def _parse_numerics(log: str) -> dict[str, dict[str, float]]:
    results = {}
    for match in COMPARISON_RE.finditer(log):
        label = match.group("label").strip().lower().replace(" ", "_")
        results[label] = {
            "cosine": float(match.group("cosine")),
            "mean_abs": float(match.group("mean_abs")),
            "max_abs": float(match.group("max_abs")),
        }
    required = {
        "vit_output_comparison_(image_embeds)",
        "backbone_output_comparison_(llm_output,_before_vl_self_attention)",
        "final_action_output_comparison",
    }
    if set(results) != required:
        raise RuntimeError(
            f"could not parse official numerical results: {sorted(results)}"
        )
    if any(value["cosine"] < 0.999 for value in results.values()):
        raise RuntimeError(f"official TensorRT component cosine gate failed: {results}")
    final = results["final_action_output_comparison"]
    if final["mean_abs"] > 0.005 or final["max_abs"] > 0.05:
        raise RuntimeError(f"official TensorRT final action gate failed: {final}")
    return results


def _runtime_packages() -> dict[str, str]:
    import flash_attn
    import tensorrt
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "flash_attn": flash_attn.__version__,
        "tensorrt": tensorrt.__version__,
        "torchcodec": __import__("torchcodec").__version__,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    attempt = args.attempt.resolve(strict=True)
    source = args.source.resolve(strict=True)
    model = args.model.resolve(strict=True)
    backbone = args.backbone.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    builder_python = _executable(args.builder_python)

    model_view = attempt / "model-view" / "libero_10"
    model_receipt = materialize_local_model_view(model, backbone, model_view)
    output = attempt / "official"
    output.mkdir(parents=True, exist_ok=False)
    command = [
        str(builder_python),
        str(source / "scripts" / "deployment" / "build_trt_pipeline.py"),
        "--model-path",
        str(model_view),
        "--dataset-path",
        str(dataset),
        "--embodiment-tag",
        "LIBERO_PANDA",
        "--output-dir",
        str(output),
        "--batch-size",
        "1",
        "--precision",
        "bf16",
        "--export-mode",
        "full_pipeline",
        "--workspace",
        "8192",
        "--warmup",
        "5",
        "--num-iterations",
        "20",
        "--steps",
        "export,build,verify,benchmark",
    ]
    (attempt / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(source),
        }
    )
    with (
        (attempt / "logs" / "pipeline.out").open("w", encoding="utf-8") as stdout,
        (attempt / "logs" / "pipeline.err").open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=source,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"official TensorRT pipeline exited {completed.returncode}")

    pipeline_log = output / "pipeline.log"
    numerics = _parse_numerics(pipeline_log.read_text(encoding="utf-8"))
    receipt = {
        "schema": "rlinf.gr00t-n1d7-official-b1-qualification.v1",
        "status": "passed",
        "command": command,
        "isaac_gr00t_revision": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime_packages": _runtime_packages(),
        "model_view": model_receipt,
        "numerics": numerics,
        "engines": _engine_inventory(output / "engines"),
        "onnx": _onnx_inventory(output / "onnx"),
        "export_metadata": json.loads(
            (output / "engines" / "export_metadata.json").read_text(encoding="utf-8")
        ),
    }
    receipt_path = attempt / "qualification.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--builder-python", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as error:
        print(f"W78 official B1 failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
