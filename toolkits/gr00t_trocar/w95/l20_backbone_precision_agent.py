#!/usr/bin/env python3
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

"""Build and qualify a true-B8 BF16 ViT plan against W98's FP32 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ISAAC_GR00T_REVISION = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"


class WorkflowError(RuntimeError):
    """Raised when a datatype-correction stage fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    os.replace(temporary, path)


def _run_stage(name: str, command: list[str], output: Path) -> dict[str, Any]:
    stages = output / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    _write(stages / f"{name}.command.json", {"argv": command})
    started = time.perf_counter()
    with (
        (stages / f"{name}.stdout").open("x", encoding="utf-8") as stdout,
        (stages / f"{name}.stderr").open("x", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    result = {
        "exit_code": completed.returncode,
        "wall_s": time.perf_counter() - started,
    }
    _write(stages / f"{name}.result.json", result)
    if completed.returncode:
        raise WorkflowError(f"stage {name} failed with code {completed.returncode}")
    return result


def _benchmark_command(
    args: argparse.Namespace,
    *,
    backend: str,
    vit_precision: str,
    engines: Path | None,
    output: Path,
) -> list[str]:
    tools = args.source / "toolkits/eos/gr00t_trocar/tensorrt"
    command = [
        sys.executable,
        str(tools / "benchmark_precision_b8.py"),
        "--source",
        str(args.gr00t_source),
        "--model",
        str(args.old_artifacts / "model-view"),
        "--collated",
        str(args.old_artifacts / "fixture/collated-inputs.pt"),
        "--output",
        str(output),
        "--backend",
        backend,
        "--vit-precision",
        vit_precision,
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.warmup),
        "--measured",
        str(args.measured),
    ]
    if engines is not None:
        command.extend(("--engines", str(engines)))
    return command


def commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    tools = args.source / "toolkits/eos/gr00t_trocar/tensorrt"
    new_onnx = args.output / "backbone-onnx-bf16"
    new_engines = args.output / "backbone-engines-bf16"
    old = args.old_artifacts
    return [
        (
            "export-bf16-vit",
            [
                sys.executable,
                str(tools / "export_true_b8.py"),
                "--source",
                str(args.gr00t_source),
                "--expected-source-revision",
                ISAAC_GR00T_REVISION,
                "--model",
                str(old / "model-view"),
                "--collated",
                str(old / "fixture/collated-inputs.pt"),
                "--fixture-receipt",
                str(old / "fixture/fixture.json"),
                "--output",
                str(new_onnx),
                "--vit-precision",
                "bf16",
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "build-bf16-vit",
            [
                sys.executable,
                str(tools / "build_true_b8.py"),
                "--onnx",
                str(new_onnx),
                "--output",
                str(new_engines),
                "--workspace",
                str(args.workspace_mib),
                "--reuse-llm-engine",
                str(old / "backbone-engines/llm_bf16.engine"),
            ],
        ),
        (
            "numerics",
            [
                sys.executable,
                str(tools / "compare_backbone_precision_b8.py"),
                "--source",
                str(args.gr00t_source),
                "--model",
                str(old / "model-view"),
                "--collated",
                str(old / "fixture/collated-inputs.pt"),
                "--raw",
                str(old / "fixture/raw-observation.npz"),
                "--fixture-receipt",
                str(old / "fixture/fixture.json"),
                "--old-engines",
                str(old / "backbone-engines"),
                "--new-engines",
                str(new_engines),
                "--output",
                str(args.output / "numerics.json"),
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "performance-eager-bf16",
            _benchmark_command(
                args,
                backend="eager",
                vit_precision="bf16",
                engines=None,
                output=args.output / "performance-eager-bf16.json",
            ),
        ),
        (
            "performance-trt-fp32-vit",
            _benchmark_command(
                args,
                backend="trt",
                vit_precision="fp32",
                engines=old / "backbone-engines",
                output=args.output / "performance-trt-fp32-vit.json",
            ),
        ),
        (
            "performance-trt-bf16-vit",
            _benchmark_command(
                args,
                backend="trt",
                vit_precision="bf16",
                engines=new_engines,
                output=args.output / "performance-trt-bf16-vit.json",
            ),
        ),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise WorkflowError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    required = (
        "model-view/config.json",
        "fixture/collated-inputs.pt",
        "fixture/raw-observation.npz",
        "fixture/fixture.json",
        "backbone-engines/vit.engine",
        "backbone-engines/llm_bf16.engine",
    )
    for relative in required:
        (args.old_artifacts / relative).resolve(strict=True)
    stages = {
        name: _run_stage(name, command, args.output)
        for name, command in commands(args)
    }
    engine_receipt_path = (
        args.output / "backbone-engines-bf16/rlinf-engine-receipt.json"
    )
    numerics_path = args.output / "numerics.json"
    performance_paths = {
        "eager_bf16": args.output / "performance-eager-bf16.json",
        "trt_fp32_vit": args.output / "performance-trt-fp32-vit.json",
        "trt_bf16_vit": args.output / "performance-trt-bf16-vit.json",
    }
    engine = json.loads(engine_receipt_path.read_text(encoding="utf-8"))
    numerics = json.loads(numerics_path.read_text(encoding="utf-8"))
    performance = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in performance_paths.items()
    }
    means = {
        name: value["timing"]["backbone_ms"]["mean_ms"]
        for name, value in performance.items()
    }
    checks = {
        "engine_receipt_passed": engine.get("status") == "passed",
        "bf16_key_gemm_count_positive": engine["precision_gates"][
            "vit_key_gemm_count"
        ]
        > 0,
        "bf16_all_key_gemms_are_bf16": not engine["precision_gates"][
            "vit_non_bf16_key_gemms"
        ],
        "new_bf16_numerics_passed": numerics["new_trt_bf16_vit"]["status"]
        == "passed",
        "new_bf16_faster_than_eager": means["trt_bf16_vit"]
        < means["eager_bf16"],
        "old_fp32_expected_bridge_casts": performance["trt_fp32_vit"][
            "runtime_cast_diagnostic"
        ]["expected_bridge_casts"]
        == 5,
        "new_bf16_zero_bridge_casts": performance["trt_bf16_vit"][
            "runtime_cast_diagnostic"
        ]["expected_bridge_casts"]
        == 0,
    }
    result = {
        "schema": "rlinf.w98.l20-bf16-vit-datatype-correction.v1",
        "status": "passed" if all(checks.values()) else "failed",
        "scope": "standalone true-B8 Backbone qualification; not PPO authority",
        "source_revision": args.rlinf_revision,
        "isaac_gr00t_revision": ISAAC_GR00T_REVISION,
        "strict_order": [name for name, _ in commands(args)],
        "warmup": args.warmup,
        "measured": args.measured,
        "checks": checks,
        "backbone_mean_ms": means,
        "stages": stages,
        "artifacts": {
            "engine_receipt": {
                "path": str(engine_receipt_path),
                "sha256": _sha256(engine_receipt_path),
            },
            "numerics": {
                "path": str(numerics_path),
                "sha256": _sha256(numerics_path),
            },
            "performance": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in performance_paths.items()
            },
        },
    }
    _write(args.output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rlinf-revision", required=True)
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--old-artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--workspace-mib", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=30)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        traceback.print_exc()
        failure = {
            "schema": "rlinf.w98.l20-bf16-vit-datatype-correction.v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if args.output.exists():
            _write(args.output / "result.json", failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
