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

"""Build and measure the W98 true-B8 executor matrix inside the L20 runtime."""

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
MIN_VIT_COSINE = 0.999
MIN_BACKBONE_COSINE = 0.9995
MIN_ACTION_COSINE = 0.999
MAX_ACTION_MEAN_ABS = 0.005
MAX_ACTION_MAX_ABS = 0.05


class WorkflowError(RuntimeError):
    """Raised when a W98 stage or qualification gate fails."""


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


def build_stage_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Return the exact ordered artifact and measurement commands."""
    source = args.source
    tools = source / "toolkits/eos/gr00t_trocar/tensorrt"
    model_view = args.output / "model-view"
    fixture = args.output / "fixture"
    backbone_onnx = args.output / "backbone-onnx"
    backbone_engines = args.output / "backbone-engines"
    dit_onnx = args.output / "dit-onnx"
    parameter_map = args.output / "refittable-dit-parameter-map.json"
    dit_engines = args.output / "dit-engines"
    lifecycle = args.output / "refit-lifecycle"

    return [
        (
            "model-view",
            [
                sys.executable,
                str(tools / "trocar_b8_model_view.py"),
                "--model",
                str(args.model),
                "--backbone",
                str(args.backbone),
                "--metadata",
                str(args.metadata),
                "--output",
                str(model_view),
            ],
        ),
        (
            "fixture",
            [
                sys.executable,
                str(tools / "trocar_b8_fixture.py"),
                "--source",
                str(args.gr00t_source),
                "--expected-source-revision",
                ISAAC_GR00T_REVISION,
                "--model",
                str(model_view),
                "--metadata",
                str(args.metadata),
                "--output",
                str(fixture),
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "backbone-export",
            [
                sys.executable,
                str(tools / "export_true_b8.py"),
                "--source",
                str(args.gr00t_source),
                "--expected-source-revision",
                ISAAC_GR00T_REVISION,
                "--model",
                str(model_view),
                "--collated",
                str(fixture / "collated-inputs.pt"),
                "--fixture-receipt",
                str(fixture / "fixture.json"),
                "--output",
                str(backbone_onnx),
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "backbone-build",
            [
                sys.executable,
                str(tools / "build_true_b8.py"),
                "--onnx",
                str(backbone_onnx),
                "--output",
                str(backbone_engines),
                "--workspace",
                str(args.workspace_mib),
            ],
        ),
        (
            "dit-export",
            [
                sys.executable,
                str(tools / "export_refittable_dit_b8.py"),
                "--source",
                str(args.gr00t_source),
                "--model",
                str(model_view),
                "--collated",
                str(fixture / "collated-inputs.pt"),
                "--fixture-receipt",
                str(fixture / "fixture.json"),
                "--output",
                str(dit_onnx),
                "--expected-source-revision",
                ISAAC_GR00T_REVISION,
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "dit-parameter-map",
            [
                sys.executable,
                str(tools / "refittable_dit_contract.py"),
                "--checkpoint",
                str(model_view),
                "--model-config",
                str(model_view / "config.json"),
                "--onnx",
                str(dit_onnx / "dit_bf16.onnx"),
                "--output",
                str(parameter_map),
            ],
        ),
        (
            "dit-build",
            [
                sys.executable,
                str(tools / "build_refittable_dit_b8.py"),
                "--onnx",
                str(dit_onnx / "dit_bf16.onnx"),
                "--parameter-map",
                str(parameter_map),
                "--output",
                str(dit_engines),
                "--workspace-mib",
                str(args.workspace_mib),
            ],
        ),
        (
            "refit-lifecycle",
            [
                sys.executable,
                str(tools / "refit_dit_lifecycle_probe.py"),
                "--source",
                str(args.gr00t_source),
                "--model",
                str(model_view),
                "--collated",
                str(fixture / "collated-inputs.pt"),
                "--engine",
                str(dit_engines / "dit_bf16_refit.engine"),
                "--engine-receipt",
                str(dit_engines / "rlinf-refittable-dit-engine-receipt.json"),
                "--parameter-map",
                str(parameter_map),
                "--output",
                str(lifecycle),
                "--seed",
                str(args.seed),
            ],
        ),
    ]


def _run_stage(name: str, command: list[str], output: Path) -> dict[str, Any]:
    stages = output / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    _write(stages / f"{name}.command.json", {"argv": command})
    started = time.perf_counter()
    with (
        (stages / f"{name}.stdout").open("x", encoding="utf-8") as stdout,
        (stages / f"{name}.stderr").open("x", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command, check=False, stdout=stdout, stderr=stderr, text=True
        )
    result = {
        "exit_code": completed.returncode,
        "wall_s": time.perf_counter() - started,
    }
    _write(stages / f"{name}.result.json", result)
    if completed.returncode:
        raise WorkflowError(f"stage {name} failed with code {completed.returncode}")
    return result


def _matrix_command(args: argparse.Namespace, source_digest: str) -> list[str]:
    tools = args.source / "toolkits/eos/gr00t_trocar/tensorrt"
    model = args.output / "model-view"
    fixture = args.output / "fixture"
    backbone_onnx = args.output / "backbone-onnx"
    backbone_engines = args.output / "backbone-engines"
    dit_engines = args.output / "dit-engines"
    parameter_map = args.output / "refittable-dit-parameter-map.json"
    engine_receipt = dit_engines / "rlinf-refittable-dit-engine-receipt.json"
    return [
        sys.executable,
        str(tools / "standalone_true_b8.py"),
        "--source",
        str(args.gr00t_source),
        "--expected-source-revision",
        ISAAC_GR00T_REVISION,
        "--rlinf-revision",
        args.rlinf_revision,
        "--model",
        str(model),
        "--engines",
        str(backbone_engines),
        "--collated",
        str(fixture / "collated-inputs.pt"),
        "--raw",
        str(fixture / "raw-observation.npz"),
        "--fixture-receipt",
        str(fixture / "fixture.json"),
        "--export-receipt",
        str(backbone_onnx / "rlinf-export-receipt.json"),
        "--engine-receipt",
        str(backbone_engines / "rlinf-engine-receipt.json"),
        "--refittable-dit-engine",
        str(dit_engines / "dit_bf16_refit.engine"),
        "--refittable-dit-receipt",
        str(engine_receipt),
        "--refittable-dit-receipt-sha256",
        _sha256(engine_receipt),
        "--refittable-dit-parameter-map",
        str(parameter_map),
        "--refittable-dit-parameter-map-sha256",
        _sha256(parameter_map),
        "--refittable-dit-source-digest",
        source_digest,
        "--seed",
        str(args.seed),
        "--matrix-warmup",
        str(args.warmup),
        "--matrix-measured",
        str(args.measured),
        "--pt2-backbone-compile-mode",
        "max-autotune-no-cudagraphs",
        "--compile-mode",
        "max-autotune",
        "--output",
        str(args.output / "executor-matrix.json"),
    ]


def _qualification(receipt: dict[str, Any]) -> dict[str, Any]:
    comparisons = receipt["comparisons"]
    action = comparisons["public_action"]
    checks = {
        "vit_cosine": comparisons["vit_image_embeds"]["cosine"] >= MIN_VIT_COSINE,
        "backbone_cosine": comparisons["pre_final_backbone"]["cosine"]
        >= MIN_BACKBONE_COSINE,
        "action_cosine": action["cosine"] >= MIN_ACTION_COSINE,
        "action_mean_abs": action["mean_abs"] <= MAX_ACTION_MEAN_ABS,
        "action_max_abs": action["max_abs"] <= MAX_ACTION_MAX_ABS,
        "matrix_present": receipt.get("executor_matrix") is not None,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "thresholds": {
            "vit_cosine_min": MIN_VIT_COSINE,
            "backbone_cosine_min": MIN_BACKBONE_COSINE,
            "action_cosine_min": MIN_ACTION_COSINE,
            "action_mean_abs_max": MAX_ACTION_MEAN_ABS,
            "action_max_abs_max": MAX_ACTION_MAX_ABS,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise WorkflowError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    stages = {}
    for name, command in build_stage_commands(args):
        stages[name] = _run_stage(name, command, args.output)

    lifecycle_path = args.output / "refit-lifecycle/refit-lifecycle-receipt.json"
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    if lifecycle.get("status") != "passed":
        raise WorkflowError("refittable DiT lifecycle did not pass")
    source_digest = lifecycle["device_weights"]["source_digest_revision_0"]
    stages["executor-matrix"] = _run_stage(
        "executor-matrix", _matrix_command(args, source_digest), args.output
    )
    matrix_path = args.output / "executor-matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    qualification = _qualification(matrix)
    result = {
        "schema": "rlinf.w98.l20-executor-matrix/v1",
        "status": "passed" if qualification["status"] == "passed" else "systems_only",
        "source": str(args.source),
        "gr00t_source": str(args.gr00t_source),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": args.seed,
        "warmup": args.warmup,
        "measured": args.measured,
        "stages": stages,
        "qualification": qualification,
        "artifacts": {
            "matrix": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
            "backbone_engine_receipt": {
                "path": str(args.output / "backbone-engines/rlinf-engine-receipt.json"),
                "sha256": _sha256(
                    args.output / "backbone-engines/rlinf-engine-receipt.json"
                ),
            },
            "dit_engine_receipt": {
                "path": str(
                    args.output / "dit-engines/rlinf-refittable-dit-engine-receipt.json"
                ),
                "sha256": _sha256(
                    args.output / "dit-engines/rlinf-refittable-dit-engine-receipt.json"
                ),
            },
            "refit_lifecycle": {
                "path": str(lifecycle_path),
                "sha256": _sha256(lifecycle_path),
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
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
            "schema": "rlinf.w98.l20-executor-matrix/v1",
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
