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

"""Capture a narrow W98 NVTX/Nsight window from qualified matrix artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from . import l20_vulkan_launcher as runtime
    from .l20_executor_matrix_launcher import _source_identity
except ImportError:
    import l20_vulkan_launcher as runtime
    from l20_executor_matrix_launcher import _source_identity

NSYS_ROOT = Path("/opt/nvidia/nsight-systems/2024.4.2")
NSYS_BINARY = NSYS_ROOT / "target-linux-x64/nsys"


class ProfileError(RuntimeError):
    """Raised when an Nsys profile input or output is incomplete."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    temporary.replace(path)


def _artifact_contract(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    files = {
        "model_view": root / "model-view",
        "collated": root / "fixture/collated-inputs.pt",
        "raw": root / "fixture/raw-observation.npz",
        "fixture_receipt": root / "fixture/fixture.json",
        "backbone_engines": root / "backbone-engines",
        "backbone_export_receipt": root / "backbone-onnx/rlinf-export-receipt.json",
        "backbone_engine_receipt": root / "backbone-engines/rlinf-engine-receipt.json",
        "dit_engine": root / "dit-engines/dit_bf16_refit.engine",
        "dit_receipt": (root / "dit-engines/rlinf-refittable-dit-engine-receipt.json"),
        "parameter_map": root / "refittable-dit-parameter-map.json",
        "lifecycle": root / "refit-lifecycle/refit-lifecycle-receipt.json",
        "matrix": root / "executor-matrix.json",
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise ProfileError(f"matrix artifact set is incomplete: {missing}")
    lifecycle = _load(files["lifecycle"])
    matrix = _load(files["matrix"])
    if lifecycle.get("status") != "passed":
        raise ProfileError("refit lifecycle receipt is not qualified")
    if matrix.get("executor_matrix", {}).get("status") != "passed":
        raise ProfileError("executor matrix mechanics did not pass")
    return {
        "root": str(root),
        "files": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in files.items()
            if path.is_file()
        },
        "source_digest": lifecycle["device_weights"]["source_digest_revision_0"],
    }


def _profile_command(contract: dict[str, Any], revision: str) -> str:
    dit_receipt_sha = contract["files"]["dit_receipt"]["sha256"]
    parameter_map_sha = contract["files"]["parameter_map"]["sha256"]
    source_digest = contract["source_digest"]
    nsys = "/opt/nvidia/nsight-systems/2024.4.2/target-linux-x64/nsys"
    python = "/isaac-sim/kit/python/bin/python3"
    tool = "/workspace/w98-src/toolkits/eos/gr00t_trocar/tensorrt/standalone_true_b8.py"
    return (
        "set -euo pipefail; "
        'export LD_LIBRARY_PATH="/w96-trt-runtime/tensorrt_libs:${LD_LIBRARY_PATH:-}"; '
        "export PYTHONPATH=/w96-overlay:/w96-trt-runtime:/workspace/gr00t-n17:"
        "/workspace/gr00t-n17/scripts/deployment:/workspace/w98-src:"
        "/workspace/w98-src/toolkits/eos/gr00t_trocar/tensorrt; "
        "export RLINF_W98_NVTX=1; mkdir -p /w96-run/nsights /w96-run/profile; "
        f"exec {nsys} profile --trace=cuda,nvtx,osrt --sample=none "
        "--cpuctxsw=none --capture-range=cudaProfilerApi "
        "--capture-range-end=stop --force-overwrite=true "
        "--output=/w96-run/nsights/w98-l20-b8-executor-matrix "
        f"{python} {tool} --source /workspace/gr00t-n17 "
        "--expected-source-revision 51d4c89f72fda44cbf77285c6a8114b52676b8a1 "
        f"--rlinf-revision {revision} --model /w98-artifacts/model-view "
        "--engines /w98-artifacts/backbone-engines "
        "--collated /w98-artifacts/fixture/collated-inputs.pt "
        "--raw /w98-artifacts/fixture/raw-observation.npz "
        "--fixture-receipt /w98-artifacts/fixture/fixture.json "
        "--export-receipt /w98-artifacts/backbone-onnx/rlinf-export-receipt.json "
        "--engine-receipt /w98-artifacts/backbone-engines/rlinf-engine-receipt.json "
        "--refittable-dit-engine /w98-artifacts/dit-engines/dit_bf16_refit.engine "
        "--refittable-dit-receipt /w98-artifacts/dit-engines/"
        "rlinf-refittable-dit-engine-receipt.json "
        f"--refittable-dit-receipt-sha256 {dit_receipt_sha} "
        "--refittable-dit-parameter-map /w98-artifacts/"
        "refittable-dit-parameter-map.json "
        f"--refittable-dit-parameter-map-sha256 {parameter_map_sha} "
        f"--refittable-dit-source-digest {source_digest} "
        "--seed 47 --warmup 1 --measured 1 --component-warmup 1 "
        "--component-measured 1 --memory-iterations 1 --paired-warmup 1 "
        "--paired-measured 1 --common-warmup 1 --common-measured 1 "
        "--matrix-warmup 1 --matrix-measured 1 "
        "--pt2-backbone-compile-mode max-autotune-no-cudagraphs "
        "--compile-mode max-autotune --profile-once --allow-systems-only "
        "--output /w96-run/profile/executor-matrix.json"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = _source_identity(args.source, args.revision)
    site = runtime._load(args.site.resolve(strict=True))
    static = runtime.validate_site(args.site)
    contract = _artifact_contract(args.artifacts)
    nsys_root = args.nsys_root.resolve(strict=True)
    if not (nsys_root / "target-linux-x64/nsys").is_file():
        raise ProfileError("Nsight Systems binary is absent")
    run_root = args.run_root.resolve()
    if run_root.exists() or "/results/W98/" not in f"{run_root}/":
        raise ProfileError(f"run root must be new and under results/W98: {run_root}")
    (run_root / "scratch/assets-cache").mkdir(parents=True)
    receipts = run_root / "receipts"
    receipts.mkdir()
    shutil.copy2(args.site, receipts / "w96-site.json")
    shutil.copy2(Path(__file__), receipts / "launcher.py")
    _write(
        receipts / "inputs.json",
        {
            "schema": "rlinf.w98.l20-executor-matrix-nsys-inputs/v1",
            "source": source,
            "w96_static": static,
            "matrix_artifacts": contract,
            "nsys_binary_sha256": _sha256(nsys_root / "target-linux-x64/nsys"),
        },
    )
    _write(
        receipts / "preflight.json",
        {
            "quota": runtime._quota_receipt(site["preflight"]["quota_mount"]),
            "gpus": runtime._gpu_receipt(),
            "image": runtime._canonical_image_matches(
                Path(site["docker"]["path"]), site
            )["checks"],
        },
    )
    container = runtime._run_container(
        site,
        run_root,
        "w98-nsys",
        _profile_command(contract, args.revision),
        extra_args=(
            "-e",
            "CUDA_VISIBLE_DEVICES=0",
            "-v",
            f"{args.source.resolve()}:/workspace/w98-src:ro",
            "-v",
            f"{args.artifacts.resolve()}:/w98-artifacts:ro",
            "-v",
            f"{nsys_root}:/opt/nvidia/nsight-systems/2024.4.2:ro",
        ),
    )
    reports = list((run_root / "nsights").glob("*.nsys-rep"))
    profile = _load(run_root / "profile/executor-matrix.json")
    records = (
        profile.get("executor_matrix", {})
        .get("lifecycle", {})
        .get("profile_once", {})
        .get("records", [])
    )
    if len(reports) != 1 or len(records) != 9:
        raise ProfileError(
            f"profile output is incomplete: reports={len(reports)} records={len(records)}"
        )
    result = {
        "schema": "rlinf.w98.l20-executor-matrix-nsys/v1",
        "status": "passed",
        "source": source,
        "container": container,
        "report": {
            "path": str(reports[0]),
            "bytes": reports[0].stat().st_size,
            "sha256": _sha256(reports[0]),
        },
        "profile_records": records,
    }
    _write(run_root / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--nsys-root", type=Path, default=NSYS_ROOT)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        print(f"W98 Nsys capture failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
