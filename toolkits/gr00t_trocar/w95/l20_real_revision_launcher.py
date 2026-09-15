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

"""Run the W98 real post-PPO 456-tensor DiT refit gate on one L20."""

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
    from .l20_executor_matrix_nsys_launcher import _artifact_contract
except ImportError:
    import l20_vulkan_launcher as runtime
    from l20_executor_matrix_launcher import _source_identity
    from l20_executor_matrix_nsys_launcher import _artifact_contract


class RefitLaunchError(RuntimeError):
    """Raised when the real-revision launch contract is incomplete."""


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
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def _revision_contract(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    sidecar = root / "action-head-model.pt"
    manifest_path = root / "manifest.json"
    if not sidecar.is_file() or not manifest_path.is_file():
        raise RefitLaunchError("real-revision sidecar or manifest is missing")
    manifest = _load(manifest_path)
    observed = _sha256(sidecar)
    if manifest.get("sidecar", {}).get("sha256") != observed:
        raise RefitLaunchError("real-revision sidecar hash differs from manifest")
    if manifest["sidecar"].get("tensor_count") != 456:
        raise RefitLaunchError("real-revision sidecar is not the 456-tensor DiT")
    return {
        "root": str(root),
        "sidecar_sha256": observed,
        "manifest_sha256": _sha256(manifest_path),
        "source": manifest["source"],
        "sidecar": manifest["sidecar"],
    }


def _probe_command(contract: dict[str, Any], checkpoint_revision: int) -> str:
    tool = (
        "/workspace/w98-src/toolkits/eos/gr00t_trocar/tensorrt/"
        "refit_dit_real_revision_probe.py"
    )
    return (
        "set -euo pipefail; "
        'export LD_LIBRARY_PATH="/w96-trt-runtime/tensorrt_libs:${LD_LIBRARY_PATH:-}"; '
        "export PYTHONPATH=/w96-overlay:/w96-trt-runtime:/workspace/gr00t-n17:"
        "/workspace/gr00t-n17/scripts/deployment:/workspace/w98-src:"
        "/workspace/w98-src/toolkits/eos/gr00t_trocar/tensorrt; "
        "exec /isaac-sim/kit/python/bin/python3 "
        f"{tool} --source /workspace/gr00t-n17 "
        "--model /w98-artifacts/model-view "
        "--collated /w98-artifacts/fixture/collated-inputs.pt "
        "--engine /w98-artifacts/dit-engines/dit_bf16_refit.engine "
        "--engine-receipt /w98-artifacts/dit-engines/"
        "rlinf-refittable-dit-engine-receipt.json "
        "--parameter-map /w98-artifacts/refittable-dit-parameter-map.json "
        "--checkpoint /w98-revision/action-head-model.pt "
        f"--checkpoint-revision {checkpoint_revision} "
        "--output /w96-run/probe"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = _source_identity(args.source, args.revision)
    site = runtime._load(args.site.resolve(strict=True))
    static = runtime.validate_site(args.site)
    artifacts = _artifact_contract(args.artifacts)
    revision = _revision_contract(args.checkpoint)
    run_root = args.run_root.resolve()
    if run_root.exists() or "/results/W98/" not in f"{run_root}/":
        raise RefitLaunchError(
            f"run root must be new and under results/W98: {run_root}"
        )
    (run_root / "scratch/assets-cache").mkdir(parents=True)
    receipts = run_root / "receipts"
    receipts.mkdir()
    shutil.copy2(args.site, receipts / "w96-site.json")
    shutil.copy2(Path(__file__), receipts / "launcher.py")
    _write(
        receipts / "inputs.json",
        {
            "schema": "rlinf.w98.l20-real-DiT-revision-inputs/v1",
            "source": source,
            "w96_static": static,
            "matrix_artifacts": artifacts,
            "real_revision": revision,
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
        "w98-real-refit",
        _probe_command(artifacts, args.checkpoint_revision),
        extra_args=(
            "-e",
            "CUDA_VISIBLE_DEVICES=0",
            "-v",
            f"{args.source.resolve()}:/workspace/w98-src:ro",
            "-v",
            f"{args.artifacts.resolve()}:/w98-artifacts:ro",
            "-v",
            f"{args.artifacts.resolve()}:/w96-run/artifacts:ro",
            "-v",
            f"{args.checkpoint.resolve()}:/w98-revision:ro",
        ),
    )
    probe_path = run_root / "probe/real-revision-refit-receipt.json"
    probe = _load(probe_path)
    if probe.get("status") != "passed":
        raise RefitLaunchError("real-revision probe did not pass")
    result = {
        "schema": "rlinf.w98.l20-real-DiT-revision/v1",
        "status": "passed",
        "source": source,
        "container": container,
        "probe": {
            "path": str(probe_path),
            "sha256": _sha256(probe_path),
            "checkpoint": probe["provenance"]["actor_checkpoint"],
            "timing": probe["timing"],
            "memory": probe["memory"],
            "fixed_probe": probe["fixed_probe"],
        },
    }
    _write(run_root / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        print(f"W98 real-revision refit failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
