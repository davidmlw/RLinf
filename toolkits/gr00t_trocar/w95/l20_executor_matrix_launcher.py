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

"""Launch the W98 executor matrix through the W96-qualified L20 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from . import l20_vulkan_launcher as runtime
except ImportError:
    import l20_vulkan_launcher as runtime

RESULT_SCHEMA = "rlinf.w98.l20-executor-matrix-launch/v1"


class LaunchError(RuntimeError):
    """Raised when the W98 launch contract is not satisfied."""


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
    temporary.replace(path)


def _source_identity(source: Path, expected_revision: str) -> dict[str, Any]:
    source = source.resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    )
    if revision != expected_revision:
        raise LaunchError(
            f"source revision mismatch: {revision} != {expected_revision}"
        )
    if status:
        raise LaunchError(f"source checkout is dirty:\n{status}")
    return {"path": str(source), "revision": revision, "clean": True}


def _extra_docker_args(site: dict[str, Any], source: Path) -> tuple[str, ...]:
    metadata = Path(site["inputs"]["trocar_metadata"])
    return (
        "-e",
        "CUDA_VISIBLE_DEVICES=0",
        "-v",
        f"{source}:/workspace/w98-src:ro",
        "-v",
        f"{metadata}:/w98-inputs/trocar/metadata.json:ro",
    )


def _agent_command(revision: str) -> str:
    return (
        'set -euo pipefail; export LD_LIBRARY_PATH="/w96-trt-runtime/'
        'tensorrt_libs:${LD_LIBRARY_PATH:-}"; '
        "export PYTHONPATH=/w96-overlay:/w96-trt-runtime:/workspace/gr00t-n17:"
        "/workspace/gr00t-n17/scripts/deployment:/workspace/w98-src:"
        "/workspace/w98-src/toolkits/eos/gr00t_trocar/tensorrt; "
        "exec /isaac-sim/kit/python/bin/python3 "
        "/workspace/w98-src/toolkits/gr00t_trocar/w95/l20_executor_matrix_agent.py "
        "--source /workspace/w98-src --gr00t-source /workspace/gr00t-n17 "
        f"--rlinf-revision {revision} "
        "--model /models/GR00T-N1.7-3B "
        "--backbone /w96-model-inputs/Cosmos-Reason2-2B "
        "--metadata /w98-inputs/trocar/metadata.json "
        "--output /w96-run/artifacts --seed 47 --workspace-mib 8192 "
        "--warmup 10 --measured 30"
    )


def _validate_new_run_root(run_root: Path) -> Path:
    run_root = run_root.resolve()
    if run_root.exists():
        raise LaunchError(f"run root already exists: {run_root}")
    if "/results/W98/" not in f"{run_root}/":
        raise LaunchError("run root must be below an owned results/W98 directory")
    return run_root


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_identity = _source_identity(args.source, args.revision)
    site = runtime._load(args.site.resolve(strict=True))
    static = runtime.validate_site(args.site)
    run_root = _validate_new_run_root(args.run_root)
    run_root.mkdir(parents=True)
    (run_root / "scratch/assets-cache").mkdir(parents=True)
    receipts = run_root / "receipts"
    receipts.mkdir()
    shutil.copy2(args.site, receipts / "w96-site.json")
    shutil.copy2(Path(__file__), receipts / "launcher.py")
    agent = Path(__file__).with_name("l20_executor_matrix_agent.py")
    shutil.copy2(agent, receipts / "agent.py")
    _write(
        receipts / "inputs.json",
        {
            "schema": "rlinf.w98.l20-executor-matrix-inputs/v1",
            "source": source_identity,
            "w96_static": static,
            "site_sha256": _sha256(args.site),
            "launcher_sha256": _sha256(Path(__file__)),
            "agent_sha256": _sha256(agent),
            "common_base": "W97 composed feature reuse source lineage",
            "standalone_boundary_excludes_feature_transport": True,
        },
    )
    preflight = {
        "quota": runtime._quota_receipt(site["preflight"]["quota_mount"]),
        "gpus": runtime._gpu_receipt(),
        "image": runtime._canonical_image_matches(Path(site["docker"]["path"]), site)[
            "checks"
        ],
    }
    _write(receipts / "preflight.json", preflight)
    container = runtime._run_container(
        site,
        run_root,
        "w98-matrix",
        _agent_command(args.revision),
        extra_args=_extra_docker_args(site, args.source.resolve()),
    )
    result_path = run_root / "artifacts/result.json"
    result = runtime._load(result_path)
    if result.get("status") not in {"passed", "systems_only"}:
        raise LaunchError(f"executor matrix did not complete: {result}")
    output = {
        "schema": RESULT_SCHEMA,
        "status": "passed",
        "disposition": result["status"],
        "source": source_identity,
        "container": container,
        "matrix_result": {"path": str(result_path), "sha256": _sha256(result_path)},
    }
    _write(run_root / "result.json", output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan:
        print(
            json.dumps(
                {
                    "source": str(args.source),
                    "revision": args.revision,
                    "site": str(args.site),
                    "run_root": str(args.run_root),
                    "container_command": _agent_command(args.revision),
                    "container_extra_args": _extra_docker_args(
                        runtime._load(args.site), args.source
                    ),
                    "container_or_gpu_started": False,
                },
                indent=2,
            )
        )
        return 0
    try:
        result = run(args)
    except Exception as error:
        print(f"W98 L20 executor matrix failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
