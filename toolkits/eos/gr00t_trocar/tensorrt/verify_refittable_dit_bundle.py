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

"""Verify a refittable TensorRT DiT build and lifecycle qualification bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"[0-9a-f]{64}")
ENGINE_RECEIPT_SCHEMA = (
    "rlinf.gr00t-n1d7-trocar-true-b8-refittable-dit-engine.v1"
)
QUALIFICATION_SCHEMA = "rlinf.gr00t-n1d7-refittable-dit-device-lifecycle.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("status") != "passed"
    ):
        raise ValueError(f"unqualified receipt: {path}")
    return value


def verify(build_root: Path, qualification_path: Path) -> dict[str, Any]:
    build_root = build_root.resolve(strict=True)
    qualification_path = qualification_path.resolve(strict=True)
    if not build_root.is_dir():
        raise ValueError("build root must be a directory")
    engine_dir = build_root / "engine"
    plan = engine_dir / "dit_bf16_refit.engine"
    engine_receipt_path = engine_dir / "rlinf-refittable-dit-engine-receipt.json"
    parameter_map_path = build_root / "refittable-dit-parameter-map.json"
    for path in (plan, engine_receipt_path, parameter_map_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required refittable DiT artifact is missing: {path}")

    engine_receipt = _load(engine_receipt_path, ENGINE_RECEIPT_SCHEMA)
    qualification = _load(qualification_path, QUALIFICATION_SCHEMA)
    hashes = {
        "engine": _sha256(plan),
        "engine_receipt": _sha256(engine_receipt_path),
        "parameter_map": _sha256(parameter_map_path),
        "qualification": _sha256(qualification_path),
    }
    expected_engine = engine_receipt.get("engine", {}).get("sha256")
    expected_parameter_map = engine_receipt.get("parameter_map", {}).get("sha256")
    if hashes["engine"] != expected_engine:
        raise ValueError("engine plan does not match its build receipt")
    if hashes["parameter_map"] != expected_parameter_map:
        raise ValueError("parameter map does not match its build receipt")

    provenance = qualification.get("provenance", {})
    if hashes["engine"] != provenance.get("engine", {}).get("sha256"):
        raise ValueError("lifecycle qualification used a different engine plan")
    if hashes["engine_receipt"] != provenance.get("engine_receipt"):
        raise ValueError("lifecycle qualification used a different engine receipt")
    if hashes["parameter_map"] != provenance.get("parameter_map"):
        raise ValueError("lifecycle qualification used a different parameter map")

    source_digest = qualification.get("device_weights", {}).get(
        "source_digest_revision_0"
    )
    if not isinstance(source_digest, str) or SHA256_RE.fullmatch(source_digest) is None:
        raise ValueError("lifecycle qualification omits revision-zero source digest")
    fixed_probe = qualification.get("fixed_probe", {})
    thresholds = fixed_probe.get("thresholds", {})
    cosine_min = float(thresholds.get("cosine_min", 0.0))
    relative_l2_max = float(thresholds.get("relative_l2_max", 1.0))
    if (
        not math.isfinite(cosine_min)
        or not math.isfinite(relative_l2_max)
        or cosine_min < 0.999
        or relative_l2_max > 0.05
    ):
        raise ValueError("lifecycle qualification used weakened numerical thresholds")

    return {
        "schema": "rlinf.gr00t-n1d7-refittable-dit-runtime-bundle.v1",
        "status": "passed",
        "scope": "experimental_approximate_behavior_only",
        "build_root": str(build_root),
        "paths": {
            "engine": str(plan),
            "engine_receipt": str(engine_receipt_path),
            "parameter_map": str(parameter_map_path),
            "qualification": str(qualification_path),
        },
        "sha256": hashes,
        "source_digest_revision_0": source_digest,
        "ppo_authority": "failed_ratio_kl_approximate_behavior_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if output.exists():
            raise ValueError(f"refusing to replace existing output: {output}")
        receipt = verify(args.build_root, args.qualification)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    except Exception as error:
        print(f"refittable DiT bundle verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
