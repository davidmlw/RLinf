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

"""Create an immutable additive correction for a qualified B1 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

EXPECTED_ENGINES = {
    "action_decoder.engine",
    "action_encoder.engine",
    "dit_bf16.engine",
    "llm_bf16.engine",
    "state_encoder.engine",
    "vit.engine",
    "vl_self_attention.engine",
}
DYNAMIC_SEQUENCE_ENGINES = {
    "dit_bf16.engine",
    "llm_bf16.engine",
    "vl_self_attention.engine",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _source_revision(request: dict[str, Any]) -> str:
    return str(request["site"]["source_revision"])


def _cleanup(job_id: str) -> dict[str, Any]:
    active = subprocess.run(
        ["squeue", "--noheader", "--jobs", job_id, "--format", "%i|%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    accounting = subprocess.run(
        [
            "sacct",
            "--jobs",
            job_id,
            "--noheader",
            "--parsable2",
            "--format=JobID,State,ExitCode,Elapsed,NodeList",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if active.stdout.strip():
        raise RuntimeError(f"fixture job remains active: {active.stdout}")
    if not any("|COMPLETED|0:0|" in line for line in accounting.stdout.splitlines()):
        raise RuntimeError("fixture job has no successful Slurm accounting record")
    return {
        "job_id": job_id,
        "active_jobs": [],
        "sacct": accounting.stdout.splitlines(),
        "endpoints_started": False,
        "post_release_process_counter": {
            "availability": "unavailable",
            "reason": "allocation released before correction was created",
        },
        "post_release_gpu_process_counter": {
            "availability": "unavailable",
            "reason": "allocation released before correction was created",
        },
    }


def _copy_receipts(source: Path, output: Path) -> dict[str, Any]:
    inventory = {}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = Path("receipts/fixture") / path.relative_to(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        inventory[str(relative)] = _inventory(destination)
    return inventory


def correct(
    bundle: Path,
    fixture_attempt: Path,
    output: Path,
    promotion_tool_revision: str,
) -> Path:
    bundle = bundle.resolve(strict=True)
    fixture_attempt = fixture_attempt.resolve(strict=True)
    base_manifest_path = bundle / "manifest.json"
    base = _load(base_manifest_path)
    fixture = _load(fixture_attempt / "fixture.json")
    fixture_result = _load(fixture_attempt / "allocation-result.json")
    fixture_request = _load(fixture_attempt / "request.json")
    if base.get("schema") != "rlinf.gr00t-n1d7-official-b1-bundle.v1":
        raise RuntimeError("correction requires the qualified v1 bundle")
    if any(
        value.get("status") != "passed" for value in (base, fixture, fixture_result)
    ):
        raise RuntimeError("base bundle or fixture witness is not qualified")

    oracle_request = _load(bundle / "receipts/oracle/request.json")
    resident_request = _load(bundle / "receipts/resident/request.json")
    oracle_revision = _source_revision(oracle_request)
    resident_revision = _source_revision(resident_request)
    if base["checks"]["rlinf_revision"] != resident_revision:
        raise RuntimeError("base revision does not match the resident tool revision")
    if promotion_tool_revision != resident_revision:
        raise RuntimeError(
            "promotion tool revision must match the recorded v1 revision"
        )

    artifact_engines = {
        path.name: _inventory(path)
        for path in sorted((bundle / "artifacts/engines").glob("*.engine"))
    }
    if set(artifact_engines) != EXPECTED_ENGINES:
        raise RuntimeError("base bundle does not contain exactly seven engines")
    if fixture["engines"] != artifact_engines:
        raise RuntimeError("fixture witness did not consume the qualified engines")
    noise = fixture["flow_noise"]
    if not noise["bitwise_replay"] or noise["actual"] != noise["seeded_replay"]:
        raise RuntimeError("fixture flow-noise replay is not bitwise identical")
    if not fixture["postprocessed_model_inputs"]["aggregate_sha256"]:
        raise RuntimeError("fixture model-input manifest is missing")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    receipts = _copy_receipts(fixture_attempt, output)
    job_id = str(fixture_request["job_id"])
    correction = {
        "schema": "rlinf.gr00t-n1d7-official-b1-bundle-correction.v2",
        "status": "passed",
        "created_unix_s": time.time(),
        "base_bundle": {
            "path": str(bundle),
            "manifest": _inventory(base_manifest_path),
        },
        "revisions": {
            "oracle_rlinf_revision": oracle_revision,
            "resident_rlinf_revision": resident_revision,
            "promotion_tool_revision": promotion_tool_revision,
            "fixture_tool_revision": _source_revision(fixture_request),
            "isaac_gr00t_revision": fixture["source_revision"],
        },
        "engine_profiles": {
            "static_batch_engines": sorted(EXPECTED_ENGINES),
            "dynamic_sequence_engines": sorted(DYNAMIC_SEQUENCE_ENGINES),
            "source": "qualified export_metadata.json and binding receipt",
        },
        "fixture_witness": {
            "attempt": str(fixture_attempt),
            "receipt": _inventory(fixture_attempt / "fixture.json"),
            "model_input_aggregate_sha256": fixture["postprocessed_model_inputs"][
                "aggregate_sha256"
            ],
            "flow_noise": noise,
            "public_action_aggregate_sha256": fixture["public_action"][
                "aggregate_sha256"
            ],
        },
        "lifecycle_availability": {
            "dependency_and_data_materialization_duration": {
                "availability": "unavailable",
                "reason": "builder and inputs pre-existed the qualified oracle attempt",
                "included_in_resident_latency": False,
            }
        },
        "cleanup": _cleanup(job_id),
        "receipts": receipts,
    }
    correction_path = output / "correction.json"
    correction_path.write_text(
        json.dumps(correction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for path in output.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    output.chmod(0o555)
    return correction_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--fixture-attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promotion-tool-revision", required=True)
    args = parser.parse_args()
    result = correct(
        args.bundle, args.fixture_attempt, args.output, args.promotion_tool_revision
    )
    print(json.dumps({"correction": str(result), **_inventory(result)}, indent=2))


if __name__ == "__main__":
    main()
