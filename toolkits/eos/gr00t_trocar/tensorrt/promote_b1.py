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

"""Promote a qualified official B1 oracle into the immutable artifact cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ORACLE_ENGINES = {
    "action_decoder.engine",
    "action_encoder.engine",
    "dit_bf16.engine",
    "llm_bf16.engine",
    "state_encoder.engine",
    "vit.engine",
    "vl_self_attention.engine",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inventory(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _validate(oracle: Path, resident: Path, site: Path) -> dict[str, object]:
    qualification = _load(oracle / "qualification.json")
    oracle_result = _load(oracle / "allocation-result.json")
    resident_result = _load(resident / "allocation-result.json")
    measurement = _load(resident / "resident.json")
    site_value = _load(site)
    if any(
        value.get("status") != "passed"
        for value in (qualification, oracle_result, resident_result, measurement)
    ):
        raise RuntimeError("oracle or resident receipt is not qualified")
    if set(qualification["engines"]) != ORACLE_ENGINES:
        raise RuntimeError("qualification does not contain exactly seven engines")
    if set(measurement["engines"]) != ORACLE_ENGINES:
        raise RuntimeError("resident run did not consume exactly seven engines")
    for name in ORACLE_ENGINES:
        expected = qualification["engines"][name]
        observed = measurement["engines"][name]
        if expected["sha256"] != observed["sha256"]:
            raise RuntimeError(f"resident engine hash differs: {name}")
    for name, arm in measurement["arms"].items():
        components = arm["components"]
        if components["warmup"] != 5 or components["measured"] != 20:
            raise RuntimeError(f"{name} component sample count differs")
        if len(arm["whole_call"]["measured_samples_ms"]) != 30:
            raise RuntimeError(f"{name} whole-call sample count differs")
        if len(arm["whole_call"]["warmup_samples_ms"]) != 10:
            raise RuntimeError(f"{name} whole-call warmup count differs")
    trt = measurement["arms"]["full_tensorrt"]
    parity = trt["vs_eager"]
    if not (
        parity["finite"]
        and parity["cosine"] >= 0.999
        and parity["mean_abs"] <= 0.005
        and parity["max_abs"] <= 0.05
    ):
        raise RuntimeError("resident TensorRT final-action gate failed")
    repeat = trt["fixed_noise_repeat"]
    if not repeat["bitwise_equal"] or repeat["max_abs"] != 0.0:
        raise RuntimeError("resident TensorRT identical-noise repeat gate failed")
    return {
        "isaac_gr00t_revision": qualification["isaac_gr00t_revision"],
        "rlinf_revision": site_value["source"]["revision"],
        "official_numerics": qualification["numerics"],
        "resident_trt_vs_eager": parity,
        "resident_trt_repeat": repeat,
        "resident_whole_call": {
            name: arm["whole_call"]["statistics"]
            for name, arm in measurement["arms"].items()
        },
    }


def _copy_receipt(source: Path, output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    return _inventory(output)


def _link_artifact(source: Path, output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, output)
    output.chmod(0o444)
    return _inventory(output)


def _slurm_cleanup(job_ids: list[str]) -> dict[str, object]:
    active = subprocess.run(
        ["squeue", "--noheader", "--jobs", ",".join(job_ids), "--format", "%i|%T"],
        check=False,
        capture_output=True,
        text=True,
    )
    accounting = subprocess.run(
        [
            "sacct",
            "--jobs",
            ",".join(job_ids),
            "--noheader",
            "--parsable2",
            "--format=JobID,State,ExitCode,Elapsed,NodeList",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if active.stdout.strip():
        raise RuntimeError(f"qualification jobs remain active: {active.stdout}")
    return {
        "job_ids": job_ids,
        "active_jobs": [],
        "squeue_return_code": active.returncode,
        "squeue_stderr": active.stderr.strip(),
        "sacct": accounting.stdout.splitlines(),
        "endpoints_started": False,
    }


def promote(oracle: Path, resident: Path, site: Path, output: Path) -> Path:
    oracle = oracle.resolve(strict=True)
    resident = resident.resolve(strict=True)
    site = site.resolve(strict=True)
    checks = _validate(oracle, resident, site)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    artifacts = {}
    for group in ("engines", "onnx"):
        source_root = oracle / "official" / group
        for source in sorted(source_root.iterdir()):
            if source.is_file():
                relative = f"artifacts/{group}/{source.name}"
                artifacts[relative] = _link_artifact(source, output / relative)

    receipts = {}
    oracle_receipts = [
        *[path for path in oracle.iterdir() if path.is_file()],
        *(oracle / "logs").iterdir(),
        oracle / "official/pipeline.log",
        oracle / "model-view/libero_10/rlinf-model-view.json",
    ]
    resident_receipts = [path for path in resident.rglob("*") if path.is_file()]
    for prefix, root, sources in (
        ("oracle", oracle, oracle_receipts),
        ("resident", resident, resident_receipts),
    ):
        for source in sorted(sources):
            relative = Path("receipts") / prefix / source.relative_to(root)
            receipts[str(relative)] = _copy_receipt(source, output / relative)
    receipts["receipts/site.json"] = _copy_receipt(site, output / "receipts/site.json")

    job_ids = [
        str(_load(oracle / "request.json")["job_id"]),
        str(_load(resident / "request.json")["job_id"]),
    ]
    manifest = {
        "schema": "rlinf.gr00t-n1d7-official-b1-bundle.v1",
        "status": "passed",
        "created_unix_s": time.time(),
        "oracle_attempt": str(oracle),
        "resident_attempt": str(resident),
        "site": str(site),
        "checks": checks,
        "cleanup": _slurm_cleanup(job_ids),
        "artifacts": artifacts,
        "receipts": receipts,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for path in output.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    output.chmod(0o555)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--resident", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = promote(args.oracle, args.resident, args.site, args.output)
    print(json.dumps({"manifest": str(manifest), **_inventory(manifest)}, indent=2))


if __name__ == "__main__":
    main()
