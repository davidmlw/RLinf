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

"""Submit and retain the official GR00T N1.7 B1 TensorRT oracle on EOS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "rlinf.eos.gr00t-tensorrt-site.v1"
RESULT_SCHEMA = "rlinf.eos.gr00t-tensorrt-result.v1"
GIT_OID_RE = re.compile(r"[0-9a-f]{40}")
HELPERS = ("prepare_builder.py", "model_view.py", "official_b1.py")


class WorkflowError(RuntimeError):
    """A fail-closed TensorRT qualification workflow error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise WorkflowError(f"git {' '.join(args)} failed in {root}: {message}")
    return completed.stdout.strip()


def _directory(value: object, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise WorkflowError(f"{label} must be an absolute directory")
    return path.resolve(strict=True)


def _file(value: object, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise WorkflowError(f"{label} must be an absolute regular file")
    return path.resolve(strict=True)


def _load(path: Path, *, verify_image: bool = True) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "slurm",
        "container",
        "source",
        "inputs",
        "builder",
        "experiment",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise WorkflowError("site has unexpected fields")
    if value["schema"] != SCHEMA:
        raise WorkflowError("site has the wrong schema")

    source = value["source"]
    root = _directory(source["root"], "source.root")
    revision = source["revision"]
    if revision == "AUTO":
        revision = _git(root, "rev-parse", "HEAD")
    if not isinstance(revision, str) or GIT_OID_RE.fullmatch(revision) is None:
        raise WorkflowError("source.revision must be AUTO or a full Git OID")
    if _git(root, "rev-parse", "HEAD") != revision:
        raise WorkflowError("RLinf source revision changed after materialization")
    ancestor = source["required_ancestor"]
    if not isinstance(ancestor, str) or GIT_OID_RE.fullmatch(ancestor) is None:
        raise WorkflowError("source.required_ancestor must be a full Git OID")
    if subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, revision],
        check=False,
    ).returncode:
        raise WorkflowError("RLinf source does not descend from the frozen W77 base")
    if source["require_clean"] and _git(root, "status", "--porcelain"):
        raise WorkflowError("RLinf source must be clean")

    image = _file(value["container"]["image"], "container.image")
    if verify_image and _sha256(image) != value["container"]["image_sha256"]:
        raise WorkflowError("container image SHA-256 mismatch")
    inputs = value["inputs"]
    gr00t = _directory(inputs["isaac_gr00t_root"], "inputs.isaac_gr00t_root")
    if _git(gr00t, "rev-parse", "HEAD") != inputs["isaac_gr00t_revision"]:
        raise WorkflowError("Isaac-GR00T revision mismatch")
    if _git(gr00t, "status", "--porcelain"):
        raise WorkflowError("Isaac-GR00T source must be clean")
    dataset = _directory(inputs["dataset_root"], "inputs.dataset_root")
    pointers = []
    for candidate in dataset.rglob("*"):
        if candidate.is_file() and candidate.stat().st_size < 1024:
            if candidate.read_bytes().startswith(b"version https://git-lfs.github.com"):
                pointers.append(str(candidate))
    if pointers:
        raise WorkflowError(f"dataset contains Git LFS pointers: {pointers[:3]}")
    _directory(inputs["model_root"], "inputs.model_root")
    _directory(inputs["backbone_root"], "inputs.backbone_root")
    _file(value["builder"]["uv"], "builder.uv")
    output = _directory(value["experiment"]["output_root"], "experiment.output_root")
    _directory(value["experiment"]["artifact_cache"], "experiment.artifact_cache")

    helper_root = root / "toolkits/eos/gr00t_trocar/tensorrt"
    value["source"]["revision"] = revision
    value["_resolved"] = {
        "site": str(resolved),
        "site_sha256": _sha256(resolved),
        "source_root": str(root),
        "source_revision": revision,
        "image": str(image),
        "dataset": str(dataset),
        "output_root": str(output),
        "launcher": str((helper_root / Path(__file__).name).resolve(strict=True)),
        "helpers": {
            name: _sha256((helper_root / name).resolve(strict=True)) for name in HELPERS
        },
    }
    return value


def _materialize(args: argparse.Namespace) -> int:
    template = Path(args.template).resolve(strict=True)
    value = json.loads(template.read_text(encoding="utf-8"))
    if value.get("source", {}).get("revision") == "AUTO":
        source = _directory(value["source"]["root"], "source.root")
        value["source"]["revision"] = _git(source, "rev-parse", "HEAD")
    output = Path(args.output)
    if not output.is_absolute():
        raise WorkflowError("materialized site path must be absolute")
    _write_new(output, value)
    site = _load(output, verify_image=not args.skip_image_hash)
    print(json.dumps(site["_resolved"], indent=2, sort_keys=True))
    return 0


def _sbatch(site: dict[str, Any]) -> list[str]:
    slurm = site["slurm"]
    experiment = site["experiment"]
    resolved = site["_resolved"]
    argv = [
        "sbatch",
        "--parsable",
        f"--job-name={experiment['name']}",
        f"--account={slurm['account']}",
        f"--partition={slurm['partition']}",
        f"--constraint={slurm['constraint']}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={slurm['cpus_per_task']}",
        f"--time={slurm['time_limit']}",
        f"--signal=B:TERM@{slurm['signal_seconds']}",
        f"--output={resolved['output_root']}/slurm-%j.out",
        f"--error={resolved['output_root']}/slurm-%j.err",
    ]
    if slurm["exclusive"]:
        argv.append("--exclusive")
    argv.extend([resolved["launcher"], "allocation-run", "--site", resolved["site"]])
    return argv


def _submit(args: argparse.Namespace) -> int:
    site = _load(Path(args.site), verify_image=not args.skip_image_hash)
    command = _sbatch(site)
    if args.dry_run:
        print(json.dumps({"command": command, "site": site["_resolved"]}, indent=2))
        return 0
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise WorkflowError(completed.stderr.strip() or "sbatch failed")
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise WorkflowError(f"sbatch returned an invalid job id: {completed.stdout!r}")
    receipt = {"job_id": job_id, "command": command, "site": site["_resolved"]}
    _write_new(
        Path(site["_resolved"]["output_root"]) / "submissions" / f"{job_id}.json",
        receipt,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _allocation_run(args: argparse.Namespace) -> int:
    site = _load(Path(args.site))
    job_id = os.environ.get("SLURM_JOB_ID")
    node_list = os.environ.get("SLURM_JOB_NODELIST")
    if not job_id or not job_id.isdigit() or not node_list:
        raise WorkflowError("allocation-run must execute inside Slurm")
    node = subprocess.check_output(
        ["scontrol", "show", "hostnames", node_list], text=True
    ).splitlines()[0]
    attempt = (
        Path(site["_resolved"]["output_root"])
        / f"{site['experiment']['name']}-{job_id}"
    )
    attempt.mkdir(mode=0o700)
    (attempt / "logs").mkdir()
    _write_new(
        attempt / "request.json",
        {"job_id": job_id, "node": node, "site": site["_resolved"]},
    )
    container = site["container"]
    command = [
        "srun",
        f"--nodelist={node}",
        "--nodes=1",
        "--ntasks=1",
        "--mpi=none",
        f"--container-image={container['image']}",
        f"--container-mounts={','.join(container['mounts'])}",
        f"--container-workdir={site['_resolved']['source_root']}",
        "--container-remap-root",
        "--no-container-mount-home",
        "python3",
        site["_resolved"]["launcher"],
        "run-agent",
        "--site",
        site["_resolved"]["site"],
        "--attempt",
        str(attempt),
    ]
    _write_new(attempt / "srun-command.json", command)
    with (
        (attempt / "logs/srun.out").open("w", encoding="utf-8") as stdout,
        (attempt / "logs/srun.err").open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    _write_new(
        attempt / "allocation-result.json",
        {
            "schema": RESULT_SCHEMA,
            "status": "passed" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "finished_unix_s": time.time(),
        },
    )
    return completed.returncode


def _logged_run(command: list[str], output: Path, error: Path) -> None:
    with (
        output.open("w", encoding="utf-8") as stdout,
        error.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode:
        raise WorkflowError(f"command failed ({completed.returncode}): {command}")


def _run_agent(args: argparse.Namespace) -> int:
    site = _load(Path(args.site))
    attempt = Path(args.attempt).resolve(strict=True)
    query = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(query, check=True, capture_output=True, text=True)
    gpus = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(gpus) != 8 or any("H100" not in line for line in gpus):
        raise WorkflowError(f"expected exactly 8 H100 GPUs, found {gpus}")
    _write_new(attempt / "gpu.json", {"query": query, "gpus": gpus})

    source = Path(site["_resolved"]["source_root"])
    tools = source / "toolkits/eos/gr00t_trocar/tensorrt"
    builder = site["builder"]
    prepare = [
        "python3",
        str(tools / "prepare_builder.py"),
        "--source",
        site["inputs"]["isaac_gr00t_root"],
        "--env-root",
        builder["env_root"],
        "--uv",
        builder["uv"],
        "--uv-cache",
        builder["uv_cache"],
    ]
    _logged_run(prepare, attempt / "logs/prepare.out", attempt / "logs/prepare.err")
    run = [
        str(Path(builder["env_root"]) / "bin/python"),
        str(tools / "official_b1.py"),
        "--attempt",
        str(attempt),
        "--source",
        site["inputs"]["isaac_gr00t_root"],
        "--model",
        site["inputs"]["model_root"],
        "--backbone",
        site["inputs"]["backbone_root"],
        "--dataset",
        site["inputs"]["dataset_root"],
        "--builder-python",
        str(Path(builder["env_root"]) / "bin/python"),
    ]
    _write_new(attempt / "oracle-command.json", run)
    _logged_run(run, attempt / "logs/oracle.out", attempt / "logs/oracle.err")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--template", required=True)
    materialize.add_argument("--output", required=True)
    materialize.add_argument("--skip-image-hash", action="store_true")
    submit = subparsers.add_parser("submit")
    submit.add_argument("--site", required=True)
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--skip-image-hash", action="store_true")
    allocation = subparsers.add_parser("allocation-run")
    allocation.add_argument("--site", required=True)
    agent = subparsers.add_parser("run-agent")
    agent.add_argument("--site", required=True)
    agent.add_argument("--attempt", required=True)
    args = parser.parse_args()
    try:
        return {
            "materialize": _materialize,
            "submit": _submit,
            "allocation-run": _allocation_run,
            "run-agent": _run_agent,
        }[args.action](args)
    except Exception as error:
        print(f"W78 TensorRT workflow failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
