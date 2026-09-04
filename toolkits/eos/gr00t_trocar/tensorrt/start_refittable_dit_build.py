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

"""Submit and run the EOS W83 refittable-DiT build-only gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

WORKSPACE = Path("/lustre/fsw/coreai_devtech_all/liweim/rlinf-workspace")
SOURCE = WORKSPACE / "RLinf"
RUN_ROOT = WORKSPACE / "runs/W83"
IMAGE = WORKSPACE / "inputs/images/rlinf-eos-system-cuda128-ubuntu2404-be095dc.sqsh"
IMAGE_SHA256 = "64bbd7bda0f8d65d298073377a3e2331e91a75c49d459893ae5b3096410b022c"
BUILDER_PYTHON = WORKSPACE / "envs/gr00t-n1d7-trt-builder-py312-cu128-v5/bin/python"
ISAAC_GR00T = WORKSPACE / "inputs/Isaac-GR00T-N1.7"
ISAAC_GR00T_REVISION = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
MODEL = WORKSPACE / "runs/W80/g0-fixture-r4/model-view"
COLLATED = WORKSPACE / "runs/W80/g1-fixture-r4-5967115/fixture/collated-inputs.pt"
FIXTURE_RECEIPT = WORKSPACE / "runs/W80/g1-fixture-r4-5967115/fixture/fixture.json"


class WorkflowError(RuntimeError):
    """Fail-closed W83 workflow error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()


def _validate_inputs(expected_revision: str) -> None:
    required = (
        SOURCE,
        IMAGE,
        BUILDER_PYTHON,
        ISAAC_GR00T,
        MODEL,
        COLLATED,
        FIXTURE_RECEIPT,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise WorkflowError(f"missing W83 build inputs: {missing}")
    if _sha256(IMAGE) != IMAGE_SHA256:
        raise WorkflowError("EOS container image SHA-256 mismatch")
    if _git_revision() != expected_revision:
        raise WorkflowError(
            f"RLInf revision mismatch: {_git_revision()} != {expected_revision}"
        )
    status = subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain"], text=True
    )
    if status:
        raise WorkflowError(f"RLInf checkout is dirty:\n{status}")


def _submit(args: argparse.Namespace) -> int:
    revision = args.revision or _git_revision()
    _validate_inputs(revision)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    launcher = (
        SOURCE / "toolkits/eos/gr00t_trocar/tensorrt/start_refittable_dit_build.py"
    )
    allocation = [
        "python3",
        str(launcher),
        "allocation-run",
        "--revision",
        revision,
        "--attempt-name",
        args.attempt_name,
    ]
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={args.attempt_name}",
        "--account=coreai_devtech_all",
        "--partition=batch",
        "--constraint=h100",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=32",
        f"--time={args.time_limit}",
        "--signal=B:TERM@600",
        "--exclusive",
        f"--output={RUN_ROOT}/slurm-%j.out",
        f"--error={RUN_ROOT}/slurm-%j.err",
        "--wrap",
        shlex.join(allocation),
    ]
    if args.dry_run:
        print(json.dumps({"command": command, "revision": revision}, indent=2))
        return 0
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise WorkflowError(completed.stderr.strip() or "sbatch failed")
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise WorkflowError(f"invalid Slurm job id: {completed.stdout!r}")
    receipt = {"job_id": job_id, "revision": revision, "command": command}
    _write_new(RUN_ROOT / f"submission-{job_id}.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _allocation_run(args: argparse.Namespace) -> int:
    _validate_inputs(args.revision)
    job_id = os.environ.get("SLURM_JOB_ID")
    node_list = os.environ.get("SLURM_JOB_NODELIST")
    if not job_id or not node_list:
        raise WorkflowError("allocation-run must execute inside Slurm")
    node = subprocess.check_output(
        ["scontrol", "show", "hostnames", node_list], text=True
    ).splitlines()[0]
    attempt = RUN_ROOT / f"{args.attempt_name}-{job_id}"
    attempt.mkdir(mode=0o700)
    launcher = (
        SOURCE / "toolkits/eos/gr00t_trocar/tensorrt/start_refittable_dit_build.py"
    )
    command = [
        "srun",
        f"--nodelist={node}",
        "--nodes=1",
        "--ntasks=1",
        "--mpi=none",
        f"--container-image={IMAGE}",
        "--container-mounts=/lustre:/lustre",
        f"--container-workdir={SOURCE}",
        "--container-remap-root",
        "--no-container-mount-home",
        str(BUILDER_PYTHON),
        str(launcher),
        "run-agent",
        "--revision",
        args.revision,
        "--attempt",
        str(attempt),
    ]
    _write_new(
        attempt / "request.json",
        {"job_id": job_id, "node": node, "revision": args.revision, "command": command},
    )
    with (
        (attempt / "srun.out").open("x", encoding="utf-8") as stdout,
        (attempt / "srun.err").open("x", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    _write_new(
        attempt / "allocation-result.json",
        {
            "status": "passed" if completed.returncode == 0 else "failed",
            "code": completed.returncode,
        },
    )
    return completed.returncode


def _stage(name: str, command: list[str], attempt: Path, env: dict[str, str]) -> None:
    _write_new(attempt / f"{name}.command.json", command)
    started = time.perf_counter()
    with (
        (attempt / f"{name}.out").open("x", encoding="utf-8") as stdout,
        (attempt / f"{name}.err").open("x", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=SOURCE,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    _write_new(
        attempt / f"{name}.result.json",
        {"return_code": completed.returncode, "wall_s": time.perf_counter() - started},
    )
    if completed.returncode:
        raise WorkflowError(f"W83 stage {name} failed with code {completed.returncode}")


def _run_agent(args: argparse.Namespace) -> int:
    _validate_inputs(args.revision)
    attempt = args.attempt.resolve(strict=True)
    tools = SOURCE / "toolkits/eos/gr00t_trocar/tensorrt"
    onnx = attempt / "onnx"
    parameter_map = attempt / "refittable-dit-parameter-map.json"
    engine = attempt / "engine"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(ISAAC_GR00T),
            "HF_HOME": str(WORKSPACE / "cache/huggingface"),
            "UV_CACHE_DIR": str(WORKSPACE / "cache/uv"),
        }
    )
    _stage(
        "export",
        [
            sys.executable,
            str(tools / "export_refittable_dit_b8.py"),
            "--source",
            str(ISAAC_GR00T),
            "--model",
            str(MODEL),
            "--collated",
            str(COLLATED),
            "--fixture-receipt",
            str(FIXTURE_RECEIPT),
            "--output",
            str(onnx),
        ],
        attempt,
        env,
    )
    _stage(
        "parameter-map",
        [
            sys.executable,
            str(tools / "refittable_dit_contract.py"),
            "--checkpoint",
            str(MODEL),
            "--model-config",
            str(MODEL / "config.json"),
            "--onnx",
            str(onnx / "dit_bf16.onnx"),
            "--output",
            str(parameter_map),
        ],
        attempt,
        env,
    )
    _stage(
        "build",
        [
            sys.executable,
            str(tools / "build_refittable_dit_b8.py"),
            "--onnx",
            str(onnx / "dit_bf16.onnx"),
            "--parameter-map",
            str(parameter_map),
            "--output",
            str(engine),
        ],
        attempt,
        env,
    )
    result = {
        "schema": "rlinf.w83-refittable-dit-build-only.v1",
        "status": "passed",
        "revision": args.revision,
        "image_sha256": IMAGE_SHA256,
        "python": sys.executable,
        "artifacts": {
            "export_receipt": _sha256(onnx / "rlinf-dit-export-receipt.json"),
            "parameter_map": _sha256(parameter_map),
            "engine_receipt": _sha256(
                engine / "rlinf-refittable-dit-engine-receipt.json"
            ),
            "engine": _sha256(engine / "dit_bf16_refit.engine"),
        },
    }
    _write_new(attempt / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--revision")
    submit.add_argument("--attempt-name", default="W83-refittable-dit-build-r1")
    submit.add_argument("--time-limit", default="01:00:00")
    submit.add_argument("--dry-run", action="store_true")
    allocation = commands.add_parser("allocation-run")
    allocation.add_argument("--revision", required=True)
    allocation.add_argument("--attempt-name", required=True)
    agent = commands.add_parser("run-agent")
    agent.add_argument("--revision", required=True)
    agent.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "submit":
            return _submit(args)
        if args.command == "allocation-run":
            return _allocation_run(args)
        return _run_agent(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W83 refittable DiT workflow failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
