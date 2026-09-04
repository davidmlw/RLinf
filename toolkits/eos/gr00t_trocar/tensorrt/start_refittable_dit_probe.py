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

"""Submit the EOS W83 device-weight refit lifecycle probe."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import traceback
from pathlib import Path

import start_refittable_dit_build as build_workflow


def _qualified_build(path: Path) -> Path:
    value = path.resolve(strict=True)
    if value.parent != build_workflow.RUN_ROOT:
        raise build_workflow.WorkflowError(
            "build attempt must be directly under runs/W83"
        )
    result = json.loads((value / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "passed":
        raise build_workflow.WorkflowError(
            "refittable DiT build attempt is not qualified"
        )
    required = (
        value / "engine/dit_bf16_refit.engine",
        value / "engine/rlinf-refittable-dit-engine-receipt.json",
        value / "refittable-dit-parameter-map.json",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise build_workflow.WorkflowError(
            f"qualified build artifacts are missing: {missing}"
        )
    return value


def _submit(args: argparse.Namespace) -> int:
    revision = args.revision or build_workflow._git_revision()
    build_workflow._validate_inputs(revision)
    qualified = _qualified_build(args.build_attempt)
    launcher = (
        build_workflow.SOURCE
        / "toolkits/eos/gr00t_trocar/tensorrt/start_refittable_dit_probe.py"
    )
    allocation = [
        "python3",
        str(launcher),
        "allocation-run",
        "--revision",
        revision,
        "--attempt-name",
        args.attempt_name,
        "--build-attempt",
        str(qualified),
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
        f"--output={build_workflow.RUN_ROOT}/slurm-%j.out",
        f"--error={build_workflow.RUN_ROOT}/slurm-%j.err",
        "--wrap",
        shlex.join(allocation),
    ]
    if args.dry_run:
        print(json.dumps({"command": command, "revision": revision}, indent=2))
        return 0
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise build_workflow.WorkflowError(completed.stderr.strip() or "sbatch failed")
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise build_workflow.WorkflowError(
            f"invalid Slurm job id: {completed.stdout!r}"
        )
    receipt = {
        "job_id": job_id,
        "revision": revision,
        "build_attempt": str(qualified),
        "command": command,
    }
    build_workflow._write_new(
        build_workflow.RUN_ROOT / f"submission-{job_id}.json", receipt
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _allocation_run(args: argparse.Namespace) -> int:
    build_workflow._validate_inputs(args.revision)
    qualified = _qualified_build(args.build_attempt)
    job_id = os.environ.get("SLURM_JOB_ID")
    node_list = os.environ.get("SLURM_JOB_NODELIST")
    if not job_id or not node_list:
        raise build_workflow.WorkflowError("allocation-run must execute inside Slurm")
    node = subprocess.check_output(
        ["scontrol", "show", "hostnames", node_list], text=True
    ).splitlines()[0]
    attempt = build_workflow.RUN_ROOT / f"{args.attempt_name}-{job_id}"
    attempt.mkdir(mode=0o700)
    launcher = (
        build_workflow.SOURCE
        / "toolkits/eos/gr00t_trocar/tensorrt/start_refittable_dit_probe.py"
    )
    command = [
        "srun",
        f"--nodelist={node}",
        "--nodes=1",
        "--ntasks=1",
        "--mpi=none",
        f"--container-image={build_workflow.IMAGE}",
        "--container-mounts=/lustre:/lustre",
        f"--container-workdir={build_workflow.SOURCE}",
        "--container-remap-root",
        "--no-container-mount-home",
        str(build_workflow.BUILDER_PYTHON),
        str(launcher),
        "run-agent",
        "--revision",
        args.revision,
        "--attempt",
        str(attempt),
        "--build-attempt",
        str(qualified),
    ]
    build_workflow._write_new(
        attempt / "request.json",
        {"job_id": job_id, "node": node, "revision": args.revision, "command": command},
    )
    with (
        (attempt / "srun.out").open("x", encoding="utf-8") as stdout,
        (attempt / "srun.err").open("x", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    build_workflow._write_new(
        attempt / "allocation-result.json",
        {
            "status": "passed" if completed.returncode == 0 else "failed",
            "code": completed.returncode,
        },
    )
    return completed.returncode


def _run_agent(args: argparse.Namespace) -> int:
    build_workflow._validate_inputs(args.revision)
    qualified = _qualified_build(args.build_attempt)
    attempt = args.attempt.resolve(strict=True)
    tools = build_workflow.SOURCE / "toolkits/eos/gr00t_trocar/tensorrt"
    output = attempt / "probe"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(build_workflow.ISAAC_GR00T),
            "HF_HOME": str(build_workflow.WORKSPACE / "cache/huggingface"),
        }
    )
    command = [
        sys.executable,
        str(tools / "refit_dit_lifecycle_probe.py"),
        "--source",
        str(build_workflow.ISAAC_GR00T),
        "--model",
        str(build_workflow.MODEL),
        "--collated",
        str(build_workflow.COLLATED),
        "--engine",
        str(qualified / "engine/dit_bf16_refit.engine"),
        "--engine-receipt",
        str(qualified / "engine/rlinf-refittable-dit-engine-receipt.json"),
        "--parameter-map",
        str(qualified / "refittable-dit-parameter-map.json"),
        "--output",
        str(output),
    ]
    build_workflow._stage("probe", command, attempt, env)
    result = {
        "schema": "rlinf.w83-refittable-dit-device-probe.v1",
        "status": "passed",
        "revision": args.revision,
        "build_attempt": str(qualified),
        "probe_receipt_sha256": build_workflow._sha256(
            output / "refit-lifecycle-receipt.json"
        ),
    }
    build_workflow._write_new(attempt / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--revision")
    submit.add_argument("--attempt-name", default="W83-refittable-dit-probe-r1")
    submit.add_argument("--time-limit", default="01:00:00")
    submit.add_argument("--build-attempt", type=Path, required=True)
    submit.add_argument("--dry-run", action="store_true")
    allocation = commands.add_parser("allocation-run")
    allocation.add_argument("--revision", required=True)
    allocation.add_argument("--attempt-name", required=True)
    allocation.add_argument("--build-attempt", type=Path, required=True)
    agent = commands.add_parser("run-agent")
    agent.add_argument("--revision", required=True)
    agent.add_argument("--attempt", type=Path, required=True)
    agent.add_argument("--build-attempt", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "submit":
            return _submit(args)
        if args.command == "allocation-run":
            return _allocation_run(args)
        return _run_agent(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W83 refit probe workflow failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
