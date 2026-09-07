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

"""Submit and run the EOS W84 true-B8 executor matrix."""

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
SOURCE = Path(__file__).resolve().parents[4]
RUN_ROOT = WORKSPACE / "runs/W84"
IMAGE = WORKSPACE / "inputs/images/rlinf-eos-system-cuda128-ubuntu2404-be095dc.sqsh"
IMAGE_SHA256 = "64bbd7bda0f8d65d298073377a3e2331e91a75c49d459893ae5b3096410b022c"
BUILDER_PYTHON = WORKSPACE / "envs/gr00t-n1d7-trt-builder-py312-cu128-v5/bin/python"
TEST_PYTHON = WORKSPACE / "envs/gr00t-n1d7-newton-py312-cu128-v1/bin/python"
ISAAC_GR00T = WORKSPACE / "inputs/Isaac-GR00T-N1.7"
ISAAC_GR00T_REVISION = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
MODEL = WORKSPACE / "runs/W80/g0-fixture-r4/model-view"
FIXTURE = WORKSPACE / "runs/W80/g1-fixture-r4-5967115/fixture"
EXPORT = WORKSPACE / "runs/W80/g2-export-build-r1-5967165/onnx"
BACKBONE_ENGINES = WORKSPACE / "runs/W80/g2-export-build-r1-5967165/engines"
REFIT_ROOT = WORKSPACE / "runs/W83/W83-refittable-dit-build-r3-5972556"
REFIT_ENGINE_RECEIPT_SHA256 = (
    "774652d469c47884c6756fe98196884df74cdc129bd611afcf7ea949be7cf024"
)
REFIT_PARAMETER_MAP_SHA256 = (
    "df7c72b90629ff6343f52c066a03d430728cc7cd605d12c1b884851fed48c935"
)
REFIT_SOURCE_DIGEST = "dcadd3c8a2bf405e53dc23aded49c536d0315f4d68f86417feb59a321bd2aaca"


class WorkflowError(RuntimeError):
    """Fail-closed W84 workflow error."""


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


def _validate_inputs(expected_revision: str) -> dict[str, Any]:
    required = {
        "source": SOURCE,
        "image": IMAGE,
        "builder_python": BUILDER_PYTHON,
        "test_python": TEST_PYTHON,
        "isaac_gr00t": ISAAC_GR00T,
        "model": MODEL,
        "collated": FIXTURE / "collated-inputs.pt",
        "raw": FIXTURE / "raw-observation.npz",
        "fixture_receipt": FIXTURE / "fixture.json",
        "export_receipt": EXPORT / "rlinf-export-receipt.json",
        "backbone_engine_receipt": BACKBONE_ENGINES / "rlinf-engine-receipt.json",
        "vit_engine": BACKBONE_ENGINES / "vit.engine",
        "llm_engine": BACKBONE_ENGINES / "llm_bf16.engine",
        "refit_engine": REFIT_ROOT / "engine/dit_bf16_refit.engine",
        "refit_engine_receipt": (
            REFIT_ROOT / "engine/rlinf-refittable-dit-engine-receipt.json"
        ),
        "refit_parameter_map": REFIT_ROOT / "refittable-dit-parameter-map.json",
        "contract": (
            SOURCE
            / "toolkits/eos/gr00t_trocar/tensorrt/contract-n1d7-b8-executor-matrix.json"
        ),
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise WorkflowError(f"missing W84 inputs: {missing}")
    if _sha256(IMAGE) != IMAGE_SHA256:
        raise WorkflowError("EOS image SHA-256 mismatch")
    if _sha256(required["refit_engine_receipt"]) != REFIT_ENGINE_RECEIPT_SHA256:
        raise WorkflowError("W83 refittable DiT receipt SHA-256 mismatch")
    if _sha256(required["refit_parameter_map"]) != REFIT_PARAMETER_MAP_SHA256:
        raise WorkflowError("W83 refittable DiT parameter map SHA-256 mismatch")
    if _git_revision() != expected_revision:
        raise WorkflowError(
            f"RLInf revision mismatch: {_git_revision()} != {expected_revision}"
        )
    status = subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain"], text=True
    )
    if status:
        raise WorkflowError(f"RLInf checkout is dirty:\n{status}")
    isaac_revision = subprocess.check_output(
        ["git", "-C", str(ISAAC_GR00T), "rev-parse", "HEAD"], text=True
    ).strip()
    if isaac_revision != ISAAC_GR00T_REVISION:
        raise WorkflowError(
            f"Isaac-GR00T revision mismatch: {isaac_revision} != {ISAAC_GR00T_REVISION}"
        )
    return {
        name: {
            "path": str(path),
            "sha256": _sha256(path) if path.is_file() else None,
        }
        for name, path in required.items()
    }


def _submit(args: argparse.Namespace) -> int:
    revision = args.revision or _git_revision()
    _validate_inputs(revision)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    launcher = SOURCE / "toolkits/eos/gr00t_trocar/tensorrt/start_executor_matrix_b8.py"
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
    inputs = _validate_inputs(args.revision)
    job_id = os.environ.get("SLURM_JOB_ID")
    node_list = os.environ.get("SLURM_JOB_NODELIST")
    if not job_id or not node_list:
        raise WorkflowError("allocation-run must execute inside Slurm")
    node = subprocess.check_output(
        ["scontrol", "show", "hostnames", node_list], text=True
    ).splitlines()[0]
    attempt = RUN_ROOT / f"{args.attempt_name}-{job_id}"
    attempt.mkdir(mode=0o700)
    launcher = SOURCE / "toolkits/eos/gr00t_trocar/tensorrt/start_executor_matrix_b8.py"
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
        {
            "job_id": job_id,
            "node": node,
            "revision": args.revision,
            "inputs": inputs,
            "command": command,
        },
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


def _logged_run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    with (
        stdout_path.open("x", encoding="utf-8") as stdout,
        stderr_path.open("x", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return {
        "command": command,
        "return_code": completed.returncode,
        "wall_s": time.perf_counter() - started,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def _run_agent(args: argparse.Namespace) -> int:
    inputs = _validate_inputs(args.revision)
    attempt = args.attempt.resolve(strict=True)
    tools = SOURCE / "toolkits/eos/gr00t_trocar/tensorrt"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": f"{SOURCE}:{ISAAC_GR00T}",
            "NO_ALBUMENTATIONS_UPDATE": "1",
            "HF_HOME": str(WORKSPACE / "cache/huggingface"),
            "UV_CACHE_DIR": str(WORKSPACE / "cache/uv"),
        }
    )
    install = _logged_run(
        ["bash", "-lc", "apt-get update && apt-get install -y libpython3.12-dev"],
        cwd=SOURCE,
        env=env,
        stdout_path=attempt / "deps.out",
        stderr_path=attempt / "deps.err",
    )
    _write_new(attempt / "deps.result.json", install)
    if install["return_code"]:
        raise WorkflowError("W84 dependency setup failed")

    tests = _logged_run(
        [
            str(TEST_PYTHON),
            "-m",
            "pytest",
            "-q",
            "tests/unit_tests/test_gr00t_n1d7_executor_matrix_b8.py",
            "tests/unit_tests/test_gr00t_n1d7_refittable_dit_contract.py",
            "tests/unit_tests/test_gr00t_n1d7_true_b8_contract.py",
            "tests/unit_tests/test_gr00t_n1d7_tensorrt_tools.py",
        ],
        cwd=SOURCE,
        env=env,
        stdout_path=attempt / "tests.out",
        stderr_path=attempt / "tests.err",
    )
    _write_new(attempt / "tests.result.json", tests)
    if tests["return_code"]:
        raise WorkflowError("W84 focused tests failed")

    command = [
        str(BUILDER_PYTHON),
        str(tools / "standalone_true_b8.py"),
        "--source",
        str(ISAAC_GR00T),
        "--model",
        str(MODEL),
        "--engines",
        str(BACKBONE_ENGINES),
        "--collated",
        str(FIXTURE / "collated-inputs.pt"),
        "--raw",
        str(FIXTURE / "raw-observation.npz"),
        "--fixture-receipt",
        str(FIXTURE / "fixture.json"),
        "--export-receipt",
        str(EXPORT / "rlinf-export-receipt.json"),
        "--engine-receipt",
        str(BACKBONE_ENGINES / "rlinf-engine-receipt.json"),
        "--refittable-dit-engine",
        str(REFIT_ROOT / "engine/dit_bf16_refit.engine"),
        "--refittable-dit-receipt",
        str(REFIT_ROOT / "engine/rlinf-refittable-dit-engine-receipt.json"),
        "--refittable-dit-receipt-sha256",
        REFIT_ENGINE_RECEIPT_SHA256,
        "--refittable-dit-parameter-map",
        str(REFIT_ROOT / "refittable-dit-parameter-map.json"),
        "--refittable-dit-parameter-map-sha256",
        REFIT_PARAMETER_MAP_SHA256,
        "--refittable-dit-source-digest",
        REFIT_SOURCE_DIGEST,
        "--matrix-warmup",
        "10",
        "--matrix-measured",
        "30",
        "--pt2-backbone-unavailable-reason",
        (
            "Torch 2.9/FlashAttention 2.8.3 varlen rejects Dynamo fake scalar "
            "max_seqlen_q/k; reproduced by W84 jobs 5986289 and 5986310 with "
            "capture_scalar_outputs false and true"
        ),
        "--output",
        str(attempt / "executor-matrix.json"),
    ]
    benchmark = _logged_run(
        command,
        cwd=SOURCE,
        env=env,
        stdout_path=attempt / "executor-matrix.out",
        stderr_path=attempt / "executor-matrix.err",
    )
    _write_new(attempt / "executor-matrix.command.json", command)
    _write_new(attempt / "executor-matrix.result.json", benchmark)
    if benchmark["return_code"]:
        raise WorkflowError("W84 executor matrix failed")

    receipt = json.loads((attempt / "executor-matrix.json").read_text(encoding="utf-8"))
    matrix = receipt.get("executor_matrix")
    if (
        receipt.get("status") != "passed"
        or not matrix
        or matrix.get("status") != "passed"
    ):
        raise WorkflowError("W84 executor matrix receipt is not qualified")
    result = {
        "schema": "rlinf.w84-b8-executor-matrix-attempt.v1",
        "status": "passed",
        "revision": args.revision,
        "image_sha256": IMAGE_SHA256,
        "inputs": inputs,
        "receipt_sha256": _sha256(attempt / "executor-matrix.json"),
    }
    _write_new(attempt / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--revision")
    submit.add_argument("--attempt-name", default="W84-b8-executor-matrix-r1")
    submit.add_argument("--time-limit", default="01:30:00")
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
        print(f"W84 executor matrix workflow failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
