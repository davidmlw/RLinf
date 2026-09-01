#!/usr/bin/env python3
"""Submit and own a single-node RLinf experiment on EOS.

The public entrypoint runs on an EOS login node. It validates immutable inputs,
submits an ``sbatch`` coordinator, enters the OCI image with one ``srun``,
starts Ray inside the container, runs the experiment until the wall-clock
deadline, and writes attempt-owned receipts. No retained agent process or
pre-created Remote MCP dock is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

SITE_SCHEMA = "rlinf.eos.site.v1"
SUBMISSION_SCHEMA = "rlinf.eos.submission.v1"
ATTEMPT_SCHEMA = "rlinf.eos.attempt.v1"
RESULT_SCHEMA = "rlinf.eos.result.v1"
BASE_REVISION = "0f9ea98c7a6d9e3ade24e8f4846c64d3b135dbcc"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_OID_RE = re.compile(r"[0-9a-f]{40}")
TIME_LIMIT_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})")


class WorkflowError(RuntimeError):
    """A fail-closed EOS workflow error."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(path: Path, value: object) -> None:
    """Create one durable JSON receipt without replacing prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(value).encode("utf-8")
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staging, path, follow_symlinks=False)
        except FileExistsError as error:
            raise WorkflowError(f"refusing to replace receipt: {path}") from error
        staging.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        staging.unlink(missing_ok=True)


def _exact_dict(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise WorkflowError(f"{label} has unexpected fields")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise WorkflowError(f"{label} must be a nonempty single-line string")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise WorkflowError(f"{label} must be a positive integer")
    return value


def _absolute_file(value: object, label: str) -> Path:
    path = Path(_string(value, label))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise WorkflowError(f"{label} must be an absolute regular file")
    return path.resolve(strict=True)


def _absolute_directory(value: object, label: str) -> Path:
    path = Path(_string(value, label))
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise WorkflowError(f"{label} must be an absolute directory")
    return path.resolve(strict=True)


def _time_limit_seconds(value: object) -> tuple[str, int]:
    text = _string(value, "slurm.time_limit")
    match = TIME_LIMIT_RE.fullmatch(text)
    if match is None:
        raise WorkflowError("slurm.time_limit must use HH:MM:SS")
    hours, minutes, seconds = (int(item) for item in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise WorkflowError("slurm.time_limit contains an invalid minute or second")
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise WorkflowError("slurm.time_limit must be positive")
    return text, total


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise WorkflowError(f"git {' '.join(args)} failed in {root}: {diagnostic}")
    return completed.stdout.strip()


def _verify_sha256(path: Path, expected: object, label: str) -> str:
    digest = _string(expected, f"{label}.sha256")
    if SHA256_RE.fullmatch(digest) is None:
        raise WorkflowError(f"{label}.sha256 must be lowercase SHA-256")
    actual = _sha256_file(path)
    if actual != digest:
        raise WorkflowError(
            f"{label} SHA-256 mismatch: expected {digest}, found {actual}"
        )
    return actual


def _baseline_contract(config: Path) -> dict[str, int | str]:
    """Parse and validate the canonical feature-free chunk16 workload."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys,yaml; "
                "print(json.dumps(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))))"
            ),
            str(config),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise WorkflowError(f"runtime Python cannot parse config: {probe.stderr.strip()}")
    try:
        cfg = json.loads(probe.stdout)
    except json.JSONDecodeError as error:
        raise WorkflowError("runtime Python returned invalid config JSON") from error
    try:
        algorithm = cfg["algorithm"]
        env = cfg["env"]["train"]
        actor = cfg["actor"]
        actor_model = actor["model"]
        rollout = cfg["rollout"]
        chunks = int(actor_model["num_action_chunks"])
        envs = int(env["total_num_envs"])
        physical = envs * int(env["max_steps_per_rollout_epoch"]) * int(
            algorithm["rollout_epoch"]
        )
        decisions = physical // chunks
        global_batch = int(actor["global_batch_size"])
        micro_batch = int(actor["micro_batch_size"])
        update_epochs = int(algorithm["update_epoch"])
        optimizer_updates = decisions // global_batch * update_epochs
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise WorkflowError("config is missing a canonical workload field") from error
    expected = {
        "chunks": 16,
        "envs": 64,
        "physical_actions": 65_536,
        "policy_decisions": 4_096,
        "global_batch": 2_048,
        "micro_batch": 128,
        "update_epochs": 4,
        "optimizer_updates": 8,
    }
    actual = {
        "chunks": chunks,
        "envs": envs,
        "physical_actions": physical,
        "policy_decisions": decisions,
        "global_batch": global_batch,
        "micro_batch": micro_batch,
        "update_epochs": update_epochs,
        "optimizer_updates": optimizer_updates,
    }
    if actual != expected:
        raise WorkflowError(f"config workload mismatch: expected {expected}, found {actual}")
    semantic = {
        "reward_type": algorithm.get("reward_type"),
        "logprob_type": algorithm.get("logprob_type"),
        "actor_lr": actor.get("optim", {}).get("lr"),
    }
    if semantic != {
        "reward_type": "chunk_level",
        "logprob_type": "action_level",
        "actor_lr": 2e-5,
    }:
        raise WorkflowError(f"config algorithm mismatch: {semantic}")
    forbidden = {
        "reuse_rollout_backbone_features",
        "rollout_backbone_feature_transport",
        "pinned_feature_ipc_batch_blocks",
        "pinned_feature_verify_trajectory",
        "borrowed_feature_ipc_verify_trajectory",
        "gpu_feature_capacity_probe",
    }
    present = forbidden.intersection(actor_model) | forbidden.intersection(rollout)
    if present:
        raise WorkflowError(f"feature-reuse fields are forbidden in baseline: {sorted(present)}")
    return {**actual, **semantic}


def _load_site(
    path: Path,
    *,
    verify_image: bool = True,
    validate_contract: bool = True,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise WorkflowError(f"site is not valid JSON: {resolved}") from error
    site = _exact_dict(
        value,
        {"schema", "slurm", "container", "source", "runtime", "experiment"},
        "site",
    )
    if site["schema"] != SITE_SCHEMA:
        raise WorkflowError(f"unsupported site schema: {site['schema']!r}")

    slurm = _exact_dict(
        site["slurm"],
        {
            "account",
            "partition",
            "constraint",
            "nodes",
            "gpus_per_node",
            "cpus_per_task",
            "time_limit",
            "exclusive",
            "signal_seconds",
        },
        "slurm",
    )
    for key in ("account", "partition", "constraint"):
        text = _string(slurm[key], f"slurm.{key}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
            raise WorkflowError(f"slurm.{key} contains an unsafe character")
    if slurm["nodes"] != 1:
        raise WorkflowError("this launcher requires exactly one EOS node")
    if slurm["gpus_per_node"] != 8:
        raise WorkflowError("the canonical Trocar baseline requires 8 GPUs")
    _positive_int(slurm["cpus_per_task"], "slurm.cpus_per_task")
    time_limit, time_limit_s = _time_limit_seconds(slurm["time_limit"])
    if time_limit != "04:00:00":
        raise WorkflowError("W73 requires a four-hour Slurm allocation")
    if not isinstance(slurm["exclusive"], bool):
        raise WorkflowError("slurm.exclusive must be boolean")
    signal_seconds = _positive_int(slurm["signal_seconds"], "slurm.signal_seconds")
    if signal_seconds >= time_limit_s:
        raise WorkflowError("slurm.signal_seconds must be below the time limit")

    container = _exact_dict(
        site["container"],
        {"image", "image_sha256", "mounts", "workdir"},
        "container",
    )
    image = _absolute_file(container["image"], "container.image")
    image_sha256 = _string(container["image_sha256"], "container.image_sha256")
    if SHA256_RE.fullmatch(image_sha256) is None:
        raise WorkflowError("container.image_sha256 must be lowercase SHA-256")
    if verify_image:
        _verify_sha256(image, image_sha256, "container.image")
    mounts = container["mounts"]
    if (
        not isinstance(mounts, list)
        or not mounts
        or any(not isinstance(item, str) or not item for item in mounts)
    ):
        raise WorkflowError("container.mounts must be a nonempty string array")
    if len(mounts) != len(set(mounts)):
        raise WorkflowError("container.mounts contains duplicates")
    workdir = _absolute_directory(container["workdir"], "container.workdir")

    source = _exact_dict(
        site["source"],
        {"root", "revision", "base_revision", "require_clean"},
        "source",
    )
    source_root = _absolute_directory(source["root"], "source.root")
    revision = _string(source["revision"], "source.revision")
    base_revision = _string(source["base_revision"], "source.base_revision")
    if GIT_OID_RE.fullmatch(revision) is None or GIT_OID_RE.fullmatch(base_revision) is None:
        raise WorkflowError("source revisions must be full lowercase Git OIDs")
    if base_revision != BASE_REVISION:
        raise WorkflowError(f"source.base_revision must be {BASE_REVISION}")
    actual_revision = _git_output(source_root, "rev-parse", "HEAD")
    if actual_revision != revision:
        raise WorkflowError(
            f"source revision mismatch: expected {revision}, found {actual_revision}"
        )
    ancestor = subprocess.run(
        ["git", "-C", str(source_root), "merge-base", "--is-ancestor", base_revision, revision],
        check=False,
    )
    if ancestor.returncode != 0:
        raise WorkflowError("source base is not an ancestor of source revision")
    if not isinstance(source["require_clean"], bool):
        raise WorkflowError("source.require_clean must be boolean")
    if source["require_clean"] and _git_output(source_root, "status", "--short"):
        raise WorkflowError(f"source worktree is not clean: {source_root}")

    runtime = _exact_dict(
        site["runtime"],
        {
            "python",
            "poiesis_root",
            "gr00t_root",
            "model_root",
            "hf_cache",
            "task_overlay_root",
            "sanitized_tray_usd",
            "python_deps",
            "prepare_command",
        },
        "runtime",
    )
    python = Path(_string(runtime["python"], "runtime.python"))
    if not python.is_absolute():
        raise WorkflowError("runtime.python must be absolute")
    for key in (
        "poiesis_root",
        "gr00t_root",
        "model_root",
        "hf_cache",
        "task_overlay_root",
    ):
        _absolute_directory(runtime[key], f"runtime.{key}")
    _absolute_file(runtime["sanitized_tray_usd"], "runtime.sanitized_tray_usd")
    python_deps = runtime["python_deps"]
    if not isinstance(python_deps, list) or any(
        not isinstance(item, str) for item in python_deps
    ):
        raise WorkflowError("runtime.python_deps must be a directory array")
    for index, item in enumerate(python_deps):
        _absolute_directory(item, f"runtime.python_deps[{index}]")
    prepare_command = runtime["prepare_command"]
    if (
        not isinstance(prepare_command, list)
        or not prepare_command
        or any(not isinstance(item, str) or not item for item in prepare_command)
    ):
        raise WorkflowError("runtime.prepare_command must be a nonempty string array")

    experiment = _exact_dict(
        site["experiment"],
        {
            "name",
            "config",
            "config_sha256",
            "runner",
            "runner_sha256",
            "output_root",
            "max_steps",
            "save_interval",
            "workload_seconds",
            "shutdown_grace_seconds",
        },
        "experiment",
    )
    name = _string(experiment["name"], "experiment.name")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name):
        raise WorkflowError("experiment.name is not a safe attempt prefix")
    config = _absolute_file(experiment["config"], "experiment.config")
    runner = _absolute_file(experiment["runner"], "experiment.runner")
    _verify_sha256(config, experiment["config_sha256"], "experiment.config")
    _verify_sha256(runner, experiment["runner_sha256"], "experiment.runner")
    contract = _baseline_contract(config) if validate_contract else {"status": "deferred"}
    output_root = _absolute_directory(experiment["output_root"], "experiment.output_root")
    _positive_int(experiment["max_steps"], "experiment.max_steps")
    _positive_int(experiment["save_interval"], "experiment.save_interval")
    workload_seconds = _positive_int(
        experiment["workload_seconds"], "experiment.workload_seconds"
    )
    shutdown_grace = _positive_int(
        experiment["shutdown_grace_seconds"],
        "experiment.shutdown_grace_seconds",
    )
    if shutdown_grace < 120:
        raise WorkflowError("experiment.shutdown_grace_seconds must be at least 120")
    if workload_seconds + shutdown_grace > time_limit_s:
        raise WorkflowError("workload plus shutdown grace exceeds Slurm time limit")

    site["_resolved"] = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "launcher": str(Path(__file__).resolve(strict=True)),
        "launcher_sha256": _sha256_file(Path(__file__).resolve(strict=True)),
        "source_root": str(source_root),
        "source_revision": revision,
        "image": str(image),
        "workdir": str(workdir),
        "output_root": str(output_root),
        "workload_contract": contract,
    }
    return site


def _container_args(site: Mapping[str, Any]) -> list[str]:
    container = site["container"]
    return [
        f"--container-image={container['image']}",
        f"--container-mounts={','.join(container['mounts'])}",
        f"--container-workdir={container['workdir']}",
        "--container-remap-root",
        "--no-container-mount-home",
    ]


def _submission_argv(site: Mapping[str, Any]) -> list[str]:
    slurm = site["slurm"]
    experiment = site["experiment"]
    resolved = site["_resolved"]
    output_root = resolved["output_root"]
    argv = [
        "sbatch",
        "--parsable",
        f"--job-name=rlinf-{experiment['name']}"[:140],
        f"--partition={slurm['partition']}",
        f"--account={slurm['account']}",
        "--nodes=1",
        "--ntasks=1",
        "--ntasks-per-node=1",
        f"--gpus-per-node={slurm['gpus_per_node']}",
        f"--cpus-per-task={slurm['cpus_per_task']}",
        f"--time={slurm['time_limit']}",
        f"--signal=B:TERM@{slurm['signal_seconds']}",
        f"--constraint={slurm['constraint']}",
        f"--output={output_root}/slurm-%j.out",
        f"--error={output_root}/slurm-%j.err",
    ]
    if slurm["exclusive"]:
        argv.append("--exclusive")
    argv.extend(
        [
            resolved["launcher"],
            "allocation-run",
            "--site",
            resolved["path"],
        ]
    )
    return argv


def _submit(args: argparse.Namespace) -> int:
    if os.environ.get("SLURM_JOB_ID"):
        raise WorkflowError("submit must run outside a Slurm allocation")
    site = _load_site(Path(args.site), verify_image=not args.skip_image_hash)
    argv = _submission_argv(site)
    if args.dry_run:
        print(
            _canonical_json(
                {
                    "schema": SUBMISSION_SCHEMA,
                    "status": "dry_run",
                    "site": site["_resolved"],
                    "command": argv,
                }
            ),
            end="",
        )
        return 0
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise WorkflowError(f"sbatch submission failed: {diagnostic}")
    response = completed.stdout.strip()
    job_id = response.split(";", 1)[0]
    if not job_id.isdigit():
        raise WorkflowError(f"sbatch returned an invalid job id: {response!r}")
    receipt = {
        "schema": SUBMISSION_SCHEMA,
        "slurm_job_id": job_id,
        "site": site["_resolved"],
        "command": argv,
        "timestamp_unix_s": time.time(),
    }
    receipt_path = (
        Path(site["_resolved"]["output_root"])
        / "submissions"
        / f"{site['experiment']['name']}-{job_id}.json"
    )
    _write_new_json(receipt_path, receipt)
    print(_canonical_json({**receipt, "receipt": str(receipt_path)}), end="")
    return 0


def _validate(args: argparse.Namespace) -> int:
    site = _load_site(Path(args.site), verify_image=not args.skip_image_hash)
    print(
        _canonical_json(
            {
                "schema": SITE_SCHEMA,
                "status": "valid",
                "site": site["_resolved"],
                "submission_command": _submission_argv(site),
            }
        ),
        end="",
    )
    return 0


def _materialize(args: argparse.Namespace) -> int:
    """Resolve source and file hashes into an immutable site manifest."""
    template = Path(args.template).resolve(strict=True)
    try:
        value = json.loads(template.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise WorkflowError(f"site template is not valid JSON: {template}") from error
    site = _exact_dict(
        value,
        {"schema", "slurm", "container", "source", "runtime", "experiment"},
        "site template",
    )
    if site["schema"] != SITE_SCHEMA:
        raise WorkflowError("site template has the wrong schema")
    source = _exact_dict(
        site["source"],
        {"root", "revision", "base_revision", "require_clean"},
        "source",
    )
    root = _absolute_directory(source["root"], "source.root")
    if source["revision"] not in {"AUTO", _git_output(root, "rev-parse", "HEAD")}:
        raise WorkflowError("template source.revision is neither AUTO nor current HEAD")
    source["revision"] = _git_output(root, "rev-parse", "HEAD")
    source["base_revision"] = BASE_REVISION
    experiment = _exact_dict(
        site["experiment"],
        {
            "name",
            "config",
            "config_sha256",
            "runner",
            "runner_sha256",
            "output_root",
            "max_steps",
            "save_interval",
            "workload_seconds",
            "shutdown_grace_seconds",
        },
        "experiment",
    )
    config = _absolute_file(experiment["config"], "experiment.config")
    runner = _absolute_file(experiment["runner"], "experiment.runner")
    experiment["config_sha256"] = _sha256_file(config)
    experiment["runner_sha256"] = _sha256_file(runner)
    output = Path(args.output)
    if not output.is_absolute():
        raise WorkflowError("materialized site output must be absolute")
    _write_new_json(output, site)
    materialized = _load_site(output, verify_image=not args.skip_image_hash)
    print(
        _canonical_json(
            {
                "schema": SITE_SCHEMA,
                "status": "materialized",
                "site": materialized["_resolved"],
            }
        ),
        end="",
    )
    return 0


def _discover_allocation() -> tuple[str, str]:
    job_id = os.environ.get("SLURM_JOB_ID")
    node_list = os.environ.get("SLURM_JOB_NODELIST")
    if not job_id or not job_id.isdigit() or not node_list:
        raise WorkflowError("allocation-run must execute inside sbatch")
    output = subprocess.check_output(["scontrol", "show", "hostnames", node_list])
    nodes = [item.strip() for item in output.decode().splitlines() if item.strip()]
    if len(nodes) != 1:
        raise WorkflowError(f"expected one allocated node, found {nodes}")
    return job_id, nodes[0]


def _attempt_root(site: Mapping[str, Any], job_id: str) -> Path:
    output_root = Path(site["_resolved"]["output_root"])
    root = output_root / f"{site['experiment']['name']}-{job_id}"
    try:
        root.mkdir(mode=0o700)
    except FileExistsError as error:
        raise WorkflowError(f"attempt root already exists: {root}") from error
    (root / "logs").mkdir()
    return root


def _allocation_deadline(site: Mapping[str, Any]) -> int:
    now = int(time.time())
    experiment = site["experiment"]
    requested = now + int(experiment["workload_seconds"])
    slurm_end = os.environ.get("SLURM_JOB_END_TIME")
    if slurm_end and slurm_end.isdigit():
        requested = min(
            requested,
            int(slurm_end) - int(experiment["shutdown_grace_seconds"]),
        )
    if requested <= now:
        raise WorkflowError("allocation has no remaining workload window")
    return requested


def _allocation_run(args: argparse.Namespace) -> int:
    site = _load_site(Path(args.site))
    job_id, node = _discover_allocation()
    attempt = _attempt_root(site, job_id)
    deadline = _allocation_deadline(site)
    request = {
        "schema": ATTEMPT_SCHEMA,
        "slurm_job_id": job_id,
        "node": node,
        "site": site["_resolved"],
        "deadline_unix_s": deadline,
        "slurm_job_end_time": os.environ.get("SLURM_JOB_END_TIME"),
        "timestamp_unix_s": time.time(),
    }
    _write_new_json(attempt / "request.json", request)
    argv = [
        "srun",
        f"--nodelist={node}",
        "--nodes=1",
        "--ntasks=1",
        "--mpi=none",
        f"--gpus={site['slurm']['gpus_per_node']}",
        *_container_args(site),
        "python3",
        site["_resolved"]["launcher"],
        "run-agent",
        "--site",
        site["_resolved"]["path"],
        "--attempt-root",
        str(attempt),
        "--expected-node",
        node,
        "--deadline-unix-s",
        str(deadline),
    ]
    _write_new_json(attempt / "srun-command.json", argv)
    with (attempt / "logs" / "srun.out").open("w", encoding="utf-8") as stdout, (
        attempt / "logs" / "srun.err"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, text=True)

        def terminate(_signum: int, _frame: object) -> None:
            if process.poll() is None:
                process.terminate()

        previous = signal.signal(signal.SIGTERM, terminate)
        try:
            return_code = process.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)
    _write_new_json(
        attempt / "allocation-result.json",
        {
            "schema": RESULT_SCHEMA,
            "status": "exited" if return_code == 0 else "failed",
            "srun_return_code": return_code,
            "timestamp_unix_s": time.time(),
        },
    )
    return return_code


def _run_command(
    argv: Sequence[str],
    *,
    output: Path,
    error: Path,
    env: Mapping[str, str] | None = None,
) -> int:
    with output.open("w", encoding="utf-8") as stdout, error.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            list(argv), check=False, stdout=stdout, stderr=stderr, text=True, env=env
        )
    return completed.returncode


def _gpu_inventory() -> list[str]:
    completed = subprocess.run(
        ["nvidia-smi", "-L"], check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise WorkflowError(f"nvidia-smi failed: {completed.stderr.strip()}")
    gpus = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(gpus) != 8:
        raise WorkflowError(f"expected 8 visible GPUs, found {len(gpus)}")
    return gpus


def _ray_stop(python: str, attempt: Path, label: str) -> int:
    return _run_command(
        [python, "-m", "ray.scripts.scripts", "stop", "--force"],
        output=attempt / "logs" / f"ray-{label}.out",
        error=attempt / "logs" / f"ray-{label}.err",
    )


def _run_agent(args: argparse.Namespace) -> int:
    site = _load_site(
        Path(args.site), verify_image=False, validate_contract=False
    )
    attempt = Path(args.attempt_root).resolve(strict=True)
    expected_parent = Path(site["_resolved"]["output_root"]).resolve(strict=True)
    if attempt.parent != expected_parent or attempt.is_symlink():
        raise WorkflowError("attempt root is outside the configured output root")
    hostname = socket.gethostname()
    if hostname.split(".", 1)[0] != args.expected_node:
        raise WorkflowError(
            f"run-agent expected {args.expected_node}, executing on {hostname}"
        )
    deadline = int(args.deadline_unix_s)
    if deadline <= int(time.time()):
        raise WorkflowError("run-agent received an expired deadline")
    runtime = site["runtime"]
    experiment = site["experiment"]
    gpus = _gpu_inventory()
    _write_new_json(
        attempt / "preflight.json",
        {
            "schema": ATTEMPT_SCHEMA,
            "hostname": hostname,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus": gpus,
            "source_revision": site["_resolved"]["source_revision"],
            "deadline_unix_s": deadline,
            "timestamp_unix_s": time.time(),
        },
    )
    prepare_out = attempt / "logs" / "runtime-prepare.out"
    prepare_err = attempt / "logs" / "runtime-prepare.err"
    remaining = max(1, deadline - int(time.time()))
    with prepare_out.open("w", encoding="utf-8") as stdout, prepare_err.open(
        "w", encoding="utf-8"
    ) as stderr:
        try:
            prepared = subprocess.run(
                runtime["prepare_command"],
                check=False,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired as error:
            raise WorkflowError("runtime preparation exceeded the workload deadline") from error
    if prepared.returncode != 0:
        raise WorkflowError("runtime preparation failed")
    python_path = _absolute_file(runtime["python"], "runtime.python")
    if not os.access(python_path, os.X_OK):
        raise WorkflowError("runtime.python is not executable after preparation")
    python = str(python_path)
    _ray_stop(python, attempt, "prestop")
    node_ip = socket.gethostbyname(hostname)
    ray_start = _run_command(
        [
            python,
            "-m",
            "ray.scripts.scripts",
            "start",
            "--head",
            f"--node-ip-address={node_ip}",
            "--port=6379",
            "--num-gpus=8",
        ],
        output=attempt / "logs" / "ray-start.out",
        error=attempt / "logs" / "ray-start.err",
    )
    if ray_start != 0:
        raise WorkflowError("Ray head failed to start")
    ray_status = _run_command(
        [python, "-m", "ray.scripts.scripts", "status"],
        output=attempt / "logs" / "ray-status.out",
        error=attempt / "logs" / "ray-status.err",
    )
    if ray_status != 0:
        _ray_stop(python, attempt, "status-failure-stop")
        raise WorkflowError("Ray status gate failed")

    run_env = dict(os.environ)
    run_env.update(
        {
            "W73_ATTEMPT_ROOT": str(attempt),
            "W73_SOURCE_ROOT": site["source"]["root"],
            "W73_CONFIG": experiment["config"],
            "W73_RUNTIME_PYTHON": python,
            "W73_POIESIS_ROOT": runtime["poiesis_root"],
            "W73_GROOT_ROOT": runtime["gr00t_root"],
            "W73_MODEL_ROOT": runtime["model_root"],
            "W73_HF_CACHE": runtime["hf_cache"],
            "W73_TASK_OVERLAY_ROOT": runtime["task_overlay_root"],
            "W73_SANITIZED_TRAY_USD": runtime["sanitized_tray_usd"],
            "W73_PYTHON_DEPS": os.pathsep.join(runtime["python_deps"]),
            "W73_MAX_STEPS": str(experiment["max_steps"]),
            "W73_SAVE_INTERVAL": str(experiment["save_interval"]),
            "W73_DEADLINE_UNIX_S": str(deadline),
        }
    )
    training_out = (attempt / "logs" / "training.out").open("w", encoding="utf-8")
    training_err = (attempt / "logs" / "training.err").open("w", encoding="utf-8")
    process: subprocess.Popen[str] | None = None
    timed_out = False
    child_code: int | None = None
    try:
        process = subprocess.Popen(
            ["bash", experiment["runner"]],
            stdout=training_out,
            stderr=training_err,
            text=True,
            env=run_env,
            start_new_session=True,
        )
        while process.poll() is None and time.time() < deadline:
            time.sleep(2)
        if process.poll() is None:
            timed_out = True
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
        child_code = process.wait()
    finally:
        training_out.close()
        training_err.close()
        ray_stop = _ray_stop(python, attempt, "stop")
    status = "deadline_reached" if timed_out else "completed"
    if not timed_out and child_code != 0:
        status = "failed"
    result = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "training_return_code": child_code,
        "deadline_reached": timed_out,
        "ray_stop_return_code": ray_stop,
        "timestamp_unix_s": time.time(),
    }
    _write_new_json(attempt / "run-result.json", result)
    return 0 if status in {"completed", "deadline_reached"} and ray_stop == 0 else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--site", required=True)
    validate.add_argument("--skip-image-hash", action="store_true")
    validate.set_defaults(handler=_validate)

    materialize = commands.add_parser("materialize")
    materialize.add_argument("--template", required=True)
    materialize.add_argument("--output", required=True)
    materialize.add_argument("--skip-image-hash", action="store_true")
    materialize.set_defaults(handler=_materialize)

    submit = commands.add_parser("submit")
    submit.add_argument("--site", required=True)
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--skip-image-hash", action="store_true")
    submit.set_defaults(handler=_submit)

    allocation = commands.add_parser("allocation-run")
    allocation.add_argument("--site", required=True)
    allocation.set_defaults(handler=_allocation_run)

    agent = commands.add_parser("run-agent")
    agent.add_argument("--site", required=True)
    agent.add_argument("--attempt-root", required=True)
    agent.add_argument("--expected-node", required=True)
    agent.add_argument("--deadline-unix-s", required=True, type=int)
    agent.set_defaults(handler=_run_agent)
    return parser


def main() -> int:
    try:
        args = _parser().parse_args()
        return int(args.handler(args))
    except (OSError, subprocess.SubprocessError, WorkflowError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
