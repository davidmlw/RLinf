#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

SITE_SCHEMA = "rlinf.eos.site.v3"
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


def _absolute_executable(value: object, label: str) -> Path:
    """Validate an executable while preserving a virtualenv's symlink path."""
    path = Path(_string(value, label))
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise WorkflowError(f"{label} must be an absolute executable file")
    return path


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


def _git_dependency_inputs_sha256(root: Path) -> str:
    listing = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-s",
            "--",
            "pyproject.toml",
            "requirements",
        ],
        check=False,
        capture_output=True,
    )
    if listing.returncode != 0 or not listing.stdout:
        raise WorkflowError(f"cannot inventory RLinf dependency inputs in {root}")
    return hashlib.sha256(listing.stdout).hexdigest()


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


def _validate_provenance(value: object) -> dict[str, object]:
    provenance = _exact_dict(value, {"git_checkouts", "files"}, "provenance")

    git_checkouts = provenance["git_checkouts"]
    if not isinstance(git_checkouts, list) or not git_checkouts:
        raise WorkflowError("provenance.git_checkouts must be a nonempty array")
    resolved_git: list[dict[str, str]] = []
    git_names: set[str] = set()
    git_roots: set[Path] = set()
    for index, raw in enumerate(git_checkouts):
        label = f"provenance.git_checkouts[{index}]"
        checkout = _exact_dict(
            raw, {"name", "root", "revision", "require_clean"}, label
        )
        name = _string(checkout["name"], f"{label}.name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            raise WorkflowError(f"{label}.name is unsafe")
        if name in git_names:
            raise WorkflowError(f"duplicate provenance Git name: {name}")
        root = _absolute_directory(checkout["root"], f"{label}.root")
        if root in git_roots:
            raise WorkflowError(f"duplicate provenance Git root: {root}")
        revision = _string(checkout["revision"], f"{label}.revision")
        if GIT_OID_RE.fullmatch(revision) is None:
            raise WorkflowError(f"{label}.revision must be a full lowercase Git OID")
        actual_revision = _git_output(root, "rev-parse", "HEAD")
        if actual_revision != revision:
            raise WorkflowError(
                f"{name} revision mismatch: expected {revision}, found {actual_revision}"
            )
        if not isinstance(checkout["require_clean"], bool):
            raise WorkflowError(f"{label}.require_clean must be boolean")
        if checkout["require_clean"] and _git_output(
            root,
            "status",
            "--short",
            "--untracked-files=no",
            "--ignore-submodules=untracked",
        ):
            raise WorkflowError(f"provenance Git tracked content is not clean: {root}")
        git_names.add(name)
        git_roots.add(root)
        resolved_git.append(
            {"name": name, "root": str(root), "revision": actual_revision}
        )

    files = provenance["files"]
    if not isinstance(files, list) or not files:
        raise WorkflowError("provenance.files must be a nonempty array")
    resolved_files: list[dict[str, str]] = []
    file_names: set[str] = set()
    file_paths: set[Path] = set()
    for index, raw in enumerate(files):
        label = f"provenance.files[{index}]"
        artifact = _exact_dict(raw, {"name", "path", "sha256"}, label)
        name = _string(artifact["name"], f"{label}.name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            raise WorkflowError(f"{label}.name is unsafe")
        if name in file_names:
            raise WorkflowError(f"duplicate provenance file name: {name}")
        path = _absolute_file(artifact["path"], f"{label}.path")
        if path in file_paths:
            raise WorkflowError(f"duplicate provenance file path: {path}")
        digest = _verify_sha256(path, artifact["sha256"], label)
        file_names.add(name)
        file_paths.add(path)
        resolved_files.append({"name": name, "path": str(path), "sha256": digest})

    return {"git_checkouts": resolved_git, "files": resolved_files}


def _baseline_contract(config: Path) -> dict[str, bool | float | int | str]:
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
        raise WorkflowError(
            f"runtime Python cannot parse config: {probe.stderr.strip()}"
        )
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
        runner = cfg["runner"]
        eval_env = cfg["env"]["eval"]
        chunks = int(actor_model["num_action_chunks"])
        envs = int(env["total_num_envs"])
        physical = (
            envs
            * int(env["max_steps_per_rollout_epoch"])
            * int(algorithm["rollout_epoch"])
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
        raise WorkflowError(
            f"config workload mismatch: expected {expected}, found {actual}"
        )
    semantic = {
        "reward_type": algorithm.get("reward_type"),
        "logprob_type": algorithm.get("logprob_type"),
        "actor_lr": actor.get("optim", {}).get("lr"),
        "eval_interval": runner.get("val_check_interval"),
        "eval_envs": eval_env.get("total_num_envs"),
        "eval_fixed_resets": eval_env.get("use_fixed_reset_state_ids"),
        "eval_video": eval_env.get("video_cfg", {}).get("save_video"),
    }
    if semantic != {
        "reward_type": "chunk_level",
        "logprob_type": "action_level",
        "actor_lr": 2e-5,
        "eval_interval": 5,
        "eval_envs": 8,
        "eval_fixed_resets": True,
        "eval_video": True,
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
        raise WorkflowError(
            f"feature-reuse fields are forbidden in baseline: {sorted(present)}"
        )
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
        {
            "schema",
            "slurm",
            "container",
            "source",
            "provenance",
            "runtime",
            "experiment",
        },
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
        {
            "oci_reference",
            "registry_digest",
            "image",
            "image_sha256",
            "mounts",
            "workdir",
        },
        "container",
    )
    _string(container["oci_reference"], "container.oci_reference")
    registry_digest = _string(container["registry_digest"], "container.registry_digest")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", registry_digest) is None:
        raise WorkflowError("container.registry_digest must be a sha256 OCI digest")
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
    if (
        GIT_OID_RE.fullmatch(revision) is None
        or GIT_OID_RE.fullmatch(base_revision) is None
    ):
        raise WorkflowError("source revisions must be full lowercase Git OIDs")
    if base_revision != BASE_REVISION:
        raise WorkflowError(f"source.base_revision must be {BASE_REVISION}")
    actual_revision = _git_output(source_root, "rev-parse", "HEAD")
    if actual_revision != revision:
        raise WorkflowError(
            f"source revision mismatch: expected {revision}, found {actual_revision}"
        )
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "merge-base",
            "--is-ancestor",
            base_revision,
            revision,
        ],
        check=False,
    )
    if ancestor.returncode != 0:
        raise WorkflowError("source base is not an ancestor of source revision")
    if not isinstance(source["require_clean"], bool):
        raise WorkflowError("source.require_clean must be boolean")
    if source["require_clean"] and _git_output(source_root, "status", "--short"):
        raise WorkflowError(f"source worktree is not clean: {source_root}")

    provenance = _validate_provenance(site["provenance"])

    runtime = _exact_dict(
        site["runtime"],
        {
            "python",
            "env_root",
            "runtime_spec",
            "runtime_spec_sha256",
            "prepare_script",
            "prepare_script_sha256",
            "uv_cache",
            "flash_attn_wheel",
            "flash_attn_wheel_sha256",
            "torchcodec_wheel",
            "torchcodec_wheel_sha256",
            "isaaclab_root",
            "gr00t_root",
            "model_root",
            "hf_cache",
            "task_overlay_root",
            "sanitized_tray_usd",
            "python_deps",
        },
        "runtime",
    )
    python = Path(_string(runtime["python"], "runtime.python"))
    if not python.is_absolute():
        raise WorkflowError("runtime.python must be absolute")
    env_root = Path(_string(runtime["env_root"], "runtime.env_root"))
    uv_cache = Path(_string(runtime["uv_cache"], "runtime.uv_cache"))
    if not env_root.is_absolute() or not uv_cache.is_absolute():
        raise WorkflowError("runtime env_root and uv_cache must be absolute")
    if python != env_root / "bin" / "python":
        raise WorkflowError("runtime.python must be env_root/bin/python")
    runtime_spec = _absolute_file(runtime["runtime_spec"], "runtime.runtime_spec")
    prepare_script = _absolute_file(runtime["prepare_script"], "runtime.prepare_script")
    _verify_sha256(runtime_spec, runtime["runtime_spec_sha256"], "runtime.runtime_spec")
    _verify_sha256(
        prepare_script,
        runtime["prepare_script_sha256"],
        "runtime.prepare_script",
    )
    flash_attn_wheel = _absolute_file(
        runtime["flash_attn_wheel"], "runtime.flash_attn_wheel"
    )
    _verify_sha256(
        flash_attn_wheel,
        runtime["flash_attn_wheel_sha256"],
        "runtime.flash_attn_wheel",
    )
    torchcodec_wheel = _absolute_file(
        runtime["torchcodec_wheel"], "runtime.torchcodec_wheel"
    )
    _verify_sha256(
        torchcodec_wheel,
        runtime["torchcodec_wheel_sha256"],
        "runtime.torchcodec_wheel",
    )
    try:
        runtime_spec_value = json.loads(runtime_spec.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise WorkflowError("runtime spec is not valid JSON") from error
    runtime_spec_value = _exact_dict(
        runtime_spec_value,
        {
            "schema",
            "system_image_registry_digest",
            "python_version",
            "torch_version",
            "torchvision_version",
            "torchaudio_version",
            "torch_backend",
            "flash_attn_version",
            "flash_attn_wheel_filename",
            "flash_attn_wheel_sha256",
            "torchcodec_version",
            "torchcodec_backend",
            "torchcodec_wheel_filename",
            "torchcodec_wheel_sha256",
            "hydra_core_version",
            "numpy_version",
            "transformers_version",
            "isaaclab_revision",
            "gr00t_revision",
            "installer",
            "installer_arguments",
        },
        "runtime spec",
    )
    if runtime_spec_value.get("schema") != "rlinf.eos.python-runtime.v1":
        raise WorkflowError("runtime spec schema mismatch")
    if runtime_spec_value.get("system_image_registry_digest") != registry_digest:
        raise WorkflowError("runtime spec and system image digest disagree")
    provenance_revisions = {
        item["name"]: item["revision"] for item in provenance["git_checkouts"]
    }
    if runtime_spec_value.get("isaaclab_revision") != provenance_revisions.get(
        "isaaclab"
    ):
        raise WorkflowError("runtime spec and IsaacLab provenance disagree")
    if runtime_spec_value.get("gr00t_revision") != provenance_revisions.get(
        "isaac-gr00t"
    ):
        raise WorkflowError("runtime spec and GR00T provenance disagree")
    for key in (
        "python_version",
        "torch_version",
        "torchvision_version",
        "torchaudio_version",
        "torch_backend",
        "flash_attn_version",
        "flash_attn_wheel_filename",
        "flash_attn_wheel_sha256",
        "torchcodec_version",
        "torchcodec_backend",
        "torchcodec_wheel_filename",
        "torchcodec_wheel_sha256",
        "hydra_core_version",
        "numpy_version",
        "transformers_version",
        "installer",
    ):
        _string(runtime_spec_value[key], f"runtime spec.{key}")
    if flash_attn_wheel.name != runtime_spec_value["flash_attn_wheel_filename"]:
        raise WorkflowError("runtime spec and FlashAttention wheel filename disagree")
    if (
        runtime["flash_attn_wheel_sha256"]
        != runtime_spec_value["flash_attn_wheel_sha256"]
    ):
        raise WorkflowError("runtime spec and FlashAttention wheel hash disagree")
    if runtime_spec_value["installer"] != "requirements/install.sh":
        raise WorkflowError("runtime spec uses an unsupported installer")
    if runtime_spec_value["torchcodec_backend"] != "cpu":
        raise WorkflowError("runtime spec uses an unsupported TorchCodec backend")
    if torchcodec_wheel.name != runtime_spec_value["torchcodec_wheel_filename"]:
        raise WorkflowError("runtime spec and TorchCodec wheel filename disagree")
    if (
        runtime["torchcodec_wheel_sha256"]
        != runtime_spec_value["torchcodec_wheel_sha256"]
    ):
        raise WorkflowError("runtime spec and TorchCodec wheel hash disagree")
    expected_installer_arguments = [
        "--no-root",
        "--platform",
        "nvidia",
        "--python",
        runtime_spec_value["python_version"],
        "--torch",
        runtime_spec_value["torch_version"],
        "--no-flash-attn",
        "embodied",
        "--model",
        "gr00t",
        "--env",
        "isaaclab",
    ]
    if runtime_spec_value["installer_arguments"] != expected_installer_arguments:
        raise WorkflowError(
            "runtime spec installer arguments disagree with the launcher"
        )
    for key in (
        "isaaclab_root",
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
    contract = (
        _baseline_contract(config) if validate_contract else {"status": "deferred"}
    )
    output_root = _absolute_directory(
        experiment["output_root"], "experiment.output_root"
    )
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

    launcher = source_root / "toolkits/eos/start_rlinf.py"
    if not launcher.is_file() or launcher.is_symlink():
        raise WorkflowError(f"source launcher is missing or invalid: {launcher}")
    launcher = launcher.resolve(strict=True)
    site["_resolved"] = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "launcher": str(launcher),
        "launcher_sha256": _sha256_file(launcher),
        "source_root": str(source_root),
        "source_revision": revision,
        "image": str(image),
        "oci_reference": container["oci_reference"],
        "registry_digest": registry_digest,
        "workdir": str(workdir),
        "output_root": str(output_root),
        "workload_contract": contract,
        "provenance": provenance,
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
        {
            "schema",
            "slurm",
            "container",
            "source",
            "provenance",
            "runtime",
            "experiment",
        },
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
    runtime = site["runtime"]
    runtime_spec = _absolute_file(runtime["runtime_spec"], "runtime.runtime_spec")
    prepare_script = _absolute_file(runtime["prepare_script"], "runtime.prepare_script")
    if runtime["runtime_spec_sha256"] not in {
        "AUTO",
        _sha256_file(runtime_spec),
    }:
        raise WorkflowError("template runtime spec hash is neither AUTO nor current")
    if runtime["prepare_script_sha256"] not in {
        "AUTO",
        _sha256_file(prepare_script),
    }:
        raise WorkflowError("template prepare script hash is neither AUTO nor current")
    runtime["runtime_spec_sha256"] = _sha256_file(runtime_spec)
    runtime["prepare_script_sha256"] = _sha256_file(prepare_script)
    if args.smoke:
        experiment.update(
            {
                "name": f"{experiment['name']}-smoke",
                "max_steps": 1,
                "save_interval": 1,
                "workload_seconds": 5_400,
            }
        )
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
    with (
        (attempt / "logs" / "srun.out").open("w", encoding="utf-8") as stdout,
        (attempt / "logs" / "srun.err").open("w", encoding="utf-8") as stderr,
    ):
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
    timeout: int | None = None,
) -> int:
    with (
        output.open("w", encoding="utf-8") as stdout,
        error.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            list(argv),
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=env,
            start_new_session=timeout is not None,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise


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


def _ray_worker_environment(site: Mapping[str, Any]) -> dict[str, str]:
    runtime = site["runtime"]
    python_paths = [
        *runtime["python_deps"],
        runtime["gr00t_root"],
        site["source"]["root"],
        runtime["task_overlay_root"],
        str(Path(runtime["isaaclab_root"]) / "source"),
    ]
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "HF_HOME": runtime["hf_cache"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "RAY_DEDUP_LOGS": "0",
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
            "RLINF_CODE_WORKING_DIR": "0",
            "RLINF_EXT_MODULE": "w68_rlinf_extension",
            "RLINF_CONFIG_FILE": site["experiment"]["config"],
            "W68_ISAACLAB_SOURCE_ROOT": str(Path(runtime["isaaclab_root"]) / "source"),
            "W68_OVERLAY_ROOT": runtime["task_overlay_root"],
            "W68_SANITIZED_TRAY_USD": runtime["sanitized_tray_usd"],
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
        }
    )
    return env


def _archive_ray_failure_logs(ray_temp: Path, attempt: Path) -> Path | None:
    logs = ray_temp / "session_latest" / "logs"
    if not logs.is_dir():
        return None
    destination = attempt / "runtime" / "ray-failure-logs.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        archive.add(logs, arcname="logs")
    return destination


def _run_agent(args: argparse.Namespace) -> int:
    site = _load_site(Path(args.site), verify_image=False, validate_contract=False)
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
    runtime_spec_value = json.loads(
        Path(runtime["runtime_spec"]).read_text(encoding="utf-8")
    )
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
    prepare_env = dict(os.environ)
    prepare_env.update(
        {
            "W73_SOURCE_ROOT": site["source"]["root"],
            "W73_ISAACLAB_ROOT": runtime["isaaclab_root"],
            "W73_GROOT_ROOT": runtime["gr00t_root"],
            "W73_RUNTIME_ROOT": runtime["env_root"],
            "W73_RUNTIME_SPEC": runtime["runtime_spec"],
            "W73_RUNTIME_SPEC_SHA256": runtime["runtime_spec_sha256"],
            "W73_UV_CACHE": runtime["uv_cache"],
            "W73_FLASH_ATTN_WHEEL": runtime["flash_attn_wheel"],
            "W73_FLASH_ATTN_WHEEL_SHA256": runtime["flash_attn_wheel_sha256"],
            "W73_TORCHCODEC_WHEEL": runtime["torchcodec_wheel"],
            "W73_TORCHCODEC_WHEEL_SHA256": runtime[
                "torchcodec_wheel_sha256"
            ],
        }
    )
    remaining = max(1, deadline - int(time.time()))
    try:
        prepare_code = _run_command(
            ["bash", runtime["prepare_script"]],
            output=attempt / "logs" / "runtime-prepare.out",
            error=attempt / "logs" / "runtime-prepare.err",
            env=prepare_env,
            timeout=remaining,
        )
    except subprocess.TimeoutExpired as error:
        raise WorkflowError(
            "RLinf runtime preparation exceeded the workload deadline"
        ) from error
    if prepare_code != 0:
        raise WorkflowError("RLinf shared runtime preparation failed")
    runtime_manifest = Path(runtime["env_root"]) / "rlinf-runtime-manifest.json"
    try:
        runtime_manifest_value = json.loads(
            runtime_manifest.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError("RLinf runtime manifest is missing or invalid") from error
    runtime_manifest_value = _exact_dict(
        runtime_manifest_value,
        {
            "schema",
            "runtime_spec_sha256",
            "prepare_script_sha256",
            "rlinf_dependency_inputs_sha256",
            "requirements_freeze_sha256",
            "python",
            "torch",
            "torchvision",
            "torchaudio",
            "torch_cuda",
            "flash_attn",
            "flash_attn_wheel_sha256",
            "torchcodec",
            "torchcodec_wheel_sha256",
            "hydra_core",
            "numpy",
            "transformers",
            "ray",
            "isaaclab",
            "isaaclab_newton",
            "gr00t",
            "isaaclab_revision",
            "gr00t_revision",
        },
        "runtime manifest",
    )
    if runtime_manifest_value["schema"] != "rlinf.eos.python-runtime-manifest.v1":
        raise WorkflowError("RLinf runtime manifest schema mismatch")
    if runtime_manifest_value["runtime_spec_sha256"] != runtime["runtime_spec_sha256"]:
        raise WorkflowError("RLinf runtime manifest spec hash mismatch")
    if (
        runtime_manifest_value["prepare_script_sha256"]
        != runtime["prepare_script_sha256"]
    ):
        raise WorkflowError("RLinf runtime manifest prepare-script hash mismatch")
    dependency_inputs_sha = _git_dependency_inputs_sha256(Path(site["source"]["root"]))
    if (
        runtime_manifest_value["rlinf_dependency_inputs_sha256"]
        != dependency_inputs_sha
    ):
        raise WorkflowError("RLinf runtime manifest dependency-input hash mismatch")
    freeze = Path(runtime["env_root"]) / "requirements.freeze.txt"
    if (
        not freeze.is_file()
        or _sha256_file(freeze) != runtime_manifest_value["requirements_freeze_sha256"]
    ):
        raise WorkflowError("RLinf runtime package freeze hash mismatch")
    expected_revisions = {
        "isaaclab_revision": _git_output(
            Path(runtime["isaaclab_root"]), "rev-parse", "HEAD"
        ),
        "gr00t_revision": _git_output(Path(runtime["gr00t_root"]), "rev-parse", "HEAD"),
    }
    for key, expected in expected_revisions.items():
        if runtime_manifest_value[key] != expected:
            raise WorkflowError(f"RLinf runtime manifest {key} mismatch")
    if (
        runtime_manifest_value["flash_attn_wheel_sha256"]
        != runtime["flash_attn_wheel_sha256"]
    ):
        raise WorkflowError("RLinf runtime manifest FlashAttention wheel hash mismatch")
    if (
        runtime_manifest_value["torchcodec_wheel_sha256"]
        != runtime["torchcodec_wheel_sha256"]
    ):
        raise WorkflowError("RLinf runtime manifest TorchCodec wheel hash mismatch")
    expected_packages = {
        "flash_attn": runtime_spec_value["flash_attn_version"],
        "torchcodec": runtime_spec_value["torchcodec_version"],
        "hydra_core": runtime_spec_value["hydra_core_version"],
        "numpy": runtime_spec_value["numpy_version"],
        "transformers": runtime_spec_value["transformers_version"],
    }
    for key, expected in expected_packages.items():
        if runtime_manifest_value[key] != expected:
            raise WorkflowError(f"RLinf runtime manifest {key} mismatch")
    _write_new_json(
        attempt / "runtime-provenance.json",
        {
            "schema": ATTEMPT_SCHEMA,
            "runtime_spec_sha256": runtime["runtime_spec_sha256"],
            "prepare_script_sha256": runtime["prepare_script_sha256"],
            "runtime_manifest": runtime_manifest_value,
            "timestamp_unix_s": time.time(),
        },
    )
    python_path = _absolute_executable(runtime["python"], "runtime.python")
    python = str(python_path)
    runtime_probe = _run_command(
        [
            python,
            "-c",
            (
                "import flash_attn, hydra, importlib.metadata, json, numpy, ray, torch, torchcodec; "
                "from torchcodec.decoders import VideoDecoder; "
                "from transformers.image_utils import VideoInput; "
                "assert torch.cuda.is_available(), "
                "'PyTorch cannot access an allocation GPU'; "
                "probe = torch.ones(1, device='cuda'); "
                "print(json.dumps({'python': __import__('sys').version, "
                "'ray': ray.__version__, 'torch': torch.__version__, "
                "'flash_attn': flash_attn.__version__, "
                "'torchcodec': torchcodec.__version__, "
                "'hydra_core': importlib.metadata.version('hydra-core'), "
                "'numpy': numpy.__version__, "
                "'transformers': importlib.metadata.version('transformers'), "
                "'isaaclab': importlib.metadata.version('isaaclab'), "
                "'cuda': torch.version.cuda, "
                "'cuda_available': torch.cuda.is_available(), "
                "'cuda_device': torch.cuda.get_device_name(), "
                "'cuda_probe': probe.item()}))"
            ),
        ],
        output=attempt / "logs" / "runtime-probe.out",
        error=attempt / "logs" / "runtime-probe.err",
    )
    if runtime_probe != 0:
        raise WorkflowError("RLinf runtime dependency probe failed")
    seed_test = _run_command(
        [
            python,
            "-m",
            "pytest",
            "-q",
            str(
                Path(site["source"]["root"])
                / "tests/unit_tests/test_convergence_seed.py"
            ),
        ],
        output=attempt / "logs" / "seed-test.out",
        error=attempt / "logs" / "seed-test.err",
    )
    if seed_test != 0:
        raise WorkflowError("deterministic seed preflight failed")
    _ray_stop(python, attempt, "prestop")
    ray_env = _ray_worker_environment(site)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id or not slurm_job_id.isdecimal():
        raise WorkflowError("run-agent requires a numeric SLURM_JOB_ID")
    ray_temp = Path("/workspace") / f"w73-ray-{slurm_job_id}"
    if ray_temp.exists():
        raise WorkflowError(f"Ray temporary directory already exists: {ray_temp}")
    ray_temp.mkdir(parents=True)
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
            f"--temp-dir={ray_temp}",
        ],
        output=attempt / "logs" / "ray-start.out",
        error=attempt / "logs" / "ray-start.err",
        env=ray_env,
    )
    if ray_start != 0:
        shutil.rmtree(ray_temp, ignore_errors=True)
        raise WorkflowError("Ray head failed to start")
    ray_status = _run_command(
        [python, "-m", "ray.scripts.scripts", "status"],
        output=attempt / "logs" / "ray-status.out",
        error=attempt / "logs" / "ray-status.err",
        env=ray_env,
    )
    if ray_status != 0:
        _ray_stop(python, attempt, "status-failure-stop")
        _archive_ray_failure_logs(ray_temp, attempt)
        shutil.rmtree(ray_temp, ignore_errors=True)
        raise WorkflowError("Ray status gate failed")

    run_env = dict(ray_env)
    run_env.update(
        {
            "W73_ATTEMPT_ROOT": str(attempt),
            "W73_SOURCE_ROOT": site["source"]["root"],
            "W73_CONFIG": experiment["config"],
            "W73_RUNTIME_PYTHON": python,
            "W73_ISAACLAB_ROOT": runtime["isaaclab_root"],
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
        if timed_out or child_code != 0:
            _archive_ray_failure_logs(ray_temp, attempt)
        shutil.rmtree(ray_temp, ignore_errors=True)
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
    materialize.add_argument(
        "--smoke",
        action="store_true",
        help="limit the materialized attempt to one step and 90 minutes",
    )
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
