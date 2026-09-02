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

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "toolkits/eos/start_rlinf.py"
SPEC = importlib.util.spec_from_file_location("start_rlinf_eos", LAUNCHER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _site(tmp_path: Path, *, config: Path | None = None) -> Path:
    image = tmp_path / "image.sqsh"
    image.write_bytes(b"image")
    flash_attn_wheel = tmp_path / "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
    flash_attn_wheel.write_bytes(b"wheel")
    torchcodec_wheel = (
        tmp_path / "torchcodec-0.11.1-cp312-cp312-manylinux_2_28_x86_64.whl"
    )
    torchcodec_wheel.write_bytes(b"torchcodec-wheel")
    healthcare_assets_archive = tmp_path / "w68-healthcare-assets.tar"
    healthcare_assets_archive.write_bytes(b"healthcare-assets")
    git_lfs_bin = tmp_path / "git-lfs"
    git_lfs_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    git_lfs_bin.chmod(0o755)
    runtime_python_target = Path(sys.executable).resolve(strict=True)
    for name in (
        "isaaclab",
        "gr00t",
        "runtime-env",
        "uv-cache",
        "model",
        "hf-cache",
        "overlay",
        "python-deps",
        "output",
    ):
        (tmp_path / name).mkdir()
    (tmp_path / "runtime-env" / "bin").mkdir()
    runtime_python = tmp_path / "runtime-env" / "bin" / "python"
    runtime_python.symlink_to(runtime_python_target)
    dependency_revisions = {}
    for name in ("isaaclab", "gr00t"):
        root = tmp_path / name
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "config", "user.name", "W73 Test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "config", "user.email", "w73@example.test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "commit", "-q", "--allow-empty", "-m", name],
            check=True,
        )
        dependency_revisions[name] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    registry_digest = f"sha256:{'1' * 64}"
    runtime_spec = tmp_path / "runtime-spec.json"
    runtime_spec_value = json.loads(
        (ROOT / "toolkits/eos/gr00t_trocar/runtime-spec.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_spec_value.update(
        {
            "system_image_registry_digest": registry_digest,
            "isaaclab_revision": dependency_revisions["isaaclab"],
            "gr00t_revision": dependency_revisions["gr00t"],
            "flash_attn_wheel_sha256": _sha256(flash_attn_wheel),
            "torchcodec_wheel_sha256": _sha256(torchcodec_wheel),
        }
    )
    runtime_spec.write_text(
        json.dumps(runtime_spec_value),
        encoding="utf-8",
    )
    tray = tmp_path / "tray.usd"
    tray.write_text("usd", encoding="utf-8")
    model_manifest = tmp_path / "model-manifest.json"
    model_manifest.write_text("{}\n", encoding="utf-8")
    deps_manifest = tmp_path / "python-deps.sha256"
    deps_manifest.write_text("fixture\n", encoding="utf-8")
    config = config or ROOT / "toolkits/eos/gr00t_trocar/config-baseline-chunk16.yaml"
    runner = ROOT / "toolkits/eos/gr00t_trocar/run_baseline.sh"
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    value = {
        "schema": MODULE.SITE_SCHEMA,
        "slurm": {
            "account": "coreai_devtech_all",
            "partition": "batch",
            "constraint": "h100",
            "nodes": 1,
            "gpus_per_node": 8,
            "cpus_per_task": 64,
            "time_limit": "04:00:00",
            "exclusive": True,
            "signal_seconds": 600,
        },
        "container": {
            "oci_reference": "registry.example/rlinf:test",
            "registry_digest": registry_digest,
            "image": str(image),
            "image_sha256": _sha256(image),
            "mounts": [f"{tmp_path}:{tmp_path}"],
            "workdir": str(ROOT),
        },
        "source": {
            "root": str(ROOT),
            "revision": revision,
            "base_revision": MODULE.BASE_REVISION,
            "require_clean": False,
        },
        "provenance": {
            "git_checkouts": [
                {
                    "name": "rlinf-fixture",
                    "root": str(ROOT),
                    "revision": revision,
                    "require_clean": False,
                },
                {
                    "name": "isaaclab",
                    "root": str(tmp_path / "isaaclab"),
                    "revision": dependency_revisions["isaaclab"],
                    "require_clean": True,
                },
                {
                    "name": "isaac-gr00t",
                    "root": str(tmp_path / "gr00t"),
                    "revision": dependency_revisions["gr00t"],
                    "require_clean": True,
                },
            ],
            "files": [
                {
                    "name": "model-manifest",
                    "path": str(model_manifest),
                    "sha256": _sha256(model_manifest),
                },
                {
                    "name": "python-deps-manifest",
                    "path": str(deps_manifest),
                    "sha256": _sha256(deps_manifest),
                },
                {
                    "name": "tray-usd",
                    "path": str(tray),
                    "sha256": _sha256(tray),
                },
                {
                    "name": "flash-attn-h100-wheel",
                    "path": str(flash_attn_wheel),
                    "sha256": _sha256(flash_attn_wheel),
                },
                {
                    "name": "torchcodec-cpu-wheel",
                    "path": str(torchcodec_wheel),
                    "sha256": _sha256(torchcodec_wheel),
                },
                {
                    "name": "w68-healthcare-assets",
                    "path": str(healthcare_assets_archive),
                    "sha256": _sha256(healthcare_assets_archive),
                },
                {
                    "name": "git-lfs-client",
                    "path": str(git_lfs_bin),
                    "sha256": _sha256(git_lfs_bin),
                },
            ],
        },
        "runtime": {
            "python": str(runtime_python),
            "env_root": str(tmp_path / "runtime-env"),
            "runtime_spec": str(runtime_spec),
            "runtime_spec_sha256": _sha256(runtime_spec),
            "prepare_script": str(
                ROOT / "toolkits/eos/gr00t_trocar/prepare_runtime.sh"
            ),
            "prepare_script_sha256": _sha256(
                ROOT / "toolkits/eos/gr00t_trocar/prepare_runtime.sh"
            ),
            "uv_cache": str(tmp_path / "uv-cache"),
            "git_lfs_bin": str(git_lfs_bin),
            "git_lfs_bin_sha256": _sha256(git_lfs_bin),
            "flash_attn_wheel": str(flash_attn_wheel),
            "flash_attn_wheel_sha256": _sha256(flash_attn_wheel),
            "torchcodec_wheel": str(torchcodec_wheel),
            "torchcodec_wheel_sha256": _sha256(torchcodec_wheel),
            "healthcare_assets_archive": str(healthcare_assets_archive),
            "healthcare_assets_archive_sha256": _sha256(healthcare_assets_archive),
            "isaaclab_root": str(tmp_path / "isaaclab"),
            "gr00t_root": str(tmp_path / "gr00t"),
            "model_root": str(tmp_path / "model"),
            "hf_cache": str(tmp_path / "hf-cache"),
            "task_overlay_root": str(tmp_path / "overlay"),
            "sanitized_tray_usd": str(tray),
            "python_deps": [str(tmp_path / "python-deps")],
        },
        "experiment": {
            "name": "W73-test",
            "config": str(config),
            "config_sha256": _sha256(config),
            "runner": str(runner),
            "runner_sha256": _sha256(runner),
            "output_root": str(tmp_path / "output"),
            "max_steps": 100000,
            "val_check_interval": 5,
            "save_interval": 5,
            "newton_num_substeps": 2,
            "resume_dir": None,
            "debug_nonfinite": False,
            "workload_seconds": 13200,
            "shutdown_grace_seconds": 600,
        },
    }
    path = tmp_path / "site.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_site_freezes_canonical_chunk16_contract(tmp_path: Path) -> None:
    site = MODULE._load_site(_site(tmp_path))

    assert site["_resolved"]["workload_contract"] == {
        "chunks": 16,
        "envs": 64,
        "physical_actions": 65536,
        "policy_decisions": 4096,
        "global_batch": 2048,
        "micro_batch": 128,
        "update_epochs": 4,
        "optimizer_updates": 8,
        "reward_type": "chunk_level",
        "logprob_type": "action_level",
        "actor_lr": 2e-5,
        "eval_interval": 5,
        "eval_envs": 8,
        "eval_fixed_resets": True,
        "eval_video": True,
        "newton_num_substeps": 2,
    }
    command = MODULE._submission_argv(site)
    assert "--time=04:00:00" in command
    assert "--gpus-per-node=8" not in command
    assert "--constraint=h100" in command
    assert "--exclusive" in command
    assert "--signal=B:TERM@600" in command
    assert f"--export=ALL,W73_GIT_LFS_BIN={site['runtime']['git_lfs_bin']}" in command
    assert site["_resolved"]["launcher"] == str(LAUNCHER)
    assert command[-4:] == [
        str(LAUNCHER),
        "allocation-run",
        "--site",
        str(site["_resolved"]["path"]),
    ]


def test_bootstrap_git_lfs_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    git_lfs_bin = tmp_path / "git-lfs"
    git_lfs_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    git_lfs_bin.chmod(0o755)
    monkeypatch.setenv("W73_GIT_LFS_BIN", str(git_lfs_bin))
    monkeypatch.setenv("PATH", "/usr/bin")

    MODULE._bootstrap_git_lfs_path()

    assert os.environ["PATH"] == f"{tmp_path}:/usr/bin"


def test_site_rejects_feature_reuse_in_baseline(tmp_path: Path) -> None:
    original = (
        ROOT / "toolkits/eos/gr00t_trocar/config-baseline-chunk16.yaml"
    ).read_text(encoding="utf-8")
    modified = original.replace(
        "    model_type: gr00t\n",
        "    model_type: gr00t\n    reuse_rollout_backbone_features: true\n",
        1,
    )
    config = tmp_path / "feature.yaml"
    config.write_text(modified, encoding="utf-8")

    with pytest.raises(MODULE.WorkflowError, match="feature-reuse fields"):
        MODULE._load_site(_site(tmp_path, config=config))


def test_site_rejects_external_provenance_drift(tmp_path: Path) -> None:
    site_path = _site(tmp_path)
    value = json.loads(site_path.read_text(encoding="utf-8"))
    value["provenance"]["files"][0]["sha256"] = "0" * 64
    site_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(MODULE.WorkflowError, match="SHA-256 mismatch"):
        MODULE._load_site(site_path)


def test_site_rejects_incompatible_save_and_evaluation_intervals(
    tmp_path: Path,
) -> None:
    site_path = _site(tmp_path)
    value = json.loads(site_path.read_text(encoding="utf-8"))
    value["experiment"]["val_check_interval"] = 3
    site_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(MODULE.WorkflowError, match="must be divisible"):
        MODULE._load_site(site_path)


def test_site_accepts_owned_resume_checkpoint(tmp_path: Path) -> None:
    site_path = _site(tmp_path)
    value = json.loads(site_path.read_text(encoding="utf-8"))
    checkpoint = tmp_path / "output" / "prior" / "global_step_5"
    (checkpoint / "actor").mkdir(parents=True)
    value["experiment"]["resume_dir"] = str(checkpoint)
    value["experiment"]["debug_nonfinite"] = True
    site_path.write_text(json.dumps(value), encoding="utf-8")

    site = MODULE._load_site(site_path)

    assert site["experiment"]["resume_dir"] == str(checkpoint)
    assert site["experiment"]["debug_nonfinite"] is True


def test_site_rejects_resume_checkpoint_outside_output_root(tmp_path: Path) -> None:
    site_path = _site(tmp_path)
    value = json.loads(site_path.read_text(encoding="utf-8"))
    checkpoint = tmp_path / "external" / "global_step_5"
    (checkpoint / "actor").mkdir(parents=True)
    value["experiment"]["resume_dir"] = str(checkpoint)
    site_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(MODULE.WorkflowError, match="under experiment.output_root"):
        MODULE._load_site(site_path)


def test_virtualenv_python_symlink_is_preserved(tmp_path: Path) -> None:
    python_link = tmp_path / "python"
    python_link.symlink_to(Path(sys.executable).resolve(strict=True))

    assert MODULE._absolute_executable(str(python_link), "python") == python_link


def test_deadline_reserves_slurm_shutdown_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = MODULE._load_site(_site(tmp_path))
    monkeypatch.setattr(MODULE.time, "time", lambda: 1_000_000.0)
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1005000")

    assert MODULE._allocation_deadline(site) == 1_004_400


def test_ray_workers_inherit_source_and_task_environment(tmp_path: Path) -> None:
    site = MODULE._load_site(_site(tmp_path))

    env = MODULE._ray_worker_environment(site)
    assert env["PYTHONPATH"].split(os.pathsep) == [
        site["runtime"]["python_deps"][0],
        site["runtime"]["gr00t_root"],
        site["source"]["root"],
        site["runtime"]["task_overlay_root"],
        str(Path(site["runtime"]["isaaclab_root"]) / "source"),
    ]
    assert env["RLINF_EXT_MODULE"] == "w68_rlinf_extension"
    assert env["RLINF_CONFIG_FILE"] == site["experiment"]["config"]
    assert env["W68_SANITIZED_TRAY_USD"] == site["runtime"]["sanitized_tray_usd"]


def test_ray_failure_logs_are_archived_before_cleanup(tmp_path: Path) -> None:
    ray_temp = tmp_path / "ray"
    logs = ray_temp / "session_latest" / "logs"
    logs.mkdir(parents=True)
    (logs / "worker-1.err").write_text("root cause\n", encoding="utf-8")
    attempt = tmp_path / "attempt"

    archive_path = MODULE._archive_ray_failure_logs(ray_temp, attempt)

    assert archive_path == attempt / "runtime" / "ray-failure-logs.tar.gz"
    assert archive_path.is_file()
    with tarfile.open(archive_path, "r:gz") as archive:
        assert archive.extractfile("logs/worker-1.err").read() == b"root cause\n"


def test_receipts_are_create_only(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    MODULE._write_new_json(path, {"value": 1})

    with pytest.raises(MODULE.WorkflowError, match="refusing to replace"):
        MODULE._write_new_json(path, {"value": 2})


def test_prepare_runtime_reuses_only_matching_package_freeze(tmp_path: Path) -> None:
    revisions: dict[str, str] = {}
    for name in ("source", "isaaclab", "gr00t"):
        root = tmp_path / name
        root.mkdir()
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "config", "user.name", "W73 Test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "config", "user.email", "w73@example.test"],
            check=True,
        )
        if name == "source":
            (root / "requirements").mkdir()
            (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
            (root / "requirements" / "fixture.txt").write_text("fixture\n")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(root), "commit", "-q", "--allow-empty", "-m", name],
            check=True,
        )
        revisions[name] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()

    flash_attn_wheel = tmp_path / "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
    flash_attn_wheel.write_bytes(b"wheel")
    torchcodec_wheel = (
        tmp_path / "torchcodec-0.11.1-cp312-cp312-manylinux_2_28_x86_64.whl"
    )
    torchcodec_wheel.write_bytes(b"torchcodec-wheel")
    spec = json.loads(
        (ROOT / "toolkits/eos/gr00t_trocar/runtime-spec.json").read_text(
            encoding="utf-8"
        )
    )
    spec["isaaclab_revision"] = revisions["isaaclab"]
    spec["gr00t_revision"] = revisions["gr00t"]
    spec["flash_attn_wheel_sha256"] = _sha256(flash_attn_wheel)
    spec["torchcodec_wheel_sha256"] = _sha256(torchcodec_wheel)
    spec_path = tmp_path / "runtime-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    runtime = tmp_path / "envs" / "runtime"
    (runtime / "bin").mkdir(parents=True)
    fake_python = runtime / "bin" / "python"
    fake_python.write_text("#!/bin/sh\ncat >/dev/null\n", encoding="utf-8")
    fake_python.chmod(0o755)
    freeze = runtime / "requirements.freeze.txt"
    freeze.write_text("torch==fixture\n", encoding="utf-8")
    script = ROOT / "toolkits/eos/gr00t_trocar/prepare_runtime.sh"
    dependency_listing = subprocess.check_output(
        [
            "git",
            "-C",
            str(tmp_path / "source"),
            "ls-files",
            "-s",
            "--",
            "pyproject.toml",
            "requirements",
        ]
    )
    manifest = {
        "schema": "rlinf.eos.python-runtime-manifest.v1",
        "runtime_spec_sha256": _sha256(spec_path),
        "prepare_script_sha256": _sha256(script),
        "rlinf_dependency_inputs_sha256": hashlib.sha256(
            dependency_listing
        ).hexdigest(),
        "requirements_freeze_sha256": _sha256(freeze),
        "isaaclab_revision": revisions["isaaclab"],
        "gr00t_revision": revisions["gr00t"],
        "flash_attn_wheel_sha256": _sha256(flash_attn_wheel),
        "torchcodec_wheel_sha256": _sha256(torchcodec_wheel),
        "transformers": spec["transformers_version"],
    }
    (runtime / "rlinf-runtime-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    env = dict(os.environ)
    env.update(
        {
            "W73_SOURCE_ROOT": str(tmp_path / "source"),
            "W73_ISAACLAB_ROOT": str(tmp_path / "isaaclab"),
            "W73_GROOT_ROOT": str(tmp_path / "gr00t"),
            "W73_RUNTIME_ROOT": str(runtime),
            "W73_RUNTIME_SPEC": str(spec_path),
            "W73_RUNTIME_SPEC_SHA256": _sha256(spec_path),
            "W73_UV_CACHE": str(tmp_path / "uv-cache"),
            "W73_FLASH_ATTN_WHEEL": str(flash_attn_wheel),
            "W73_FLASH_ATTN_WHEEL_SHA256": _sha256(flash_attn_wheel),
            "W73_TORCHCODEC_WHEEL": str(torchcodec_wheel),
            "W73_TORCHCODEC_WHEEL_SHA256": _sha256(torchcodec_wheel),
        }
    )
    reused = subprocess.run(
        ["bash", str(script)], check=False, capture_output=True, text=True, env=env
    )
    assert reused.returncode == 0
    assert "reusing verified RLinf runtime" in reused.stdout

    freeze.write_text("torch==tampered\n", encoding="utf-8")
    rejected = subprocess.run(
        ["bash", str(script)], check=False, capture_output=True, text=True, env=env
    )
    assert rejected.returncode != 0
    assert "runtime package freeze hash mismatch" in rejected.stderr


def test_prepare_runtime_requires_uv_managed_python() -> None:
    script = (ROOT / "toolkits/eos/gr00t_trocar/prepare_runtime.sh").read_text(
        encoding="utf-8"
    )

    assert 'UV_PYTHON_INSTALL_DIR="$runtime_parent/.uv-python"' in script
    assert "UV_PYTHON_PREFERENCE=only-managed" in script
    assert (
        'cd "$runtime_parent"\n'
        'git -C "$W73_SOURCE_ROOT" worktree remove --force "$build_source"' in script
    )
    assert '"$W73_TORCHCODEC_WHEEL"' in script
    assert "--no-deps" in script
    assert '"transformers==$(spec_value transformers_version)"' in script
    assert "from torchcodec.decoders import VideoDecoder" in script
    assert "from transformers.image_utils import VideoInput" in script


def test_runtime_contract_pins_torchcodec_for_torch_211() -> None:
    spec = json.loads(
        (ROOT / "toolkits/eos/gr00t_trocar/runtime-spec.json").read_text(
            encoding="utf-8"
        )
    )
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert spec["torch_version"] == "2.11.0"
    assert spec["torchcodec_version"] == "0.11.1+cpu"
    assert spec["torchcodec_backend"] == "cpu"
    assert spec["transformers_version"] == "4.51.3"
    assert (
        spec["torchcodec_wheel_filename"]
        == "torchcodec-0.11.1-cp312-cp312-manylinux_2_28_x86_64.whl"
    )
    assert spec["torchcodec_wheel_sha256"] == (
        "6c26e90e7aa982302644d0af8cb706318682bb390f48a80ecbfeab03499acd04"
    )
    assert "from torchcodec.decoders import VideoDecoder" in launcher
    assert "from transformers.image_utils import VideoInput" in launcher


def test_n1d7_runtime_and_model_contract_are_frozen() -> None:
    spec = json.loads(
        (ROOT / "toolkits/eos/gr00t_trocar/runtime-spec-n1d7.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (ROOT / "toolkits/eos/gr00t_trocar/model-manifest-n1d7.json").read_text(
            encoding="utf-8"
        )
    )
    script = (ROOT / "toolkits/eos/gr00t_trocar/prepare_runtime.sh").read_text(
        encoding="utf-8"
    )

    assert spec["torch_version"] == "2.11.0"
    assert spec["transformers_version"] == "4.57.3"
    assert spec["installer_model"] == "gr00t_n1d7"
    assert spec["gr00t_revision"] == (
        "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
    )
    assert 'installer_model=$(spec_value_or installer_model gr00t)' in script
    assert 'embodied --model "$installer_model" --env isaaclab' in script

    assert manifest["model"]["revision"] == (
        "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
    )
    assert manifest["backbone"]["revision"] == (
        "9ce19a195e423419c349abfc86fd07178b230561"
    )
    assert manifest["model"]["files"]["model-00001-of-00002.safetensors"] == (
        "8a1a1d8a33c99103c7c80c136073c5bb8bfe9ca8f7a970c93c033ea89742906d"
    )
    assert manifest["backbone"]["files"]["model.safetensors"] == (
        "fa5a6e6ef4fce40216b185cc48a3b24d31637ac3e2ba69c107ed1f389c1e6ede"
    )


def test_eos_template_pins_newton_tray_textures() -> None:
    template = json.loads(
        (ROOT / "toolkits/eos/gr00t_trocar/site.eos.template.json").read_text(
            encoding="utf-8"
        )
    )
    files = {item["name"]: item for item in template["provenance"]["files"]}

    assert files["trocar-tray-box-base-color-texture"]["sha256"] == (
        "43ee84268e161a10ece13fff539388e5fa807f6fffb3e0314366867336c7d092"
    )
    assert files["trocar-tray-box-normal-texture"]["sha256"] == (
        "742bc21bf7b4fcaec7a7670466a6aa91e6d2910cd2b2449fd1155b958249824e"
    )
    assert files["trocar-tray-box-orm-texture"]["sha256"] == (
        "688701d0450d5f913134f978793d0c3e4423cc7c81d0397aec295a19dfe18bd3"
    )
    assert files["w68-healthcare-assets"]["sha256"] == (
        "9289b4e37b64a4fbe86f1a030393179dbcb2215f283a5411ca46529f5fe8bf13"
    )


def test_runner_stages_healthcare_assets_before_training() -> None:
    runner = (ROOT / "toolkits/eos/gr00t_trocar/run_baseline.sh").read_text(
        encoding="utf-8"
    )

    extract = 'tar -xf "$W73_HEALTHCARE_ASSETS_ARCHIVE" -C "$short_tmp"'
    launch = '"$W73_RUNTIME_PYTHON" examples/embodiment/train_embodied_agent.py'
    assert extract in runner
    assert runner.index(extract) < runner.index(launch)


def test_runner_localizes_train_and_eval_videos() -> None:
    runner = (ROOT / "toolkits/eos/gr00t_trocar/run_baseline.sh").read_text(
        encoding="utf-8"
    )

    assert (
        'env.train.video_cfg.video_base_dir="$W73_ATTEMPT_ROOT/output/video/train"'
        in runner
    )
    assert (
        'env.eval.video_cfg.video_base_dir="$W73_ATTEMPT_ROOT/output/video/eval"'
        in runner
    )
    assert 'runner.resume_dir="$W73_RESUME_DIR"' in runner
    assert 'runner.debug_nonfinite="$W73_DEBUG_NONFINITE"' in runner
    assert 'export RLINF_DEBUG_NONFINITE="$W73_DEBUG_NONFINITE"' in runner
    assert 'export W68_NEWTON_NUM_SUBSTEPS="$W73_NEWTON_NUM_SUBSTEPS"' in runner
    assert "W73_NEWTON_NUM_SUBSTEPS=%s" in runner


def test_prepare_runtime_build_isolated_from_canonical_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    isaaclab = tmp_path / "isaaclab"
    gr00t = tmp_path / "gr00t"
    revisions: dict[str, str] = {}
    for name, root in (("source", source), ("isaaclab", isaaclab), ("gr00t", gr00t)):
        root.mkdir()
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "config", "user.name", "W73 Test"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "config", "user.email", "w73@example.test"],
            check=True,
        )
        if name == "source":
            (root / "requirements").mkdir()
            installer = root / "requirements" / "install.sh"
            installer.write_text(
                "#!/bin/sh\n"
                "while [ $# -gt 0 ]; do\n"
                '  if [ "$1" = --venv ]; then runtime=$2; shift 2; else shift; fi\n'
                "done\n"
                'mkdir -p "$runtime/bin"\n'
                'cp "$FAKE_RUNTIME_PYTHON" "$runtime/bin/python"\n'
                'chmod +x "$runtime/bin/python"\n',
                encoding="utf-8",
            )
            installer.chmod(0o755)
            (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(root), "commit", "-q", "--allow-empty", "-m", name],
            check=True,
        )
        revisions[name] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()

    flash_attn_wheel = tmp_path / "flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl"
    flash_attn_wheel.write_bytes(b"wheel")
    torchcodec_wheel = (
        tmp_path / "torchcodec-0.11.1-cp312-cp312-manylinux_2_28_x86_64.whl"
    )
    torchcodec_wheel.write_bytes(b"torchcodec-wheel")
    spec = json.loads(
        (ROOT / "toolkits/eos/gr00t_trocar/runtime-spec.json").read_text(
            encoding="utf-8"
        )
    )
    spec["isaaclab_revision"] = revisions["isaaclab"]
    spec["gr00t_revision"] = revisions["gr00t"]
    spec["flash_attn_wheel_sha256"] = _sha256(flash_attn_wheel)
    spec["torchcodec_wheel_sha256"] = _sha256(torchcodec_wheel)
    spec_path = tmp_path / "runtime-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/bin/sh\nif [ \"$1 $2\" = 'pip freeze' ]; then echo fixture==1; fi\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "cat >/dev/null\n"
        'case "${2:-}" in\n'
        '  *.json) printf \'%s\\n\' "{\\"schema\\":\\"rlinf.eos.python-runtime-manifest.v1\\",\\"runtime_spec_sha256\\":\\"$3\\"}" > "$2" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    runtime = tmp_path / "envs" / "runtime"
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_RUNTIME_PYTHON": str(fake_python),
            "W73_SOURCE_ROOT": str(source),
            "W73_ISAACLAB_ROOT": str(isaaclab),
            "W73_GROOT_ROOT": str(gr00t),
            "W73_RUNTIME_ROOT": str(runtime),
            "W73_RUNTIME_SPEC": str(spec_path),
            "W73_RUNTIME_SPEC_SHA256": _sha256(spec_path),
            "W73_UV_CACHE": str(tmp_path / "uv-cache"),
            "W73_FLASH_ATTN_WHEEL": str(flash_attn_wheel),
            "W73_FLASH_ATTN_WHEEL_SHA256": _sha256(flash_attn_wheel),
            "W73_TORCHCODEC_WHEEL": str(torchcodec_wheel),
            "W73_TORCHCODEC_WHEEL_SHA256": _sha256(torchcodec_wheel),
        }
    )
    script = ROOT / "toolkits/eos/gr00t_trocar/prepare_runtime.sh"
    prepared = subprocess.run(
        ["bash", str(script)], check=False, capture_output=True, text=True, env=env
    )

    assert prepared.returncode == 0, prepared.stderr
    assert (runtime / "rlinf-runtime-manifest.json").is_file()
    assert not (runtime.parent / ".runtime.source").exists()
    assert (
        subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        )
        == ""
    )
    assert (
        len(
            subprocess.check_output(
                ["git", "-C", str(source), "worktree", "list", "--porcelain"],
                text=True,
            ).split("worktree ")
        )
        == 2
    )


def test_dry_run_does_not_call_sbatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site_path = _site(tmp_path)
    real_run = MODULE.subprocess.run

    def guarded_run(argv: object, *args: object, **kwargs: object) -> object:
        if isinstance(argv, list) and argv and argv[0] == "sbatch":
            raise AssertionError("dry-run must not execute sbatch")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(MODULE.subprocess, "run", guarded_run)
    args = argparse.Namespace(site=str(site_path), dry_run=True, skip_image_hash=False)
    assert MODULE._submit(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "dry_run"


def test_materialize_smoke_has_bounded_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_path = _site(tmp_path)
    template = tmp_path / "template.json"
    value = json.loads(site_path.read_text(encoding="utf-8"))
    value["source"]["revision"] = "AUTO"
    value["runtime"]["runtime_spec_sha256"] = "AUTO"
    value["runtime"]["prepare_script_sha256"] = "AUTO"
    value["experiment"]["config_sha256"] = "AUTO"
    value["experiment"]["runner_sha256"] = "AUTO"
    template.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "smoke.json"
    args = argparse.Namespace(
        template=str(template),
        output=str(output),
        skip_image_hash=False,
        smoke=True,
    )

    assert MODULE._materialize(args) == 0
    smoke = json.loads(output.read_text(encoding="utf-8"))
    assert smoke["experiment"]["name"] == "W73-test-smoke"
    assert smoke["experiment"]["max_steps"] == 1
    assert smoke["experiment"]["val_check_interval"] == 1
    assert smoke["experiment"]["save_interval"] == 1
    assert smoke["experiment"]["workload_seconds"] == 5_400
    assert (
        MODULE._load_site(output)["_resolved"]["workload_contract"]["eval_interval"]
        == 1
    )
    assert smoke["runtime"]["runtime_spec_sha256"] == _sha256(
        Path(smoke["runtime"]["runtime_spec"])
    )
    assert smoke["runtime"]["prepare_script_sha256"] == _sha256(
        ROOT / "toolkits/eos/gr00t_trocar/prepare_runtime.sh"
    )


def test_materialize_overrides_checkpoint_and_evaluation_intervals(
    tmp_path: Path,
) -> None:
    site_path = _site(tmp_path)
    template = tmp_path / "template.json"
    value = json.loads(site_path.read_text(encoding="utf-8"))
    value["source"]["revision"] = "AUTO"
    value["runtime"]["runtime_spec_sha256"] = "AUTO"
    value["runtime"]["prepare_script_sha256"] = "AUTO"
    value["experiment"]["config_sha256"] = "AUTO"
    value["experiment"]["runner_sha256"] = "AUTO"
    template.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "long-run.json"
    args = argparse.Namespace(
        template=str(template),
        output=str(output),
        skip_image_hash=False,
        smoke=False,
        max_steps=120,
        val_check_interval=5,
        save_interval=20,
        newton_num_substeps=4,
    )

    assert MODULE._materialize(args) == 0
    materialized = json.loads(output.read_text(encoding="utf-8"))
    assert materialized["experiment"]["max_steps"] == 120
    assert materialized["experiment"]["val_check_interval"] == 5
    assert materialized["experiment"]["save_interval"] == 20
    assert materialized["experiment"]["newton_num_substeps"] == 4
    contract = MODULE._load_site(output)["_resolved"]["workload_contract"]
    assert contract["eval_interval"] == 5
    assert contract["newton_num_substeps"] == 4
