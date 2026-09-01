import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
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
    runtime_python = Path(sys.executable).resolve(strict=True)
    for name in (
        "poiesis",
        "gr00t",
        "model",
        "hf-cache",
        "overlay",
        "python-deps",
        "output",
    ):
        (tmp_path / name).mkdir()
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
            "registry_digest": f"sha256:{'1' * 64}",
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
                }
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
            ],
        },
        "runtime": {
            "python": str(runtime_python),
            "poiesis_root": str(tmp_path / "poiesis"),
            "gr00t_root": str(tmp_path / "gr00t"),
            "model_root": str(tmp_path / "model"),
            "hf_cache": str(tmp_path / "hf-cache"),
            "task_overlay_root": str(tmp_path / "overlay"),
            "sanitized_tray_usd": str(tray),
            "python_deps": [str(tmp_path / "python-deps")],
            "prepare_command": ["/usr/local/bin/poiesis-w63-prepare"],
        },
        "experiment": {
            "name": "W73-test",
            "config": str(config),
            "config_sha256": _sha256(config),
            "runner": str(runner),
            "runner_sha256": _sha256(runner),
            "output_root": str(tmp_path / "output"),
            "max_steps": 100000,
            "save_interval": 5,
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
    }
    command = MODULE._submission_argv(site)
    assert "--time=04:00:00" in command
    assert "--gpus-per-node=8" in command
    assert "--signal=B:TERM@600" in command


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


def test_deadline_reserves_slurm_shutdown_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = MODULE._load_site(_site(tmp_path))
    monkeypatch.setattr(MODULE.time, "time", lambda: 1_000_000.0)
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1005000")

    assert MODULE._allocation_deadline(site) == 1_004_400


def test_receipts_are_create_only(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    MODULE._write_new_json(path, {"value": 1})

    with pytest.raises(MODULE.WorkflowError, match="refusing to replace"):
        MODULE._write_new_json(path, {"value": 2})


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
    assert smoke["experiment"]["save_interval"] == 1
    assert smoke["experiment"]["workload_seconds"] == 5_400
