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

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "toolkits/eos/gr00t_trocar/tensorrt"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from toolkits.eos.gr00t_trocar.tensorrt import (  # noqa: E402
    builder_probe,
    model_view,
    official_b1,
    prepare_builder,
    resident_b1,
    start_official_b1,
)

OFFICIAL_NUMERICS = """
[6a] ViT output comparison (image_embeds):
  Cosine Similarity: 0.999586
  L1 Mean Error:     0.010622
  L∞ Max Error:      0.567877

[6b] Backbone output comparison (LLM output, before vl_self_attention):
  Cosine Similarity: 0.999982
  L1 Mean Error:     0.105064
  L∞ Max Error:      7.000000

[6b] Final action output comparison:
  Cosine Similarity: 0.999993
  L1 Mean Error:     0.000817
  L∞ Max Error:      0.006330
"""


def _git_init(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "W78"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "w78@example.test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "--allow-empty", "-m", "fixture"],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def test_parse_official_numerics_and_enforce_cosine_gate() -> None:
    parsed = official_b1._parse_numerics(OFFICIAL_NUMERICS)

    assert parsed["final_action_output_comparison"]["max_abs"] == 0.00633
    with pytest.raises(RuntimeError, match="cosine gate failed"):
        official_b1._parse_numerics(OFFICIAL_NUMERICS.replace("0.999586", "0.998999"))


def test_onnx_inventory_includes_external_data(tmp_path: Path) -> None:
    for name in (*official_b1.EXPECTED_ONNX, "dit_bf16.onnx.data"):
        (tmp_path / name).write_bytes(name.encode())

    inventory = official_b1._onnx_inventory(tmp_path)

    assert set(inventory) == {*official_b1.EXPECTED_ONNX, "dit_bf16.onnx.data"}


def test_builder_python_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    target = tmp_path / "python-target"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(target)

    assert official_b1._executable(venv_python) == venv_python


def test_builder_runtime_library_paths_are_shared_by_probe_and_pipeline(
    tmp_path: Path,
) -> None:
    env_root = tmp_path / "env"
    site_packages = env_root / "lib/python3.12/site-packages"
    expected = [site_packages / "torch/lib", site_packages / "torchcodec"]
    for path in expected:
        path.mkdir(parents=True)

    assert builder_probe.runtime_library_paths(env_root) == expected
    environment = prepare_builder._isolated_environment(
        {"PYTHONPATH": "wrong-prefix", "LD_LIBRARY_PATH": "/system/lib"},
        env_root,
    )
    assert "PYTHONPATH" not in environment
    assert environment["LD_LIBRARY_PATH"].split(os.pathsep) == [
        *(str(path) for path in expected),
        "/system/lib",
    ]


def test_builder_runtime_library_paths_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="expected one builder site-packages"):
        builder_probe.runtime_library_paths(tmp_path / "missing")


def test_ldd_normalization_removes_only_aslr_addresses() -> None:
    output = "\n".join(
        (
            "linux-vdso.so.1 (0x00007fff1234)",
            "libtorch.so => /venv/torch/lib/libtorch.so (0x00007abc5678)",
            "libmissing.so => not found",
        )
    )

    assert builder_probe._normalize_ldd(output) == [
        "linux-vdso.so.1",
        "libtorch.so => /venv/torch/lib/libtorch.so",
        "libmissing.so => not found",
    ]


def test_resident_statistics_retain_raw_distribution() -> None:
    statistics = resident_b1._statistics([1.0, 2.0, 3.0, 4.0])

    assert statistics["count"] == 4
    assert statistics["p50_ms"] == 2.5
    assert statistics["p95_ms"] == pytest.approx(3.85)


def test_model_view_preserves_selector_suffix_and_hashes_weights(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    backbone = tmp_path / "backbone"
    output = tmp_path / "view"
    model.mkdir()
    backbone.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_name": "nvidia/Cosmos-Reason2-2B"}), encoding="utf-8"
    )
    (model / "processor_config.json").write_text(
        json.dumps({"processor_kwargs": {"model_name": "old"}}), encoding="utf-8"
    )
    (model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (backbone / "config.json").write_text("{}", encoding="utf-8")

    receipt = model_view.materialize_local_model_view(model, backbone, output)

    assert receipt["selector_preserves_repository_suffix"] is True
    assert receipt["source_files"]["model-00001-of-00001.safetensors"]["bytes"] == 7
    assert (output / "model-00001-of-00001.safetensors").is_symlink()
    generated = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert generated["model_name"].endswith("nvidia/Cosmos-Reason2-2B")


def _site(tmp_path: Path) -> Path:
    source = tmp_path / "RLinf"
    helpers = source / "toolkits/eos/gr00t_trocar/tensorrt"
    helpers.mkdir(parents=True)
    dataset_objects = [
        (f"objects/object-{index:02d}.bin", f"object-{index}\n".encode())
        for index in range(15)
    ]
    for name in (*start_official_b1.HELPERS, "start_official_b1.py"):
        if name == "libero-b1-lfs.json":
            continue
        (helpers / name).write_text(f"# {name}\n", encoding="utf-8")
    lfs_manifest = {
        "schema": "rlinf.gr00t-n1d7-libero-b1-lfs.v1",
        "objects": [
            {
                "path": path,
                "lfs_oid": f"sha256:{hashlib.sha256(content).hexdigest()}",
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for path, content in dataset_objects
        ],
    }
    lfs_manifest_path = helpers / "libero-b1-lfs.json"
    lfs_manifest_path.write_text(json.dumps(lfs_manifest), encoding="utf-8")
    revision = _git_init(source)
    gr00t = tmp_path / "Isaac-GR00T"
    (gr00t / "demo").mkdir(parents=True)
    gr00t_revision = _git_init(gr00t)
    model = tmp_path / "model"
    backbone = tmp_path / "backbone"
    dataset = tmp_path / "dataset"
    output = tmp_path / "runs"
    cache = tmp_path / "artifacts"
    for root in (model, backbone, dataset, output, cache):
        root.mkdir()
    for relative, content in dataset_objects:
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    image = tmp_path / "image.sqsh"
    image.write_bytes(b"image")
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\n", encoding="utf-8")
    uv.chmod(0o755)
    torchcodec_wheel = tmp_path / "torchcodec.whl"
    torchcodec_wheel.write_bytes(b"torchcodec")
    value = {
        "schema": start_official_b1.SCHEMA,
        "slurm": {
            "account": "account",
            "partition": "batch",
            "constraint": "h100",
            "cpus_per_task": 32,
            "time_limit": "04:00:00",
            "signal_seconds": 600,
            "exclusive": True,
        },
        "container": {
            "oci_reference": "registry/image:test",
            "registry_digest": f"sha256:{'1' * 64}",
            "image": str(image),
            "image_sha256": start_official_b1._sha256(image),
            "mounts": [f"{tmp_path}:{tmp_path}"],
        },
        "source": {
            "root": str(source),
            "revision": revision,
            "required_ancestor": revision,
            "require_clean": True,
        },
        "inputs": {
            "isaac_gr00t_root": str(gr00t),
            "isaac_gr00t_revision": gr00t_revision,
            "model_root": str(model),
            "model_revision": "model-revision",
            "backbone_root": str(backbone),
            "dataset_root": str(dataset),
            "dataset_lfs_manifest": str(lfs_manifest_path),
            "dataset_lfs_manifest_sha256": start_official_b1._sha256(lfs_manifest_path),
        },
        "builder": {
            "env_root": str(tmp_path / "env"),
            "uv": str(uv),
            "uv_cache": str(tmp_path / "uv-cache"),
            "torchcodec_wheel": str(torchcodec_wheel),
            "torchcodec_wheel_sha256": start_official_b1._sha256(torchcodec_wheel),
        },
        "experiment": {
            "name": "W79-test",
            "output_root": str(output),
            "artifact_cache": str(cache),
        },
    }
    site = tmp_path / "site.json"
    site.write_text(json.dumps(value), encoding="utf-8")
    return site


def test_site_validation_and_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site = _site(tmp_path)
    loaded = start_official_b1._load(site)

    assert loaded["_resolved"]["source_revision"] == loaded["source"]["revision"]
    assert loaded["_resolved"]["launcher"].endswith("/start_official_b1.py")
    args = argparse.Namespace(site=str(site), dry_run=True, skip_image_hash=False)
    assert start_official_b1._submit(args) == 0
    command = json.loads(capsys.readouterr().out)["command"]
    assert command[0] == "sbatch"
    assert "--constraint=h100" in command


def test_site_rejects_git_lfs_pointer(tmp_path: Path) -> None:
    site = _site(tmp_path)
    value = json.loads(site.read_text(encoding="utf-8"))
    dataset = Path(value["inputs"]["dataset_root"])
    (dataset / "pointer").write_text(
        "version https://git-lfs.github.com/spec/v1\n", encoding="utf-8"
    )

    with pytest.raises(start_official_b1.WorkflowError, match="Git LFS pointers"):
        start_official_b1._load(site)


def test_site_rejects_dataset_content_hash_mismatch(tmp_path: Path) -> None:
    site = _site(tmp_path)
    value = json.loads(site.read_text(encoding="utf-8"))
    dataset = Path(value["inputs"]["dataset_root"])
    (dataset / "objects/object-00.bin").write_bytes(b"changed")

    with pytest.raises(start_official_b1.WorkflowError, match="SHA-256 mismatch"):
        start_official_b1._load(site)


def test_resident_submit_requires_qualified_exact_engine_bundle(tmp_path: Path) -> None:
    site_path = _site(tmp_path)
    site = start_official_b1._load(site_path)
    output = Path(site["_resolved"]["output_root"])
    oracle = output / "oracle"
    oracle.mkdir()
    (oracle / "qualification.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "engines": {name: {} for name in start_official_b1.ORACLE_ENGINES},
            }
        ),
        encoding="utf-8",
    )
    (oracle / "allocation-result.json").write_text(
        json.dumps({"status": "passed"}), encoding="utf-8"
    )

    resolved = start_official_b1._qualified_oracle(oracle, output)
    command = start_official_b1._resident_sbatch(site, resolved)

    assert resolved == oracle.resolve()
    assert "resident-allocation-run" in command
    assert str(resolved) in command

    qualification = json.loads((oracle / "qualification.json").read_text())
    qualification["engines"].pop("vit.engine")
    (oracle / "qualification.json").write_text(
        json.dumps(qualification), encoding="utf-8"
    )
    with pytest.raises(start_official_b1.WorkflowError, match="exact seven engines"):
        start_official_b1._qualified_oracle(oracle, output)


def test_builder_environment_removes_python_path_and_user_site() -> None:
    environment = prepare_builder._isolated_environment(
        {"PYTHONPATH": "/wrong/prefix", "KEEP": "value"}
    )

    assert "PYTHONPATH" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PIP_NO_CACHE_DIR"] == "1"
    assert environment["KEEP"] == "value"


def test_shipped_libero_manifest_binds_fifteen_lfs_objects() -> None:
    value = json.loads((TOOLS / "libero-b1-lfs.json").read_text(encoding="utf-8"))

    assert value["schema"] == "rlinf.gr00t-n1d7-libero-b1-lfs.v1"
    assert len(value["objects"]) == 15
    assert all(
        entry["lfs_oid"] == f"sha256:{entry['sha256']}" for entry in value["objects"]
    )
