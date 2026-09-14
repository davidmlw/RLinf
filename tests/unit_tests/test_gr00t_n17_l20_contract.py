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

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "toolkits/gr00t_trocar/w95/contract-v1.json"
BASE_CONFIG = ROOT / "toolkits/gr00t_trocar/config-n1d7-vulkan-control.yaml"
MODULE_PATH = ROOT / "toolkits/gr00t_trocar/w95/contract.py"
TREE_MANIFEST_PATH = ROOT / "toolkits/gr00t_trocar/w95/tree_manifest.py"
GIT_ATTESTATION_PATH = ROOT / "toolkits/gr00t_trocar/w95/git_tree_attestation.py"
L20_LAUNCHER_PATH = ROOT / "toolkits/gr00t_trocar/w95/l20_vulkan_launcher.py"
L20_RUNTIME_PROBE_PATH = ROOT / "toolkits/gr00t_trocar/w95/l20_runtime_probe.py"
L20_Q2_SMOKE_PATH = ROOT / "toolkits/gr00t_trocar/w95/l20_q2_smoke.py"
L20_RUNTIME_SPEC_PATH = (
    ROOT / "toolkits/gr00t_trocar/w95/runtime-spec-n1d7-l20-w88.json"
)
EOS_RUNTIME_SPEC_PATH = ROOT / "toolkits/eos/gr00t_trocar/runtime-spec-n1d7.json"
OVERLAY_REQUIREMENTS_PATH = (
    ROOT / "toolkits/gr00t_trocar/w95/python-overlay-requirements-w88.txt"
)


def _module():
    spec = importlib.util.spec_from_file_location("w95_contract", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree_manifest_module():
    spec = importlib.util.spec_from_file_location(
        "w95_tree_manifest", TREE_MANIFEST_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _base():
    return yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))


def test_contract_is_n17_l20_vulkan_and_rooted_at_fixed_base() -> None:
    contract = _contract()
    assert contract["model"]["family"] == "GR00T N1.7"
    assert contract["source"]["rlinf_base_sha"] == (
        "0f9ea98c7a6d9e3ade24e8f4846c64d3b135dbcc"
    )
    assert contract["workload"]["hardware"] == "8x NVIDIA L20 SM89"
    assert contract["workload"]["renderer"] == "Vulkan/RTX"
    assert contract["artifact_policy"]["forbid_sm90_plans"] is True


def test_l20_runtime_uses_image_torch_and_isolated_tensorrt() -> None:
    runtime = json.loads(L20_RUNTIME_SPEC_PATH.read_text(encoding="utf-8"))
    assert runtime["scope"] == "w95_l20_vulkan_w88_reproduction"
    assert runtime["image_owned_packages"]["torch"]["expected_version"] == (
        "2.10.0+cu128"
    )
    assert "torch" in runtime["python_overlay"]["forbidden_distributions"]
    assert "tensorrt" in runtime["python_overlay"]["forbidden_distributions"]
    assert {"numpy", "pandas"} <= set(
        runtime["python_overlay"]["forbidden_distributions"]
    )
    assert {"numpy", "pandas"} <= set(
        runtime["python_overlay"]["forbidden_top_level_paths"]
    )
    assert runtime["tensorrt_runtime"]["container_path"] == "/w96-trt-runtime"
    assert runtime["pythonpath"] == [
        "/w96-overlay",
        "/w96-trt-runtime",
        "/workspace/gr00t-n17",
        "/workspace/rlinf-src",
    ]


def test_eos_torch_211_runtime_is_not_w96_l20_authority() -> None:
    runtime = json.loads(EOS_RUNTIME_SPEC_PATH.read_text(encoding="utf-8"))
    assert runtime["torch_version"] == "2.11.0"
    assert runtime["scope"] == "eos_h100_newton_uv_runtime"
    assert runtime["w96_l20_authority"] is False


def test_l20_overlay_requirements_are_frozen_without_runtime_authorities() -> None:
    requirements = [
        line.strip()
        for line in OVERLAY_REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    normalized = {
        requirement.split("==", 1)[0].lower().replace("_", "-")
        for requirement in requirements
    }
    assert len(requirements) == 74
    assert len(normalized) == len(requirements)
    assert {"transformers", "tokenizers", "albumentations"} <= normalized
    assert normalized.isdisjoint(
        {
            "torch",
            "torchvision",
            "torchaudio",
            "triton",
            "tensorrt",
            "tensorrt-cu12",
            "tensorrt-cu12-bindings",
            "tensorrt-cu12-libs",
            "numpy",
            "pandas",
        }
    )


def test_contract_freezes_true_b8_chunk16_counts() -> None:
    contract = _contract()
    assert contract["model"]["true_b8"]["batch"] == 8
    assert contract["model"]["executed_action_chunks"] == 16
    assert contract["workload"]["global_envs"] == 64
    assert contract["workload"]["physical_actions_per_outer_step"] == 65536
    assert contract["workload"]["policy_decisions_per_outer_step"] == 4096
    assert contract["workload"]["optimizer_updates_per_outer_step"] == 8


def test_b8_and_b32_authority_is_explicit_and_not_interchangeable() -> None:
    profiles = _contract()["profiles"]
    assert profiles["absolute_correctness_b8"] == {
        "actor_micro_batch_size": 8,
        "gradient_accumulation_microsteps_per_update": 32,
        "authority": "absolute_ppo_authority",
        "allowed_claim": (
            "PPO PASS only after the registered same-revision and optimizer gates pass"
        ),
    }
    assert profiles["baseline_relative_throughput_b32"]["actor_micro_batch_size"] == 32
    assert (
        profiles["baseline_relative_throughput_b32"]["authority"]
        == "baseline_relative_only"
    )
    assert (
        "cannot qualify"
        in profiles["baseline_relative_throughput_b32"]["allowed_claim"]
    )


def test_all_profile_and_arm_configs_render_and_validate() -> None:
    module = _module()
    contract = _contract()
    for profile in contract["profiles"]:
        for arm in contract["arms"]:
            rendered = module.render(_base(), contract, profile, arm)
            assert module.validate(rendered, contract, profile, arm) == []


def test_all_off_and_composed_differ_only_in_registered_paths() -> None:
    module = _module()
    contract = _contract()
    profile = "absolute_correctness_b8"
    all_off = module.render(_base(), contract, profile, "all_off")
    composed = module.render(_base(), contract, profile, "composed")

    def flatten(value, prefix=""):
        result = {}
        for key, child in value.items():
            dotted = f"{prefix}.{key}" if prefix else key
            if isinstance(child, dict):
                result.update(flatten(child, dotted))
            else:
                result[dotted] = child
        return result

    left = flatten(all_off)
    right = flatten(composed)
    changed = {
        path
        for path in left.keys() | right.keys()
        if left.get(path, object()) != right.get(path, object())
    }
    assert changed == {
        "runner.logger.experiment_name",
        "rollout.model.skip_unused_lm_head",
        "actor.model.skip_unused_lm_head",
        "actor.model.rollout_backbone_feature_transport",
    }


def test_b8_replay_cannot_validate_b32_config() -> None:
    module = _module()
    contract = _contract()
    b8 = module.render(_base(), contract, "absolute_correctness_b8", "composed")
    errors = module.validate(
        b8, contract, "baseline_relative_throughput_b32", "composed"
    )
    assert errors == ["actor.micro_batch_size: expected 32, got 8"]


def test_w71_measurement_and_profiler_separation_are_frozen() -> None:
    contract = _contract()
    measurement = contract["measurement"]
    assert measurement["schema"] == "vla-rl.performance-measurement/v1"
    assert measurement["warmup_steps"] == [0]
    assert measurement["measured_steps"] == [1, 2, 3, 4]
    assert "never sum" in measurement["rollout_env_aggregation"]
    profiler = contract["profiler"]
    assert profiler["clean_performance_contains_nsys"] is False
    assert profiler["profiled_samples_are_headline"] is False
    assert profiler["vulkan_trace"].startswith("separate short Env-only")


def test_existing_sm89_vit_is_not_silently_qualified() -> None:
    policy = _contract()["artifact_policy"]
    assert policy["existing_sm89_vit_disposition"] == (
        "component_gate_failed_rebuild_or_requalify_before_use"
    )
    assert policy["existing_sm89_vit_cosine"] == 0.99690463


def test_immutable_tree_manifest_records_files_modes_and_symlinks(tmp_path) -> None:
    module = _tree_manifest_module()
    root = tmp_path / "bundle"
    nested = root / "nested"
    nested.mkdir(parents=True)
    payload = nested / "payload.txt"
    payload.write_text("payload\n", encoding="ascii")
    payload.chmod(0o444)
    (root / "payload-link").symlink_to("nested/payload.txt")

    manifest = module.create_manifest(root)
    entries = {entry["relative_path"]: entry for entry in manifest["entries"]}
    assert manifest["schema"] == "rlinf.immutable-tree-manifest/v1"
    assert entries["nested/payload.txt"]["mode"] == "0444"
    assert entries["nested/payload.txt"]["sha256"] == (
        "d4e4877bac978b7952f0d544fc52ebff5411d351d129f1f056fa43f11da9af2b"
    )
    assert entries["payload-link"]["type"] == "symlink"
    assert entries["payload-link"]["symlink_target"] == "nested/payload.txt"
    assert module.verify_manifest(root, manifest) == []

    payload.chmod(0o644)
    assert module.verify_manifest(root, manifest) == [
        "immutable tree differs at manifest field: tree_sha256",
        "immutable tree differs at manifest field: entries",
    ]


def test_immutable_tree_manifest_rejects_manifest_inside_tree(tmp_path) -> None:
    module = _tree_manifest_module()
    root = tmp_path / "bundle"
    root.mkdir()
    assert module._is_within(root / "manifest.json", root) is True
    assert module._is_within(tmp_path / "manifest.json", root) is False


def test_immutable_tree_materialization_copies_exact_manifest(tmp_path) -> None:
    module = _tree_manifest_module()
    source = tmp_path / "source" / "bundle"
    nested = source / "nested"
    nested.mkdir(parents=True)
    payload = nested / "payload.txt"
    payload.write_text("payload\n", encoding="ascii")
    payload.chmod(0o600)
    (source / "payload-link").symlink_to("nested/payload.txt")
    manifest = module.create_manifest(source)

    destination = tmp_path / "destination" / "bundle"
    module.materialize_manifest(source, destination, manifest)

    assert module.verify_manifest(destination, manifest) == []
    assert (destination / "nested/payload.txt").read_text(encoding="ascii") == (
        "payload\n"
    )
    assert (destination / "payload-link").readlink() == Path("nested/payload.txt")


def test_immutable_tree_materialization_refuses_existing_destination(tmp_path) -> None:
    module = _tree_manifest_module()
    source = tmp_path / "source" / "bundle"
    source.mkdir(parents=True)
    manifest = module.create_manifest(source)
    destination = tmp_path / "destination" / "bundle"
    destination.mkdir(parents=True)

    try:
        module.materialize_manifest(source, destination, manifest)
    except ValueError as error:
        assert str(error) == f"destination already exists: {destination}"
    else:
        raise AssertionError("existing destination must fail closed")


def test_git_tree_attestation_does_not_depend_on_bundle_git_context(
    tmp_path,
) -> None:
    module = _load_module("w96_git_tree_attestation", GIT_ATTESTATION_PATH)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "w96@example.test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "W96 test"],
        check=True,
    )
    script = repo / "run.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
    script.chmod(0o755)
    (repo / "payload.txt").write_text("payload\n", encoding="ascii")
    (repo / "payload-link").symlink_to("payload.txt")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)

    attestation = module.create_attestation(repo, "HEAD")
    bundle = tmp_path / "parent-repo" / "bundle"
    bundle.parent.mkdir()
    subprocess.run(["git", "init", "-q", str(bundle.parent)], check=True)
    shutil.copytree(repo, bundle, ignore=shutil.ignore_patterns(".git"), symlinks=True)
    assert not (bundle / ".git").exists()
    assert module.verify_attestation(bundle, attestation)["status"] == "passed"

    (bundle / "extra.txt").write_text("extra\n", encoding="ascii")
    failed = module.verify_attestation(bundle, attestation)
    assert failed["status"] == "failed"
    assert failed["extra"] == ["extra.txt"]


def test_w96_launcher_quota_parser_is_fail_closed() -> None:
    module = _load_module("w96_l20_launcher", L20_LAUNCHER_PATH)
    output = (
        "Disk quotas for usr 123:\n"
        "Filesystem kbytes quota limit grace files quota limit grace\n"
        "/home/liweim 334046588 1073741824 1153433600 - 1 2 3 -\n"
    )
    assert module._parse_quota_line(output, "/home/liweim") == {
        "used_kib": 334046588,
        "soft_quota_kib": 1073741824,
        "hard_limit_kib": 1153433600,
    }
    try:
        module._parse_quota_line("no quota row\n", "/home/liweim")
    except module.LaunchError as error:
        assert "cannot parse" in str(error)
    else:
        raise AssertionError("missing quota row must fail closed")


def test_w96_docker_command_uses_only_w96_authorities(tmp_path) -> None:
    module = _load_module("w96_l20_launcher_args", L20_LAUNCHER_PATH)
    names = (
        "rlinf_source",
        "gr00t_source",
        "python_overlay",
        "tensorrt_runtime",
        "model",
        "backbone_model",
        "resolved_config",
        "extension",
        "assets_override",
    )
    inputs = {name: str(tmp_path / name) for name in names}
    site = {
        "docker": {"path": "/home/liweim/bin/docker"},
        "image": {"reference": "example.invalid/image@sha256:abc"},
        "inputs": inputs,
    }
    args = module._common_docker_args(site, tmp_path / "run", "w96-test")
    joined = " ".join(args)
    assert module.EXPECTED_PYTHONPATH in joined
    assert "/workspace/rlinf-src:ro" in joined
    assert "/workspace/gr00t-n17:ro" in joined
    assert "/w96-overlay:ro" in joined
    assert "/w96-trt-runtime:ro" in joined
    assert "/tmp/Assets" in joined
    assert "VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json" in joined
    assert "/w88-overlay" not in joined
    assert "W88_" not in joined


def test_w96_static_site_validates_all_nine_immutable_roots(tmp_path) -> None:
    launcher = _load_module("w96_l20_launcher_site", L20_LAUNCHER_PATH)
    tree_manifest = _tree_manifest_module()
    git_attestation = _load_module(
        "w96_git_tree_attestation_site", GIT_ATTESTATION_PATH
    )

    repo = tmp_path / "source-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "w96@example.test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "W96 test"],
        check=True,
    )
    (repo / "rlinf").mkdir()
    (repo / "rlinf/__init__.py").write_text("\n", encoding="ascii")
    contract_directory = repo / "toolkits/gr00t_trocar/w95"
    contract_directory.mkdir(parents=True)
    shutil.copy2(CONTRACT_PATH, contract_directory / "contract-v1.json")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    immutable = tmp_path / "immutable"
    rlinf_source = immutable / "rlinf-source"
    shutil.copytree(repo, rlinf_source, ignore=shutil.ignore_patterns(".git"))

    roots = {
        "assets-final-readonly.json": immutable / "assets",
        "config.json": immutable / "config",
        "models-GR00T-N1.7-Cosmos.json": immutable / "models",
        "overrides.json": immutable / "overrides",
        "python-w96-overlay.json": immutable / "python-overlay",
        "python-wheelhouse.json": immutable / "wheelhouse",
        "runtime-tensorrt-10.15.1.29.json": immutable / "tensorrt-runtime",
        "sources-isaac-gr00t.json": immutable / "gr00t-source",
        "sources-rlinf.json": rlinf_source,
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    (roots["assets-final-readonly.json"] / "asset.usd").write_text(
        "asset\n", encoding="ascii"
    )
    contract_module = _module()
    resolved_config = roots["config.json"] / "resolved.yaml"
    resolved_config.write_text(
        yaml.safe_dump(
            contract_module.render(
                _base(),
                _contract(),
                "absolute_correctness_b8",
                "all_off",
            ),
            sort_keys=False,
        ),
        encoding="ascii",
    )
    metadata = roots["overrides.json"] / "metadata.json"
    metadata.write_text("{}\n", encoding="ascii")
    extension = roots["overrides.json"] / "extension.py"
    extension.write_text("\n", encoding="ascii")
    assets_override = roots["overrides.json"] / "assets.py"
    assets_override.write_text("\n", encoding="ascii")
    model = roots["models-GR00T-N1.7-Cosmos.json"] / "GR00T-N1.7-3B"
    backbone = roots["models-GR00T-N1.7-Cosmos.json"] / "Cosmos-Reason2-2B"
    model.mkdir()
    backbone.mkdir()
    (model / "config.json").write_text(
        '{"model_name":"nvidia/Cosmos-Reason2-2B"}\n', encoding="ascii"
    )
    (backbone / "config.json").write_text("{}\n", encoding="ascii")

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    manifest_lines = []
    for name, root in sorted(roots.items()):
        path = manifests / name
        path.write_text(
            json.dumps(tree_manifest.create_manifest(root), sort_keys=True) + "\n",
            encoding="ascii",
        )
        manifest_lines.append(f"{launcher._sha256(path)}  {path}")
    manifest_set = manifests / "SHA256SUMS"
    manifest_set.write_text("\n".join(manifest_lines) + "\n", encoding="ascii")

    attestation_path = tmp_path / "rlinf-git-tree.json"
    attestation_path.write_text(
        json.dumps(git_attestation.create_attestation(repo, revision)) + "\n",
        encoding="ascii",
    )
    runtime = json.loads(L20_RUNTIME_SPEC_PATH.read_text(encoding="ascii"))
    image_receipt = tmp_path / "image.json"
    image_receipt.write_text(
        json.dumps(
            {
                "repo_digest": runtime["container_image"]["repo_digest"],
                "image_id": runtime["container_image"]["image_id"],
                "architecture": "amd64",
                "os": "linux",
                "hashes": {
                    "canonical_identity_sha256": runtime["container_image"][
                        "canonical_identity_sha256"
                    ]
                },
            }
        )
        + "\n",
        encoding="ascii",
    )
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nexit 1\n", encoding="ascii")
    docker.chmod(0o755)
    run_base = tmp_path / "runs"
    run_base.mkdir()
    site = {
        "schema": launcher.SITE_SCHEMA,
        "runtime_spec": {
            "path": str(L20_RUNTIME_SPEC_PATH),
            "sha256": launcher._sha256(L20_RUNTIME_SPEC_PATH),
        },
        "image": {
            "reference": runtime["container_image"]["repo_digest"],
            "canonical_receipt": str(image_receipt),
            "canonical_receipt_sha256": launcher._sha256(image_receipt),
        },
        "manifest_set": {
            "path": str(manifest_set),
            "sha256": launcher._sha256(manifest_set),
            "roots": {name: str(root) for name, root in roots.items()},
        },
        "source": {
            "root": str(rlinf_source),
            "revision": revision,
            "attestation": str(attestation_path),
            "attestation_sha256": launcher._sha256(attestation_path),
        },
        "inputs": {
            "rlinf_source": str(rlinf_source),
            "gr00t_source": str(roots["sources-isaac-gr00t.json"]),
            "model": str(model),
            "backbone_model": str(backbone),
            "python_overlay": str(roots["python-w96-overlay.json"]),
            "tensorrt_runtime": str(roots["runtime-tensorrt-10.15.1.29.json"]),
            "config_root": str(roots["config.json"]),
            "overrides_root": str(roots["overrides.json"]),
            "asset_seed": str(roots["assets-final-readonly.json"]),
            "resolved_config": str(resolved_config),
            "trocar_metadata": str(metadata),
            "extension": str(extension),
            "assets_override": str(assets_override),
        },
        "docker": {
            "path": str(docker),
            "sha256": launcher._sha256(docker),
        },
        "run_root_base": str(run_base),
        "preflight": {"quota_mount": "/home/liweim"},
        "workload": {
            "profile": "absolute_correctness_b8",
            "arm": "all_off",
        },
    }
    site_path = tmp_path / "site.json"
    site_path.write_text(json.dumps(site) + "\n", encoding="ascii")

    receipt = launcher.validate_site(site_path)
    assert receipt["status"] == "passed"
    assert receipt["source"]["revision"] == revision
    assert set(receipt["tree_receipts"]) == launcher.REQUIRED_MANIFESTS


def test_q1_probe_is_pre_isaac_and_pre_ray() -> None:
    source = L20_RUNTIME_PROBE_PATH.read_text(encoding="utf-8")
    assert "import isaaclab" not in source
    assert "import ray" not in source
    assert "vulkaninfo" not in source
    assert "vkCreateInstance" in source
    assert "vkEnumeratePhysicalDevices" in source
    assert "vkGetPhysicalDeviceProperties" in source
    assert "vkDestroyInstance" in source
    assert '"pre_isaac_pre_ray": True' in source


class _FakeVulkanApi:
    def __init__(
        self,
        loader_path: Path,
        *,
        create_return: int = 0,
        count_return: int = 0,
        enumerate_return: int = 0,
        property_error_index: int | None = None,
        destroy_error: bool = False,
    ) -> None:
        self.loader_receipt = {
            "requested": "libvulkan.so.1",
            "resolved_path": str(loader_path),
            "soname": "libvulkan.so.1",
            "size": 7,
            "sha256": "a" * 64,
        }
        self.create_return = create_return
        self.count_return = count_return
        self.enumerate_return = enumerate_return
        self.property_error_index = property_error_index
        self.destroy_error = destroy_error
        self.destroy_calls = 0

    def create_instance(self):
        return self.create_return, object() if self.create_return == 0 else None

    def enumerate_count(self, _instance):
        return self.count_return, 8

    def enumerate_devices(self, _instance, count):
        return self.enumerate_return, list(range(1, count + 1)), count

    def get_properties(self, device):
        if device - 1 == self.property_error_index:
            raise RuntimeError("injected property failure")
        return {
            "api_version": 1,
            "driver_version": 2,
            "vendor_id": 0x10DE,
            "device_id": 3,
            "device_type": 2,
            "device_name": "NVIDIA L20",
            "pipeline_cache_uuid": f"{device:032x}",
        }

    def destroy_instance(self, _instance):
        self.destroy_calls += 1
        if self.destroy_error:
            raise RuntimeError("injected destroy failure")


def _vulkan_test_setup(tmp_path, monkeypatch):
    module = _load_module("w96_l20_runtime_probe", L20_RUNTIME_PROBE_PATH)
    loader = tmp_path / "libvulkan.so.1"
    loader.write_bytes(b"fixture")
    module.EXPECTED_DRIVER_LIBRARY_ROOTS = (str(tmp_path),)
    monkeypatch.setenv("VK_DRIVER_FILES", "/etc/vulkan/icd.d/nvidia_icd.json")
    return module, loader


def test_q1_rejects_unregistered_driver_paths(tmp_path) -> None:
    module = _load_module("w96_l20_runtime_probe", L20_RUNTIME_PROBE_PATH)
    library = tmp_path / "libcuda.so.1"
    library.write_bytes(b"fixture")
    assert module._driver_library_path_allowed(str(library)) is False
    module.EXPECTED_DRIVER_LIBRARY_ROOTS = (str(tmp_path),)
    assert module._driver_library_path_allowed(str(library)) is True


def test_q1_vulkan_loader_accepts_exact_l20_inventory_and_destroys(
    tmp_path, monkeypatch
) -> None:
    module, loader = _vulkan_test_setup(tmp_path, monkeypatch)
    api = _FakeVulkanApi(loader)
    receipt = module._vulkan_receipt(api)
    assert receipt["status"] == "passed"
    assert len(receipt["devices"]) == 8
    assert receipt["calls"]["create_instance"]["return_code"] == 0
    assert receipt["calls"]["enumerate_count"] == {"return_code": 0, "count": 8}
    assert receipt["calls"]["enumerate_devices"]["returned_count"] == 8
    assert receipt["calls"]["destroy_instance"]["status"] == "passed"
    assert api.destroy_calls == 1


def test_q1_vulkan_create_failure_does_not_destroy_null_instance(
    tmp_path, monkeypatch
) -> None:
    module, loader = _vulkan_test_setup(tmp_path, monkeypatch)
    api = _FakeVulkanApi(loader, create_return=-9)
    receipt = module._vulkan_receipt(api)
    assert receipt["status"] == "failed"
    assert "vkCreateInstance failed" in receipt["error"]
    assert receipt["calls"]["destroy_instance"] == {
        "attempted": False,
        "status": "not_applicable",
    }
    assert api.destroy_calls == 0


def test_q1_vulkan_enumerate_failure_still_destroys_instance(
    tmp_path, monkeypatch
) -> None:
    module, loader = _vulkan_test_setup(tmp_path, monkeypatch)
    api = _FakeVulkanApi(loader, enumerate_return=-3)
    receipt = module._vulkan_receipt(api)
    assert receipt["status"] == "failed"
    assert "vkEnumeratePhysicalDevices list failed" in receipt["error"]
    assert receipt["calls"]["destroy_instance"]["status"] == "passed"
    assert api.destroy_calls == 1


def test_q1_vulkan_property_failure_still_destroys_instance(
    tmp_path, monkeypatch
) -> None:
    module, loader = _vulkan_test_setup(tmp_path, monkeypatch)
    api = _FakeVulkanApi(loader, property_error_index=3)
    receipt = module._vulkan_receipt(api)
    assert receipt["status"] == "failed"
    assert "vkGetPhysicalDeviceProperties failed for index 3" in receipt["error"]
    assert receipt["calls"]["get_properties"][-1]["status"] == "failed"
    assert receipt["calls"]["destroy_instance"]["status"] == "passed"
    assert api.destroy_calls == 1


def test_q1_vulkan_rejects_non_nvidia_device(tmp_path, monkeypatch) -> None:
    module, loader = _vulkan_test_setup(tmp_path, monkeypatch)
    api = _FakeVulkanApi(loader)
    original = api.get_properties

    def mixed_properties(device):
        properties = original(device)
        if device == 7:
            properties.update(
                vendor_id=0x10005,
                device_type=4,
                device_name="llvmpipe",
            )
        return properties

    api.get_properties = mixed_properties
    receipt = module._vulkan_receipt(api)
    assert receipt["status"] == "failed"
    assert "inventory does not match 8 L20s" in receipt["error"]
    assert api.destroy_calls == 1


def test_q1_vulkan_destroy_failure_fails_receipt(tmp_path, monkeypatch) -> None:
    module, loader = _vulkan_test_setup(tmp_path, monkeypatch)
    api = _FakeVulkanApi(loader, destroy_error=True)
    receipt = module._vulkan_receipt(api)
    assert receipt["status"] == "failed"
    assert "injected destroy failure" in receipt["error"]
    assert receipt["calls"]["destroy_instance"]["status"] == "failed"
    assert api.destroy_calls == 1


def test_q1_python_executable_requires_literal_and_symlink_identity(
    tmp_path,
) -> None:
    module = _load_module("w96_l20_runtime_probe", L20_RUNTIME_PROBE_PATH)
    target = tmp_path / "python3.12"
    target.write_bytes(b"python")
    expected = tmp_path / "python3"
    expected.symlink_to(target.name)

    receipt = module._python_executable_receipt(str(expected), str(expected))
    assert receipt["status"] == "passed"
    assert receipt["literal_matches"] is True
    assert receipt["resolved_matches"] is True
    assert receipt["same_file"] is True
    assert receipt["expected_resolved"] == str(target)

    wrong_literal = module._python_executable_receipt(str(target), str(expected))
    assert wrong_literal["status"] == "failed"
    assert wrong_literal["literal_matches"] is False
    assert wrong_literal["resolved_matches"] is True
    assert wrong_literal["same_file"] is True


def test_q2_smoke_is_eager_true_b8_without_trt_or_nsys() -> None:
    source = L20_Q2_SMOKE_PATH.read_text(encoding="utf-8")
    assert '"backend": "pytorch_eager"' in source
    assert '"batch_size": 8' in source
    assert '"executed_action_chunks": 16' in source
    assert "policy.get_action(observation)" in source
    assert "env.step(action)" in source
    assert "setup_tensorrt_engines" not in source
    assert "import ray" not in source
    assert "nsys" not in source.lower()
    env_start = source.index("def run_env(")
    assert source.index("_isaac_launcher_contract_receipt()", env_start) < source.index(
        "from isaaclab.app import AppLauncher", env_start
    )
    assert "/workspace/isaaclab/apps/isaaclab.python.headless.rendering.kit" in source
    assert '"app_created": False' in source


def test_q2_argv_sets_exact_isaac_launcher_environment(tmp_path) -> None:
    module = _load_module("w96_l20_launcher_q2_args", L20_LAUNCHER_PATH)
    smoke = _load_module("w96_q2_smoke_q2_args", L20_Q2_SMOKE_PATH)
    assert module.Q2_ISAAC_ENVIRONMENT == smoke.ISAAC_LAUNCHER_ENVIRONMENT
    names = (
        "rlinf_source",
        "gr00t_source",
        "python_overlay",
        "tensorrt_runtime",
        "model",
        "backbone_model",
        "resolved_config",
        "extension",
        "assets_override",
        "trocar_metadata",
    )
    inputs = {name: str(tmp_path / name) for name in names}
    site = {
        "docker": {"path": "/fixture/docker"},
        "image": {"reference": "fixture@example"},
        "inputs": inputs,
    }
    common = module._common_docker_args(site, tmp_path, "q1-container")
    argv = module._common_docker_args(
        site,
        tmp_path,
        "q2-container",
        extra_args=module._q2_extra_docker_args(inputs),
    )
    for name, value in {
        "ISAAC_PATH": "/isaac-sim",
        "EXP_PATH": "/isaac-sim/apps",
        "CARB_APP_PATH": "/isaac-sim/kit",
    }.items():
        assignment = f"{name}={value}"
        assert assignment not in common
        assert argv.count(assignment) == 1
        assert argv[argv.index(assignment) - 1] == "-e"
    assert not any("setup_conda_env.sh" in argument for argument in argv)


def test_q2_uses_standard_isaac_wrapper_only_for_bootstrap_and_env() -> None:
    module = _load_module("w96_l20_launcher_q2_interpreters", L20_LAUNCHER_PATH)
    source = L20_LAUNCHER_PATH.read_text(encoding="utf-8")
    q2_source = source[source.index("def run_q2(") :]
    runtime = q2_source.index("l20_runtime_probe.py")
    bootstrap = q2_source.index("bootstrap --output /w96-run/q2-bootstrap.json")
    model = q2_source.index("model --model /models/GR00T-N1.7-3B")
    env = q2_source.index("env --output /w96-run/q2-env.json")
    assert runtime < bootstrap < model < env
    assert module.EXPECTED_ISAAC_PYTHON_WRAPPER == "/isaac-sim/python.sh"
    assert (
        '{EXPECTED_ISAAC_PYTHON_WRAPPER} "\n'
        '        "/workspace/rlinf-src/toolkits/gr00t_trocar/w95/l20_q2_smoke.py "\n'
        '        "bootstrap'
    ) in q2_source
    assert q2_source.count('f"{EXPECTED_ISAAC_PYTHON_WRAPPER} "') == 2
    assert "setup_conda_env.sh" not in q2_source


def test_q2_bootstrap_rejects_missing_wrapper_and_foreign_origins(
    tmp_path, monkeypatch
) -> None:
    module = _load_module("w96_q2_bootstrap_helpers", L20_Q2_SMOKE_PATH)
    missing = module._file_receipt(tmp_path / "missing-python.sh")
    assert missing["status"] == "failed"

    foreign = tmp_path / "foreign" / "module.py"
    foreign.parent.mkdir()
    foreign.write_text("", encoding="ascii")
    fake_module = type("FakeModule", (), {"__file__": str(foreign)})()
    monkeypatch.setattr(module.importlib, "import_module", lambda _name: fake_module)
    module.MODULE_AUTHORITIES = {"isaacsim": (str(tmp_path / "isaac-sim"),)}
    origin = module._module_receipt("isaacsim")
    assert origin["status"] == "failed"
    assert origin["authority_matches"] is False


def test_q2_bootstrap_find_spec_branch_resolves_registered_origin(
    tmp_path, monkeypatch
) -> None:
    module = _load_module("w96_q2_bootstrap_find_spec", L20_Q2_SMOKE_PATH)
    authority = tmp_path / "registered"
    origin = authority / "package" / "__init__.py"
    origin.parent.mkdir(parents=True)
    origin.write_text("", encoding="ascii")
    spec = type("FakeSpec", (), {"origin": str(origin)})()
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _name: spec)
    module.MODULE_AUTHORITIES = {"isaaclab": (str(authority),)}

    receipt = module._module_receipt("isaaclab", import_now=False)
    assert receipt["status"] == "passed"
    assert receipt["resolution"] == "find_spec"
    assert receipt["authority_matches"] is True


def test_q2_pre_app_contract_accepts_exact_readable_paths(
    tmp_path, monkeypatch
) -> None:
    module = _load_module("w96_q2_smoke_contract", L20_Q2_SMOKE_PATH)
    isaac = tmp_path / "isaac-sim"
    apps = isaac / "apps"
    kit = isaac / "kit"
    apps.mkdir(parents=True)
    kit.mkdir()
    experience = tmp_path / "isaaclab.python.headless.rendering.kit"
    experience.write_text("[package]\n", encoding="ascii")
    module.ISAAC_LAUNCHER_ENVIRONMENT = {
        "ISAAC_PATH": str(isaac),
        "EXP_PATH": str(apps),
        "CARB_APP_PATH": str(kit),
    }
    module.ISAAC_RENDERING_EXPERIENCE = experience
    for name, value in module.ISAAC_LAUNCHER_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    receipt = module._isaac_launcher_contract_receipt()
    assert receipt["status"] == "passed"
    assert receipt["setup_conda_env_sourced"] is False
    assert receipt["rendering_experience"]["status"] == "passed"
    assert all(
        item["literal_matches"] and item["readable_and_searchable"]
        for item in receipt["environment"].values()
    )


def test_q2_pre_app_contract_rejects_missing_and_mismatched_values(
    tmp_path, monkeypatch
) -> None:
    module = _load_module("w96_q2_smoke_contract_failure", L20_Q2_SMOKE_PATH)
    isaac = tmp_path / "isaac-sim"
    apps = isaac / "apps"
    kit = isaac / "kit"
    apps.mkdir(parents=True)
    kit.mkdir()
    experience = tmp_path / "isaaclab.python.headless.rendering.kit"
    experience.write_text("[package]\n", encoding="ascii")
    module.ISAAC_LAUNCHER_ENVIRONMENT = {
        "ISAAC_PATH": str(isaac),
        "EXP_PATH": str(apps),
        "CARB_APP_PATH": str(kit),
    }
    module.ISAAC_RENDERING_EXPERIENCE = experience
    for name, value in module.ISAAC_LAUNCHER_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    monkeypatch.delenv("EXP_PATH")
    missing = module._isaac_launcher_contract_receipt()
    assert missing["status"] == "failed"
    assert missing["environment"]["EXP_PATH"]["observed_literal"] is None
    assert missing["environment"]["EXP_PATH"]["status"] == "failed"

    monkeypatch.setenv("EXP_PATH", str(tmp_path / "wrong-apps"))
    mismatch = module._isaac_launcher_contract_receipt()
    assert mismatch["status"] == "failed"
    assert mismatch["environment"]["EXP_PATH"]["literal_matches"] is False
    assert mismatch["environment"]["EXP_PATH"]["status"] == "failed"

    monkeypatch.setenv("EXP_PATH", str(apps))
    experience.unlink()
    missing_experience = module._isaac_launcher_contract_receipt()
    assert missing_experience["status"] == "failed"
    assert missing_experience["rendering_experience"]["status"] == "failed"


def test_q2_requalifies_runtime_and_normalizes_output_ownership() -> None:
    source = L20_LAUNCHER_PATH.read_text(encoding="utf-8")
    q2_start = source.index("def run_q2(")
    q2_source = source[q2_start:]
    assert "l20_runtime_probe.py" in q2_source
    assert 'runtime = _load(run_root / "q2-runtime.json")' in q2_source
    assert "ownership-normalization-command.json" in source
    assert '"/bin/chown"' in source
    assert '"requires_gpu": False' in source


def test_container_request_uses_all_gpus_without_device_ids(tmp_path) -> None:
    module = _load_module("w96_l20_launcher_gpu_request", L20_LAUNCHER_PATH)
    inputs = {
        name: f"/fixture/{name}"
        for name in (
            "rlinf_source",
            "gr00t_source",
            "python_overlay",
            "tensorrt_runtime",
            "model",
            "backbone_model",
            "resolved_config",
            "extension",
            "assets_override",
        )
    }
    site = {
        "docker": {"path": "/fixture/docker"},
        "image": {"reference": "fixture@example"},
        "inputs": inputs,
    }
    argv = module._common_docker_args(site, tmp_path, "fixture-container")
    gpu_index = argv.index("--gpus")
    assert argv[gpu_index : gpu_index + 2] == ["--gpus", "all"]
    assert not any(arg.startswith("device=") for arg in argv)


def _exercise_failed_container_cleanup(tmp_path, monkeypatch, responses):
    module = _load_module("w96_l20_launcher_cleanup", L20_LAUNCHER_PATH)
    run_root = tmp_path / "run"
    (run_root / "receipts").mkdir(parents=True)
    site = {
        "docker": {"path": "/fixture/docker"},
        "image": {"reference": "fixture@example"},
    }
    monkeypatch.setattr(
        module,
        "_common_docker_args",
        lambda *_args, **_kwargs: ["/fixture/docker", "run", "fixture@example"],
    )
    calls = []

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        if not responses:
            raise AssertionError(f"unexpected subprocess call: {argv}")
        return responses.pop(0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    try:
        module._run_container(site, run_root, "q1", "true")
    except module.LaunchError as error:
        return (
            str(error),
            json.loads(
                (run_root / "receipts/cleanup.json").read_text(encoding="ascii")
            ),
            calls,
        )
    raise AssertionError("cleanup failure did not fail closed")


def test_container_remove_failure_skips_ownership_normalization(
    tmp_path, monkeypatch
) -> None:
    container = "w96-q1-run"
    completed = subprocess.CompletedProcess
    error, cleanup, calls = _exercise_failed_container_cleanup(
        tmp_path,
        monkeypatch,
        [
            completed([], 1, "", f"Error: No such object: {container}"),
            completed([], 0, "workload\n", ""),
            completed([], 0, '{"State":{"Running":true}}', ""),
            completed([], 1, "", "permission denied"),
            completed([], 0, '{"State":{"Running":true}}', ""),
        ],
    )
    assert "container removal failed" in error
    assert cleanup["container_remove_exit_code"] == 1
    assert cleanup["container_absent_confirmed"] is False
    assert cleanup["ownership_normalization_attempted"] is False
    assert cleanup["ownership_exit_code"] is None
    assert not (tmp_path / "run/q1-receipt.json").exists()
    assert len(calls) == 5


def test_container_still_present_after_remove_skips_ownership_normalization(
    tmp_path, monkeypatch
) -> None:
    container = "w96-q1-run"
    completed = subprocess.CompletedProcess
    error, cleanup, calls = _exercise_failed_container_cleanup(
        tmp_path,
        monkeypatch,
        [
            completed([], 1, "", f"Error: No such object: {container}"),
            completed([], 0, "workload\n", ""),
            completed([], 0, '{"State":{"Running":true}}', ""),
            completed([], 0, container, ""),
            completed([], 0, '{"State":{"Running":true}}', ""),
        ],
    )
    assert "absence was not confirmed" in error
    assert cleanup["container_remove_exit_code"] == 0
    assert cleanup["post_remove_inspect_exit_code"] == 0
    assert cleanup["container_absent_confirmed"] is False
    assert cleanup["ownership_normalization_attempted"] is False
    assert not (tmp_path / "run/q1-receipt.json").exists()
    assert len(calls) == 5


def test_run_tree_ownership_audit_rejects_unreadable_output(tmp_path) -> None:
    module = _load_module("w96_l20_launcher_ownership", L20_LAUNCHER_PATH)
    output = tmp_path / "output.json"
    output.write_text("{}\n", encoding="ascii")
    assert module._audit_run_ownership(tmp_path)["status"] == "passed"
    output.chmod(0)
    try:
        receipt = module._audit_run_ownership(tmp_path)
        assert receipt["status"] == "failed"
        assert receipt["unreadable"] == ["output.json"]
    finally:
        output.chmod(0o600)
