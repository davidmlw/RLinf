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
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "toolkits/gr00t_trocar/w95/contract-v1.json"
BASE_CONFIG = ROOT / "toolkits/gr00t_trocar/config-n1d7-vulkan-control.yaml"
MODULE_PATH = ROOT / "toolkits/gr00t_trocar/w95/contract.py"
TREE_MANIFEST_PATH = ROOT / "toolkits/gr00t_trocar/w95/tree_manifest.py"
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
    assert "cannot qualify" in profiles["baseline_relative_throughput_b32"][
        "allowed_claim"
    ]


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
