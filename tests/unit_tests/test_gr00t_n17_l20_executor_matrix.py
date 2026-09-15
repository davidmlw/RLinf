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

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "toolkits/gr00t_trocar/w95"


def _load(name: str):
    sys.path.insert(0, str(TOOLS))
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"w98_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agent_freezes_ordered_build_and_measurement_stages(tmp_path: Path) -> None:
    agent = _load("l20_executor_matrix_agent")
    args = argparse.Namespace(
        source=Path("/workspace/w98-src"),
        gr00t_source=Path("/workspace/gr00t-n17"),
        model=Path("/models/GR00T-N1.7-3B"),
        backbone=Path("/w96-model-inputs/Cosmos-Reason2-2B"),
        metadata=Path("/w98-inputs/trocar/metadata.json"),
        output=tmp_path,
        seed=47,
        workspace_mib=8192,
        rlinf_revision="abc123",
    )
    stages = agent.build_stage_commands(args)
    assert [name for name, _ in stages] == [
        "model-view",
        "fixture",
        "backbone-export",
        "backbone-build",
        "dit-export",
        "dit-parameter-map",
        "dit-build",
        "refit-lifecycle",
    ]
    flattened = "\n".join(" ".join(command) for _, command in stages)
    assert "--workspace 8192" in flattened
    assert "--workspace-mib 8192" in flattened
    assert "--expected-source-revision 51d4c89f" in flattened


def test_launcher_uses_one_l20_and_w96_runtime() -> None:
    launcher = _load("l20_executor_matrix_launcher")
    command = launcher._agent_command("abc123")
    assert "CUDA_VISIBLE_DEVICES" not in command
    assert "/w96-overlay:/w96-trt-runtime" in command
    assert "/workspace/gr00t-n17/scripts/deployment" in command
    assert "--warmup 10 --measured 30" in command
    assert "--rlinf-revision abc123" in command
    args = launcher._extra_docker_args(
        {"inputs": {"trocar_metadata": "/authority/trocar.json"}},
        Path("/source"),
    )
    assert args[:2] == ("-e", "CUDA_VISIBLE_DEVICES=0")
    assert "/source:/workspace/w98-src:ro" in args


def test_w98_qualification_is_stricter_than_legacy_vit_gate() -> None:
    agent = _load("l20_executor_matrix_agent")
    assert agent.MIN_VIT_COSINE == 0.999
    receipt = {
        "comparisons": {
            "vit_image_embeds": {"cosine": 0.998},
            "pre_final_backbone": {"cosine": 0.9999},
            "public_action": {"cosine": 0.9999, "mean_abs": 0.001, "max_abs": 0.01},
        },
        "executor_matrix": {},
    }
    result = agent._qualification(receipt)
    assert result["status"] == "failed"
    assert result["checks"]["vit_cosine"] is False


def test_launcher_rejects_non_w98_result_path(tmp_path: Path) -> None:
    launcher = _load("l20_executor_matrix_launcher")
    try:
        launcher._validate_new_run_root(tmp_path / "attempt")
    except launcher.LaunchError as error:
        assert "results/W98" in str(error)
    else:
        raise AssertionError("non-W98 result path was accepted")


def test_source_identity_accepts_external_attestation(tmp_path: Path) -> None:
    identity = _load("../../eos/gr00t_trocar/tensorrt/source_identity")
    result = identity.resolve_source_revision(tmp_path, "51d4c89")
    assert result["revision"] == "51d4c89"
    assert result["authority"] == "external_tree_attestation"


def test_refittable_compute_capability_comes_from_receipt(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "toolkits/eos/gr00t_trocar/tensorrt"))
    matrix = _load("../../eos/gr00t_trocar/tensorrt/standalone_true_b8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": ("rlinf.gr00t-n1d7-trocar-true-b8-refittable-dit-engine.v1"),
                "status": "passed",
                "runtime": {"compute_capability": [8, 9]},
            }
        ),
        encoding="ascii",
    )
    assert matrix._refittable_compute_capability(receipt) == [8, 9]


def test_nsys_command_uses_narrow_profiler_api_window() -> None:
    launcher = _load("l20_executor_matrix_nsys_launcher")
    command = launcher._profile_command(
        {
            "files": {
                "dit_receipt": {"sha256": "a" * 64},
                "parameter_map": {"sha256": "b" * 64},
            },
            "source_digest": "c" * 64,
        },
        "d" * 40,
    )
    assert "--capture-range=cudaProfilerApi" in command
    assert "--profile-once --allow-systems-only" in command
    assert "RLINF_W98_NVTX=1" in command


def test_summary_selects_two_part_three_backend_headline() -> None:
    summary = _load("../../eos/gr00t_trocar/tensorrt/summarize_l20_executor_matrix")
    rows = []
    for partition, arms, stage in (
        ("frozen_backbone", ("eager_backbone", "pt2_backbone"), "backbone_ms"),
        ("pure_dit", ("eager", "pt2"), "dit_ms"),
    ):
        for arm in arms:
            rows.append({"partition": partition, "arm": arm, "stage": stage})
    assert summary._headline(rows) == rows


def test_real_revision_extractor_freezes_dit_keyspace() -> None:
    extractor = _load("../../eos/gr00t_trocar/tensorrt/extract_action_head_revision")
    assert extractor.PREFIX == "action_head.model."
    assert extractor.EXPECTED_TENSORS == 456
