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
