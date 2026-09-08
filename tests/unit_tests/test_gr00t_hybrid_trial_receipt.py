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

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = (
    ROOT
    / "toolkits/eos/gr00t_trocar/tensorrt/qualify_hybrid_trial.py"
)
SPEC = importlib.util.spec_from_file_location("qualify_hybrid_trial", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path, *, fallback: float = 0.0) -> tuple[Path, Path, Path]:
    standalone = tmp_path / "standalone.json"
    engine = tmp_path / "rlinf-engine-receipt.json"
    training = tmp_path / "training.out"
    engine.write_text(
        json.dumps({"status": "passed", "silent_fallback": False}),
        encoding="utf-8",
    )
    receipt_sha256 = _sha256(engine)
    standalone.write_text(
        json.dumps(
            {
                "status": "passed",
                "provenance": {
                    "artifacts": {
                        "engine_receipt": {"sha256": receipt_sha256}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    lines = []
    for rank in range(2):
        engine_stats = {
            "load_count": 1,
            "context_count": 1,
            "resident_host_sync_count": 0,
        }
        backbone = {
            "receipt_sha256": receipt_sha256,
            "vit": engine_stats,
            "llm": engine_stats,
        }
        for stage in (
            "initialized",
            "revision_adopted",
            "revision_adopted",
            "rollout_complete",
            "rollout_complete",
            "closing",
            "closed",
        ):
            payload = {
                "stage": stage,
                "rank": rank,
                "compiled_dit": {
                    "enabled": False,
                    "mode": None,
                    "unique_graphs": 0,
                    "parameter_count": 0,
                },
                "tensorrt_backbone": backbone,
            }
            lines.append(f"prefix RLINF_HYBRID_RUNTIME {json.dumps(payload)}")
        for step in range(2):
            lines.extend(
                [
                    f"PINNED_FEATURE_STREAM_DONE rank={rank} lease=l{step}",
                    f"PINNED_FEATURE_ROUTE_VALID rank={rank} step={step}",
                    f"PINNED_FEATURE_TRAINING_DONE rank={rank} step={step}",
                ]
            )
    identity = {
        "schema": "rlinf.pre-update-policy-identity.v1",
        "passed": True,
        "actor_world_size": 2,
    }
    lines.append(f"RLINF_PRE_UPDATE_IDENTITY {json.dumps(identity)}")
    lines.extend(
        [f"actor/reuse_feature_fallbacks={fallback}" for _ in range(2)]
    )
    lines.extend(
        [
            "actor/grad_norm=1.25 actor/policy_loss=-0.2 actor/total_loss=0.3",
            "actor/grad_norm=0.75 actor/policy_loss=0.1 actor/total_loss=0.2",
        ]
    )
    training.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return standalone, engine, training


def test_qualified_trial_passes_all_runtime_and_ppo_gates(tmp_path: Path) -> None:
    standalone, engine, training = _inputs(tmp_path)

    receipt = MODULE.qualify(
        standalone,
        engine,
        training,
        world_size=2,
        min_outer_steps=2,
    )

    assert receipt["status"] == "passed"
    assert receipt["failures"] == []
    assert receipt["gates"]["feature_fallback_values"] == [0.0, 0.0]
    assert receipt["gates"]["training_metrics"] == {
        "counts": {"grad_norm": 2, "policy_loss": 2, "total_loss": 2},
        "nonfinite_steps": {
            "grad_norm": [],
            "policy_loss": [],
            "total_loss": [],
        },
    }
    assert receipt["gates"]["runtime"]["rank_stage_counts"]["0"] == {
        "closed": 1,
        "closing": 1,
        "initialized": 1,
        "revision_adopted": 2,
        "rollout_complete": 2,
    }
    assert receipt["gates"]["runtime"]["completed_outer_steps"] == 2


def test_trial_fails_closed_on_nonzero_feature_fallback(tmp_path: Path) -> None:
    standalone, engine, training = _inputs(tmp_path, fallback=1.0)

    receipt = MODULE.qualify(
        standalone,
        engine,
        training,
        world_size=2,
        min_outer_steps=2,
    )

    assert receipt["status"] == "failed"
    assert "feature reuse reported a nonzero fallback count" in receipt["failures"]


def test_trial_fails_closed_on_missing_rank_lifecycle(tmp_path: Path) -> None:
    standalone, engine, training = _inputs(tmp_path)
    text = training.read_text(encoding="utf-8")
    training.write_text(
        "\n".join(line for line in text.splitlines() if '"rank": 1' not in line),
        encoding="utf-8",
    )

    receipt = MODULE.qualify(
        standalone,
        engine,
        training,
        world_size=2,
        min_outer_steps=2,
    )

    assert receipt["status"] == "failed"
    assert "rank 1 has no initialized telemetry" in receipt["failures"]


def test_trial_fails_closed_on_mixed_standalone_engine_bundle(tmp_path: Path) -> None:
    standalone, engine, training = _inputs(tmp_path)
    value = json.loads(standalone.read_text(encoding="utf-8"))
    value["provenance"]["artifacts"]["engine_receipt"]["sha256"] = "0" * 64
    standalone.write_text(json.dumps(value), encoding="utf-8")

    receipt = MODULE.qualify(
        standalone,
        engine,
        training,
        world_size=2,
        min_outer_steps=2,
    )

    assert receipt["status"] == "failed"
    assert (
        "standalone qualification used a different engine receipt"
        in receipt["failures"]
    )


def test_trial_fails_closed_on_compiled_dit(tmp_path: Path) -> None:
    standalone, engine, training = _inputs(tmp_path)
    text = training.read_text(encoding="utf-8")
    training.write_text(
        text.replace('"enabled": false', '"enabled": true'),
        encoding="utf-8",
    )

    receipt = MODULE.qualify(
        standalone,
        engine,
        training,
        world_size=2,
        min_outer_steps=2,
    )

    assert receipt["status"] == "failed"
    assert "rank 0 enabled an unsupported compiled DiT" in receipt["failures"]


def test_trial_fails_closed_on_tensorrt_dit(tmp_path: Path) -> None:
    standalone, engine, training = _inputs(tmp_path)
    text = training.read_text(encoding="utf-8")
    training.write_text(
        text.replace(
            '"tensorrt_backbone":',
            '"tensorrt_dit": {"enabled": true}, "tensorrt_backbone":',
        ),
        encoding="utf-8",
    )

    receipt = MODULE.qualify(
        standalone,
        engine,
        training,
        world_size=2,
        min_outer_steps=2,
    )

    assert receipt["status"] == "failed"
    assert "rank 0 enabled an unsupported TensorRT DiT" in receipt["failures"]


def test_trial_fails_closed_on_nonfinite_or_missing_training_metrics(
    tmp_path: Path,
) -> None:
    standalone, engine, training = _inputs(tmp_path)
    text = training.read_text(encoding="utf-8")
    training.write_text(
        text.replace("actor/grad_norm=1.25", "actor/grad_norm=nan").replace(
            "actor/total_loss=0.2", "missing_total_loss=0.2"
        ),
        encoding="utf-8",
    )

    receipt = MODULE.qualify(
        standalone,
        engine,
        training,
        world_size=2,
        min_outer_steps=2,
    )

    assert receipt["status"] == "failed"
    assert "actor/grad_norm: nonfinite at steps [0]" in receipt["failures"]
    assert (
        "actor/total_loss: expected 2 values, found 1"
        in receipt["failures"]
    )


def test_trial_uses_observed_steps_not_only_requested_minimum(tmp_path: Path) -> None:
    standalone, engine, training = _inputs(tmp_path)
    text = training.read_text(encoding="utf-8")
    training.write_text(
        text.replace("actor/total_loss=0.2", "missing_total_loss=0.2"),
        encoding="utf-8",
    )

    receipt = MODULE.qualify(
        standalone,
        engine,
        training,
        world_size=2,
        min_outer_steps=1,
    )

    assert receipt["status"] == "failed"
    assert "actor/total_loss: expected 2 values, found 1" in receipt["failures"]
