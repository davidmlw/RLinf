#!/usr/bin/env python3
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

"""Create a fail-closed receipt for an RLinf GR00T TensorRT hybrid trial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

JSON_MARKERS = {
    "runtime": "RLINF_HYBRID_RUNTIME ",
    "identity": "RLINF_PRE_UPDATE_IDENTITY ",
}
RANK_MARKERS = {
    "feature_stream": re.compile(r"PINNED_FEATURE_STREAM_DONE .*?rank=(\d+)\b"),
    "feature_route": re.compile(r"PINNED_FEATURE_ROUTE_VALID .*?rank=(\d+)\b"),
    "feature_training": re.compile(r"PINNED_FEATURE_TRAINING_DONE .*?rank=(\d+)\b"),
}
FALLBACK_RE = re.compile(
    r"actor/reuse_feature_fallbacks=(?P<value>"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
TRAINING_METRIC_RE = re.compile(
    r"actor/(?P<name>grad_norm|policy_loss|total_loss)="
    r"(?P<value>[-+A-Za-z0-9.eE]+)"
)
TRAINING_METRICS = ("grad_norm", "policy_loss", "total_loss")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _marker_json(lines: list[str], marker: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    values = []
    for line in lines:
        position = line.find(marker)
        if position < 0:
            continue
        payload = line[position + len(marker) :].lstrip()
        try:
            value, _ = decoder.raw_decode(payload)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON after {marker.strip()}: {line!r}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{marker.strip()} payload must be an object")
        values.append(value)
    return values


def _rank_counts(lines: list[str], pattern: re.Pattern[str]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line in lines:
        match = pattern.search(line)
        if match is not None:
            counts[int(match.group(1))] += 1
    return counts


def _training_metric_gates(
    lines: list[str], *, min_outer_steps: int
) -> tuple[dict[str, Any], list[str]]:
    values: dict[str, list[float]] = {name: [] for name in TRAINING_METRICS}
    for line in lines:
        for match in TRAINING_METRIC_RE.finditer(line):
            values[match.group("name")].append(float(match.group("value")))

    failures = []
    nonfinite_steps = {}
    for name, metric_values in values.items():
        if len(metric_values) < min_outer_steps:
            failures.append(
                f"actor/{name}: expected at least {min_outer_steps} values, "
                f"found {len(metric_values)}"
            )
        indices = [
            step for step, value in enumerate(metric_values) if not math.isfinite(value)
        ]
        nonfinite_steps[name] = indices
        if indices:
            failures.append(f"actor/{name}: nonfinite at steps {indices}")
    return {
        "counts": {name: len(metric_values) for name, metric_values in values.items()},
        "nonfinite_steps": nonfinite_steps,
    }, failures


def _runtime_gates(
    records: list[dict[str, Any]],
    *,
    expected_ranks: set[int],
    expected_receipt: str,
    min_outer_steps: int,
) -> tuple[dict[str, Any], list[str]]:
    failures = []
    by_rank_stage: dict[int, Counter[str]] = defaultdict(Counter)
    for record in records:
        rank = int(record.get("rank", -1))
        stage = str(record.get("stage", ""))
        by_rank_stage[rank][stage] += 1
        if rank not in expected_ranks:
            failures.append(f"unexpected hybrid runtime rank {rank}")
            continue
        compiled_dit = record.get("compiled_dit")
        if not isinstance(compiled_dit, dict):
            failures.append(f"rank {rank} omitted compiled DiT telemetry")
        elif compiled_dit.get("enabled") is not False:
            failures.append(f"rank {rank} enabled an unsupported compiled DiT")
        if record.get("tensorrt_dit") is not None:
            failures.append(f"rank {rank} enabled an unsupported TensorRT DiT")
        backbone = record.get("tensorrt_backbone")
        if not isinstance(backbone, dict):
            failures.append(f"rank {rank} omitted TensorRT backbone telemetry")
            continue
        if backbone.get("receipt_sha256") != expected_receipt:
            failures.append(f"rank {rank} used a different engine receipt")
        for engine_name in ("vit", "llm"):
            engine = backbone.get(engine_name)
            if not isinstance(engine, dict):
                failures.append(f"rank {rank} omitted {engine_name} telemetry")
                continue
            if engine.get("load_count") != 1 or engine.get("context_count") != 1:
                failures.append(
                    f"rank {rank} {engine_name} load/context count is not one"
                )
            if engine.get("resident_host_sync_count") != 0:
                failures.append(f"rank {rank} {engine_name} synchronized the host")

    for rank in sorted(expected_ranks):
        stages = by_rank_stage.get(rank, Counter())
        if stages["initialized"] < 1:
            failures.append(f"rank {rank} has no initialized telemetry")
        if stages["revision_adopted"] < min_outer_steps:
            failures.append(
                f"rank {rank} has fewer than {min_outer_steps} revision adoptions"
            )
        if stages["rollout_complete"] < min_outer_steps:
            failures.append(
                f"rank {rank} has fewer than {min_outer_steps} completed rollouts"
            )
        if stages["closing"] != 1 or stages["closed"] != 1:
            failures.append(f"rank {rank} has incomplete shutdown telemetry")

    return {
        "record_count": len(records),
        "rank_stage_counts": {
            str(rank): dict(sorted(stages.items()))
            for rank, stages in sorted(by_rank_stage.items())
        },
    }, failures


def qualify(
    standalone_path: Path,
    engine_receipt_path: Path,
    training_log_path: Path,
    *,
    world_size: int,
    min_outer_steps: int,
) -> dict[str, Any]:
    if world_size <= 0 or min_outer_steps <= 0:
        raise ValueError("world size and minimum outer steps must be positive")
    standalone = _load_json(standalone_path)
    engine_receipt = _load_json(engine_receipt_path)
    lines = training_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    expected_ranks = set(range(world_size))
    engine_receipt_sha256 = _sha256(engine_receipt_path)
    failures = []

    if standalone.get("status") != "passed":
        failures.append("standalone qualification did not pass")
    standalone_engine = (
        standalone.get("provenance", {})
        .get("artifacts", {})
        .get("engine_receipt", {})
    )
    if standalone_engine.get("sha256") != engine_receipt_sha256:
        failures.append("standalone qualification used a different engine receipt")
    if engine_receipt.get("status") != "passed":
        failures.append("engine receipt did not pass")
    if engine_receipt.get("silent_fallback") is not False:
        failures.append("engine receipt permits silent fallback")

    runtime_records = _marker_json(lines, JSON_MARKERS["runtime"])
    runtime, runtime_failures = _runtime_gates(
        runtime_records,
        expected_ranks=expected_ranks,
        expected_receipt=engine_receipt_sha256,
        min_outer_steps=min_outer_steps,
    )
    failures.extend(runtime_failures)

    identities = _marker_json(lines, JSON_MARKERS["identity"])
    if not identities:
        failures.append("training log has no global pre-update identity receipt")
    for identity in identities:
        if identity.get("schema") != "rlinf.pre-update-policy-identity.v1":
            failures.append("pre-update identity receipt has the wrong schema")
        if identity.get("passed") is not True:
            failures.append("pre-update identity gate did not pass")
        if identity.get("actor_world_size") != world_size:
            failures.append("pre-update identity actor world size mismatch")

    stream_counts = {
        name: _rank_counts(lines, pattern)
        for name, pattern in RANK_MARKERS.items()
    }
    for name, counts in stream_counts.items():
        for rank in sorted(expected_ranks):
            if counts[rank] < min_outer_steps:
                failures.append(
                    f"rank {rank} has fewer than {min_outer_steps} {name} records"
                )

    fallback_values = [
        float(match.group("value"))
        for line in lines
        for match in FALLBACK_RE.finditer(line)
    ]
    if len(fallback_values) < min_outer_steps:
        failures.append("training log has too few feature-fallback metrics")
    if any(value != 0.0 for value in fallback_values):
        failures.append("feature reuse reported a nonzero fallback count")

    training_metrics, training_metric_failures = _training_metric_gates(
        lines, min_outer_steps=min_outer_steps
    )
    failures.extend(training_metric_failures)

    deduplicated_failures = list(dict.fromkeys(failures))
    return {
        "schema": "rlinf.gr00t-n1d7-hybrid-trial.v1",
        "status": "passed" if not deduplicated_failures else "failed",
        "scope": "TensorRT ViT+LLM, eager PyTorch action head, exact feature reuse",
        "inputs": {
            "standalone": {
                "path": str(standalone_path),
                "sha256": _sha256(standalone_path),
            },
            "engine_receipt": {
                "path": str(engine_receipt_path),
                "sha256": engine_receipt_sha256,
            },
            "training_log": {
                "path": str(training_log_path),
                "sha256": _sha256(training_log_path),
            },
        },
        "contract": {
            "world_size": world_size,
            "minimum_outer_steps": min_outer_steps,
            "compiled_dit": False,
            "feature_transport": "borrowed_ipc_pinned",
            "silent_fallback": False,
        },
        "gates": {
            "standalone_passed": standalone.get("status") == "passed",
            "standalone_engine_receipt_matches": (
                standalone_engine.get("sha256") == engine_receipt_sha256
            ),
            "engine_receipt_passed": engine_receipt.get("status") == "passed",
            "runtime": runtime,
            "pre_update_identity_receipts": identities,
            "feature_rank_counts": {
                name: {
                    str(rank): counts[rank] for rank in sorted(expected_ranks)
                }
                for name, counts in stream_counts.items()
            },
            "feature_fallback_values": fallback_values,
            "training_metrics": training_metrics,
        },
        "failures": deduplicated_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone", type=Path, required=True)
    parser.add_argument("--engine-receipt", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--min-outer-steps", type=int, default=2)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if output.exists():
            raise ValueError(f"refusing to replace existing output: {output}")
        receipt = qualify(
            args.standalone.resolve(strict=True),
            args.engine_receipt.resolve(strict=True),
            args.training_log.resolve(strict=True),
            world_size=args.world_size,
            min_outer_steps=args.min_outer_steps,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception as error:
        print(f"hybrid trial qualification failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
