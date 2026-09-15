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

"""Create durable CSV, JSON, and Markdown views of a W98 L20 matrix receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = receipt["executor_matrix"]
    rows = []
    for partition, value in matrix["matrices"].items():
        reference = value["reference"]
        for arm, stages in value["arms"].items():
            for stage, timing in stages.items():
                relative = value["relative_to_reference"][arm][stage]
                rows.append(
                    {
                        "partition": partition,
                        "arm": arm,
                        "reference": reference,
                        "stage": stage,
                        "mean_ms": timing["mean_ms"],
                        "p50_ms": timing["p50_ms"],
                        "p95_ms": timing["p95_ms"],
                        "sample_std_ms": timing["sample_std_ms"],
                        "cv": timing["cv"],
                        "speedup_vs_reference": relative["speedup_vs_reference"],
                        "latency_reduction_pct": 100
                        * relative["latency_reduction_fraction_vs_reference"],
                    }
                )
    for arm, timing in matrix["pure_dit"]["arms"].items():
        eager = matrix["pure_dit"]["arms"]["eager"]["mean_ms"]
        rows.append(
            {
                "partition": "pure_dit",
                "arm": arm,
                "reference": "eager",
                "stage": "dit_ms",
                "mean_ms": timing["mean_ms"],
                "p50_ms": timing["p50_ms"],
                "p95_ms": timing["p95_ms"],
                "sample_std_ms": timing["sample_std_ms"],
                "cv": timing["cv"],
                "speedup_vs_reference": eager / timing["mean_ms"],
                "latency_reduction_pct": 100 * (eager - timing["mean_ms"]) / eager,
            }
        )
    return rows


def _headline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {
        ("frozen_backbone", "eager_backbone", "backbone_ms"),
        ("frozen_backbone", "pt2_backbone", "backbone_ms"),
        ("frozen_backbone", "tensorrt_backbone", "backbone_ms"),
        ("refittable_action_head", "eager_action_head", "action_head_ms"),
        ("refittable_action_head", "pt2_action_head", "action_head_ms"),
        (
            "refittable_action_head",
            "refittable_tensorrt_action_head",
            "action_head_ms",
        ),
        ("pure_dit", "eager", "dit_ms"),
        ("pure_dit", "pt2", "dit_ms"),
        ("pure_dit", "refittable_tensorrt", "dit_ms"),
        (
            "whole_model_diagonal",
            "eager_backbone_eager_head",
            "total_ms",
        ),
        ("whole_model_diagonal", "pt2_backbone_pt2_head", "total_ms"),
        (
            "whole_model_diagonal",
            "tensorrt_backbone_refittable_head",
            "total_ms",
        ),
    }
    return [
        row for row in rows if (row["partition"], row["arm"], row["stage"]) in selected
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.mkdir(parents=True)
    receipt = json.loads(source.read_text(encoding="utf-8"))
    rows = _rows(receipt)
    headline = _headline(rows)
    with (output / "matrix.csv").open("x", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = receipt["comparisons"]
    matrix = receipt["executor_matrix"]
    summary = {
        "schema": "rlinf.w98.l20-executor-matrix-summary/v1",
        "status": "systems_only",
        "input": {"path": str(source), "sha256": _sha256(source)},
        "headline": headline,
        "numerics": {
            "vit_image_embeds": comparisons["vit_image_embeds"],
            "pre_final_backbone": comparisons["pre_final_backbone"],
            "public_action": comparisons["public_action"],
        },
        "matrix_gates": matrix["gates"],
        "ppo_authority": receipt["ppo_statistics"],
        "disposition": {
            "complete_systems_matrix": True,
            "deployable_new_backend_cells": [],
            "reason": (
                "TensorRT ViT component cosine fails and PT2/TRT PPO authority "
                "remains pending W99"
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    labels = {
        "eager_backbone": "Eager Backbone",
        "pt2_backbone": "PT2 Backbone",
        "tensorrt_backbone": "TensorRT Backbone",
        "eager_action_head": "Eager Action Head",
        "pt2_action_head": "PT2 Action Head",
        "refittable_tensorrt_action_head": "Refittable TRT Action Head",
        "eager": "Eager DiT",
        "pt2": "PT2 DiT",
        "refittable_tensorrt": "Refittable TRT DiT",
        "eager_backbone_eager_head": "Eager/Eager whole core",
        "pt2_backbone_pt2_head": "PT2/PT2 whole core",
        "tensorrt_backbone_refittable_head": "TRT/TRT whole core",
    }
    lines = [
        "# W98 L20 True-B8 Executor Matrix",
        "",
        "Clean CUDA-event data use one L20 (SM89), B=8, three cameras, L=208, "
        "explicit fixed noise, 10 warmups and 30 balanced measurements per arm.",
        "",
        "| Partition | Arm | Mean ms | P50 ms | P95 ms | Delta vs eager | Speedup |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in headline:
        lines.append(
            f"| {row['partition']} | {labels[row['arm']]} | "
            f"{row['mean_ms']:.3f} | {row['p50_ms']:.3f} | "
            f"{row['p95_ms']:.3f} | {row['latency_reduction_pct']:+.2f}% | "
            f"{row['speedup_vs_reference']:.3f}x |"
        )
    vit = comparisons["vit_image_embeds"]
    action = comparisons["public_action"]
    lines.extend(
        [
            "",
            "## Disposition",
            "",
            f"- TensorRT ViT cosine is {vit['cosine']:.9f}, below the frozen "
            "0.999 gate; the TensorRT cells are systems-only.",
            f"- Final public action still matches closely: cosine "
            f"{action['cosine']:.9f}, mean/max absolute error "
            f"{action['mean_abs']:.6f}/{action['max_abs']:.6f}.",
            "- PT2 is the only backend that improves the complete L20 model-core "
            "in this run. PPO ratio/KL authority is intentionally deferred to W99.",
            "- W97 feature reuse is the composed system baseline. This standalone "
            "matrix measures the rollout model executors and does not add feature "
            "reuse speedup a second time.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
