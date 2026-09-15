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

"""Summarize the nine outer W98 NVTX ranges from Nsys CSV reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PREFIXES = (
    "W98/frozen_backbone/",
    "W98/refittable_action_head/",
    "W98/whole_model_diagonal/",
)


def _range_name(value: str) -> str:
    return value.removeprefix(":")


def _selected(value: str) -> bool:
    return any(_range_name(value).startswith(prefix) for prefix in PREFIXES)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _summarize(
    projected_rows: list[dict[str, str]], kernel_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    projected = {
        _range_name(row["Range"]): row
        for row in projected_rows
        if _selected(row["Range"])
    }
    kernels: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in kernel_rows:
        if _selected(row["NVTX Range"]):
            kernels[_range_name(row["NVTX Range"])].append(row)
    if set(projected) != set(kernels) or len(projected) != 9:
        raise RuntimeError(
            "expected the same nine W98 outer ranges in projected and kernel reports"
        )
    summaries = []
    top_rows = []
    for name, row in sorted(projected.items()):
        partition, arm = name.removeprefix("W98/").split("/", 1)
        entries = kernels[name]
        ranked = sorted(
            entries, key=lambda item: int(item["Total Time (ns)"]), reverse=True
        )
        summaries.append(
            {
                "partition": partition,
                "arm": arm,
                "projected_gpu_ms": int(row["Total Proj Time (ns)"]) / 1e6,
                "cpu_range_ms": int(row["Total Range Time (ns)"]) / 1e6,
                "gpu_ops": int(row["Total GPU Ops"]),
                "kernel_launches": sum(int(item["Kern Inst"]) for item in entries),
                "unique_kernel_rows": len(entries),
                "summed_kernel_ms": sum(
                    int(item["Total Time (ns)"]) for item in entries
                )
                / 1e6,
            }
        )
        for rank, item in enumerate(ranked[:10], 1):
            top_rows.append(
                {
                    "partition": partition,
                    "arm": arm,
                    "rank": rank,
                    "instances": int(item["Kern Inst"]),
                    "total_ms": int(item["Total Time (ns)"]) / 1e6,
                    "kernel": item["Kernel Name"],
                }
            )
    return summaries, top_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.mkdir(parents=True)
    summaries, top_rows = _summarize(
        _read(args.projected.resolve(strict=True)),
        _read(args.kernels.resolve(strict=True)),
    )
    _write_csv(output / "range-summary.csv", summaries)
    _write_csv(output / "top-kernels.csv", top_rows)
    by_key = {(row["partition"], row["arm"]): row for row in summaries}
    eager = by_key[("whole_model_diagonal", "eager_backbone_eager_head")]
    pt2 = by_key[("whole_model_diagonal", "pt2_backbone_pt2_head")]
    trt = by_key[("whole_model_diagonal", "tensorrt_backbone_refittable_head")]
    result = {
        "schema": "rlinf.w98.l20-executor-matrix-nsys-summary/v1",
        "status": "passed",
        "scope": "single_profiled_call_per_arm_explanatory_not_headline_timing",
        "ranges": summaries,
        "whole_model_launch_reduction": {
            "pt2_vs_eager_pct": 100
            * (eager["kernel_launches"] - pt2["kernel_launches"])
            / eager["kernel_launches"],
            "tensorrt_vs_eager_pct": 100
            * (eager["kernel_launches"] - trt["kernel_launches"])
            / eager["kernel_launches"],
        },
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    lines = [
        "# W98 Narrow Nsys Summary",
        "",
        "These are one-call profiled ranges and explain the clean CUDA-event matrix; "
        "Nsys timings are not headline latency.",
        "",
        "| Partition | Arm | Projected GPU ms | Kernel launches | Unique kernel rows |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['partition']} | {row['arm']} | "
            f"{row['projected_gpu_ms']:.3f} | {row['kernel_launches']} | "
            f"{row['unique_kernel_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- PT2/PT2 reduces whole-model kernel launches by "
            f"{result['whole_model_launch_reduction']['pt2_vs_eager_pct']:.2f}%.",
            f"- TRT/TRT reduces whole-model kernel launches by only "
            f"{result['whole_model_launch_reduction']['tensorrt_vs_eager_pct']:.2f}%.",
            "- Use `top-kernels.csv` to attribute each outer range; the clean matrix "
            "remains the latency authority.",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projected", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
