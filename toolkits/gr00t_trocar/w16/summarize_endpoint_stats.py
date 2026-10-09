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

"""Build a compact matched-endpoint summary from W16 Nsys CSV exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROLES = ("ActorGroup", "RolloutGroup", "EnvGroup")
RANGES = {
    "actor_training_s": ("ActorGroup", ":run_training", "Avg (ns)"),
    "actor_recv_traj_s": ("ActorGroup", ":actor/recv_traj", "Avg (ns)"),
    "weight_sync_s": (
        "ActorGroup",
        ":actor/sync_model_to_rollout",
        "Avg (ns)",
    ),
    "rollout_generate_s": ("RolloutGroup", ":rollout/generate", "Avg (ns)"),
    "rollout_predict_ms": ("RolloutGroup", ":predict", "Avg (ns)"),
    "env_interact_s": ("EnvGroup", ":interact", "Avg (ns)"),
    "env_step_s": ("EnvGroup", ":env_interact_step", "Avg (ns)"),
    "env_send_trajectory_s": (
        "EnvGroup",
        ":env/send_rollout_trajectories",
        "Avg (ns)",
    ),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _role_dir(run_root: Path, role: str) -> Path:
    matches = list((run_root / "analysis/nsys-stats").glob(f"*{role}*"))
    if len(matches) != 1:
        raise ValueError(f"expected one {role} stats directory, got {matches}")
    return matches[0]


def _range_value(run_root: Path, role: str, name: str, column: str) -> float:
    rows = _read_csv(_role_dir(run_root, role) / "stats_nvtx_sum.csv")
    matches = [row for row in rows if row["Range"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one {role} range {name}, got {len(matches)}")
    scale = 1e6 if name == ":predict" else 1e9
    return float(matches[0][column]) / scale


def _kernel_stats(run_root: Path, role: str) -> dict[str, float | int]:
    rows = _read_csv(_role_dir(run_root, role) / "stats_cuda_gpu_kern_sum.csv")
    total_ns = sum(float(row["Total Time (ns)"]) for row in rows)
    nccl_ns = sum(
        float(row["Total Time (ns)"])
        for row in rows
        if "nccl" in row["Name"].lower()
    )
    return {
        "aggregate_kernel_s": total_ns / 1e9,
        "aggregate_non_nccl_kernel_s": (total_ns - nccl_ns) / 1e9,
        "kernel_instances": sum(int(row["Instances"]) for row in rows),
    }


def _memory_stats(run_root: Path) -> dict[str, float]:
    receipt = json.loads(
        (run_root / "receipts/gpu-sampler.json").read_text(encoding="utf-8")
    )
    if receipt["status"] != "passed":
        raise ValueError(f"GPU sampler failed for {run_root}")
    peaks = [float(value) for value in receipt["per_gpu_peak_memory_mib"].values()]
    if len(peaks) != 8:
        raise ValueError(f"expected eight GPU peaks for {run_root}")
    return {
        "gpu_peak_memory_min_mib": min(peaks),
        "gpu_peak_memory_mean_mib": sum(peaks) / len(peaks),
        "gpu_peak_memory_max_mib": max(peaks),
    }


def collect(run_root: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, (role, name, column) in RANGES.items():
        values[key] = _range_value(run_root, role, name, column)
    values["roles"] = {
        role: _kernel_stats(run_root, role) for role in ROLES
    }
    values.update(_memory_stats(run_root))
    return values


def _delta(original: float, optimized: float) -> float:
    return (optimized / original - 1.0) * 100.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--optimized-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"output directory exists: {args.output_dir}")

    endpoints = {
        "original": collect(args.original_run),
        "optimized": collect(args.optimized_run),
    }
    args.output_dir.mkdir(parents=True)

    metrics = [
        ("Actor training", "actor_training_s", "s/step"),
        ("Actor receive trajectory", "actor_recv_traj_s", "s/step"),
        ("Weight sync", "weight_sync_s", "s/step"),
        ("Rollout generate", "rollout_generate_s", "s/step"),
        ("Rollout predict", "rollout_predict_ms", "ms/call"),
        ("Env interact", "env_interact_s", "s/step"),
        ("Env physical step", "env_step_s", "s/call"),
        ("Env send trajectory", "env_send_trajectory_s", "s/step"),
    ]
    rows = []
    for label, key, unit in metrics:
        original = float(endpoints["original"][key])
        optimized = float(endpoints["optimized"][key])
        rows.append(
            {
                "metric": label,
                "unit": unit,
                "original": original,
                "optimized": optimized,
                "delta_percent": _delta(original, optimized),
            }
        )
    for role in ROLES:
        for label, key, unit in (
            (
                f"{role} aggregate non-NCCL kernel",
                "aggregate_non_nccl_kernel_s",
                "s/window",
            ),
            (f"{role} kernel launches", "kernel_instances", "count/window"),
        ):
            original = float(endpoints["original"]["roles"][role][key])
            optimized = float(endpoints["optimized"]["roles"][role][key])
            rows.append(
                {
                    "metric": label,
                    "unit": unit,
                    "original": original,
                    "optimized": optimized,
                    "delta_percent": _delta(original, optimized),
                }
            )

    with (args.output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema": "rlinf.w16.endpoint-nsys-summary/v1",
        "capture": {
            "steps": [1, 2],
            "continuous": True,
            "rank": 0,
            "roles": list(ROLES),
        },
        "run_roots": {
            "original": str(args.original_run),
            "optimized": str(args.optimized_run),
        },
        "endpoints": endpoints,
        "metrics": rows,
        "interpretation": {
            "headline_source": "clean W13/W15 runs, not profiled W16 wall time",
            "actor": "feature reuse removes frozen-backbone recomputation",
            "rollout": (
                "TensorRT Backbone and refittable TensorRT DiT reduce "
                "prediction work"
            ),
            "environment": "physical simulation kernel work is effectively unchanged",
            "correctness": (
                "optimized endpoint is systems-only because its W15 PPO "
                "identity gate failed"
            ),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# W16 RTX 6000 Nsight Endpoint Comparison",
        "",
        (
            "The window covers zero-based steps 1-2 after step-0 warmup for "
            "rank 0 of Actor, Rollout and Env."
        ),
        (
            "Profiled wall times explain mechanisms; clean W13/W15 runs "
            "remain the performance authority."
        ),
        "",
        "| Metric | Original | Optimized | Delta |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} ({row['unit']}) | {row['original']:.3f} | "
            f"{row['optimized']:.3f} | {row['delta_percent']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- Actor training falls because rollout-produced frozen "
                "Backbone features replace Actor-side Backbone recomputation."
            ),
            (
                "- Rollout prediction and kernel work fall after BF16 "
                "TensorRT ViT+LLM and refittable TensorRT DiT replace eager "
                "execution."
            ),
            (
                "- Env physical-step wall and aggregate kernel time are "
                "essentially unchanged; Env remains the next optimization "
                "target."
            ),
            (
                "- The optimized endpoint is systems-only: W15's pre-update "
                "PPO ratio/KL identity gate failed for refittable TensorRT DiT."
            ),
            "",
        ]
    )
    readme = args.output_dir / "README.md"
    readme.write_text("\n".join(lines), encoding="utf-8")
    with (args.output_dir / "SHA256SUMS").open("w", encoding="utf-8") as stream:
        for path in (readme, args.output_dir / "metrics.csv", summary_path):
            stream.write(f"{_sha256(path)}  {path.name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
