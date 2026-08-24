#!/usr/bin/env python3
"""Validate RLinf W71 events and produce unified per-step measurements."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PHYSICAL_ACTIONS = 65_536
POLICY_DECISIONS = 4_096
DEFAULT_MEASURED_STEPS = (1, 2, 3, 4)


def load_events(path: Path) -> list[dict]:
    events = []
    for source in sorted(path.glob("events-*.jsonl")):
        for line_number, line in enumerate(source.read_text().splitlines(), 1):
            event = json.loads(line)
            if event.get("schema") != "rlinf.performance-event/v1":
                raise ValueError(f"{source}:{line_number}: wrong schema")
            if event.get("status") != "ok":
                raise ValueError(f"{source}:{line_number}: non-ok event")
            events.append(event)
    if not events:
        raise ValueError(f"no event files found under {path}")
    for field in ("run_id", "boot_id", "hostname"):
        values = {event[field] for event in events}
        if len(values) != 1:
            raise ValueError(f"mixed {field}: {sorted(values)}")
    return events


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError(f"negative interval: {start}..{end}")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_union_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in merge_intervals(intervals))


def interval_intersection_ns(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> int:
    a = merge_intervals(left)
    b = merge_intervals(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        total += max(0, end - start)
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def event_key(event: dict, *, include_request: bool) -> tuple:
    key = (event["outer_step"], event["rank"])
    if include_request:
        metadata = event.get("metadata", {})
        key += (event["request_id"], metadata.get("pipeline_stage", 0))
    return key


def pair_stage(
    events: list[dict], stage: str, *, include_request: bool
) -> dict[tuple, tuple[int, int]]:
    endpoints = defaultdict(dict)
    for event in events:
        if event["stage"] != stage:
            continue
        key = event_key(event, include_request=include_request)
        endpoint = event["event"]
        if endpoint in endpoints[key]:
            raise ValueError(f"duplicate {stage} {endpoint}: {key}")
        endpoints[key][endpoint] = event["timestamp_ns"]
    result = {}
    for key, pair in endpoints.items():
        if set(pair) != {"start", "end"}:
            raise ValueError(f"incomplete {stage}: {key}: {sorted(pair)}")
        if pair["end"] < pair["start"]:
            raise ValueError(f"negative {stage}: {key}")
        result[key] = (pair["start"], pair["end"])
    return result


def rank_distribution(intervals: list[tuple[int, int, int]]) -> dict:
    values = sorted((end - start, rank) for rank, start, end in intervals)
    durations = [value for value, _ in values]

    def nearest_rank(percentile: float) -> int:
        index = math.ceil(percentile * len(values)) - 1
        return values[max(index, 0)][0]

    return {
        "raw_rank_duration_ns": [
            {"rank": rank, "duration_ns": end - start}
            for rank, start, end in sorted(intervals)
        ],
        "min_ns": min(durations),
        "mean_ns": statistics.fmean(durations),
        "p50_nearest_rank_ns": nearest_rank(0.50),
        "p95_nearest_rank_ns": nearest_rank(0.95),
        "max_ns": max(durations),
        "skew_ns": max(durations) - min(durations),
    }


def summarize_steps(events: list[dict]) -> dict[int, dict]:
    outer = pair_stage(events, "step.resident_outer", include_request=False)
    publication = pair_stage(events, "revision.publication", include_request=False)
    region = pair_stage(events, "rollout_env.region", include_request=False)
    advantage = pair_stage(events, "advantage.compute", include_request=False)
    trainer = pair_stage(events, "trainer.update", include_request=False)
    rollout_service = pair_stage(events, "rollout.service", include_request=True)
    env_service = pair_stage(events, "environment.service", include_request=True)
    policy_wait = pair_stage(events, "environment.policy_wait", include_request=True)
    request_wait = pair_stage(events, "rollout.request_wait", include_request=True)

    steps = sorted(step for step, rank in outer if rank == 0)
    result = {}
    for step in steps:
        outer_start, outer_end = outer[(step, 0)]
        region_start, region_end = region[(step, 0)]
        rollout_intervals = [
            interval for key, interval in rollout_service.items() if key[0] == step
        ]
        env_intervals = [
            interval for key, interval in env_service.items() if key[0] == step
        ]
        all_service = rollout_intervals + env_intervals
        for start, end in all_service:
            if start < region_start or end > region_end:
                raise ValueError(
                    f"step {step}: service interval outside rollout_env.region"
                )

        rollout_union = interval_union_ns(rollout_intervals)
        env_union = interval_union_ns(env_intervals)
        combined_union = interval_union_ns(all_service)
        overlap = interval_intersection_ns(rollout_intervals, env_intervals)
        region_wall = region_end - region_start
        if combined_union > region_wall:
            raise ValueError(f"step {step}: service union exceeds region")

        def stage_ranks(pairs: dict) -> dict:
            intervals = [
                (key[1], start, end)
                for key, (start, end) in pairs.items()
                if key[0] == step
            ]
            return rank_distribution(intervals)

        outer_wall = outer_end - outer_start
        result[step] = {
            "resident_outer_ns": outer_wall,
            "physical_actions_per_second": PHYSICAL_ACTIONS / (outer_wall / 1e9),
            "policy_decisions_per_second": POLICY_DECISIONS / (outer_wall / 1e9),
            "revision_publication_ns": publication[(step, 0)][1]
            - publication[(step, 0)][0],
            "rollout_env_region_ns": region_wall,
            "rollout_service_union_ns": rollout_union,
            "environment_service_union_ns": env_union,
            "combined_service_union_ns": combined_union,
            "service_overlap_ns": overlap,
            "uncovered_region_ns": region_wall - combined_union,
            "rollout_service_interval_count": len(rollout_intervals),
            "environment_service_interval_count": len(env_intervals),
            "environment_policy_wait": stage_ranks(policy_wait),
            "rollout_request_wait": stage_ranks(request_wait),
            "advantage_compute": stage_ranks(advantage),
            "trainer_update": stage_ranks(trainer),
        }
    return result


def repetition_statistics(
    steps: dict[int, dict], measured_steps: tuple[int, ...]
) -> dict:
    missing = sorted(set(measured_steps) - set(steps))
    if missing:
        raise ValueError(f"missing measured steps: {missing}")
    if len(measured_steps) < 2:
        raise ValueError("at least two measured steps are required")
    fields = (
        "resident_outer_ns",
        "physical_actions_per_second",
        "policy_decisions_per_second",
        "revision_publication_ns",
        "rollout_env_region_ns",
        "rollout_service_union_ns",
        "environment_service_union_ns",
        "combined_service_union_ns",
        "service_overlap_ns",
        "uncovered_region_ns",
    )
    result = {}
    for field in fields:
        values = [steps[step][field] for step in measured_steps]
        mean = statistics.fmean(values)
        result[field] = {
            "values": values,
            "mean": mean,
            "sample_std": statistics.stdev(values),
            "cv": statistics.stdev(values) / mean if mean else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--measured-steps",
        default=",".join(str(step) for step in DEFAULT_MEASURED_STEPS),
        help="comma-separated zero-based outer steps included in statistics",
    )
    parser.add_argument(
        "--warmup-step",
        type=int,
        default=0,
        help="zero-based warmup step, or -1 when a smoke has no excluded warmup",
    )
    args = parser.parse_args()

    measured_steps = tuple(int(step) for step in args.measured_steps.split(","))
    if not measured_steps or len(set(measured_steps)) != len(measured_steps):
        raise ValueError("measured steps must be a non-empty unique sequence")

    events = load_events(args.event_dir)
    steps = summarize_steps(events)
    receipt = {
        "schema": "rlinf.unified-performance-receipt/v1",
        "run_id": events[0]["run_id"],
        "hostname": events[0]["hostname"],
        "boot_id": events[0]["boot_id"],
        "warmup_step": args.warmup_step if args.warmup_step >= 0 else None,
        "measured_steps": list(measured_steps),
        "counts": {
            "physical_actions_per_outer_step": PHYSICAL_ACTIONS,
            "policy_decisions_per_outer_step": POLICY_DECISIONS,
        },
        "steps": steps,
        "statistics": repetition_statistics(steps, measured_steps),
        "unavailable": {
            "rollout_env.causal_critical_path": "dependency edges not recorded",
            "policy.inference_device": "no complete device-boundary event pair",
            "policy.model_forward_device": "no equivalent RLinf model-only CUDA interval",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
