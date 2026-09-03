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

"""Compare eager and persistent TensorRT-backbone N1.7 Trocar true-B8."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from persistent_trt import PersistentEngine
from resident_b1 import _compare_actions
from trocar_b8_model_view import LANGUAGE_KEY, STATE_ACTION_ORDER


def _compare_array(reference: Any, candidate: Any) -> dict[str, Any]:
    return _compare_actions({"value": reference}, {"value": candidate})


def _statistics(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "samples_ms": values,
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "mean_ms": mean,
        "sample_std_ms": sample_std,
        "cv": sample_std / mean if mean else None,
    }


def _process_rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is absent from /proc/self/status")


def _cuda_memory() -> dict[str, int]:
    import torch

    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def _measure_memory(call: Any, iterations: int) -> dict[str, Any]:
    """Measure a separate untimed memory probe without polluting latency samples."""

    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cuda_before = _cuda_memory()
    rss_before = _process_rss_bytes()
    rss_peak = [rss_before]
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.wait(0.001):
            rss_peak[0] = max(rss_peak[0], _process_rss_bytes())

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    try:
        for _ in range(iterations):
            call()
        torch.cuda.synchronize()
    finally:
        stop.set()
        sampler.join()
    rss_after = _process_rss_bytes()
    rss_peak[0] = max(rss_peak[0], rss_after)
    return {
        "iterations": iterations,
        "host_rss_before_bytes": rss_before,
        "host_rss_peak_bytes": rss_peak[0],
        "host_rss_peak_delta_bytes": rss_peak[0] - rss_before,
        "host_rss_after_bytes": rss_after,
        "cuda_allocated_before_bytes": cuda_before["allocated_bytes"],
        "cuda_reserved_before_bytes": cuda_before["reserved_bytes"],
        "cuda_allocated_after_bytes": torch.cuda.memory_allocated(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_reserved_after_bytes": torch.cuda.memory_reserved(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "note": "dedicated probe; host RSS sampled every 1 ms",
    }


def _measure_model_memory(policy: Any, inputs: Any, iterations: int) -> dict[str, Any]:
    import torch

    def call() -> None:
        with torch.inference_mode():
            policy.model.get_action(inputs)

    return _measure_memory(call, iterations)


def _measure_whole_memory(
    policy: Any, observation: dict[str, Any], iterations: int
) -> dict[str, Any]:
    return _measure_memory(lambda: policy.get_action(observation), iterations)


def _raw_observation(raw_path: Path, receipt_path: Path) -> dict[str, Any]:
    import numpy as np

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    prompt = receipt["raw_observation"]["prompt"]
    with np.load(raw_path) as archive:
        video = {
            key.removeprefix("video."): archive[key]
            for key in archive.files
            if key.startswith("video.")
        }
        state = {
            key.removeprefix("state."): archive[key]
            for key in archive.files
            if key.startswith("state.")
        }
    return {
        "video": video,
        "state": state,
        "language": {LANGUAGE_KEY: [[prompt] for _ in range(8)]},
    }


def _public_action(policy: Any, normalized_action: Any, observation: dict) -> Any:
    import numpy as np

    decoded = policy.processor.decode_action(
        normalized_action.float().cpu().numpy(),
        policy.embodiment_tag,
        observation["state"],
    )
    result = np.concatenate([decoded[key] for key in STATE_ACTION_ORDER], axis=-1)
    if result.shape != (8, 16, 28):
        raise RuntimeError(f"unexpected public action shape: {result.shape}")
    return result


def _fixed_model_call(policy: Any, inputs: Any, seed: int) -> tuple[Any, Any, Any]:
    import torch

    captured = {}

    def backbone_hook(_module: Any, _args: Any, output: Any) -> None:
        captured["backbone"] = output.backbone_features.detach().cpu().clone()

    hook = policy.model.backbone.register_forward_hook(backbone_hook)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        with torch.inference_mode():
            output = policy.model.get_action(inputs)
        torch.cuda.synchronize()
    finally:
        hook.remove()
    return (
        output.action_pred.detach().cpu().clone(),
        captured["backbone"],
        output,
    )


def _measure_model(policy: Any, inputs: Any, warmup: int, measured: int) -> dict:
    import torch

    def one() -> float:
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        with torch.inference_mode():
            policy.model.get_action(inputs)
        torch.cuda.synchronize()
        return (time.perf_counter_ns() - started) / 1_000_000

    warmup_values = [one() for _ in range(warmup)]
    measured_values = [one() for _ in range(measured)]
    return {
        "boundary": "cuda_sync; Gr00tN1d7.get_action(collated); cuda_sync",
        "warmup_ms": warmup_values,
        "measured": _statistics(measured_values),
    }


def _measure_whole(policy: Any, observation: dict, warmup: int, measured: int) -> dict:
    import torch

    def one() -> float:
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        policy.get_action(observation)
        torch.cuda.synchronize()
        return (time.perf_counter_ns() - started) / 1_000_000

    warmup_values = [one() for _ in range(warmup)]
    measured_values = [one() for _ in range(measured)]
    return {
        "boundary": "cuda_sync; Gr00tPolicy.get_action(raw_observation); cuda_sync",
        "warmup_ms": warmup_values,
        "measured": _statistics(measured_values),
    }


def _measure_components(
    policy: Any, observation: dict, warmup: int, measured: int
) -> dict[str, Any]:
    from benchmark_inference import benchmark_components  # noqa: PLC0415

    values = benchmark_components(
        policy, observation, num_iterations=measured, warmup=warmup
    )
    raw = {name: samples.tolist() for name, samples in values.items()}
    raw["constructed_sum"] = (
        values["data_processing"] + values["backbone"] + values["action_head"]
    ).tolist()
    return {
        "boundary": "upstream benchmark_inference.benchmark_components",
        "constructed_sum_is_not_measured_e2e": True,
        "warmup": warmup,
        "measured": measured,
        "raw_samples_ms": raw,
        "statistics": {name: _statistics(samples) for name, samples in raw.items()},
    }


def _load_policy(source: Path, model: Path) -> Any:
    deployment = source / "scripts/deployment"
    sys.path.insert(0, str(deployment))
    from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: PLC0415

    return Gr00tPolicy(
        embodiment_tag="NEW_EMBODIMENT",
        model_path=str(model),
        device="cuda",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    source = args.source.resolve(strict=True)
    model = args.model.resolve(strict=True)
    engines = args.engines.resolve(strict=True)
    collated_path = args.collated.resolve(strict=True)
    raw_path = args.raw.resolve(strict=True)
    fixture_receipt = args.fixture_receipt.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    collated = torch.load(collated_path, map_location="cpu", weights_only=False)
    inputs = collated["inputs"] if "inputs" in collated else collated
    observation = _raw_observation(raw_path, fixture_receipt)

    eager_load_started = time.perf_counter_ns()
    eager = _load_policy(source, model)
    torch.cuda.synchronize()
    eager_load_ms = (time.perf_counter_ns() - eager_load_started) / 1_000_000
    eager_vit = {}

    def eager_vit_hook(_module: Any, _args: Any, output_value: Any) -> None:
        value = output_value[0] if isinstance(output_value, tuple) else output_value
        eager_vit["image_embeds"] = value.detach().cpu().clone()

    vit_hook = eager.model.backbone.model.model.visual.register_forward_hook(
        eager_vit_hook
    )
    try:
        eager_first_started = time.perf_counter_ns()
        eager_action, eager_backbone, _ = _fixed_model_call(eager, inputs, args.seed)
        eager_first_ms = (time.perf_counter_ns() - eager_first_started) / 1_000_000
    finally:
        vit_hook.remove()
    eager_public = _public_action(eager, eager_action, observation)
    eager_resident_memory = {
        "host_rss_bytes": _process_rss_bytes(),
        "cuda": _cuda_memory(),
    }
    eager_components = _measure_components(
        eager, observation, args.component_warmup, args.component_measured
    )
    eager_model_timing = _measure_model(eager, inputs, args.warmup, args.measured)
    eager_whole_timing = _measure_whole(eager, observation, args.warmup, args.measured)

    eager_model_memory = _measure_model_memory(eager, inputs, args.memory_iterations)
    eager_whole_memory = _measure_whole_memory(
        eager, observation, args.memory_iterations
    )
    del eager
    gc.collect()
    torch.cuda.empty_cache()

    hybrid_load_started = time.perf_counter_ns()
    hybrid = _load_policy(source, model)
    torch.cuda.synchronize()
    hybrid_load_ms = (time.perf_counter_ns() - hybrid_load_started) / 1_000_000
    deployment = source / "scripts/deployment"
    sys.path.insert(0, str(deployment))
    import trt_model_forward  # noqa: PLC0415

    trt_model_forward.Engine = PersistentEngine
    setup_started = time.perf_counter_ns()
    trt_model_forward.setup_tensorrt_engines(hybrid, str(engines), mode="vit_llm_only")
    torch.cuda.synchronize()
    engine_setup_ms = (time.perf_counter_ns() - setup_started) / 1_000_000
    hybrid_first_started = time.perf_counter_ns()
    hybrid_action, hybrid_backbone, _ = _fixed_model_call(hybrid, inputs, args.seed)
    hybrid_first_ms = (time.perf_counter_ns() - hybrid_first_started) / 1_000_000
    hybrid_public = _public_action(hybrid, hybrid_action, observation)
    vit_engine = hybrid.model.backbone.vit_engine
    llm_engine = hybrid.model.backbone.llm_engine
    torch.cuda.synchronize()
    hybrid_vit = vit_engine.last_outputs["image_embeds"].detach().cpu().clone()

    repeat_action, _repeat_backbone, _ = _fixed_model_call(hybrid, inputs, args.seed)
    fixed_noise_repeat = _compare_array(
        hybrid_action.float().numpy(), repeat_action.float().numpy()
    )
    allocation_floor = {
        "vit": vit_engine.allocation_count,
        "llm": llm_engine.allocation_count,
    }
    hybrid_resident_memory = {
        "host_rss_bytes": _process_rss_bytes(),
        "cuda": _cuda_memory(),
    }
    hybrid_components = _measure_components(
        hybrid, observation, args.component_warmup, args.component_measured
    )
    hybrid_model_timing = _measure_model(hybrid, inputs, args.warmup, args.measured)
    hybrid_whole_timing = _measure_whole(
        hybrid, observation, args.warmup, args.measured
    )

    hybrid_model_memory = _measure_model_memory(hybrid, inputs, args.memory_iterations)
    hybrid_whole_memory = _measure_whole_memory(
        hybrid, observation, args.memory_iterations
    )
    telemetry = {
        "vit": vit_engine.telemetry(),
        "llm": llm_engine.telemetry(),
        "allocation_count_after_model_timing": allocation_floor,
    }

    comparisons = {
        "vit_image_embeds": _compare_array(
            eager_vit["image_embeds"].float().numpy(), hybrid_vit.float().numpy()
        ),
        "pre_final_backbone": _compare_array(
            eager_backbone.float().numpy(), hybrid_backbone.float().numpy()
        ),
        "normalized_action": _compare_array(
            eager_action.float().numpy(), hybrid_action.float().numpy()
        ),
        "public_action": _compare_array(eager_public, hybrid_public),
        "fixed_noise_hybrid_repeat": fixed_noise_repeat,
    }
    gates = {
        "finite": all(value["finite"] for value in comparisons.values()),
        "vit_cosine": comparisons["vit_image_embeds"]["cosine"] >= 0.997,
        "backbone_cosine": comparisons["pre_final_backbone"]["cosine"] >= 0.9995,
        "public_action": (
            comparisons["public_action"]["cosine"] >= 0.999
            and comparisons["public_action"]["mean_abs"] <= 0.005
            and comparisons["public_action"]["max_abs"] <= 0.05
        ),
        "fixed_noise_repeat": fixed_noise_repeat["bitwise_equal"],
        "persistent_lifecycle": (
            telemetry["vit"]["load_count"] == 1
            and telemetry["llm"]["load_count"] == 1
            and telemetry["vit"]["context_count"] == 1
            and telemetry["llm"]["context_count"] == 1
            and telemetry["vit"]["stream_sync_count"] == 0
            and telemetry["llm"]["stream_sync_count"] == 0
            and telemetry["vit"]["allocation_count"] == allocation_floor["vit"]
            and telemetry["llm"]["allocation_count"] == allocation_floor["llm"]
        ),
    }
    status = "passed" if all(gates.values()) else "failed"
    receipt = {
        "schema": "rlinf.gr00t-n1d7-trocar-true-b8-standalone-hybrid.v1",
        "status": status,
        "semantic_scope": "standalone systems and public-action qualification",
        "ppo_statistics": {
            "availability": "unavailable",
            "reason": (
                "the official standalone Gr00tPolicy has no RLInf value head or "
                "behavior/current PPO logprob path; these gates remain in W78"
            ),
        },
        "batch_size": 8,
        "single_policy_call": True,
        "b1x8": False,
        "seed": args.seed,
        "comparisons": comparisons,
        "gates": gates,
        "timing": {
            "eager": {
                "cold": {
                    "model_load_ms": eager_load_ms,
                    "first_fixed_noise_model_call_ms": eager_first_ms,
                },
                "components": eager_components,
                "model_only": eager_model_timing,
                "whole_call": eager_whole_timing,
            },
            "hybrid": {
                "cold": {
                    "model_load_ms": hybrid_load_ms,
                    "engine_deserialize_context_setup_ms": engine_setup_ms,
                    "first_fixed_noise_model_call_ms": hybrid_first_ms,
                },
                "components": hybrid_components,
                "model_only": hybrid_model_timing,
                "whole_call": hybrid_whole_timing,
            },
        },
        "memory": {
            "eager": {
                "resident_after_first_call": eager_resident_memory,
                "model_only_probe": eager_model_memory,
                "whole_call_probe": eager_whole_memory,
            },
            "hybrid": {
                "resident_after_first_call": hybrid_resident_memory,
                "model_only_probe": hybrid_model_memory,
                "whole_call_probe": hybrid_whole_memory,
            },
            "host_comparison_note": (
                "arms run sequentially in one process; compare scoped RSS deltas, "
                "not absolute RSS, because the host allocator may retain storage"
            ),
        },
        "telemetry": telemetry,
    }
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trt_model_forward.close_tensorrt_engines(hybrid)
    if status != "passed":
        raise RuntimeError(f"standalone true-B8 gates failed: {gates}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engines", type=Path, required=True)
    parser.add_argument("--collated", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--fixture-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=30)
    parser.add_argument("--component-warmup", type=int, default=5)
    parser.add_argument("--component-measured", type=int, default=20)
    parser.add_argument("--memory-iterations", type=int, default=3)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W80 standalone true-B8 failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
