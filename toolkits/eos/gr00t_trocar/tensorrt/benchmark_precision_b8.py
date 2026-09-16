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

"""Measure true-B8 GR00T backends without numerical checks in the hot path."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any

from common_boundary_b8 import make_explicit_noise_head, prepare_cuda_inputs
from persistent_trt import PersistentEngine


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
        "mean_ms": mean,
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "sample_std_ms": sample_std,
        "cv": sample_std / mean if mean else None,
    }


def _tree_to_dtype(value: Any, dtype: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(_tree_to_dtype(item, dtype) for item in value)
    if isinstance(value, list):
        return [_tree_to_dtype(item, dtype) for item in value]
    if isinstance(value, dict):
        return {key: _tree_to_dtype(item, dtype) for key, item in value.items()}
    return value


def _install_fp32_visual_bridge(policy: Any) -> None:
    """Run eager ViT in FP32 and hand BF16 tensors back to the BF16 LLM."""

    import torch

    visual = policy.model.backbone.model.model.visual
    visual.float()
    original_forward = visual.forward

    def fp32_forward(_self: Any, *args: Any, **kwargs: Any) -> Any:
        fp32_args = _tree_to_dtype(args, torch.float32)
        fp32_kwargs = _tree_to_dtype(kwargs, torch.float32)
        result = original_forward(*fp32_args, **fp32_kwargs)
        return _tree_to_dtype(result, torch.bfloat16)

    visual.forward = types.MethodType(fp32_forward, visual)


def _load_policy(source: Path, model: Path) -> Any:
    deployment = source / "scripts/deployment"
    sys.path.insert(0, str(deployment))
    from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: PLC0415

    return Gr00tPolicy(
        embodiment_tag="NEW_EMBODIMENT",
        model_path=str(model),
        device="cuda",
    )


def _setup_trt(policy: Any, source: Path, engines: Path) -> tuple[Any, Any, Any]:
    deployment = source / "scripts/deployment"
    sys.path.insert(0, str(deployment))
    import trt_model_forward  # noqa: PLC0415

    trt_model_forward.Engine = PersistentEngine
    trt_model_forward.setup_tensorrt_engines(
        policy, str(engines), mode="vit_llm_only"
    )
    return (
        trt_model_forward,
        policy.model.backbone.vit_engine,
        policy.model.backbone.llm_engine,
    )


def _timed_sample(
    model: Any,
    explicit_head: Any,
    prepared: tuple[Any, Any],
    initial_actions: Any,
) -> tuple[float, float, float]:
    """Time only Backbone and Action Head; perform no output validation."""

    import torch

    start = torch.cuda.Event(enable_timing=True)
    backbone_done = torch.cuda.Event(enable_timing=True)
    action_done = torch.cuda.Event(enable_timing=True)
    backbone_inputs, action_inputs = prepared
    with torch.inference_mode():
        start.record()
        backbone_output = model.backbone(backbone_inputs)
        backbone_done.record()
        explicit_head(
            backbone_output["backbone_features"],
            backbone_output["backbone_attention_mask"],
            backbone_output["image_mask"],
            action_inputs["state"],
            action_inputs["embodiment_id"],
            initial_actions,
        )
        action_done.record()
    action_done.synchronize()
    backbone_ms = start.elapsed_time(backbone_done)
    head_ms = backbone_done.elapsed_time(action_done)
    return backbone_ms, head_ms, start.elapsed_time(action_done)


def _measure(
    model: Any,
    explicit_head: Any,
    prepared: tuple[Any, Any],
    initial_actions: Any,
    warmup: int,
    measured: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        _timed_sample(model, explicit_head, prepared, initial_actions)
    samples = {"backbone_ms": [], "action_head_ms": [], "total_ms": []}
    for _ in range(measured):
        values = _timed_sample(model, explicit_head, prepared, initial_actions)
        for name, value in zip(samples, values, strict=True):
            samples[name].append(value)
    return {name: _statistics(values) for name, values in samples.items()}


def _dtype_counts(module: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for parameter in module.parameters():
        name = str(parameter.dtype)
        result[name] = result.get(name, 0) + 1
    return result


def _runtime_cast_diagnostic(
    model: Any,
    prepared: tuple[Any, Any],
    vit_engine: Any | None,
    llm_engine: Any | None,
) -> dict[str, Any]:
    import torch

    if vit_engine is None or llm_engine is None:
        return {
            "scope": "TensorRT precision bridge only",
            "expected_bridge_casts": 0,
            "observed_cast_operator_calls": 0,
            "observed_cast_operator_device_ms": 0.0,
            "operators": [],
        }
    pixel_dtype = prepared[0]["pixel_values"].dtype
    vit_dtype = vit_engine.dtype_of("pixel_values")
    llm_dtype = llm_engine.dtype_of("inputs_embeds")
    expected = int(pixel_dtype != vit_dtype) + 4 * int(vit_dtype != llm_dtype)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        with torch.inference_mode():
            model.backbone(prepared[0])
        torch.cuda.synchronize()
    operators = []
    for event in profile.key_averages():
        if event.key not in {"aten::to", "aten::_to_copy", "aten::copy_"}:
            continue
        device_us = float(
            getattr(event, "self_device_time_total", 0.0)
            or getattr(event, "self_cuda_time_total", 0.0)
        )
        operators.append(
            {
                "name": event.key,
                "count": event.count,
                "self_device_ms": device_us / 1000.0,
            }
        )
    return {
        "scope": (
            "diagnostic outside the timed hot path; expected bridge casts are the "
            "explicit pixel/image/deepstack precision conversions in trt_model_forward"
        ),
        "prepared_pixel_dtype": str(pixel_dtype),
        "vit_input_dtype": str(vit_dtype),
        "llm_input_dtype": str(llm_dtype),
        "expected_bridge_casts": expected,
        "observed_cast_operator_calls": sum(item["count"] for item in operators),
        "observed_cast_operator_device_ms": sum(
            item["self_device_ms"] for item in operators
        ),
        "operators": operators,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    source = args.source.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    collated_path = args.collated.resolve(strict=True)
    engines = args.engines.resolve(strict=True) if args.engines else None
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    if args.backend == "trt" and engines is None:
        raise ValueError("--engines is required for the TensorRT backend")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    collated = torch.load(collated_path, map_location="cpu", weights_only=False)
    inputs = collated["inputs"] if "inputs" in collated else collated

    load_started = time.perf_counter_ns()
    policy = _load_policy(source, model_path)
    if args.backend == "eager" and args.vit_precision == "fp32":
        _install_fp32_visual_bridge(policy)
    trt_api = vit_engine = llm_engine = None
    if args.backend == "trt":
        trt_api, vit_engine, llm_engine = _setup_trt(policy, source, engines)
        actual_precision = {
            torch.bfloat16: "bf16",
            torch.float32: "fp32",
        }[vit_engine.dtype_of("pixel_values")]
        if actual_precision != args.vit_precision:
            raise RuntimeError(
                "ViT engine precision is "
                f"{actual_precision}, expected {args.vit_precision}"
            )
    prepared = prepare_cuda_inputs(policy.model, inputs)
    explicit_head = make_explicit_noise_head(policy.model.action_head)
    initial_actions = torch.randn(
        (8, 40, 132),
        dtype=torch.bfloat16,
        device="cuda",
    )
    torch.cuda.synchronize()
    load_ms = (time.perf_counter_ns() - load_started) / 1_000_000

    visual_dtypes = (
        {"TensorRT": 1}
        if args.backend == "trt"
        else _dtype_counts(policy.model.backbone.model.model.visual)
    )
    timing = _measure(
        policy.model,
        explicit_head,
        prepared,
        initial_actions,
        args.warmup,
        args.measured,
    )
    cast_diagnostic = _runtime_cast_diagnostic(
        policy.model, prepared, vit_engine, llm_engine
    )
    telemetry = None
    if vit_engine is not None and llm_engine is not None:
        telemetry = {
            "vit": vit_engine.telemetry(),
            "llm": llm_engine.telemetry(),
        }
        trt_api.close_tensorrt_engines(policy)

    receipt = {
        "schema": "rlinf.gr00t-n1d7-b8-precision-performance.v1",
        "status": "completed",
        "scope": "performance-only; numerical validation intentionally excluded",
        "backend": args.backend,
        "vit_precision": args.vit_precision,
        "llm_precision": "bf16",
        "action_head_precision": "bf16",
        "batch_size": 8,
        "warmup": args.warmup,
        "measured": args.measured,
        "seed_used_only_to_materialize_fixed_initial_actions": args.seed,
        "hot_path": (
            "CUDA-resident prepared inputs + fixed explicit noise -> normalized "
            "action; Backbone and Action Head CUDA events; no output copy or comparison"
        ),
        "load_and_prepare_ms": load_ms,
        "dtype_evidence": {
            "visual_parameters": visual_dtypes,
            "language_parameters": _dtype_counts(
                policy.model.backbone.model.model.language_model
            )
            if args.backend == "eager"
            else {"TensorRT": 1},
            "action_head_parameters": _dtype_counts(policy.model.action_head),
            "prepared_pixel_values": str(prepared[0]["pixel_values"].dtype),
            "fixed_initial_actions": str(initial_actions.dtype),
        },
        "timing": timing,
        "runtime_cast_diagnostic": cast_diagnostic,
        "trt_telemetry_after_measurement": telemetry,
        "cuda": {
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--collated", type=Path, required=True)
    parser.add_argument("--engines", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("eager", "trt"), required=True)
    parser.add_argument("--vit-precision", choices=("bf16", "fp32"), required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=30)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"precision performance benchmark failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
