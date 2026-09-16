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
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
import traceback
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from common_boundary_b8 import make_explicit_noise_head, prepare_cuda_inputs
from persistent_trt import PersistentEngine

VISION_ATTENTION_CLASS = "Qwen3VLVisionAttention"
EXPECTED_VISION_SEGMENTS = 24
EXPECTED_VISION_SEQUENCE_LENGTH = 256


class _CudaTimedForward:
    """Collect CUDA-event samples around one DiT forward invocation."""

    def __init__(self, forward: Any) -> None:
        self.forward = forward
        self.events: list[tuple[Any, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = self.forward(*args, **kwargs)
        end.record()
        self.events.append((start, end))
        return output

    def take_samples(self) -> list[float]:
        if self.events:
            self.events[-1][1].synchronize()
        samples = [float(start.elapsed_time(end)) for start, end in self.events]
        self.events.clear()
        return samples


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uniform_vision_sequence_length(
    grid_rows: Sequence[Sequence[int]],
) -> tuple[int, int]:
    lengths = []
    for row in grid_rows:
        if len(row) != 3:
            raise ValueError(f"image_grid_thw row must have three values: {row!r}")
        temporal, height, width = (int(value) for value in row)
        if temporal < 1 or height < 1 or width < 1:
            raise ValueError(f"image_grid_thw values must be positive: {row!r}")
        lengths.extend([height * width] * temporal)
    if not lengths:
        raise ValueError("image_grid_thw must contain at least one vision segment")
    unique_lengths = set(lengths)
    if len(unique_lengths) != 1:
        raise ValueError(
            "PT2 static vision adapter requires one sequence length: "
            f"{sorted(unique_lengths)}"
        )
    return lengths[0], len(lengths)


def _install_static_vision_flash_attention(
    backbone: Any,
    prepared: tuple[Any, Any],
) -> tuple[dict[str, Any], Any]:
    """Freeze true-B8 visual geometry without changing the attention backend."""

    backbone_inputs, _ = prepared
    image_grid_thw = backbone_inputs.get("image_grid_thw")
    if image_grid_thw is None:
        raise RuntimeError("PT2 Backbone requires image_grid_thw")
    grid_rows = image_grid_thw.detach().cpu().tolist()
    sequence_length, segment_count = _uniform_vision_sequence_length(grid_rows)
    if (
        sequence_length != EXPECTED_VISION_SEQUENCE_LENGTH
        or segment_count != EXPECTED_VISION_SEGMENTS
    ):
        raise RuntimeError(
            "PT2 Backbone visual geometry changed: "
            f"segments={segment_count}, sequence_length={sequence_length}"
        )
    vision_modules = sum(
        type(module).__name__ == VISION_ATTENTION_CLASS for module in backbone.modules()
    )
    if vision_modules < 1:
        raise RuntimeError("PT2 Backbone found no Qwen vision attention modules")

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: PLC0415

    local_mapping = ALL_ATTENTION_FUNCTIONS._local_mapping
    had_local_override = "flash_attention_2" in local_mapping
    previous_local = local_mapping.get("flash_attention_2")
    original = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

    def static_vision_flash_attention(
        module: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if type(module).__name__ == VISION_ATTENTION_CLASS:
            kwargs["max_length_q"] = sequence_length
            kwargs["max_length_k"] = sequence_length
        return original(module, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = static_vision_flash_attention

    def restore() -> None:
        if had_local_override:
            ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = previous_local
        else:
            del ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

    return (
        {
            "policy": "fixture-validated static Qwen vision FlashAttention max length",
            "sequence_length": sequence_length,
            "segment_count": segment_count,
            "vision_attention_modules": vision_modules,
            "kernel": "flash_attention_2",
            "changes_attention_backend": False,
        },
        restore,
    )


def _load_refittable_runtime(source: Path) -> Any:
    spec = importlib.util.spec_from_file_location("w03_refittable_tensorrt_dit", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load refittable TensorRT DiT from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _measure_pure_dit(
    model: Any,
    explicit_head: Any,
    prepared: tuple[Any, Any],
    initial_actions: Any,
    warmup: int,
    measured: int,
) -> dict[str, Any]:
    """Measure DiT separately after the uninstrumented main benchmark."""

    import torch

    backbone_inputs, action_inputs = prepared
    with torch.inference_mode():
        backbone_output = model.backbone(backbone_inputs)
    torch.cuda.synchronize()

    action_model = model.action_head.model
    original_forward = action_model.forward
    timer = _CudaTimedForward(original_forward)
    action_model.forward = timer

    def call_head() -> None:
        with torch.inference_mode():
            explicit_head(
                backbone_output["backbone_features"],
                backbone_output["backbone_attention_mask"],
                backbone_output["image_mask"],
                action_inputs["state"],
                action_inputs["embodiment_id"],
                initial_actions,
            )

    try:
        for _ in range(warmup):
            call_head()
        timer.take_samples()
        for _ in range(measured):
            call_head()
        samples = timer.take_samples()
    finally:
        action_model.forward = original_forward
    expected = measured * model.action_head.num_inference_timesteps
    if len(samples) != expected:
        raise RuntimeError(f"unexpected DiT sample count: {len(samples)} != {expected}")
    return {
        "boundary": "one B8 DiT invocation; collected after the main benchmark",
        "instrumented_main_timing": False,
        "invocations_per_action_head": model.action_head.num_inference_timesteps,
        "statistics": _statistics(samples),
    }


def _dtype_counts(module: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for parameter in module.parameters():
        name = str(parameter.dtype)
        result[name] = result.get(name, 0) + 1
    return result


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
    if args.backend == "pt2" and args.vit_precision != "bf16":
        raise ValueError("the PT2 Backbone arm supports the production BF16 path only")
    refit_arguments = (
        args.refittable_runtime_source,
        args.refittable_dit_engine,
        args.refittable_dit_receipt,
        args.refittable_dit_parameter_map,
    )
    if args.action_head_backend == "refittable-trt" and not all(refit_arguments):
        raise ValueError(
            "the refittable TensorRT Action Head requires runtime, engine, receipt, "
            "and parameter-map arguments"
        )

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
                f"ViT engine precision is {actual_precision}, expected {args.vit_precision}"
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

    backbone_compile_receipt = None
    restore_vision_flash_attention = None
    if args.backend == "pt2":
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        adapter, restore_vision_flash_attention = _install_static_vision_flash_attention(
            policy.model.backbone, prepared
        )
        compile_started = time.perf_counter_ns()
        policy.model.backbone.forward = torch.compile(
            policy.model.backbone.forward,
            mode=args.backbone_compile_mode,
            dynamic=False,
        )
        _timed_sample(policy.model, explicit_head, prepared, initial_actions)
        torch.cuda.synchronize()
        backbone_compile_receipt = {
            "mode": args.backbone_compile_mode,
            "first_call_wall_ms": (time.perf_counter_ns() - compile_started)
            / 1_000_000,
            "unique_graphs_after_first_call": int(
                torch._dynamo.utils.counters["stats"]["unique_graphs"]
            ),
            "graph_breaks_after_first_call": {
                str(key): int(value)
                for key, value in torch._dynamo.utils.counters["graph_break"].items()
            },
            "static_vision_adapter": adapter,
        }

    compile_receipt = None
    refittable_dit = None
    refit_receipt = None
    if args.action_head_backend == "pt2":
        torch._dynamo.reset()
        compile_started = time.perf_counter_ns()
        action_model = policy.model.action_head.model
        action_model.forward = torch.compile(
            action_model.forward,
            mode=args.compile_mode,
        )
        _timed_sample(policy.model, explicit_head, prepared, initial_actions)
        torch.cuda.synchronize()
        compile_receipt = {
            "mode": args.compile_mode,
            "first_call_wall_ms": (time.perf_counter_ns() - compile_started)
            / 1_000_000,
            "unique_graphs_after_first_call": int(
                torch._dynamo.utils.counters["stats"]["unique_graphs"]
            ),
        }
    elif args.action_head_backend == "refittable-trt":
        runtime_source = args.refittable_runtime_source.resolve(strict=True)
        engine_path = args.refittable_dit_engine.resolve(strict=True)
        receipt_path = args.refittable_dit_receipt.resolve(strict=True)
        parameter_map_path = args.refittable_dit_parameter_map.resolve(strict=True)
        runtime = _load_refittable_runtime(runtime_source)
        parameter_map = json.loads(parameter_map_path.read_text(encoding="utf-8"))
        action_model = policy.model.action_head.model
        source_digest = runtime._ordered_source_digest(
            action_model, parameter_map["dit_refit"]["entries"]
        )
        import tensorrt as trt  # noqa: PLC0415

        setup_started = time.perf_counter_ns()
        refittable_dit = runtime.RefittableTensorRTDiT(
            action_model,
            {
                "engine_path": str(engine_path),
                "receipt_path": str(receipt_path),
                "receipt_sha256": _sha256(receipt_path),
                "parameter_map_path": str(parameter_map_path),
                "parameter_map_sha256": _sha256(parameter_map_path),
                "source_digest_revision_0": source_digest,
                "revision": 0,
                "runtime_version": trt.__version__,
                "runtime_distribution": args.refittable_runtime_distribution,
                "compute_capability": list(torch.cuda.get_device_capability()),
                "online_refit": True,
                "probe_each_revision": True,
                "minimum_free_device_bytes": args.refittable_minimum_free_bytes,
                "ppo_authority_status": (
                    "failed_ratio_kl_approximate_behavior_only"
                ),
                "lineage_receipt_mode": "gpu_transform_validation",
                "shadow_eager": False,
            },
        )
        refittable_dit.verify_revision(0)
        action_model.forward = refittable_dit
        _timed_sample(policy.model, explicit_head, prepared, initial_actions)
        torch.cuda.synchronize()
        refittable_dit.verify_revision(1)
        _timed_sample(policy.model, explicit_head, prepared, initial_actions)
        torch.cuda.synchronize()
        setup_telemetry = refittable_dit.telemetry()
        refit_receipt = {
            "scope": (
                "double-slot online-refittable TensorRT DiT; setup, initial probe, "
                "and one real 456-weight refit/adopt excluded from hot timing"
            ),
            "runtime_source": str(runtime_source),
            "runtime_source_sha256": _sha256(runtime_source),
            "source_digest_revision_0": source_digest,
            "setup_and_refit_wall_ms": (
                time.perf_counter_ns() - setup_started
            )
            / 1_000_000,
            "active_revision_before_measurement": setup_telemetry["active_revision"],
            "active_slot_before_measurement": setup_telemetry["active_slot"],
            "online_refit": setup_telemetry["online_refit"],
            "refit_records": setup_telemetry["refit_records"],
            "memory": setup_telemetry["memory"],
        }

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
    pure_dit = _measure_pure_dit(
        policy.model,
        explicit_head,
        prepared,
        initial_actions,
        args.diagnostic_warmup,
        args.diagnostic_measured,
    )
    if compile_receipt is not None:
        compile_receipt["unique_graphs_after_measurement"] = int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"]
        )
    if backbone_compile_receipt is not None:
        backbone_compile_receipt["unique_graphs_after_measurement"] = int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"]
        )
    telemetry = None
    if vit_engine is not None and llm_engine is not None:
        telemetry = {
            "vit": vit_engine.telemetry(),
            "llm": llm_engine.telemetry(),
        }
        trt_api.close_tensorrt_engines(policy)
    if refittable_dit is not None:
        refit_receipt["telemetry_after_measurement"] = refittable_dit.telemetry()
        refittable_dit.close()
    if restore_vision_flash_attention is not None:
        restore_vision_flash_attention()

    receipt = {
        "schema": "rlinf.gr00t-n1d7-b8-precision-performance.v1",
        "status": "completed",
        "scope": "performance-only; numerical validation intentionally excluded",
        "backend": args.backend,
        "vit_precision": args.vit_precision,
        "llm_precision": "bf16",
        "action_head_precision": "bf16",
        "action_head_backend": args.action_head_backend,
        "batch_size": 8,
        "warmup": args.warmup,
        "measured": args.measured,
        "seed_used_only_to_materialize_fixed_initial_actions": args.seed,
        "hot_path": (
            "CUDA-resident prepared inputs + fixed explicit noise -> normalized action; "
            "Backbone and Action Head CUDA events; no output copy or comparison"
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
        "diagnostic": {"pure_dit": pure_dit},
        "compile": compile_receipt,
        "backbone_compile": backbone_compile_receipt,
        "refittable_dit": refit_receipt,
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
    parser.add_argument("--backend", choices=("eager", "pt2", "trt"), required=True)
    parser.add_argument("--vit-precision", choices=("bf16", "fp32"), required=True)
    parser.add_argument(
        "--action-head-backend",
        choices=("eager", "pt2", "refittable-trt"),
        default="eager",
    )
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument(
        "--backbone-compile-mode", default="max-autotune-no-cudagraphs"
    )
    parser.add_argument("--refittable-runtime-source", type=Path)
    parser.add_argument("--refittable-dit-engine", type=Path)
    parser.add_argument("--refittable-dit-receipt", type=Path)
    parser.add_argument("--refittable-dit-parameter-map", type=Path)
    parser.add_argument(
        "--refittable-runtime-distribution", default="tensorrt-cu12"
    )
    parser.add_argument(
        "--refittable-minimum-free-bytes", type=int, default=8 << 30
    )
    parser.add_argument("--diagnostic-warmup", type=int, default=2)
    parser.add_argument("--diagnostic-measured", type=int, default=30)
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
