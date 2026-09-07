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

"""Matched true-B8 Backbone and refittable Action Head executor matrix."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from importlib import util
from pathlib import Path
from typing import Any

from common_boundary_b8 import call_with_explicit_noise, cuda_event_call

StageCall = Callable[[], tuple[Any, dict[str, float]]]
TensorRTPhase = Callable[[str, int, Callable[[], Any]], Any]


def _load_refittable_tensorrt_dit() -> Any:
    """Load the narrow runtime without importing RLinf's model registry."""

    source = (
        Path(__file__).resolve().parents[4]
        / "rlinf/models/embodiment/gr00t/gr00t_n1d7/tensorrt_dit.py"
    )
    spec = util.spec_from_file_location("w84_refittable_tensorrt_dit", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load refittable TensorRT DiT from {source}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RefittableTensorRTDiT


def balanced_orders(names: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Return a balanced forward/reverse rotation schedule."""

    names = tuple(names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("executor matrix needs at least two distinct arms")
    forward = tuple(names[offset:] + names[:offset] for offset in range(len(names)))
    reverse_names = tuple(reversed(names))
    reverse = tuple(
        reverse_names[offset:] + reverse_names[:offset]
        for offset in range(len(reverse_names))
    )
    return forward + reverse


def _statistics(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty sample set")
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    mean = statistics.fmean(ordered)
    sample_std = statistics.stdev(ordered) if len(ordered) > 1 else 0.0
    return {
        "count": len(ordered),
        "samples_ms": list(values),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "mean_ms": mean,
        "sample_std_ms": sample_std,
        "cv": sample_std / mean if mean else None,
    }


def _compare(reference: Any, candidate: Any) -> dict[str, Any]:
    import torch

    reference = reference.detach().float()
    candidate = candidate.detach().float()
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"executor output shape mismatch: {reference.shape} != {candidate.shape}"
        )
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    difference = candidate - reference
    absolute = difference.abs()
    reference_norm = torch.linalg.vector_norm(reference)
    candidate_norm = torch.linalg.vector_norm(candidate)
    denominator = reference_norm * candidate_norm
    cosine = (
        float(torch.sum(reference * candidate) / denominator)
        if float(denominator) != 0.0
        else None
    )
    return {
        "finite": finite,
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "cosine": cosine,
        "mean_abs": float(absolute.mean()),
        "max_abs": float(absolute.max()),
        "relative_l2": (
            float(torch.linalg.vector_norm(difference) / reference_norm)
            if float(reference_norm) != 0.0
            else None
        ),
    }


def measure_stage_matrix(
    arms: Mapping[str, StageCall],
    *,
    reference: str,
    warmup: int,
    measured: int,
    boundary: str,
) -> dict[str, Any]:
    """Measure all arms under one balanced schedule and one CUDA-event boundary."""

    names = tuple(arms)
    if reference not in arms:
        raise ValueError(f"matrix reference is absent: {reference}")
    if warmup < 0 or measured < 1:
        raise ValueError("matrix warmup/measured counts are invalid")
    schedule = balanced_orders(names)
    for index in range(warmup):
        for name in schedule[index % len(schedule)]:
            arms[name]()

    stages = ("backbone_ms", "action_head_ms", "total_ms")
    samples = {name: {stage: [] for stage in stages} for name in names}
    closure_error = {name: [] for name in names}
    outputs = {}
    orders = []
    for index in range(measured):
        order = schedule[index % len(schedule)]
        orders.append(list(order))
        for name in order:
            output, timing = arms[name]()
            if set(timing) != set(stages):
                raise RuntimeError(f"unexpected CUDA stage set for {name}: {timing}")
            outputs[name] = output.detach().clone()
            for stage in stages:
                samples[name][stage].append(float(timing[stage]))
            closure_error[name].append(
                float(
                    timing["backbone_ms"]
                    + timing["action_head_ms"]
                    - timing["total_ms"]
                )
            )

    summaries = {
        name: {stage: _statistics(values) for stage, values in arm.items()}
        for name, arm in samples.items()
    }
    reference_summary = summaries[reference]
    relative = {}
    comparisons = {}
    for name in names:
        relative[name] = {}
        for stage in stages:
            baseline = reference_summary[stage]["mean_ms"]
            candidate = summaries[name][stage]["mean_ms"]
            relative[name][stage] = {
                "speedup_vs_reference": baseline / candidate,
                "latency_reduction_fraction_vs_reference": (baseline - candidate)
                / baseline,
            }
        comparisons[name] = _compare(outputs[reference], outputs[name])

    max_closure_error = max(
        abs(value) for values in closure_error.values() for value in values
    )
    return {
        "boundary": boundary,
        "order_policy": "balanced forward/reverse rotations",
        "orders": orders,
        "warmup": warmup,
        "measured": measured,
        "reference": reference,
        "arms": summaries,
        "relative_to_reference": relative,
        "output_comparisons": comparisons,
        "stage_closure_error_ms": {
            name: _statistics(values) for name, values in closure_error.items()
        },
        "max_abs_stage_closure_error_ms": max_closure_error,
    }


def _cuda_memory() -> dict[str, int]:
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def _parameter_contract(module: Any) -> dict[str, tuple[int, int]]:
    return {
        name: (id(parameter), parameter.data_ptr())
        for name, parameter in module.named_parameters()
    }


def _first_refit_parameter(executor: Any) -> tuple[str, Any]:
    parameters = dict(executor.action_model.named_parameters())
    for entry in executor.artifacts["entries"]:
        name = entry["source_fqn"].removeprefix("action_head.model.")
        if name in parameters and parameters[name].numel():
            return name, parameters[name]
    raise RuntimeError("refittable TensorRT DiT has no mutable source parameter")


def _exercise_refit_lifecycle(executor: Any) -> dict[str, Any]:
    """Exercise both engine slots cold and warm, ending at original weights."""

    import torch

    name, parameter = _first_refit_parameter(executor)
    flat = parameter.view(-1)
    original = flat[0].detach().clone()
    changed = (original.float() + 0.125).to(dtype=parameter.dtype)
    if bool(torch.equal(original, changed)):
        changed = (original.float() + 1.0).to(dtype=parameter.dtype)
    if bool(torch.equal(original, changed)):
        raise RuntimeError("synthetic refit delta did not change the BF16 source value")

    records_before = len(executor.refit_records)
    try:
        with torch.no_grad():
            flat[0].copy_(changed)
        executor.verify_revision(1)
        with torch.no_grad():
            flat[0].copy_(original)
        executor.verify_revision(2)
        with torch.no_grad():
            flat[0].copy_(changed)
        executor.verify_revision(3)
        with torch.no_grad():
            flat[0].copy_(original)
        executor.verify_revision(4)
    finally:
        with torch.no_grad():
            flat[0].copy_(original)

    records = executor.refit_records[records_before:]
    if executor.active_revision != 4 or len(records) != 4:
        raise RuntimeError("refittable TensorRT DiT did not complete four revisions")
    if any(record.get("revision") != index for index, record in enumerate(records, 1)):
        raise RuntimeError("refittable TensorRT DiT revision order changed")
    return {
        "scope": "one BF16 DiT element toggled and restored across both engine slots",
        "parameter": name,
        "revisions": [record["revision"] for record in records],
        "final_revision": executor.active_revision,
        "final_source_restored_bitwise": bool(torch.equal(flat[0], original)),
        "records": records,
    }


def _make_call(
    policy: Any,
    prepared: Any,
    explicit_head: Any,
    initial_actions: Any,
    *,
    backbone_forward: Any,
    action_forward: Any,
) -> StageCall:
    def call() -> tuple[Any, dict[str, float]]:
        policy.model.backbone.forward = backbone_forward
        policy.model.action_head.model.forward = action_forward
        return cuda_event_call(
            policy.model,
            explicit_head,
            prepared,
            initial_actions,
        )

    return call


def run_executor_matrix(
    *,
    eager_policy: Any,
    trt_backbone_policy: Any,
    eager_prepared: Any,
    trt_prepared: Any,
    eager_explicit_head: Any,
    trt_explicit_head: Any,
    initial_actions: Any,
    compile_mode: str,
    warmup: int,
    measured: int,
    refittable_dit_config: Mapping[str, Any],
    trt_backbone_phase: TensorRTPhase,
) -> dict[str, Any]:
    """Run the W84 Backbone, Action Head, and diagonal executor matrices."""

    import torch

    RefittableTensorRTDiT = _load_refittable_tensorrt_dit()

    eager_backbone = eager_policy.model.backbone
    eager_action_model = eager_policy.model.action_head.model
    trt_backbone_action_model = trt_backbone_policy.model.action_head.model
    original_eager_backbone_forward = eager_backbone.forward
    original_eager_action_forward = eager_action_model.forward
    original_trt_action_forward = trt_backbone_action_model.forward
    trt_backbone_forward = trt_backbone_policy.model.backbone.forward
    parameter_contracts_before = {
        "eager_action_head": _parameter_contract(eager_action_model),
        "trt_backbone_action_head": _parameter_contract(trt_backbone_action_model),
    }
    lifecycle = {"memory_before_setup": _cuda_memory()}
    trt_dit = None
    try:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        compile_started = time.perf_counter()
        compiled_backbone_forward = torch.compile(
            original_eager_backbone_forward,
            mode=compile_mode,
            dynamic=False,
        )
        eager_backbone.forward = compiled_backbone_forward
        eager_action_model.forward = original_eager_action_forward
        call_with_explicit_noise(
            eager_policy.model,
            eager_explicit_head,
            eager_prepared,
            initial_actions,
        )
        torch.cuda.synchronize()
        lifecycle["pt2_backbone_first_call_wall_ms"] = (
            time.perf_counter() - compile_started
        ) * 1000
        lifecycle["unique_graphs_after_backbone_first"] = int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"]
        )

        compile_started = time.perf_counter()
        compiled_eager_action_forward = torch.compile(
            original_eager_action_forward,
            mode=compile_mode,
            dynamic=False,
        )
        eager_backbone.forward = original_eager_backbone_forward
        eager_action_model.forward = compiled_eager_action_forward
        call_with_explicit_noise(
            eager_policy.model,
            eager_explicit_head,
            eager_prepared,
            initial_actions,
        )
        torch.cuda.synchronize()
        lifecycle["pt2_eager_policy_head_first_call_wall_ms"] = (
            time.perf_counter() - compile_started
        ) * 1000
        lifecycle["unique_graphs_after_eager_head_first"] = int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"]
        )

        compile_started = time.perf_counter()
        compiled_trt_action_forward = torch.compile(
            original_trt_action_forward,
            mode=compile_mode,
            dynamic=False,
        )
        trt_backbone_action_model.forward = compiled_trt_action_forward
        trt_backbone_phase(
            "w84_pt2_trt_backbone_head_first_call",
            1,
            lambda: call_with_explicit_noise(
                trt_backbone_policy.model,
                trt_explicit_head,
                trt_prepared,
                initial_actions,
            ),
        )
        torch.cuda.synchronize()
        lifecycle["pt2_trt_policy_head_first_call_wall_ms"] = (
            time.perf_counter() - compile_started
        ) * 1000
        lifecycle["unique_graphs_after_all_first_calls"] = int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"]
        )

        trt_backbone_action_model.forward = original_trt_action_forward
        trt_dit_setup_started = time.perf_counter()
        trt_dit = RefittableTensorRTDiT(
            trt_backbone_action_model,
            refittable_dit_config,
        )
        trt_dit.verify_revision(0)
        trt_backbone_action_model.forward = trt_dit
        trt_backbone_phase(
            "w84_refittable_dit_initial_live_probe",
            1,
            lambda: call_with_explicit_noise(
                trt_backbone_policy.model,
                trt_explicit_head,
                trt_prepared,
                initial_actions,
            ),
        )
        torch.cuda.synchronize()
        lifecycle["refittable_dit_setup_and_probe_wall_ms"] = (
            time.perf_counter() - trt_dit_setup_started
        ) * 1000
        lifecycle["synthetic_refit"] = _exercise_refit_lifecycle(trt_dit)
        lifecycle["memory_after_setup"] = _cuda_memory()

        eager_eager = _make_call(
            eager_policy,
            eager_prepared,
            eager_explicit_head,
            initial_actions,
            backbone_forward=original_eager_backbone_forward,
            action_forward=original_eager_action_forward,
        )
        pt2_backbone_eager_head = _make_call(
            eager_policy,
            eager_prepared,
            eager_explicit_head,
            initial_actions,
            backbone_forward=compiled_backbone_forward,
            action_forward=original_eager_action_forward,
        )
        trt_backbone_eager_head = _make_call(
            trt_backbone_policy,
            trt_prepared,
            trt_explicit_head,
            initial_actions,
            backbone_forward=trt_backbone_forward,
            action_forward=original_trt_action_forward,
        )
        trt_backbone_pt2_head = _make_call(
            trt_backbone_policy,
            trt_prepared,
            trt_explicit_head,
            initial_actions,
            backbone_forward=trt_backbone_forward,
            action_forward=compiled_trt_action_forward,
        )
        trt_backbone_refittable_head = _make_call(
            trt_backbone_policy,
            trt_prepared,
            trt_explicit_head,
            initial_actions,
            backbone_forward=trt_backbone_forward,
            action_forward=trt_dit,
        )
        pt2_pt2 = _make_call(
            eager_policy,
            eager_prepared,
            eager_explicit_head,
            initial_actions,
            backbone_forward=compiled_backbone_forward,
            action_forward=compiled_eager_action_forward,
        )

        per_arm_calls = warmup + measured
        backbone_matrix = trt_backbone_phase(
            "w84_backbone_matrix",
            per_arm_calls,
            lambda: measure_stage_matrix(
                {
                    "eager_backbone": eager_eager,
                    "pt2_backbone": pt2_backbone_eager_head,
                    "tensorrt_backbone": trt_backbone_eager_head,
                },
                reference="eager_backbone",
                warmup=warmup,
                measured=measured,
                boundary=(
                    "CUDA-resident model inputs -> pre-final hidden state; "
                    "identical eager Action Head continues to normalized action"
                ),
            ),
        )
        action_head_matrix = trt_backbone_phase(
            "w84_action_head_matrix",
            3 * per_arm_calls,
            lambda: measure_stage_matrix(
                {
                    "eager_action_head": trt_backbone_eager_head,
                    "pt2_action_head": trt_backbone_pt2_head,
                    "refittable_tensorrt_action_head": (trt_backbone_refittable_head),
                },
                reference="eager_action_head",
                warmup=warmup,
                measured=measured,
                boundary=(
                    "retained TensorRT pre-final hidden state + CUDA state/noise -> "
                    "normalized action; PT2/TRT replace DiT, remainder stays PyTorch"
                ),
            ),
        )
        diagonal_matrix = trt_backbone_phase(
            "w84_diagonal_matrix",
            per_arm_calls,
            lambda: measure_stage_matrix(
                {
                    "eager_backbone_eager_head": eager_eager,
                    "pt2_backbone_pt2_head": pt2_pt2,
                    "tensorrt_backbone_refittable_head": (trt_backbone_refittable_head),
                },
                reference="eager_backbone_eager_head",
                warmup=warmup,
                measured=measured,
                boundary=(
                    "preloaded CUDA inputs + explicit fixed noise -> normalized action"
                ),
            ),
        )
        unique_graphs_after_measurement = int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"]
        )
        parameter_contracts_after = {
            "eager_action_head": _parameter_contract(eager_action_model),
            "trt_backbone_action_head": _parameter_contract(trt_backbone_action_model),
        }
        trt_telemetry_before_close = trt_dit.telemetry()

        matrices = {
            "frozen_backbone": backbone_matrix,
            "refittable_action_head": action_head_matrix,
            "whole_model_diagonal": diagonal_matrix,
        }
        gates = {
            "all_outputs_finite": all(
                comparison["finite"]
                for matrix in matrices.values()
                for comparison in matrix["output_comparisons"].values()
            ),
            "stage_boundaries_close": all(
                matrix["max_abs_stage_closure_error_ms"] <= 0.01
                for matrix in matrices.values()
            ),
            "pt2_compiled_graphs_exist": (
                lifecycle["unique_graphs_after_all_first_calls"] > 0
            ),
            "no_pt2_recompile_during_measurement": (
                unique_graphs_after_measurement
                == lifecycle["unique_graphs_after_all_first_calls"]
            ),
            "parameter_and_storage_identity_preserved": (
                parameter_contracts_before == parameter_contracts_after
            ),
            "refit_revisions_complete": (
                trt_telemetry_before_close["active_revision"] == 4
                and trt_telemetry_before_close["phase"] == "idle"
                and lifecycle["synthetic_refit"]["final_source_restored_bitwise"]
            ),
            "refit_engine_is_online_double_slot": (
                trt_telemetry_before_close["online_refit"]
                and len(trt_telemetry_before_close["engines"]) == 2
            ),
        }
        trt_dit.close()
        trt_telemetry_after_close = trt_dit.telemetry()
        gates["refit_engines_closed"] = trt_telemetry_after_close["closed"] and all(
            engine["closed"] for engine in trt_telemetry_after_close["engines"]
        )
        trt_dit = None
        lifecycle["unique_graphs_after_measurement"] = unique_graphs_after_measurement
        return {
            "schema": "rlinf.gr00t-n1d7-b8-executor-matrix.v1",
            "status": "passed" if all(gates.values()) else "failed",
            "scope": {
                "backbone": "immutable ViT+LLM to pre-final hidden state",
                "action_head": (
                    "hot-updateable head; TensorRT arm refits DiT while the "
                    "remaining head stays PyTorch"
                ),
                "ppo_authority": (
                    "standalone timing only; known PT2/TRT DiT ratio/KL failures "
                    "remain external qualification state"
                ),
            },
            "compile_mode": compile_mode,
            "matrices": matrices,
            "lifecycle": lifecycle,
            "refittable_dit_telemetry": {
                "before_close": trt_telemetry_before_close,
                "after_close": trt_telemetry_after_close,
            },
            "gates": gates,
        }
    finally:
        eager_backbone.forward = original_eager_backbone_forward
        eager_action_model.forward = original_eager_action_forward
        trt_backbone_action_model.forward = original_trt_action_forward
        if trt_dit is not None:
            trt_dit.close()
