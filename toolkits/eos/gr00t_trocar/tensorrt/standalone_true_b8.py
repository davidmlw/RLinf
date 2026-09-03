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
import hashlib
import json
import math
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from common_boundary_b8 import (
    call_with_explicit_noise,
    cuda_event_call,
    make_explicit_noise_head,
    prepare_cuda_inputs,
    prepared_manifest,
    tensor_manifest,
)
from persistent_trt import PersistentEngine
from resident_b1 import _compare_actions
from trocar_b8_model_view import LANGUAGE_KEY, STATE_ACTION_ORDER


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema or value.get("status") != "passed":
        raise RuntimeError(f"unqualified receipt {path}: {value.get('schema')}")
    return value


def _require_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def _verified_provenance(
    source: Path,
    model: Path,
    collated: Path,
    raw: Path,
    fixture_path: Path,
    export_path: Path,
    engine_path: Path,
    engines: Path,
) -> dict[str, Any]:
    fixture = _load_json(fixture_path, "rlinf.gr00t-n1d7-trocar-true-b8-fixture.v1")
    export = _load_json(export_path, "rlinf.gr00t-n1d7-trocar-true-b8-onnx.v1")
    engine = _load_json(engine_path, "rlinf.gr00t-n1d7-trocar-true-b8-engines.v1")
    if fixture.get("batch_size") != 8 or fixture.get("b1x8") is not False:
        raise RuntimeError("fixture is not a qualified genuine B8 call")
    if export.get("batch_size") != 8 or export.get("b1x8") is not False:
        raise RuntimeError("export receipt is not a qualified genuine B8 export")
    if (
        engine.get("static_batch") != 8
        or engine.get("sequence_opt") != 208
        or engine.get("silent_fallback") is not False
    ):
        raise RuntimeError("engine receipt does not enforce static B8/no fallback")

    artifacts = {
        "fixture_receipt": _require_hash(
            fixture_path,
            export["metadata"]["fixture_receipt_sha256"],
            "fixture receipt",
        ),
        "collated": _require_hash(
            collated, fixture["artifacts"]["collated-inputs.pt"]["sha256"], "collated"
        ),
        "raw_observation": _require_hash(
            raw, fixture["artifacts"]["raw-observation.npz"]["sha256"], "raw input"
        ),
        "model_view_receipt": _require_hash(
            model / "rlinf-model-view.json",
            fixture["model_view_receipt_sha256"],
            "model view receipt",
        ),
        "export_receipt": {
            "path": str(export_path),
            "bytes": export_path.stat().st_size,
            "sha256": _sha256(export_path),
        },
        "engine_receipt": {
            "path": str(engine_path),
            "bytes": engine_path.stat().st_size,
            "sha256": _sha256(engine_path),
        },
    }
    if export["metadata"]["collated_sha256"] != artifacts["collated"]["sha256"]:
        raise RuntimeError("export receipt does not bind the supplied collated input")
    if (
        export["metadata"]["model_view_receipt_sha256"]
        != artifacts["model_view_receipt"]["sha256"]
    ):
        raise RuntimeError("export receipt does not bind the supplied model view")

    export_files = {}
    for name, metadata in export["files"].items():
        export_files[name] = _require_hash(
            export_path.parent / name, metadata["sha256"], f"export {name}"
        )
    if (
        export_files["export_metadata.json"]["sha256"]
        != engine["export_metadata_sha256"]
    ):
        raise RuntimeError("engine receipt does not bind the exported metadata")

    expected_engines = {"vit.engine", "llm_bf16.engine"}
    if set(engine["engines"]) != expected_engines:
        raise RuntimeError(
            "engine receipt does not contain the exact two-engine bundle"
        )
    actual_engine_files = {path.name for path in engines.glob("*.engine")}
    if actual_engine_files != expected_engines:
        raise RuntimeError(
            "runtime directory does not contain the exact two-engine bundle"
        )
    engine_files = {
        name: _require_hash(
            engines / name, engine["engines"][name]["sha256"], f"engine {name}"
        )
        for name in sorted(expected_engines)
    }
    engine_metadata = _require_hash(
        engines / "export_metadata.json",
        engine["export_metadata_sha256"],
        "runtime export metadata",
    )
    source_revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        source_revision != fixture["source_revision"]
        or source_revision != export["source_revision"]
    ):
        raise RuntimeError("Isaac-GR00T revision differs across fixture/export/runtime")
    return {
        "status": "passed",
        "isaac_gr00t_revision": source_revision,
        "rlinf_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "artifacts": artifacts,
        "export_files": export_files,
        "engine_files": engine_files,
        "engine_metadata": engine_metadata,
    }


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

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
        "device_used_bytes": total_bytes - free_bytes,
        "device_used_source": "cudaMemGetInfo",
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


def _fixed_model_call_with_captured_noise(
    policy: Any, inputs: Any, seed: int
) -> tuple[Any, Any]:
    """Capture the deployment head's only random tensor outside timed regions."""

    import torch

    original_randn = torch.randn
    captured = []

    def capture_randn(*args: Any, **kwargs: Any) -> Any:
        value = original_randn(*args, **kwargs)
        captured.append(value.detach().clone())
        return value

    torch.randn = capture_randn
    try:
        action, _backbone, _output = _fixed_model_call(policy, inputs, seed)
    finally:
        torch.randn = original_randn
    if len(captured) != 1:
        raise RuntimeError(
            f"expected exactly one deployment flow-noise tensor, found {len(captured)}"
        )
    return action, captured[0]


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


def _timed_model_call(policy: Any, inputs: Any) -> float:
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    with torch.inference_mode():
        policy.model.get_action(inputs)
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - started) / 1_000_000


def _timed_whole_call(policy: Any, observation: dict[str, Any]) -> float:
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    policy.get_action(observation)
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - started) / 1_000_000


def _set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _paired_timing(
    eager: Any,
    hybrid: Any,
    argument: Any,
    call: Any,
    warmup: int,
    measured: int,
    seed: int,
    boundary: str,
) -> dict[str, Any]:
    """Run counterbalanced AB/BA pairs with identical per-arm noise seeds."""

    order = []
    for index in range(warmup):
        pair = (("eager", eager), ("hybrid", hybrid))
        if index % 2:
            pair = tuple(reversed(pair))
        for _name, policy in pair:
            _set_seed(seed + index)
            call(policy, argument)

    samples = {"eager": [], "hybrid": []}
    for index in range(measured):
        pair = (("eager", eager), ("hybrid", hybrid))
        if index % 2:
            pair = tuple(reversed(pair))
        order.append([name for name, _policy in pair])
        for name, policy in pair:
            _set_seed(seed + warmup + index)
            samples[name].append(call(policy, argument))
    deltas = [
        reference - candidate
        for reference, candidate in zip(samples["eager"], samples["hybrid"])
    ]
    eager_stats = _statistics(samples["eager"])
    hybrid_stats = _statistics(samples["hybrid"])
    return {
        "boundary": boundary,
        "order_policy": "alternating AB/BA",
        "orders": order,
        "same_noise_seed_per_pair": True,
        "eager": eager_stats,
        "hybrid": hybrid_stats,
        "eager_minus_hybrid_ms": _statistics(deltas),
        "speedup": eager_stats["mean_ms"] / hybrid_stats["mean_ms"],
        "latency_reduction_fraction": (eager_stats["mean_ms"] - hybrid_stats["mean_ms"])
        / eager_stats["mean_ms"],
    }


def _paired_cuda_stage_timing(
    arms: dict[str, Any],
    warmup: int,
    measured: int,
    boundary: str,
) -> dict[str, Any]:
    """Run counterbalanced CUDA-event calls and retain natural stage timings."""

    names = tuple(arms)
    if len(names) != 2:
        raise ValueError("paired CUDA timing requires exactly two arms")
    for index in range(warmup):
        order = names if index % 2 == 0 else tuple(reversed(names))
        for name in order:
            arms[name]()

    orders = []
    samples = {
        name: {stage: [] for stage in ("backbone_ms", "action_head_ms", "total_ms")}
        for name in names
    }
    outputs = {}
    for index in range(measured):
        order = names if index % 2 == 0 else tuple(reversed(names))
        orders.append(list(order))
        for name in order:
            output, stages = arms[name]()
            outputs[name] = output.detach()
            for stage, value in stages.items():
                samples[name][stage].append(value)

    reference, candidate = names
    deltas = [
        left - right
        for left, right in zip(
            samples[reference]["total_ms"], samples[candidate]["total_ms"]
        )
    ]
    return {
        "boundary": boundary,
        "order_policy": "alternating AB/BA",
        "orders": orders,
        "warmup": warmup,
        "measured": measured,
        "arms": {
            name: {stage: _statistics(values) for stage, values in stages.items()}
            for name, stages in samples.items()
        },
        "reference_minus_candidate_ms": _statistics(deltas),
        "speedup": (
            statistics.fmean(samples[reference]["total_ms"])
            / statistics.fmean(samples[candidate]["total_ms"])
        ),
        "outputs": outputs,
    }


def _engine_counts(vit_engine: Any, llm_engine: Any) -> dict[str, int]:
    return {
        "vit": vit_engine.execute_count,
        "llm": llm_engine.execute_count,
    }


def _engine_phase(
    phases: dict[str, Any],
    name: str,
    vit_engine: Any,
    llm_engine: Any,
    expected: int,
    call: Any,
) -> Any:
    before = _engine_counts(vit_engine, llm_engine)
    result = call()
    after = _engine_counts(vit_engine, llm_engine)
    delta = {key: after[key] - before[key] for key in before}
    phases[name] = {
        "before": before,
        "after": after,
        "delta": delta,
        "expected": expected,
    }
    if delta != {"vit": expected, "llm": expected}:
        raise RuntimeError(f"TRT execute delta mismatch in {name}: {delta}")
    return result


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
    export_receipt = args.export_receipt.resolve(strict=True)
    engine_receipt = args.engine_receipt.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = _verified_provenance(
        source,
        model,
        collated_path,
        raw_path,
        fixture_receipt,
        export_receipt,
        engine_receipt,
        engines,
    )

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
    vit_engine = hybrid.model.backbone.vit_engine
    llm_engine = hybrid.model.backbone.llm_engine
    hybrid_structure = {
        "vit_engine_exact_type": type(vit_engine) is PersistentEngine,
        "llm_engine_exact_type": type(llm_engine) is PersistentEngine,
        "pytorch_vit_removed": not hasattr(hybrid.model.backbone.model.model, "visual"),
        "pytorch_llm_layers_removed": not hasattr(
            hybrid.model.backbone.model.model.language_model, "layers"
        ),
        "pytorch_llm_norm_removed": not hasattr(
            hybrid.model.backbone.model.model.language_model, "norm"
        ),
        "trt_backbone_forward_installed": (
            getattr(hybrid.model.backbone.forward, "func", None)
            is trt_model_forward.qwen3_backbone_full_trt_forward
        ),
    }
    if not all(hybrid_structure.values()):
        raise RuntimeError(
            f"hybrid structure has an eager fallback: {hybrid_structure}"
        )
    engine_phases: dict[str, Any] = {}
    hybrid_first_started = time.perf_counter_ns()
    hybrid_action, hybrid_backbone, _ = _engine_phase(
        engine_phases,
        "first_fixed_model_call",
        vit_engine,
        llm_engine,
        1,
        lambda: _fixed_model_call(hybrid, inputs, args.seed),
    )
    hybrid_first_ms = (time.perf_counter_ns() - hybrid_first_started) / 1_000_000
    hybrid_public = _public_action(hybrid, hybrid_action, observation)
    torch.cuda.synchronize()
    hybrid_vit = vit_engine.last_outputs["image_embeds"].detach().cpu().clone()

    repeat_action, _repeat_backbone, _ = _engine_phase(
        engine_phases,
        "repeat_fixed_model_call",
        vit_engine,
        llm_engine,
        1,
        lambda: _fixed_model_call(hybrid, inputs, args.seed),
    )
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
    hybrid_components = _engine_phase(
        engine_phases,
        "components",
        vit_engine,
        llm_engine,
        args.component_warmup + args.component_measured,
        lambda: _measure_components(
            hybrid, observation, args.component_warmup, args.component_measured
        ),
    )
    hybrid_model_timing = _engine_phase(
        engine_phases,
        "sequential_model_timing",
        vit_engine,
        llm_engine,
        args.warmup + args.measured,
        lambda: _measure_model(hybrid, inputs, args.warmup, args.measured),
    )
    hybrid_whole_timing = _engine_phase(
        engine_phases,
        "sequential_whole_timing",
        vit_engine,
        llm_engine,
        args.warmup + args.measured,
        lambda: _measure_whole(hybrid, observation, args.warmup, args.measured),
    )

    hybrid_model_memory = _engine_phase(
        engine_phases,
        "model_memory_probe",
        vit_engine,
        llm_engine,
        args.memory_iterations,
        lambda: _measure_model_memory(hybrid, inputs, args.memory_iterations),
    )
    hybrid_whole_memory = _engine_phase(
        engine_phases,
        "whole_memory_probe",
        vit_engine,
        llm_engine,
        args.memory_iterations,
        lambda: _measure_whole_memory(hybrid, observation, args.memory_iterations),
    )

    paired_eager = _load_policy(source, model)
    torch.cuda.synchronize()
    paired_model_timing = _engine_phase(
        engine_phases,
        "paired_model_timing",
        vit_engine,
        llm_engine,
        args.paired_warmup + args.paired_measured,
        lambda: _paired_timing(
            paired_eager,
            hybrid,
            inputs,
            _timed_model_call,
            args.paired_warmup,
            args.paired_measured,
            args.seed + 10_000,
            "counterbalanced cuda_sync; model.get_action(collated); cuda_sync",
        ),
    )
    paired_whole_timing = _engine_phase(
        engine_phases,
        "paired_whole_timing",
        vit_engine,
        llm_engine,
        args.paired_warmup + args.paired_measured,
        lambda: _paired_timing(
            paired_eager,
            hybrid,
            observation,
            _timed_whole_call,
            args.paired_warmup,
            args.paired_measured,
            args.seed + 20_000,
            "counterbalanced cuda_sync; policy.get_action(raw); cuda_sync",
        ),
    )

    common_seed = args.seed + 30_000
    eager_prepared = prepare_cuda_inputs(paired_eager.model, inputs)
    hybrid_prepared = prepare_cuda_inputs(hybrid.model, inputs)
    eager_prepared_manifest = prepared_manifest(*eager_prepared)
    hybrid_prepared_manifest = prepared_manifest(*hybrid_prepared)
    if eager_prepared_manifest != hybrid_prepared_manifest:
        raise RuntimeError("eager and hybrid CUDA-prepared inputs differ")

    eager_explicit_head = make_explicit_noise_head(paired_eager.model.action_head)
    hybrid_explicit_head = make_explicit_noise_head(hybrid.model.action_head)
    eager_official_action, initial_actions = _fixed_model_call_with_captured_noise(
        paired_eager, inputs, common_seed
    )
    hybrid_official_action, hybrid_initial_actions = _engine_phase(
        engine_phases,
        "common_boundary_official_noise_capture",
        vit_engine,
        llm_engine,
        1,
        lambda: _fixed_model_call_with_captured_noise(hybrid, inputs, common_seed),
    )
    initial_actions_manifest = tensor_manifest(initial_actions)
    if initial_actions_manifest != tensor_manifest(hybrid_initial_actions):
        raise RuntimeError("eager and hybrid official calls generated different noise")

    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state().clone()
    eager_explicit_action = call_with_explicit_noise(
        paired_eager.model,
        eager_explicit_head,
        eager_prepared,
        initial_actions,
    )
    torch.cuda.synchronize()
    hybrid_explicit_action = _engine_phase(
        engine_phases,
        "common_boundary_explicit_equivalence",
        vit_engine,
        llm_engine,
        1,
        lambda: call_with_explicit_noise(
            hybrid.model,
            hybrid_explicit_head,
            hybrid_prepared,
            initial_actions,
        ),
    )
    torch.cuda.synchronize()
    explicit_rng_unchanged = bool(
        torch.equal(cpu_rng_before, torch.random.get_rng_state())
        and torch.equal(cuda_rng_before, torch.cuda.get_rng_state())
    )
    noise_unchanged = initial_actions_manifest == tensor_manifest(initial_actions)

    common_eager_hybrid = _engine_phase(
        engine_phases,
        "common_boundary_eager_hybrid_timing",
        vit_engine,
        llm_engine,
        args.common_warmup + args.common_measured,
        lambda: _paired_cuda_stage_timing(
            {
                "pytorch_eager": lambda: cuda_event_call(
                    paired_eager.model,
                    eager_explicit_head,
                    eager_prepared,
                    initial_actions,
                ),
                "tensorrt_backbone_eager_head": lambda: cuda_event_call(
                    hybrid.model,
                    hybrid_explicit_head,
                    hybrid_prepared,
                    initial_actions,
                ),
            },
            args.common_warmup,
            args.common_measured,
            (
                "preloaded contiguous CUDA tensors + explicit initial noise -> "
                "normalized action; one natural call with CUDA-event stages"
            ),
        ),
    )
    common_eager_hybrid_outputs = common_eager_hybrid.pop("outputs")

    action_model = hybrid.model.action_head.model
    original_action_model_forward = action_model.forward
    torch._dynamo.reset()
    compile_started = time.perf_counter_ns()
    compiled_action_model_forward = torch.compile(
        original_action_model_forward, mode=args.compile_mode
    )
    action_model.forward = compiled_action_model_forward
    compiled_first_action = _engine_phase(
        engine_phases,
        "common_boundary_compile_first_call",
        vit_engine,
        llm_engine,
        1,
        lambda: call_with_explicit_noise(
            hybrid.model,
            hybrid_explicit_head,
            hybrid_prepared,
            initial_actions,
        ),
    )
    torch.cuda.synchronize()
    compile_first_call_wall_ms = (time.perf_counter_ns() - compile_started) / 1_000_000
    unique_graphs_after_first = int(
        torch._dynamo.utils.counters["stats"]["unique_graphs"]
    )

    def hybrid_eager_head_call() -> tuple[Any, dict[str, float]]:
        action_model.forward = original_action_model_forward
        return cuda_event_call(
            hybrid.model,
            hybrid_explicit_head,
            hybrid_prepared,
            initial_actions,
        )

    def hybrid_compiled_head_call() -> tuple[Any, dict[str, float]]:
        action_model.forward = compiled_action_model_forward
        return cuda_event_call(
            hybrid.model,
            hybrid_explicit_head,
            hybrid_prepared,
            initial_actions,
        )

    common_head_executor = _engine_phase(
        engine_phases,
        "common_boundary_head_executor_timing",
        vit_engine,
        llm_engine,
        2 * (args.common_warmup + args.common_measured),
        lambda: _paired_cuda_stage_timing(
            {
                "tensorrt_backbone_eager_head": hybrid_eager_head_call,
                "tensorrt_backbone_compile_dit_head": hybrid_compiled_head_call,
            },
            args.common_warmup,
            args.common_measured,
            (
                "preloaded contiguous CUDA tensors + explicit initial noise -> "
                "normalized action; identical TRT backbone; eager versus "
                "torch.compile DiT forward"
            ),
        ),
    )
    action_model.forward = original_action_model_forward
    common_head_executor_outputs = common_head_executor.pop("outputs")
    unique_graphs_after_measurement = int(
        torch._dynamo.utils.counters["stats"]["unique_graphs"]
    )

    common_comparisons = {
        "eager_official_vs_explicit": _compare_array(
            eager_official_action.float().numpy(),
            eager_explicit_action.detach().float().cpu().numpy(),
        ),
        "hybrid_official_vs_explicit": _compare_array(
            hybrid_official_action.float().numpy(),
            hybrid_explicit_action.detach().float().cpu().numpy(),
        ),
        "eager_vs_hybrid_explicit": _compare_array(
            common_eager_hybrid_outputs["pytorch_eager"].float().cpu().numpy(),
            common_eager_hybrid_outputs["tensorrt_backbone_eager_head"]
            .float()
            .cpu()
            .numpy(),
        ),
        "hybrid_eager_vs_compile_dit": _compare_array(
            common_head_executor_outputs["tensorrt_backbone_eager_head"]
            .float()
            .cpu()
            .numpy(),
            common_head_executor_outputs["tensorrt_backbone_compile_dit_head"]
            .float()
            .cpu()
            .numpy(),
        ),
        "compile_first_vs_measured": _compare_array(
            compiled_first_action.detach().float().cpu().numpy(),
            common_head_executor_outputs["tensorrt_backbone_compile_dit_head"]
            .float()
            .cpu()
            .numpy(),
        ),
    }
    common_gates = {
        "prepared_cuda_inputs_identical": (
            eager_prepared_manifest == hybrid_prepared_manifest
        ),
        "explicit_noise_shape": initial_actions_manifest["shape"] == [8, 40, 132],
        "explicit_noise_is_cuda_contiguous": (
            initial_actions_manifest["device"] == "cuda"
            and initial_actions_manifest["contiguous"]
        ),
        "explicit_path_does_not_advance_rng": explicit_rng_unchanged,
        "explicit_noise_not_mutated": noise_unchanged,
        "eager_official_equivalence": common_comparisons["eager_official_vs_explicit"][
            "bitwise_equal"
        ],
        "hybrid_official_equivalence": common_comparisons[
            "hybrid_official_vs_explicit"
        ]["bitwise_equal"],
        "compiled_head_action": (
            common_comparisons["hybrid_eager_vs_compile_dit"]["finite"]
            and common_comparisons["hybrid_eager_vs_compile_dit"]["cosine"] >= 0.999
            and common_comparisons["hybrid_eager_vs_compile_dit"]["mean_abs"] <= 0.005
            and common_comparisons["hybrid_eager_vs_compile_dit"]["max_abs"] <= 0.05
        ),
        "compiled_head_stable": common_comparisons["compile_first_vs_measured"][
            "bitwise_equal"
        ],
        "no_compile_rebuild_during_measurement": (
            unique_graphs_after_first > 0
            and unique_graphs_after_measurement == unique_graphs_after_first
        ),
    }
    paired_eager = None
    gc.collect()
    telemetry_before_close = {
        "vit": vit_engine.telemetry(),
        "llm": llm_engine.telemetry(),
        "allocation_floor_after_fixed_calls": allocation_floor,
        "execute_phases": engine_phases,
        "hybrid_structure": hybrid_structure,
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
    trt_model_forward.close_tensorrt_engines(hybrid)
    telemetry_after_close = {
        "vit": vit_engine.telemetry(),
        "llm": llm_engine.telemetry(),
    }
    expected_execute_count = sum(phase["expected"] for phase in engine_phases.values())
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
        "provenance_chain": provenance["status"] == "passed",
        "no_eager_backbone_fallback": all(hybrid_structure.values()),
        "engine_execution": (
            telemetry_before_close["vit"]["execute_count"]
            == expected_execute_count
            == telemetry_before_close["llm"]["execute_count"]
            and all(
                phase["delta"] == {"vit": phase["expected"], "llm": phase["expected"]}
                for phase in engine_phases.values()
            )
        ),
        "persistent_lifecycle": (
            telemetry_before_close["vit"]["load_count"] == 1
            and telemetry_before_close["llm"]["load_count"] == 1
            and telemetry_before_close["vit"]["context_count"] == 1
            and telemetry_before_close["llm"]["context_count"] == 1
            and telemetry_before_close["vit"]["resident_host_sync_count"] == 0
            and telemetry_before_close["llm"]["resident_host_sync_count"] == 0
            and telemetry_before_close["vit"]["allocation_count"]
            == allocation_floor["vit"]
            and telemetry_before_close["llm"]["allocation_count"]
            == allocation_floor["llm"]
            and telemetry_before_close["vit"]["event_record_count"]
            == telemetry_before_close["vit"]["execute_count"]
            and telemetry_before_close["llm"]["event_record_count"]
            == telemetry_before_close["llm"]["execute_count"]
            and telemetry_before_close["vit"]["stream_handle"]
            == telemetry_before_close["llm"]["stream_handle"]
        ),
        "close_lifetime": (
            telemetry_after_close["vit"]["closed"]
            and telemetry_after_close["llm"]["closed"]
            and telemetry_after_close["vit"]["close_event_sync_count"] == 1
            and telemetry_after_close["llm"]["close_event_sync_count"] == 1
        ),
    }
    status = (
        "passed" if all(gates.values()) and all(common_gates.values()) else "failed"
    )
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
        "provenance": provenance,
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
            "counterbalanced_paired": {
                "model_only": paired_model_timing,
                "whole_call": paired_whole_timing,
            },
            "common_cuda_boundary": {
                "contract": (
                    "preloaded contiguous CUDA tensors + explicit initial noise "
                    "-> normalized deployment action"
                ),
                "excludes": [
                    "raw observation preprocessing",
                    "CPU-to-GPU input transfer",
                    "implicit RNG generation",
                    "public action decode",
                    "PPO transition noise, logprob, and value",
                ],
                "prepared_inputs": eager_prepared_manifest,
                "initial_actions": initial_actions_manifest,
                "eager_vs_hybrid": common_eager_hybrid,
                "head_executor": common_head_executor,
                "compile": {
                    "mode": args.compile_mode,
                    "first_call_wall_ms": compile_first_call_wall_ms,
                    "unique_graphs_after_first": unique_graphs_after_first,
                    "unique_graphs_after_measurement": (
                        unique_graphs_after_measurement
                    ),
                },
                "comparisons": common_comparisons,
                "gates": common_gates,
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
        "telemetry": {
            "expected_execute_count_per_engine": expected_execute_count,
            "before_close": telemetry_before_close,
            "after_close": telemetry_after_close,
        },
    }
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if status != "passed":
        raise RuntimeError(
            "standalone true-B8 gates failed: "
            f"legacy={gates}, common_boundary={common_gates}"
        )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engines", type=Path, required=True)
    parser.add_argument("--collated", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--fixture-receipt", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--engine-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=30)
    parser.add_argument("--component-warmup", type=int, default=5)
    parser.add_argument("--component-measured", type=int, default=20)
    parser.add_argument("--memory-iterations", type=int, default=3)
    parser.add_argument("--paired-warmup", type=int, default=10)
    parser.add_argument("--paired-measured", type=int, default=30)
    parser.add_argument("--common-warmup", type=int, default=10)
    parser.add_argument("--common-measured", type=int, default=30)
    parser.add_argument("--compile-mode", default="max-autotune")
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
