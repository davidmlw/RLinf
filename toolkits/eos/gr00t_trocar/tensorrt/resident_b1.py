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

"""Record raw component and resident whole-call timings for the B1 oracle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EXPECTED_ENGINES = {
    "action_decoder.engine",
    "action_encoder.engine",
    "dit_bf16.engine",
    "llm_bf16.engine",
    "state_encoder.engine",
    "vit.engine",
    "vl_self_attention.engine",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _statistics(samples_ms: list[float]) -> dict[str, float | int]:
    values = [float(value) for value in samples_ms]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("timing samples must be finite and non-empty")
    ordered = sorted(values)

    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent / 100
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    mean = math.fsum(values) / len(values)
    std = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))
    return {
        "count": len(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": mean,
        "std_ms": std,
        "p50_ms": percentile(50),
        "p95_ms": percentile(95),
        "p99_ms": percentile(99),
    }


def _array_manifest(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _observation_manifest(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        group: {key: _array_manifest(value) for key, value in sorted(values.items())}
        if group != "language"
        else dict(sorted(values.items()))
        for group, values in sorted(observation.items())
    }


def _flatten_action(action: dict[str, Any]) -> Any:
    import numpy as np

    return np.concatenate(
        [
            np.asarray(action[key], dtype=np.float64).reshape(-1)
            for key in sorted(action)
        ]
    )


def _action_manifest(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "keys": {key: _array_manifest(value) for key, value in sorted(action.items())},
        "aggregate": _array_manifest(_flatten_action(action)),
    }


def _compare_actions(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np

    bitwise_equal = sorted(reference) == sorted(candidate) and all(
        np.ascontiguousarray(reference[key]).dtype
        == np.ascontiguousarray(candidate[key]).dtype
        and np.ascontiguousarray(reference[key]).shape
        == np.ascontiguousarray(candidate[key]).shape
        and np.ascontiguousarray(reference[key]).tobytes()
        == np.ascontiguousarray(candidate[key]).tobytes()
        for key in reference
    )
    lhs = _flatten_action(reference)
    rhs = _flatten_action(candidate)
    if lhs.shape != rhs.shape:
        raise RuntimeError(f"action shapes differ: {lhs.shape} != {rhs.shape}")
    delta = rhs - lhs
    denominator = max(float(np.linalg.norm(lhs)), np.finfo(np.float64).eps)
    cosine_denominator = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    cosine = (
        1.0
        if cosine_denominator == 0 and np.array_equal(lhs, rhs)
        else (
            float(np.dot(lhs, rhs) / cosine_denominator)
            if cosine_denominator
            else math.nan
        )
    )
    return {
        "finite": bool(np.isfinite(lhs).all() and np.isfinite(rhs).all()),
        "bitwise_equal": bitwise_equal,
        "cosine": cosine,
        "mean_abs": float(np.abs(delta).mean()),
        "max_abs": float(np.abs(delta).max()),
        "relative_l2": float(np.linalg.norm(delta) / denominator),
    }


def _measure_call(
    policy: Any, observation: dict[str, Any]
) -> tuple[dict[str, Any], float]:
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    action, _ = policy.get_action(observation)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return action, elapsed_ms


def _fixed_action(
    api: dict[str, Any], policy: Any, observation: dict[str, Any], seed: int
) -> tuple[dict[str, Any], float]:
    api["set_seed"](seed)
    return _measure_call(policy, observation)


def _whole_call(
    policy: Any, observation: dict[str, Any], warmup: int, iterations: int
) -> dict[str, Any]:
    warmup_ms = [_measure_call(policy, observation)[1] for _ in range(warmup)]
    measured_ms = [_measure_call(policy, observation)[1] for _ in range(iterations)]
    return {
        "boundary": "cuda_sync; Gr00tPolicy.get_action(observation); cuda_sync",
        "warmup_samples_ms": warmup_ms,
        "measured_samples_ms": measured_ms,
        "statistics": _statistics(measured_ms),
    }


def _components(
    api: dict[str, Any], policy: Any, observation: dict[str, Any], warmup: int
) -> dict[str, Any]:
    measured = api["benchmark_components"](
        policy, observation, num_iterations=20, warmup=warmup
    )
    raw = {name: values.tolist() for name, values in measured.items()}
    raw["constructed_sum"] = (
        measured["data_processing"] + measured["backbone"] + measured["action_head"]
    ).tolist()
    return {
        "boundary": "upstream benchmark_inference.benchmark_components",
        "warmup": warmup,
        "measured": 20,
        "raw_samples_ms": raw,
        "statistics": {name: _statistics(values) for name, values in raw.items()},
    }


def _load_api(source: Path) -> dict[str, Any]:
    deployment = source / "scripts/deployment"
    sys.path.insert(0, str(deployment))
    from benchmark_inference import (  # noqa: PLC0415
        EmbodimentTag,
        Gr00tPolicy,
        LeRobotEpisodeLoader,
        benchmark_components,
        extract_step_data,
        prepare_model_inputs,
        set_seed,
    )
    from trt_model_forward import (  # noqa: PLC0415
        close_tensorrt_engines,
        setup_tensorrt_engines,
    )

    return {
        "EmbodimentTag": EmbodimentTag,
        "Gr00tPolicy": Gr00tPolicy,
        "LeRobotEpisodeLoader": LeRobotEpisodeLoader,
        "benchmark_components": benchmark_components,
        "close_tensorrt_engines": close_tensorrt_engines,
        "extract_step_data": extract_step_data,
        "prepare_model_inputs": prepare_model_inputs,
        "set_seed": set_seed,
        "setup_tensorrt_engines": setup_tensorrt_engines,
    }


def _load_policy(api: dict[str, Any], model: Path) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    policy = api["Gr00tPolicy"](
        model_path=str(model),
        embodiment_tag=api["EmbodimentTag"].LIBERO_PANDA,
        device="cuda",
        strict=True,
    )
    return policy, (time.perf_counter_ns() - started) / 1_000_000


def _fixture(api: dict[str, Any], policy: Any, dataset_root: Path) -> dict[str, Any]:
    import numpy as np

    modality_config = policy.get_modality_config()
    dataset = api["LeRobotEpisodeLoader"](
        dataset_path=str(dataset_root), modality_configs=modality_config
    )
    step = api["extract_step_data"](
        dataset[0],
        step_index=0,
        modality_configs=modality_config,
        embodiment_tag=api["EmbodimentTag"].LIBERO_PANDA,
        allow_padding=False,
    )
    return {
        "video": {key: np.stack(value)[None] for key, value in step.images.items()},
        "state": {key: value[None] for key, value in step.states.items()},
        "language": {modality_config["language"].modality_keys[0]: [[step.text]]},
    }


def _release() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    source = args.source.resolve(strict=True)
    model = args.model.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    engines = args.engines.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    api = _load_api(source)
    engine_paths = sorted(engines.glob("*.engine"))
    if {path.name for path in engine_paths} != EXPECTED_ENGINES:
        raise RuntimeError("resident benchmark requires the exact seven-engine bundle")
    seed = 20260903
    receipt: dict[str, Any] = {
        "schema": "rlinf.gr00t-n1d7-official-b1-resident.v1",
        "status": "running",
        "source_revision": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": str(model),
        "dataset": str(dataset),
        "engines": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in engine_paths
        },
        "device": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "fixed_noise_seed": seed,
        "arms": {},
    }

    api["set_seed"](42)
    eager, eager_load_ms = _load_policy(api, model)
    observation = _fixture(api, eager, dataset)
    receipt["observation"] = _observation_manifest(observation)
    eager_action, eager_first_ms = _fixed_action(api, eager, observation, seed)
    api["set_seed"](42)
    receipt["arms"]["eager"] = {
        "model_load_ms": eager_load_ms,
        "first_fixed_noise_whole_call_ms": eager_first_ms,
        "fixed_noise_action": _action_manifest(eager_action),
        "whole_call": _whole_call(eager, observation, 10, 30),
        "components": _components(api, eager, observation, 5),
    }
    del eager
    _release()

    api["set_seed"](42)
    compiled, compiled_load_ms = _load_policy(api, model)
    compile_started = time.perf_counter_ns()
    compiled.model.action_head.model.forward = torch.compile(
        compiled.model.action_head.model.forward, mode="max-autotune"
    )
    compile_wrap_ms = (time.perf_counter_ns() - compile_started) / 1_000_000
    compiled_action, compiled_first_ms = _fixed_action(api, compiled, observation, seed)
    api["set_seed"](42)
    receipt["arms"]["compile"] = {
        "model_load_ms": compiled_load_ms,
        "compile_wrap_ms": compile_wrap_ms,
        "first_fixed_noise_whole_call_ms": compiled_first_ms,
        "fixed_noise_action": _action_manifest(compiled_action),
        "vs_eager": _compare_actions(eager_action, compiled_action),
        "whole_call": _whole_call(compiled, observation, 10, 30),
        "components": _components(api, compiled, observation, 5),
    }
    del compiled
    _release()

    api["set_seed"](42)
    trt_policy, trt_load_ms = _load_policy(api, model)
    setup_started = time.perf_counter_ns()
    api["setup_tensorrt_engines"](trt_policy, str(engines), mode="n17_full_pipeline")
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter_ns() - setup_started) / 1_000_000
    trt_action, trt_first_ms = _fixed_action(api, trt_policy, observation, seed)
    repeated_action, repeated_ms = _fixed_action(api, trt_policy, observation, seed)
    api["set_seed"](42)
    receipt["arms"]["full_tensorrt"] = {
        "model_load_ms": trt_load_ms,
        "engine_deserialize_context_setup_ms": setup_ms,
        "first_fixed_noise_whole_call_ms": trt_first_ms,
        "repeat_fixed_noise_whole_call_ms": repeated_ms,
        "fixed_noise_action": _action_manifest(trt_action),
        "fixed_noise_repeat": _compare_actions(trt_action, repeated_action),
        "vs_eager": _compare_actions(eager_action, trt_action),
        "whole_call": _whole_call(trt_policy, observation, 10, 30),
        "components": _components(api, trt_policy, observation, 5),
    }
    api["close_tensorrt_engines"](trt_policy)
    del trt_policy
    _release()
    receipt["status"] = "passed"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--engines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as error:
        print(f"W79 resident B1 failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
