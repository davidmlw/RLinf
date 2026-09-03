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

"""Capture the official B1 post-processor inputs and actual flow noise."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from resident_b1 import (
    EXPECTED_ENGINES,
    _fixture,
    _load_api,
    _load_policy,
    _observation_manifest,
    _sha256,
)


def _bytes_manifest(value: Any) -> dict[str, Any]:
    import torch

    tensor = value.detach().contiguous()
    host_bytes = tensor.cpu().view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "bytes": len(host_bytes),
        "sha256": hashlib.sha256(host_bytes).hexdigest(),
        "finite": bool(torch.isfinite(tensor).all())
        if tensor.is_floating_point()
        else None,
    }


def _tree_manifest(value: Any, path: str = "") -> dict[str, Any]:
    import torch

    if isinstance(value, torch.Tensor):
        return {path: _bytes_manifest(value)}
    if isinstance(value, dict) or hasattr(value, "items"):
        result = {}
        for key, child in sorted(value.items()):
            child_path = f"{path}.{key}" if path else str(key)
            result.update(_tree_manifest(child, child_path))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, child in enumerate(value):
            result.update(_tree_manifest(child, f"{path}[{index}]"))
        return result
    payload = json.dumps(value, sort_keys=True, default=str).encode()
    return {
        path: {
            "type": type(value).__name__,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    }


def _aggregate_hash(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _rng_manifest() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        "torch_cpu": _bytes_manifest(torch.get_rng_state()),
        "torch_cuda": [
            _bytes_manifest(state) for state in torch.cuda.get_rng_state_all()
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    source = args.source.resolve(strict=True)
    model = args.model.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    engines = args.engines.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    engine_paths = sorted(engines.glob("*.engine"))
    if {path.name for path in engine_paths} != EXPECTED_ENGINES:
        raise RuntimeError("fixture witness requires the exact seven-engine bundle")

    api = _load_api(source)
    api["set_seed"](42)
    policy, model_load_ms = _load_policy(api, model)
    observation = _fixture(api, policy, dataset)
    collated_inputs, _ = api["prepare_model_inputs"](
        policy, observation, return_states=True
    )
    input_manifest = _tree_manifest(collated_inputs)
    setup_started = torch.cuda.Event(enable_timing=True)
    setup_finished = torch.cuda.Event(enable_timing=True)
    setup_started.record()
    api["setup_tensorrt_engines"](policy, str(engines), mode="n17_full_pipeline")
    setup_finished.record()
    torch.cuda.synchronize()

    seed = 20260903
    api["set_seed"](seed)
    rng_before = _rng_manifest()
    captured_noise = []
    original_randn = torch.randn

    def capture_randn(*values: Any, **options: Any) -> Any:
        result = original_randn(*values, **options)
        captured_noise.append(result.detach().clone())
        return result

    torch.randn = capture_randn
    try:
        with torch.inference_mode():
            prediction = policy.model.get_action(**collated_inputs)
        torch.cuda.synchronize()
    finally:
        torch.randn = original_randn
    rng_after = _rng_manifest()
    if len(captured_noise) != 1:
        raise RuntimeError(
            f"expected one flow-noise sample, found {len(captured_noise)}"
        )

    noise = captured_noise[0]
    api["set_seed"](seed)
    replay = original_randn(size=noise.shape, dtype=noise.dtype, device=noise.device)
    if _bytes_manifest(noise)["sha256"] != _bytes_manifest(replay)["sha256"]:
        raise RuntimeError("captured flow noise does not match the seeded replay")
    action_manifest = _tree_manifest(prediction)
    api["close_tensorrt_engines"](policy)
    receipt = {
        "schema": "rlinf.gr00t-n1d7-official-b1-fixture-witness.v1",
        "status": "passed",
        "source_revision": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": str(model),
        "dataset": str(dataset),
        "model_load_ms": model_load_ms,
        "engine_setup_ms": setup_started.elapsed_time(setup_finished),
        "engines": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in engine_paths
        },
        "raw_observation": _observation_manifest(observation),
        "postprocessed_model_inputs": {
            "tensors": input_manifest,
            "aggregate_sha256": _aggregate_hash(input_manifest),
        },
        "flow_noise": {
            "seed": seed,
            "actual": _bytes_manifest(noise),
            "seeded_replay": _bytes_manifest(replay),
            "bitwise_replay": True,
        },
        "rng_state_before_model": rng_before,
        "rng_state_after_model": rng_after,
        "normalized_prediction": {
            "tensors": action_manifest,
            "aggregate_sha256": _aggregate_hash(action_manifest),
        },
    }
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
        print(f"W79 fixture witness failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
