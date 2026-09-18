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

"""Qualify N1.5 eager versus persistent TensorRT backbone on one B8 fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import OmegaConf
from transformers.feature_extraction_utils import BatchFeature


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_config(config_root: Path, model_path: Path) -> Any:
    model = OmegaConf.load(config_root / "model/gr00t.yaml")
    root = OmegaConf.create({"actor": {"model": model}})
    cfg = root.actor.model
    cfg.model_path = str(model_path)
    cfg.embodiment_tag = "isaaclab_franka"
    cfg.obs_converter_type = "isaaclab_stack_cube"
    cfg.num_action_chunks = 16
    cfg.skip_unused_lm_head = True
    OmegaConf.resolve(root)
    return cfg


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    left = reference.detach().float().flatten()
    right = candidate.detach().float().flatten()
    difference = (left - right).abs()
    norm = torch.linalg.vector_norm(left)
    return {
        "finite": bool(torch.isfinite(right).all()),
        "cosine": float(torch.nn.functional.cosine_similarity(left, right, dim=0)),
        "mean_abs": float(difference.mean()),
        "max_abs": float(difference.max()),
        "relative_l2": float(torch.linalg.vector_norm(left - right) / norm),
    }


def _clone_feature(value: BatchFeature) -> BatchFeature:
    return BatchFeature(data={name: tensor.clone() for name, tensor in value.items()})


def _seed(value: int) -> None:
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _benchmark(
    function: Callable[[], Any],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)))
    ordered = sorted(samples)
    mean = statistics.fmean(samples)
    p50 = statistics.median(samples)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return {
        "warmup": warmup,
        "iterations": iterations,
        "samples_ms": samples,
        "mean_ms": mean,
        "p50_ms": p50,
        "p95_ms": p95,
        "sample_std_ms": statistics.stdev(samples),
        "cv": statistics.stdev(samples) / mean,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    fixture_root = args.fixture.resolve(strict=True)
    engine_root = args.engine.resolve(strict=True)
    engine_receipt = engine_root / "rlinf-engine-receipt.json"
    fixture = torch.load(
        fixture_root / "backbone-inputs.pt", map_location="cpu", weights_only=True
    )
    fixture = {name: value.cuda() for name, value in fixture.items()}
    backbone_input = BatchFeature(
        data={
            name: fixture[name]
            for name in (
                "eagle_input_ids",
                "eagle_attention_mask",
                "eagle_pixel_values",
                "eagle_image_sizes",
            )
        }
    )
    action_input = BatchFeature(
        data={
            name: fixture[name]
            for name in ("state", "state_mask", "embodiment_id")
        }
    )

    cfg = _model_config(
        args.config_root.resolve(strict=True), args.model.resolve(strict=True)
    )
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    model.cuda()
    model.eval()
    with torch.inference_mode():
        eager_feature = model.backbone(backbone_input)
    eager_timing = _benchmark(
        lambda: model.backbone(backbone_input), args.warmup, args.iterations
    )
    _seed(args.seed)
    with torch.inference_mode():
        _, eager_action = model.action_head.get_rl_action(
            _clone_feature(eager_feature), action_input, mode="train"
        )

    backend_config = {
        "engine_dir": str(engine_root),
        "receipt_path": str(engine_receipt),
        "receipt_sha256": _sha256(engine_receipt),
        "static_batch_size": 8,
        "image_views": 2,
        "image_batch_size": 16,
        "sequence_length": 570,
        "runtime_version": args.tensorrt_version,
        "runtime_distribution": "tensorrt-cu12",
        "compute_capability": [8, 9],
        "components": args.components,
    }
    model.enable_tensorrt_backbone(backend_config)

    def candidate_backbone() -> BatchFeature:
        return model._forward_backbone(backbone_input)

    with torch.inference_mode():
        candidate_feature = candidate_backbone()
    candidate_timing = _benchmark(
        candidate_backbone,
        args.warmup,
        args.iterations,
    )
    _seed(args.seed)
    with torch.inference_mode():
        _, candidate_action = model.action_head.get_rl_action(
            _clone_feature(candidate_feature), action_input, mode="train"
        )

    comparisons = {
        "backbone_features": _metrics(
            eager_feature["backbone_features"],
            candidate_feature["backbone_features"],
        ),
        "actions": _metrics(
            eager_action["actions"], candidate_action["actions"]
        ),
        "prev_logprobs": _metrics(
            eager_action["prev_logprobs"], candidate_action["prev_logprobs"]
        ),
        "prev_values": _metrics(
            eager_action["prev_values"], candidate_action["prev_values"]
        ),
    }
    feature = comparisons["backbone_features"]
    action = comparisons["actions"]
    feature_pass = (
        feature["finite"]
        and feature["cosine"] >= 0.9975
        and feature["relative_l2"] <= 0.07
    )
    action_pass = (
        action["finite"]
        and action["cosine"] >= 0.999
        and action["mean_abs"] <= 0.005
        and action["relative_l2"] <= 0.015
    )
    status = "passed" if feature_pass and action_pass else "failed"
    result = {
        "schema": "rlinf.gr00t-n1d5-stack-cube-trt-qualification.v1",
        "status": status,
        "scope": "systems_only_feature_and_action_parity_not_ppo_authority",
        "components": args.components,
        "source": {
            "model_path": str(args.model.resolve(strict=True)),
            "fixture_receipt_sha256": _sha256(
                fixture_root / "backbone-abi.json"
            ),
            "engine_receipt_sha256": _sha256(engine_receipt),
        },
        "thresholds": {
            "feature_cosine_min": 0.9975,
            "feature_relative_l2_max": 0.07,
            "action_cosine_min": 0.999,
            "action_mean_abs_max": 0.005,
            "action_relative_l2_max": 0.015,
        },
        "comparisons": comparisons,
        "latency": {
            "boundary": "cuda_resident_backbone_input_to_backbone_features",
            "eager": eager_timing,
            "persistent_tensorrt": candidate_timing,
            "speedup": eager_timing["mean_ms"] / candidate_timing["mean_ms"],
            "reduction_percent": 100
            * (1 - candidate_timing["mean_ms"] / eager_timing["mean_ms"]),
        },
        "telemetry": model.hybrid_runtime_telemetry(),
    }
    receipt = output / "qualification.json"
    receipt.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model.close_hybrid_runtime()
    if status != "passed":
        raise RuntimeError(f"TensorRT qualification failed: {comparisons}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorrt-version", default="10.15.1.29")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=260918)
    parser.add_argument(
        "--components", choices=("full", "llm_only"), default="full"
    )
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W06 N1.5 TensorRT qualification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
