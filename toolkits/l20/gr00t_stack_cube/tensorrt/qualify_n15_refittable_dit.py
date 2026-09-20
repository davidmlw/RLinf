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

"""Qualify N1.5 full-TRT Backbone with eager or refittable-TRT DiT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from qualify_n15_backbone import (
    _benchmark,
    _clone_feature,
    _metrics,
    _model_config,
    _seed,
)
from transformers import BatchFeature


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model
    from rlinf.models.embodiment.gr00t.gr00t_n1d5.tensorrt_dit import (
        _ordered_source_digest,
    )

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    model_path = args.model.resolve(strict=True)
    fixture_root = args.fixture.resolve(strict=True)
    backbone_root = args.backbone_engine.resolve(strict=True)
    dit_root = args.dit_engine.resolve(strict=True)
    dit_receipt = dit_root / "rlinf-refittable-dit-engine-receipt.json"
    parameter_map_path = args.parameter_map.resolve(strict=True)
    parameter_map = json.loads(parameter_map_path.read_text(encoding="utf-8"))
    entries = parameter_map["dit_refit"]["entries"]

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
        data={name: fixture[name] for name in ("state", "state_mask", "embodiment_id")}
    )
    cfg = _model_config(args.config_root.resolve(strict=True), model_path)
    model = get_model(cfg, torch_dtype=torch.bfloat16).cuda().eval()
    backbone_receipt = backbone_root / "rlinf-engine-receipt.json"
    model.enable_tensorrt_backbone(
        {
            "engine_dir": str(backbone_root),
            "receipt_path": str(backbone_receipt),
            "receipt_sha256": _sha256(backbone_receipt),
            "static_batch_size": 8,
            "image_views": 2,
            "image_batch_size": 16,
            "sequence_length": 570,
            "runtime_version": args.tensorrt_version,
            "runtime_distribution": "tensorrt-cu12",
            "compute_capability": [8, 9],
            "components": "full",
        }
    )
    with torch.inference_mode():
        feature = model._forward_backbone(backbone_input)

    def eager_head() -> dict[str, torch.Tensor]:
        _, result = model.action_head.get_rl_action(
            _clone_feature(feature), action_input, mode="train"
        )
        return result

    _seed(args.seed)
    with torch.inference_mode():
        eager_result = eager_head()
    _seed(args.seed)
    eager_timing = _benchmark(eager_head, args.warmup, args.iterations)

    source_digest = _ordered_source_digest(model.action_head.model, entries)
    dit_config = {
        "engine_path": str(dit_root / "dit_bf16_refit.engine"),
        "receipt_path": str(dit_receipt),
        "receipt_sha256": _sha256(dit_receipt),
        "parameter_map_path": str(parameter_map_path),
        "parameter_map_sha256": _sha256(parameter_map_path),
        "source_digest_revision_0": source_digest,
        "revision": 0,
        "runtime_version": args.tensorrt_version,
        "runtime_distribution": "tensorrt-cu12",
        "compute_capability": [8, 9],
        "online_refit": True,
        "lineage_receipt_mode": "gpu_transform_validation",
        "probe_each_revision": True,
        "minimum_probe_cosine": 0.999,
        "maximum_probe_relative_l2": 0.05,
        "minimum_free_device_bytes": 8 << 30,
        "ppo_authority_status": "unqualified_requires_convergence_validation",
        "shadow_eager": False,
    }
    model.enable_tensorrt_dit(dit_config)
    model.verify_online_update_contract(0)

    def candidate_head() -> dict[str, torch.Tensor]:
        _, result = model.action_head.get_rl_action(
            _clone_feature(feature), action_input, mode="train"
        )
        return result

    _seed(args.seed)
    with torch.inference_mode():
        candidate_result = candidate_head()
    _seed(args.seed)
    candidate_timing = _benchmark(candidate_head, args.warmup, args.iterations)
    comparisons = {
        name: _metrics(eager_result[name], candidate_result[name])
        for name in ("actions", "prev_logprobs", "prev_values")
    }

    action_model = model.action_head.model
    with torch.no_grad():
        action_model.proj_out_2.bias[0].add_(torch.tensor(2**-8, device="cuda"))
    model.verify_online_update_contract(1)
    _seed(args.seed)
    with torch.inference_mode():
        revised_result = candidate_head()
    revision_delta = _metrics(candidate_result["actions"], revised_result["actions"])
    telemetry = model.hybrid_runtime_telemetry()
    dit_telemetry = telemetry["tensorrt_dit"]
    passed = (
        comparisons["actions"]["finite"]
        and comparisons["prev_logprobs"]["finite"]
        and dit_telemetry["active_revision"] == 1
        and len(dit_telemetry["refit_records"]) >= 2
        and revision_delta["relative_l2"] > 0
    )
    result = {
        "schema": "rlinf.gr00t-n1d5-stack-cube-trt-refittable-dit-qualification.v1",
        "status": "passed" if passed else "failed",
        "scope": "systems_only_requires_convergence_validation",
        "source": {
            "model": str(model_path),
            "backbone_receipt_sha256": _sha256(backbone_receipt),
            "dit_receipt_sha256": _sha256(dit_receipt),
            "parameter_map_sha256": _sha256(parameter_map_path),
            "revision_0_source_digest": source_digest,
        },
        "comparisons": comparisons,
        "revision_1_action_delta": revision_delta,
        "latency": {
            "boundary": "fixed_full_trt_backbone_feature_to_action_result",
            "trt_backbone_eager_head": eager_timing,
            "trt_backbone_refittable_trt_dit": candidate_timing,
            "speedup": eager_timing["mean_ms"] / candidate_timing["mean_ms"],
            "reduction_percent": 100
            * (1 - candidate_timing["mean_ms"] / eager_timing["mean_ms"]),
        },
        "telemetry": telemetry,
    }
    (output / "qualification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model.close_hybrid_runtime()
    if not passed:
        raise RuntimeError("N1.5 refittable TensorRT DiT qualification failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--backbone-engine", type=Path, required=True)
    parser.add_argument("--dit-engine", type=Path, required=True)
    parser.add_argument("--parameter-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorrt-version", default="10.15.1.29")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W06 N1.5 refittable DiT qualification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
