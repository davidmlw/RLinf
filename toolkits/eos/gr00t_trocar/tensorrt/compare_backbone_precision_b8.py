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

"""Compare eager and TensorRT B8 backbone precisions with fixed explicit noise."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from common_boundary_b8 import make_explicit_noise_head, prepare_cuda_inputs
from persistent_trt import PersistentEngine
from standalone_true_b8 import _compare_array, _public_action, _raw_observation

THRESHOLDS = {
    "vit_cosine_min": 0.999,
    "backbone_cosine_min": 0.9995,
    "action_cosine_min": 0.999,
    "action_mean_abs_max": 0.005,
    "action_max_abs_max": 0.05,
}


def _load_policy(source: Path, model: Path) -> Any:
    sys.path.insert(0, str(source / "scripts/deployment"))
    from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: PLC0415

    return Gr00tPolicy(
        embodiment_tag="NEW_EMBODIMENT", model_path=str(model), device="cuda"
    )


def _call(policy: Any, inputs: dict[str, Any], initial_actions: Any) -> dict[str, Any]:
    import torch

    prepared = prepare_cuda_inputs(policy.model, inputs)
    explicit_head = make_explicit_noise_head(policy.model.action_head)
    backbone_inputs, action_inputs = prepared
    with torch.inference_mode():
        backbone = policy.model.backbone(backbone_inputs)
        action = explicit_head(
            backbone["backbone_features"],
            backbone["backbone_attention_mask"],
            backbone["image_mask"],
            action_inputs["state"],
            action_inputs["embodiment_id"],
            initial_actions,
        )
    torch.cuda.synchronize()
    return {
        "backbone": backbone["backbone_features"].detach().float().cpu().numpy(),
        "action": action.detach().float().cpu(),
    }


def _eager(
    source: Path,
    model: Path,
    inputs: dict[str, Any],
    observation: dict[str, Any],
    initial_actions: Any,
) -> dict[str, Any]:
    policy = _load_policy(source, model)
    capture: dict[str, Any] = {}

    def hook(_module: Any, _args: Any, output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        capture["vit"] = value.detach().float().cpu().numpy()

    handle = policy.model.backbone.model.model.visual.register_forward_hook(hook)
    try:
        values = _call(policy, inputs, initial_actions)
    finally:
        handle.remove()
    values["vit"] = capture["vit"]
    values["public_action"] = _public_action(
        policy, values["action"], observation
    )
    del policy
    gc.collect()
    return values


def _trt(
    source: Path,
    model: Path,
    engines: Path,
    inputs: dict[str, Any],
    observation: dict[str, Any],
    initial_actions: Any,
) -> dict[str, Any]:
    import torch

    policy = _load_policy(source, model)
    sys.path.insert(0, str(source / "scripts/deployment"))
    import trt_model_forward  # noqa: PLC0415

    trt_model_forward.Engine = PersistentEngine
    trt_model_forward.setup_tensorrt_engines(
        policy, str(engines), mode="vit_llm_only"
    )
    vit = policy.model.backbone.vit_engine
    values = _call(policy, inputs, initial_actions)
    values["vit"] = vit.last_outputs["image_embeds"].detach().float().cpu().numpy()
    values["public_action"] = _public_action(
        policy, values["action"], observation
    )
    values["vit_input_dtype"] = str(vit.dtype_of("pixel_values"))
    trt_model_forward.close_tensorrt_engines(policy)
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return values


def _qualification(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    comparisons = {
        "vit_image_embeds": _compare_array(reference["vit"], candidate["vit"]),
        "pre_final_backbone": _compare_array(
            reference["backbone"], candidate["backbone"]
        ),
        "public_action": _compare_array(
            reference["public_action"], candidate["public_action"]
        ),
    }
    action = comparisons["public_action"]
    checks = {
        "vit_cosine": comparisons["vit_image_embeds"]["cosine"]
        >= THRESHOLDS["vit_cosine_min"],
        "backbone_cosine": comparisons["pre_final_backbone"]["cosine"]
        >= THRESHOLDS["backbone_cosine_min"],
        "action_cosine": action["cosine"] >= THRESHOLDS["action_cosine_min"],
        "action_mean_abs": action["mean_abs"]
        <= THRESHOLDS["action_mean_abs_max"],
        "action_max_abs": action["max_abs"] <= THRESHOLDS["action_max_abs_max"],
        "all_finite": all(value["finite"] for value in comparisons.values()),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "comparisons": comparisons,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    source = args.source.resolve(strict=True)
    model = args.model.resolve(strict=True)
    collated_path = args.collated.resolve(strict=True)
    fixture_receipt = args.fixture_receipt.resolve(strict=True)
    raw = args.raw.resolve(strict=True)
    old_engines = args.old_engines.resolve(strict=True)
    new_engines = args.new_engines.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    collated = torch.load(collated_path, map_location="cpu", weights_only=False)
    inputs = collated.get("inputs", collated)
    observation = _raw_observation(raw, fixture_receipt)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    initial_actions = torch.randn(
        (8, 40, 132), dtype=torch.bfloat16, device="cuda"
    )
    reference = _eager(source, model, inputs, observation, initial_actions)
    old = _trt(source, model, old_engines, inputs, observation, initial_actions)
    new = _trt(source, model, new_engines, inputs, observation, initial_actions)
    result = {
        "schema": "rlinf.gr00t-n1d7-b8-backbone-precision-numerics.v1",
        "status": "completed",
        "scope": "standalone numerical gate; not PPO authority",
        "batch_size": 8,
        "seed": args.seed,
        "thresholds": THRESHOLDS,
        "old_trt_fp32_vit": {
            "vit_input_dtype": old["vit_input_dtype"],
            **_qualification(reference, old),
        },
        "new_trt_bf16_vit": {
            "vit_input_dtype": new["vit_input_dtype"],
            **_qualification(reference, new),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--collated", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--fixture-receipt", type=Path, required=True)
    parser.add_argument("--old-engines", type=Path, required=True)
    parser.add_argument("--new-engines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"backbone precision comparison failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
