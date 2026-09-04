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

"""Refit a second DiT engine slot from a real post-PPO Actor checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from persistent_trt import PersistentEngine
from refit_dit_lifecycle_probe import (
    MAX_RELATIVE_L2,
    MIN_COSINE,
    _canonical_sha256,
    _capture_dit,
    _compare,
    _eager,
    _make_staging,
    _ordered_tensor_digest,
    _output_digest,
    _refit,
    _set_all_weights,
    _sha256,
    _source_state,
    _trt,
)
from refittable_dit_contract import RefitRevisionFence, double_buffer_memory_preflight


def _load_checkpoint_dit(
    checkpoint_path: Path, entries: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise RuntimeError("Actor checkpoint is not a string-keyed state dict")
    selected: dict[str, Any] = {}
    for entry in entries:
        source_fqn = entry["source_fqn"]
        if source_fqn not in state:
            raise RuntimeError(f"Actor checkpoint is missing DiT tensor {source_fqn}")
        value = state[source_fqn]
        if list(value.shape) != entry["source_shape"]:
            raise RuntimeError(f"checkpoint shape mismatch for {source_fqn}")
        if value.dtype != torch.bfloat16:
            raise RuntimeError(
                f"checkpoint dtype mismatch for {source_fqn}: {value.dtype}"
            )
        selected[source_fqn] = value
    checkpoint_dit_keys = {key for key in state if key.startswith("action_head.model.")}
    selected_keys = set(selected)
    if checkpoint_dit_keys != selected_keys:
        missing = sorted(checkpoint_dit_keys - selected_keys)
        extra = sorted(selected_keys - checkpoint_dit_keys)
        raise RuntimeError(
            "checkpoint DiT keyspace differs from the qualified parameter map: "
            f"checkpoint_only={missing[:8]}, map_only={extra[:8]}"
        )
    metadata = {
        "total_state_dict_tensors": len(state),
        "dit_tensor_count": len(selected),
        "dit_parameter_count": sum(value.numel() for value in selected.values()),
        "dit_bytes": sum(
            value.numel() * value.element_size() for value in selected.values()
        ),
    }
    return selected, metadata


def _apply_checkpoint_dit(
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch

    changed_tensors = 0
    changed_parameters = 0
    maximum_absolute_delta = 0.0
    with torch.no_grad():
        for entry in entries:
            name = entry["source_fqn"]
            destination = source[name]
            candidate = checkpoint[name].to(device=destination.device)
            delta = (candidate.float() - destination.float()).abs()
            tensor_changed = bool(torch.count_nonzero(delta))
            if tensor_changed:
                changed_tensors += 1
                changed_parameters += int(torch.count_nonzero(delta))
                maximum_absolute_delta = max(maximum_absolute_delta, float(delta.max()))
            destination.copy_(candidate)
            del candidate, delta
    torch.cuda.synchronize()
    return {
        "changed_tensors": changed_tensors,
        "changed_parameters": changed_parameters,
        "maximum_absolute_delta": maximum_absolute_delta,
    }


def _new_runtime(
    plan: Path, probe: dict[str, Any]
) -> tuple[PersistentEngine, int, int]:
    import torch

    free_before, _ = torch.cuda.mem_get_info()
    runtime = PersistentEngine(str(plan))
    for name in ("vl_embs", "image_mask", "backbone_attention_mask"):
        runtime.set_runtime_tensor_shape(name, probe[name].shape)
    free_after, _ = torch.cuda.mem_get_info()
    return runtime, free_before, free_after


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source.resolve(strict=True)
    sys.path.insert(0, str(source_root / "scripts/deployment"))
    import tensorrt as trt
    import torch
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    model = args.model.resolve(strict=True)
    collated = args.collated.resolve(strict=True)
    plan = args.engine.resolve(strict=True)
    engine_receipt_path = args.engine_receipt.resolve(strict=True)
    parameter_map_path = args.parameter_map.resolve(strict=True)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.mkdir(parents=True)

    engine_receipt = json.loads(engine_receipt_path.read_text(encoding="utf-8"))
    parameter_map = json.loads(parameter_map_path.read_text(encoding="utf-8"))
    if engine_receipt["engine"]["sha256"] != _sha256(plan):
        raise RuntimeError("engine hash differs from its qualified build receipt")
    entries = parameter_map["dit_refit"]["entries"]
    if len(entries) != 456:
        raise RuntimeError("parameter map is not the qualified 456-tensor map")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy = Gr00tPolicy(
        embodiment_tag="NEW_EMBODIMENT", model_path=str(model), device="cuda"
    )
    dit, probe = _capture_dit(policy, collated)
    del policy
    gc.collect()
    torch.cuda.empty_cache()

    active, active_free_before, active_free_after = _new_runtime(plan, probe)
    base_source = _source_state(dit, entries)
    base_order = [
        (entry["source_fqn"], base_source[entry["source_fqn"]]) for entry in entries
    ]
    source_digest_0 = _ordered_tensor_digest(base_order)
    probe_digest = _ordered_tensor_digest(sorted(probe.items()))
    eager_0 = _eager(dit, probe)
    active_0 = _trt(active, probe)
    baseline_metrics = _compare(eager_0, active_0)
    active_output_digest_0 = _output_digest(active_0)

    checkpoint, checkpoint_metadata = _load_checkpoint_dit(checkpoint_path, entries)
    checkpoint_order = [
        (entry["source_fqn"], checkpoint[entry["source_fqn"]]) for entry in entries
    ]
    expected_source_digest_1 = _ordered_tensor_digest(checkpoint_order)
    checkpoint_sha256 = _sha256(checkpoint_path)
    patch_digest = _canonical_sha256(
        {
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_dit_digest": expected_source_digest_1,
            "revision": args.checkpoint_revision,
        }
    )
    inventory_digest = engine_receipt["refitter"]["named_weight_digest"]
    prototype_digest = engine_receipt["refitter"]["classification"][
        "mapped_trainable_digest"
    ]
    fence = RefitRevisionFence(active_probe_output_digest=active_output_digest_0)
    candidate_slot = fence.admit_patch(
        args.checkpoint_revision,
        patch_digest,
        expected_source_digest_1,
        inventory_digest,
        prototype_digest,
    )

    change_summary = _apply_checkpoint_dit(base_source, checkpoint, entries)
    source_digest_1 = _ordered_tensor_digest(base_order)
    fence.mark_pytorch_applied(
        args.checkpoint_revision,
        observed_patch_digest=patch_digest,
        head_digest=source_digest_1,
        observed_source_weight_digest=source_digest_1,
    )
    if source_digest_1 != expected_source_digest_1:
        raise RuntimeError("post-apply DiT digest differs from the checkpoint")
    if change_summary["changed_tensors"] == 0:
        raise RuntimeError("real Actor checkpoint did not change any DiT tensor")

    staging, transformed_staging_bytes = _make_staging(base_source, entries)
    staging_order = [
        (entry["initializer"], staging[entry["initializer"]]) for entry in entries
    ]
    staging_digest_1 = _ordered_tensor_digest(staging_order)
    fence.register_staging_reference(source_digest_1, staging_digest_1)
    fence.mark_staging_verified(staging_digest_1)
    del checkpoint, checkpoint_order
    gc.collect()

    verification_io_bytes = sum(
        value.numel() * value.element_size() for value in [*probe.values(), active_0]
    )
    free_before_candidate, total = torch.cuda.mem_get_info()
    memory = engine_receipt["memory"]
    preflight = double_buffer_memory_preflight(
        free_before_bytes=free_before_candidate,
        total_device_bytes=total,
        candidate_engine_device_bytes=memory["deserialize_delta_bytes"],
        candidate_context_bytes=memory["context_delta_bytes"],
        refit_workspace_bytes=engine_receipt["engine"]["device_memory_size_v2"],
        transformed_staging_bytes=transformed_staging_bytes,
        verification_io_bytes=verification_io_bytes,
    )
    if not preflight["double_buffer_allowed"]:
        raise RuntimeError("qualified memory preflight rejected a second engine slot")

    candidate, candidate_free_before, candidate_free_after = _new_runtime(plan, probe)
    candidate_refitter = trt.Refitter(candidate.handle, candidate.logger)
    candidate_refitter.weights_validation = True
    actual_names = set(candidate_refitter.get_all_weights())
    mapped_names = {entry["initializer"] for entry in entries}
    if not mapped_names <= actual_names:
        raise RuntimeError("candidate refitter inventory lost mapped trainable weights")
    references = _set_all_weights(candidate_refitter, entries, staging)
    full_revision_refit = _refit(candidate_refitter)
    if full_revision_refit["missing_after"]:
        raise RuntimeError("full revision refit left missing weights")
    fence.mark_refit_complete(inventory_digest, prototype_digest)

    eager_1 = _eager(dit, probe)
    candidate_1 = _trt(candidate, probe)
    candidate_metrics = _compare(eager_1, candidate_1)
    output_change = _compare(active_0, candidate_1)
    active_after_candidate_refit = _trt(active, probe)
    old_slot_unchanged = (
        _output_digest(active_after_candidate_refit) == active_output_digest_0
    )
    if not old_slot_unchanged:
        raise RuntimeError("inactive-slot refit mutated the active engine output")
    numerics_passed = (
        candidate_metrics["finite"]
        and candidate_metrics["cosine"] >= MIN_COSINE
        and candidate_metrics["relative_l2"] <= MAX_RELATIVE_L2
        and _output_digest(candidate_1) != active_output_digest_0
    )
    metrics_digest = _canonical_sha256(candidate_metrics)
    fence.mark_probe_verified(
        probe_input_digest=probe_digest,
        pytorch_output_digest=_output_digest(eager_1),
        trt_output_digest=_output_digest(candidate_1),
        metrics_digest=metrics_digest,
        numerics_passed=numerics_passed,
        require_output_change=True,
    )
    fence.commit()
    if fence.active_slot != candidate_slot:
        raise RuntimeError("revision fence selected the wrong engine slot")
    slots = {"a": active, "b": candidate}
    fence.begin_inference()
    selected_output = _trt(slots[fence.active_slot], probe)
    fence.end_inference()
    selected_digest = _output_digest(selected_output)
    if selected_digest != _output_digest(candidate_1):
        raise RuntimeError("post-commit inference did not use the candidate slot")

    receipt = {
        "schema": "rlinf.gr00t-n1d7-refittable-dit-real-revision.v1",
        "status": "passed",
        "scope": "two_slot_real_post_PPO_DiT_revision_not_PPO_authority",
        "provenance": {
            "engine": {"path": str(plan), "sha256": _sha256(plan)},
            "engine_receipt": _sha256(engine_receipt_path),
            "parameter_map": _sha256(parameter_map_path),
            "model_config": _sha256(model / "config.json"),
            "collated": _sha256(collated),
            "actor_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
                "revision": args.checkpoint_revision,
                **checkpoint_metadata,
            },
        },
        "inventory": {
            "total_refittable": len(actual_names),
            "mapped_trainable": len(mapped_names),
            "derived_constants_retained": len(actual_names - mapped_names),
            "missing_after_full_revision_refit": full_revision_refit["missing_after"],
        },
        "device_weights": {
            "source_digest_revision_0": source_digest_0,
            "expected_source_digest_revision_1": expected_source_digest_1,
            "observed_source_digest_revision_1": source_digest_1,
            "staging_digest_revision_1": staging_digest_1,
            "transformed_staging_bytes": transformed_staging_bytes,
            **change_summary,
        },
        "revision": {
            "patch_digest": patch_digest,
            "candidate_slot": candidate_slot,
            "active_pytorch_revision": fence.active_pytorch_revision,
            "active_trt_revision": fence.active_trt_revision,
            "active_slot": fence.active_slot,
            "phase": fence.phase,
            "old_slot_unchanged_before_switch": old_slot_unchanged,
            "post_commit_selected_output_digest": selected_digest,
        },
        "fixed_probe": {
            "input_digest": probe_digest,
            "baseline_eager_vs_trt": baseline_metrics,
            "candidate_eager_vs_trt": candidate_metrics,
            "trt_revision_0_vs_1": output_change,
            "thresholds": {
                "cosine_min": MIN_COSINE,
                "relative_l2_max": MAX_RELATIVE_L2,
            },
        },
        "timing": {"full_456_tensor_revision_refit": full_revision_refit},
        "memory": {
            "active_slot_free_before": active_free_before,
            "active_slot_free_after": active_free_after,
            "free_before_candidate": free_before_candidate,
            "candidate_slot_free_before": candidate_free_before,
            "candidate_slot_free_after": candidate_free_after,
            "candidate_slot_observed_delta_bytes": (
                candidate_free_before - candidate_free_after
            ),
            "double_buffer_preflight": preflight,
        },
        "runtime": {
            "slot_a": active.telemetry(),
            "slot_b": candidate.telemetry(),
            "engine_instances": 2,
            "context_instances": 2,
        },
    }
    receipt_path = output / "real-revision-refit-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del references
    candidate.close()
    active.close()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--collated", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--engine-receipt", type=Path, required=True)
    parser.add_argument("--parameter-map", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W83 real revision probe failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
