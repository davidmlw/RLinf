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

"""Exercise device-weight refit and revision fencing on a fixed B8 DiT probe."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

from persistent_trt import PersistentEngine
from refittable_dit_contract import RefitRevisionFence, double_buffer_memory_preflight

DELTA_SOURCE_FQN = "action_head.model.proj_out_2.bias"
MIN_COSINE = 0.999
MAX_RELATIVE_L2 = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_bytes(value: Any) -> bytes:
    import torch

    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _ordered_tensor_digest(values: Iterable[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in values:
        metadata = json.dumps(
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        payload = _tensor_bytes(value)
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _compare(reference: Any, candidate: Any) -> dict[str, Any]:
    import torch

    lhs = reference.float().flatten()
    rhs = candidate.float().flatten()
    delta = rhs - lhs
    lhs_norm = torch.linalg.vector_norm(lhs)
    relative_l2 = torch.linalg.vector_norm(delta) / torch.clamp(lhs_norm, min=1e-12)
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(lhs, rhs, dim=0)),
        "mean_abs": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(relative_l2),
        "finite": bool(torch.isfinite(rhs).all()),
    }


def _output_digest(value: Any) -> str:
    return hashlib.sha256(_tensor_bytes(value)).hexdigest()


def _capture_dit(policy: Any, collated: Path) -> tuple[Any, dict[str, Any]]:
    import torch
    from export_onnx_n1d7 import DiTInputCapture
    from gr00t.policy.gr00t_policy import _rec_to_dtype

    value = torch.load(collated, map_location="cpu", weights_only=False)
    value = _rec_to_dtype(value, dtype=torch.bfloat16)
    model_inputs = value["inputs"] if "inputs" in value else value
    capture = DiTInputCapture()
    hook = policy.model.action_head.model.register_forward_pre_hook(
        capture.hook_fn, with_kwargs=True
    )
    try:
        with torch.inference_mode():
            policy.model.get_action(model_inputs)
        torch.cuda.synchronize()
    finally:
        hook.remove()
    if not capture.captured:
        raise RuntimeError("failed to capture the DiT fixed probe")
    tensors = {
        "sa_embs": capture.sa_embs.cuda().contiguous(),
        "vl_embs": capture.vl_embs.cuda().contiguous(),
        "timestep": capture.timestep.cuda().contiguous(),
        "image_mask": capture.image_mask.cuda().contiguous(),
        "backbone_attention_mask": capture.backbone_attention_mask.cuda().contiguous(),
    }
    expected = {
        "sa_embs": (8, 41, 1536),
        "vl_embs": (8, 208, 2048),
        "timestep": (8,),
        "image_mask": (8, 208),
        "backbone_attention_mask": (8, 208),
    }
    actual = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
    if actual != expected:
        raise RuntimeError(f"fixed probe ABI mismatch: {actual} != {expected}")
    return policy.model.action_head.model, tensors


def _eager(dit: Any, values: dict[str, Any]) -> Any:
    import torch

    with torch.inference_mode():
        return (
            dit(
                values["sa_embs"],
                values["vl_embs"],
                values["timestep"],
                image_mask=values["image_mask"],
                backbone_attention_mask=values["backbone_attention_mask"],
            )
            .detach()
            .clone()
        )


def _trt(engine: PersistentEngine, values: dict[str, Any]) -> Any:
    import torch

    output = engine(**values)["output"]
    torch.cuda.synchronize()
    return output.detach().clone()


def _source_state(dit: Any, entries: list[dict[str, Any]]) -> dict[str, Any]:
    state = dit.state_dict()
    prefix = "action_head.model."
    result = {}
    for entry in entries:
        source_fqn = entry["source_fqn"]
        if not source_fqn.startswith(prefix):
            raise RuntimeError(f"source tensor is outside DiT: {source_fqn}")
        name = source_fqn[len(prefix) :]
        if name not in state:
            raise RuntimeError(f"loaded DiT is missing {source_fqn}")
        value = state[name]
        if (
            list(value.shape) != entry["source_shape"]
            or str(value.dtype) != "torch.bfloat16"
        ):
            raise RuntimeError(f"loaded source tensor ABI mismatch: {source_fqn}")
        result[source_fqn] = value
    return result


def _make_staging(
    source: dict[str, Any], entries: list[dict[str, Any]]
) -> tuple[dict[str, Any], int]:
    staging = {}
    allocated = 0
    for entry in entries:
        value = source[entry["source_fqn"]]
        if entry["transform"] == "identity":
            transformed = value.contiguous()
        elif entry["transform"] == "transpose_2d":
            transformed = value.transpose(0, 1).contiguous()
            allocated += transformed.numel() * transformed.element_size()
        else:
            raise RuntimeError(f"unsupported refit transform: {entry['transform']}")
        if list(transformed.shape) != entry["initializer_shape"]:
            raise RuntimeError(f"staging shape mismatch for {entry['initializer']}")
        staging[entry["initializer"]] = transformed
    return staging, allocated


def _set_all_weights(
    refitter: Any, entries: list[dict[str, Any]], staging: dict[str, Any]
) -> list[Any]:
    import tensorrt as trt

    references = []
    for entry in entries:
        name = entry["initializer"]
        value = staging[name]
        weights = trt.Weights(trt.bfloat16, value.data_ptr(), value.numel())
        if not refitter.set_named_weights(name, weights, trt.TensorLocation.DEVICE):
            raise RuntimeError(f"TensorRT rejected device weight {name}")
        references.append(weights)
    missing = sorted(refitter.get_missing_weights())
    if missing:
        raise RuntimeError(f"TensorRT refit has missing weights: {missing[:12]}")
    return references


def _refit(refitter: Any) -> dict[str, Any]:
    import torch

    stream = torch.cuda.current_stream()
    before, _ = torch.cuda.mem_get_info()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record(stream)
    if not refitter.refit_cuda_engine_async(stream.cuda_stream):
        raise RuntimeError("TensorRT async refit returned false")
    end.record(stream)
    end.synchronize()
    wall_s = time.perf_counter() - wall_start
    after, _ = torch.cuda.mem_get_info()
    return {
        "wall_s": wall_s,
        "device_ms": float(start.elapsed_time(end)),
        "free_before_bytes": before,
        "free_after_bytes": after,
        "retained_delta_bytes": before - after,
        "missing_after": sorted(refitter.get_missing_weights()),
    }


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

    runtime = PersistentEngine(str(plan))
    for name in ("vl_embs", "image_mask", "backbone_attention_mask"):
        runtime.set_runtime_tensor_shape(name, probe[name].shape)
    refitter = trt.Refitter(runtime.handle, runtime.logger)
    refitter.weights_validation = True
    actual_names = set(refitter.get_all_weights())
    mapped_names = {entry["initializer"] for entry in entries}
    if not mapped_names <= actual_names:
        raise RuntimeError("runtime refitter inventory lost mapped trainable weights")

    source = _source_state(dit, entries)
    source_order = [
        (entry["source_fqn"], source[entry["source_fqn"]]) for entry in entries
    ]
    source_digest_0 = _ordered_tensor_digest(source_order)
    free_before_staging, total = torch.cuda.mem_get_info()
    staging, transformed_staging_bytes = _make_staging(source, entries)
    free_after_staging, _ = torch.cuda.mem_get_info()
    staging_order = [
        (entry["initializer"], staging[entry["initializer"]]) for entry in entries
    ]
    staging_digest_0 = _ordered_tensor_digest(staging_order)
    probe_digest = _ordered_tensor_digest(sorted(probe.items()))

    eager_0 = _eager(dit, probe)
    trt_0 = _trt(runtime, probe)
    references = _set_all_weights(refitter, entries, staging)
    identity_refit = _refit(refitter)
    trt_identity = _trt(runtime, probe)
    identity_metrics = _compare(trt_0, trt_identity)
    if _output_digest(trt_0) != _output_digest(trt_identity):
        raise RuntimeError("same-weight refit changed the TensorRT fixed-probe output")

    delta_entry = next(
        item for item in entries if item["source_fqn"] == DELTA_SOURCE_FQN
    )
    delta_spec = {"source_fqn": DELTA_SOURCE_FQN, "add": args.delta}
    patch_digest = _canonical_sha256(delta_spec)
    candidate_target = source[DELTA_SOURCE_FQN].detach().clone()
    with torch.no_grad():
        candidate_target.add_(args.delta)
    expected_source_order = [
        (name, candidate_target if name == DELTA_SOURCE_FQN else value)
        for name, value in source_order
    ]
    expected_staging_order = [
        (
            name,
            candidate_target if name == delta_entry["initializer"] else value,
        )
        for name, value in staging_order
    ]
    expected_source_digest_1 = _ordered_tensor_digest(expected_source_order)
    expected_staging_digest_1 = _ordered_tensor_digest(expected_staging_order)
    inventory_digest = engine_receipt["refitter"]["named_weight_digest"]
    prototype_digest = engine_receipt["refitter"]["classification"][
        "mapped_trainable_digest"
    ]
    fence = RefitRevisionFence(active_probe_output_digest=_output_digest(trt_0))
    fence.admit_patch(
        1,
        patch_digest,
        expected_source_digest_1,
        inventory_digest,
        prototype_digest,
    )
    with torch.no_grad():
        source[DELTA_SOURCE_FQN].add_(args.delta)
    source_digest_1 = _ordered_tensor_digest(source_order)
    staging_digest_1 = _ordered_tensor_digest(staging_order)
    fence.mark_pytorch_applied(
        1,
        observed_patch_digest=patch_digest,
        head_digest=source_digest_1,
        observed_source_weight_digest=source_digest_1,
    )
    if staging_digest_1 != expected_staging_digest_1:
        fence.fail_refit("actual device staging differs from admitted candidate")
        raise RuntimeError("actual device staging differs from admitted candidate")
    fence.register_staging_reference(source_digest_1, expected_staging_digest_1)
    fence.mark_staging_verified(staging_digest_1)

    delta_value = staging[delta_entry["initializer"]]
    delta_weights = trt.Weights(
        trt.bfloat16, delta_value.data_ptr(), delta_value.numel()
    )
    if not refitter.set_named_weights(
        delta_entry["initializer"], delta_weights, trt.TensorLocation.DEVICE
    ):
        raise RuntimeError("TensorRT rejected the deliberate delta weight")
    references.append(delta_weights)
    if refitter.get_missing_weights():
        raise RuntimeError("deliberate delta produced missing refit weights")
    delta_refit = _refit(refitter)
    fence.mark_refit_complete(inventory_digest, prototype_digest)
    eager_1 = _eager(dit, probe)
    trt_1 = _trt(runtime, probe)
    delta_metrics = _compare(eager_1, trt_1)
    baseline_metrics = _compare(eager_0, trt_0)
    output_change = _compare(trt_0, trt_1)
    numerics_passed = (
        delta_metrics["finite"]
        and delta_metrics["cosine"] >= MIN_COSINE
        and delta_metrics["relative_l2"] <= MAX_RELATIVE_L2
        and _output_digest(trt_0) != _output_digest(trt_1)
    )
    metrics_digest = _canonical_sha256(delta_metrics)
    fence.mark_probe_verified(
        probe_input_digest=probe_digest,
        pytorch_output_digest=_output_digest(eager_1),
        trt_output_digest=_output_digest(trt_1),
        metrics_digest=metrics_digest,
        numerics_passed=numerics_passed,
        require_output_change=True,
    )
    fence.commit()

    verification_io_bytes = sum(
        value.numel() * value.element_size()
        for value in [*probe.values(), trt_0, trt_1]
    )
    memory = engine_receipt["memory"]
    preflight = double_buffer_memory_preflight(
        free_before_bytes=free_before_staging,
        total_device_bytes=total,
        candidate_engine_device_bytes=memory["deserialize_delta_bytes"],
        candidate_context_bytes=memory["context_delta_bytes"],
        refit_workspace_bytes=max(
            identity_refit["retained_delta_bytes"],
            delta_refit["retained_delta_bytes"],
            engine_receipt["engine"]["device_memory_size_v2"],
        ),
        transformed_staging_bytes=transformed_staging_bytes,
        verification_io_bytes=verification_io_bytes,
    )
    receipt = {
        "schema": "rlinf.gr00t-n1d7-refittable-dit-device-lifecycle.v1",
        "status": "passed",
        "scope": "single_engine_device_refit_probe_not_PPO_authority",
        "provenance": {
            "engine": {"path": str(plan), "sha256": _sha256(plan)},
            "engine_receipt": _sha256(engine_receipt_path),
            "parameter_map": _sha256(parameter_map_path),
            "model_config": _sha256(model / "config.json"),
            "collated": _sha256(collated),
        },
        "inventory": {
            "total_refittable": len(actual_names),
            "mapped_trainable": len(mapped_names),
            "derived_constants_retained": len(actual_names - mapped_names),
            "missing_after_identity_refit": identity_refit["missing_after"],
            "missing_after_delta_refit": delta_refit["missing_after"],
        },
        "device_weights": {
            "source_digest_revision_0": source_digest_0,
            "source_digest_revision_1": source_digest_1,
            "staging_digest_revision_0": staging_digest_0,
            "staging_digest_revision_1": staging_digest_1,
            "transformed_staging_bytes": transformed_staging_bytes,
            "free_before_staging": free_before_staging,
            "free_after_staging": free_after_staging,
        },
        "revision": {
            "delta": delta_spec,
            "patch_digest": patch_digest,
            "active_pytorch_revision": fence.active_pytorch_revision,
            "active_trt_revision": fence.active_trt_revision,
            "active_slot": fence.active_slot,
            "phase": fence.phase,
        },
        "fixed_probe": {
            "input_digest": probe_digest,
            "baseline_eager_vs_trt": baseline_metrics,
            "identity_refit_trt_vs_trt": identity_metrics,
            "delta_eager_vs_trt": delta_metrics,
            "trt_revision_0_vs_1": output_change,
            "thresholds": {
                "cosine_min": MIN_COSINE,
                "relative_l2_max": MAX_RELATIVE_L2,
            },
        },
        "timing": {"identity_refit": identity_refit, "delta_refit": delta_refit},
        "double_buffer_preflight": preflight,
        "runtime": runtime.telemetry(),
    }
    (output / "refit-lifecycle-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    runtime.close()
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--collated", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--engine-receipt", type=Path, required=True)
    parser.add_argument("--parameter-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--delta", type=float, default=0.03125)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W83 refit lifecycle probe failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
