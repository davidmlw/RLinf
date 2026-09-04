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

"""Build the offline parameter contract for a refittable GR00T N1.7 DiT.

The legacy PyTorch ONNX exporter keeps bias and normalization initializers under
semantic names, but materializes linear weights as anonymous MatMul constants.
This module resolves both forms back to the authoritative RLInf state-dict FQN
and fails closed if any DiT tensor is missing, duplicated, or shape-incompatible.

TensorRT is intentionally not imported here. The resulting receipt is a build
input; a later GPU artifact gate must prove that ``trt.Refitter.get_all_weights``
exposes exactly the same initializer inventory and prototypes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

CHECKPOINT_DIT_PREFIX = "action_head.model."
ONNX_DIT_PREFIX = "dit."

DTYPE_BYTES = {
    "BF16": 2,
    "BOOL": 1,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}

ONNX_DTYPE_ALIASES = {
    "BFLOAT16": "BF16",
    "FLOAT16": "F16",
    "FLOAT": "F32",
    "DOUBLE": "F64",
    "INT8": "I8",
    "INT16": "I16",
    "INT32": "I32",
    "INT64": "I64",
    "UINT8": "U8",
    "UINT16": "U16",
    "UINT32": "U32",
    "UINT64": "U64",
}

REQUIRED_MODEL_CONFIG = {
    "action_horizon": 40,
    "add_pos_embed": True,
    "backbone_embedding_dim": 2048,
    "max_action_dim": 132,
    "max_num_embodiments": 32,
    "max_seq_len": 1024,
    "num_inference_timesteps": 4,
    "tune_diffusion_model": True,
    "tune_projector": True,
    "tune_vlln": True,
    "use_alternate_vl_dit": True,
}

DEFAULT_DOUBLE_BUFFER_HEADROOM_BYTES = 8 << 30


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_dtype(dtype: str) -> str:
    normalized = ONNX_DTYPE_ALIASES.get(dtype, dtype)
    if normalized not in DTYPE_BYTES:
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    return normalized


def load_checkpoint_header(path: Path) -> dict[str, dict[str, Any]]:
    """Load tensor metadata from safetensors files or a retained header JSON."""

    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        tensors = value.get("tensors")
        if not isinstance(tensors, dict):
            raise ValueError("checkpoint header JSON must contain a tensors mapping")
        return tensors

    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")

    tensors: dict[str, dict[str, Any]] = {}
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"checkpoint contains no safetensors files: {path}")
    for safetensors_path in files:
        with safetensors_path.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_length))
        header.pop("__metadata__", None)
        for name, spec in header.items():
            if name in tensors:
                raise ValueError(f"duplicate checkpoint tensor: {name}")
            tensors[name] = {
                "shape": list(spec["shape"]),
                "dtype": str(spec["dtype"]),
                "file": safetensors_path.name,
                "data_offsets": list(spec["data_offsets"]),
            }
    return tensors


def load_onnx_initializer_contract(path: Path) -> dict[str, dict[str, Any]]:
    """Load initializer metadata and consumer identity from an ONNX graph."""

    try:
        import onnx  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "onnx is required only for receipt generation; install the frozen "
            "builder version (1.20.1)"
        ) from error

    model = onnx.load(str(path), load_external_data=False)
    consumers: dict[str, list[dict[str, str]]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(
                {"name": node.name, "op_type": node.op_type}
            )

    initializers: dict[str, dict[str, Any]] = {}
    for initializer in model.graph.initializer:
        dtype = onnx.TensorProto.DataType.Name(initializer.data_type)
        initializers[initializer.name] = {
            "shape": list(initializer.dims),
            "dtype": _normalize_dtype(dtype),
            "external_data": {
                item.key: item.value for item in initializer.external_data
            },
            "consumers": consumers.get(initializer.name, []),
        }
    return initializers


def validate_model_config(path: Path) -> dict[str, Any]:
    """Freeze fields that determine DiT shape and trainable authority."""

    value = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"expected": expected, "actual": value.get(key)}
        for key, expected in REQUIRED_MODEL_CONFIG.items()
        if value.get(key) != expected
    }
    diffusion = value.get("diffusion_model_cfg", {})
    required_diffusion = {
        "attention_head_dim": 48,
        "interleave_self_attention": True,
        "num_attention_heads": 32,
        "num_layers": 32,
        "output_dim": 1024,
    }
    mismatches.update(
        {
            f"diffusion_model_cfg.{key}": {
                "expected": expected,
                "actual": diffusion.get(key),
            }
            for key, expected in required_diffusion.items()
            if diffusion.get(key) != expected
        }
    )
    if mismatches:
        raise ValueError(f"model config does not match the W83 contract: {mismatches}")
    return {
        "sha256": _file_sha256(path),
        "fields": {key: value[key] for key in sorted(REQUIRED_MODEL_CONFIG)},
        "diffusion_model_cfg": {
            key: diffusion[key] for key in sorted(required_diffusion)
        },
    }


def _source_for_initializer(
    initializer_name: str,
    initializer: Mapping[str, Any],
) -> tuple[str, str]:
    if initializer_name.startswith(ONNX_DIT_PREFIX):
        suffix = initializer_name[len(ONNX_DIT_PREFIX) :]
        return f"{CHECKPOINT_DIT_PREFIX}{suffix}", "identity"

    consumers = initializer.get("consumers", [])
    if len(consumers) != 1:
        raise ValueError(
            f"anonymous initializer {initializer_name} must have one consumer; "
            f"got {consumers}"
        )
    consumer = consumers[0]
    node_name = str(consumer.get("name", ""))
    if (
        consumer.get("op_type") != "MatMul"
        or not node_name.startswith("/dit/")
        or not node_name.endswith("/MatMul")
    ):
        raise ValueError(
            f"cannot resolve anonymous initializer {initializer_name} from "
            f"consumer {consumer}"
        )
    module_path = node_name[len("/dit/") : -len("/MatMul")].replace("/", ".")
    return f"{CHECKPOINT_DIT_PREFIX}{module_path}.weight", "transpose_2d"


def _expected_initializer_shape(
    source_shape: Sequence[int], transform: str
) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in source_shape)
    if transform == "identity":
        return shape
    if transform == "transpose_2d":
        if len(shape) != 2:
            raise ValueError(f"transpose_2d requires a matrix, got shape={shape}")
        return (shape[1], shape[0])
    raise ValueError(f"unsupported initializer transform: {transform}")


def action_head_inventory(
    checkpoint_tensors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize checkpoint-resident action-head components."""

    components: dict[str, dict[str, Any]] = {}
    for name, spec in checkpoint_tensors.items():
        if not name.startswith("action_head."):
            continue
        suffix = name[len("action_head.") :]
        component = suffix.split(".", 1)[0]
        dtype = _normalize_dtype(str(spec["dtype"]))
        count = math.prod(int(dim) for dim in spec["shape"])
        summary = components.setdefault(
            component,
            {"tensor_count": 0, "parameter_count": 0, "byte_count": 0, "dtypes": set()},
        )
        summary["tensor_count"] += 1
        summary["parameter_count"] += count
        summary["byte_count"] += count * DTYPE_BYTES[dtype]
        summary["dtypes"].add(dtype)

    for summary in components.values():
        summary["dtypes"] = sorted(summary["dtypes"])
    return {
        "components": dict(sorted(components.items())),
        "tensor_count": sum(item["tensor_count"] for item in components.values()),
        "parameter_count": sum(item["parameter_count"] for item in components.values()),
        "byte_count": sum(item["byte_count"] for item in components.values()),
    }


def build_refit_manifest(
    checkpoint_tensors: Mapping[str, Mapping[str, Any]],
    onnx_initializers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a complete ONNX-initializer to state-dict refit mapping."""

    expected_sources = {
        name for name in checkpoint_tensors if name.startswith(CHECKPOINT_DIT_PREFIX)
    }
    if not expected_sources:
        raise ValueError("checkpoint contains no action_head.model tensors")
    if not onnx_initializers:
        raise ValueError("ONNX graph contains no initializers")

    source_to_initializer: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for initializer_name, initializer in sorted(onnx_initializers.items()):
        source_name, transform = _source_for_initializer(initializer_name, initializer)
        if source_name in source_to_initializer:
            raise ValueError(
                f"source tensor {source_name} maps to multiple initializers: "
                f"{source_to_initializer[source_name]}, {initializer_name}"
            )
        source = checkpoint_tensors.get(source_name)
        if source is None:
            raise ValueError(
                f"ONNX initializer {initializer_name} maps to missing source "
                f"tensor {source_name}"
            )

        source_dtype = _normalize_dtype(str(source["dtype"]))
        initializer_dtype = _normalize_dtype(str(initializer["dtype"]))
        if source_dtype != initializer_dtype:
            raise ValueError(
                f"dtype mismatch for {source_name}: checkpoint={source_dtype}, "
                f"ONNX={initializer_dtype}"
            )
        expected_shape = _expected_initializer_shape(source["shape"], transform)
        actual_shape = tuple(int(dim) for dim in initializer["shape"])
        if actual_shape != expected_shape:
            raise ValueError(
                f"shape mismatch for {source_name}: expected ONNX "
                f"{expected_shape}, got {actual_shape}"
            )

        parameter_count = math.prod(int(dim) for dim in source["shape"])
        source_to_initializer[source_name] = initializer_name
        entries.append(
            {
                "initializer": initializer_name,
                "source_fqn": source_name,
                "transform": transform,
                "source_shape": list(source["shape"]),
                "initializer_shape": list(initializer["shape"]),
                "dtype": source_dtype,
                "parameter_count": parameter_count,
                "byte_count": parameter_count * DTYPE_BYTES[source_dtype],
            }
        )

    actual_sources = set(source_to_initializer)
    if actual_sources != expected_sources:
        missing = sorted(expected_sources - actual_sources)
        extra = sorted(actual_sources - expected_sources)
        raise ValueError(
            f"DiT refit mapping is incomplete: missing={missing[:8]}, extra={extra[:8]}"
        )

    transforms: dict[str, int] = {}
    for entry in entries:
        transform = entry["transform"]
        transforms[transform] = transforms.get(transform, 0) + 1
    mapping_sha256 = _canonical_sha256(entries)
    return {
        "schema": "rlinf.gr00t-n1d7-dit-refit-parameter-map.v1",
        "status": "passed",
        "initializer_count": len(entries),
        "source_tensor_count": len(expected_sources),
        "parameter_count": sum(item["parameter_count"] for item in entries),
        "byte_count": sum(item["byte_count"] for item in entries),
        "transforms": dict(sorted(transforms.items())),
        "mapping_sha256": mapping_sha256,
        "entries": entries,
    }


def double_buffer_memory_preflight(
    *,
    free_before_bytes: int,
    total_device_bytes: int,
    candidate_engine_device_bytes: int,
    candidate_context_bytes: int,
    refit_workspace_bytes: int,
    transformed_staging_bytes: int,
    verification_io_bytes: int,
    safety_headroom_bytes: int = DEFAULT_DOUBLE_BUFFER_HEADROOM_BYTES,
) -> dict[str, Any]:
    """Decide whether a second DiT engine slot may be allocated.

    Every device allocation is reported separately because a serialized plan's
    file size is not a reliable proxy for TensorRT's runtime allocation. Values
    for the second slot must come from a measured single-slot artifact probe.
    """

    fields = {
        "free_before_bytes": free_before_bytes,
        "total_device_bytes": total_device_bytes,
        "candidate_engine_device_bytes": candidate_engine_device_bytes,
        "candidate_context_bytes": candidate_context_bytes,
        "refit_workspace_bytes": refit_workspace_bytes,
        "transformed_staging_bytes": transformed_staging_bytes,
        "verification_io_bytes": verification_io_bytes,
        "safety_headroom_bytes": safety_headroom_bytes,
    }
    invalid = {name: value for name, value in fields.items() if value < 0}
    if invalid:
        raise ValueError(f"memory preflight values must be non-negative: {invalid}")
    if free_before_bytes > total_device_bytes:
        raise ValueError("free device memory cannot exceed total device memory")

    requested_bytes = sum(
        fields[name]
        for name in (
            "candidate_engine_device_bytes",
            "candidate_context_bytes",
            "refit_workspace_bytes",
            "transformed_staging_bytes",
            "verification_io_bytes",
        )
    )
    required_bytes = requested_bytes + safety_headroom_bytes
    qualified = free_before_bytes >= required_bytes
    return {
        **fields,
        "requested_bytes": requested_bytes,
        "required_with_headroom_bytes": required_bytes,
        "remaining_after_request_bytes": free_before_bytes - requested_bytes,
        "status": (
            "qualified_double_buffer" if qualified else "single_engine_inventory_only"
        ),
        "double_buffer_allowed": qualified,
        "double_buffer_lifecycle_gate_allowed": qualified,
        "single_engine_latency_or_authority_claim_forbidden": not qualified,
    }


@dataclass
class RefitRevisionFence:
    """Pure state machine for fail-closed PyTorch/TRT revision adoption."""

    active_pytorch_revision: int = 0
    active_trt_revision: int = 0
    active_slot: str = "a"
    active_probe_output_digest: str | None = None
    phase: str = "idle"
    candidate_revision: int | None = None
    candidate_slot: str | None = None
    source_weight_digest: str | None = None
    expected_staging_digest: str | None = None
    expected_inventory_digest: str | None = None
    expected_prototype_digest: str | None = None
    pytorch_head_digest: str | None = None
    observed_staging_digest: str | None = None
    observed_inventory_digest: str | None = None
    observed_prototype_digest: str | None = None
    probe_input_digest: str | None = None
    pytorch_probe_output_digest: str | None = None
    trt_probe_output_digest: str | None = None
    probe_metrics_digest: str | None = None
    failure_reason: str | None = None

    def begin_inference(self) -> None:
        if self.phase != "idle":
            raise RuntimeError(f"cannot infer while refit phase is {self.phase}")
        if self.active_pytorch_revision != self.active_trt_revision:
            raise RuntimeError(
                "cannot infer with mixed revisions: "
                f"pytorch={self.active_pytorch_revision}, "
                f"trt={self.active_trt_revision}"
            )
        self.phase = "inference"

    def end_inference(self) -> None:
        if self.phase != "inference":
            raise RuntimeError(f"cannot end inference while phase is {self.phase}")
        self.phase = "idle"

    def begin_refit(
        self,
        revision: int,
        source_weight_digest: str,
        expected_staging_digest: str,
        expected_inventory_digest: str,
        expected_prototype_digest: str,
    ) -> str:
        if self.phase != "idle":
            raise RuntimeError(f"cannot refit while phase is {self.phase}")
        if self.active_pytorch_revision != self.active_trt_revision:
            raise RuntimeError("cannot refit from an already mixed revision")
        if revision <= self.active_trt_revision:
            raise ValueError(
                f"refit revision must advance: active={self.active_trt_revision}, "
                f"candidate={revision}"
            )
        digests = {
            "source_weight_digest": source_weight_digest,
            "expected_staging_digest": expected_staging_digest,
            "expected_inventory_digest": expected_inventory_digest,
            "expected_prototype_digest": expected_prototype_digest,
        }
        if any(not value for value in digests.values()):
            raise ValueError(f"refit digests must not be empty: {digests}")
        self.phase = "refitting"
        self.candidate_revision = revision
        self.candidate_slot = "b" if self.active_slot == "a" else "a"
        self.source_weight_digest = source_weight_digest
        self.expected_staging_digest = expected_staging_digest
        self.expected_inventory_digest = expected_inventory_digest
        self.expected_prototype_digest = expected_prototype_digest
        self.pytorch_head_digest = None
        self.observed_staging_digest = None
        self.observed_inventory_digest = None
        self.observed_prototype_digest = None
        self.probe_input_digest = None
        self.pytorch_probe_output_digest = None
        self.trt_probe_output_digest = None
        self.probe_metrics_digest = None
        self.failure_reason = None
        return self.candidate_slot

    def mark_pytorch_applied(self, revision: int, head_digest: str) -> None:
        if self.phase != "refitting":
            raise RuntimeError(f"cannot mark PyTorch apply while phase is {self.phase}")
        if revision != self.candidate_revision or not head_digest:
            self.fail_refit("PyTorch head revision or digest mismatch")
            raise ValueError("PyTorch head does not match the candidate revision")
        self.active_pytorch_revision = revision
        self.pytorch_head_digest = head_digest
        self.phase = "pytorch_applied"

    def mark_staging_verified(self, observed_digest: str) -> None:
        if self.phase != "pytorch_applied":
            raise RuntimeError(f"cannot verify staging while phase is {self.phase}")
        self.observed_staging_digest = observed_digest
        if observed_digest != self.expected_staging_digest:
            self.fail_refit("transformed staging digest mismatch")
            raise ValueError("transformed staging digest does not match reference")
        self.phase = "staging_verified"

    def mark_refit_complete(self, inventory_digest: str, prototype_digest: str) -> None:
        if self.phase != "staging_verified":
            raise RuntimeError(f"cannot complete refit while phase is {self.phase}")
        self.observed_inventory_digest = inventory_digest
        self.observed_prototype_digest = prototype_digest
        mismatches = []
        if inventory_digest != self.expected_inventory_digest:
            mismatches.append("inventory")
        if prototype_digest != self.expected_prototype_digest:
            mismatches.append("prototype")
        if mismatches:
            self.fail_refit(f"refitter {'/'.join(mismatches)} digest mismatch")
            raise ValueError(
                f"refitter {'/'.join(mismatches)} does not match the frozen map"
            )
        self.phase = "refitted"

    def mark_probe_verified(
        self,
        *,
        probe_input_digest: str,
        pytorch_output_digest: str,
        trt_output_digest: str,
        metrics_digest: str,
        numerics_passed: bool,
    ) -> None:
        if self.phase != "refitted":
            raise RuntimeError(f"cannot verify probe while phase is {self.phase}")
        digests = {
            "probe_input_digest": probe_input_digest,
            "pytorch_output_digest": pytorch_output_digest,
            "trt_output_digest": trt_output_digest,
            "metrics_digest": metrics_digest,
        }
        if any(not value for value in digests.values()):
            self.fail_refit("probe evidence is incomplete")
            raise ValueError(f"probe digests must not be empty: {digests}")
        if not numerics_passed:
            self.fail_refit("candidate probe failed numerical thresholds")
            raise ValueError("candidate probe failed numerical thresholds")
        if (
            self.active_probe_output_digest is not None
            and trt_output_digest == self.active_probe_output_digest
        ):
            self.fail_refit("candidate output did not change across revisions")
            raise ValueError("candidate output did not change across revisions")
        self.probe_input_digest = probe_input_digest
        self.pytorch_probe_output_digest = pytorch_output_digest
        self.trt_probe_output_digest = trt_output_digest
        self.probe_metrics_digest = metrics_digest
        self.phase = "verified"

    def commit(self) -> None:
        if self.phase != "verified":
            raise RuntimeError(f"cannot commit while phase is {self.phase}")
        assert self.candidate_revision is not None
        assert self.candidate_slot is not None
        assert self.trt_probe_output_digest is not None
        self.active_trt_revision = self.candidate_revision
        self.active_slot = self.candidate_slot
        self.active_probe_output_digest = self.trt_probe_output_digest
        self._clear_candidate("idle")

    def abort_before_pytorch_apply(self) -> None:
        if self.phase != "refitting":
            raise RuntimeError(
                f"cannot cleanly abort after PyTorch apply; phase is {self.phase}"
            )
        self._clear_candidate("idle")

    def fail_refit(self, reason: str) -> None:
        if self.phase not in {
            "refitting",
            "pytorch_applied",
            "staging_verified",
            "refitted",
            "verified",
        }:
            raise RuntimeError(f"cannot fail refit while phase is {self.phase}")
        self.failure_reason = reason
        self._clear_candidate("failed_stopped")

    def _clear_candidate(self, next_phase: str) -> None:
        self.phase = next_phase
        self.candidate_revision = None
        self.candidate_slot = None
        self.source_weight_digest = None
        self.expected_staging_digest = None
        self.expected_inventory_digest = None
        self.expected_prototype_digest = None


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_tensors = load_checkpoint_header(args.checkpoint)
    model_config = validate_model_config(args.model_config)
    onnx_initializers = load_onnx_initializer_contract(args.onnx)
    parameter_map = build_refit_manifest(checkpoint_tensors, onnx_initializers)
    value = {
        "schema": "rlinf.gr00t-n1d7-refittable-dit-offline-gate.v1",
        "status": "passed",
        "inputs": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_header_sha256": (
                _file_sha256(args.checkpoint) if args.checkpoint.is_file() else None
            ),
            "model_config": str(args.model_config.resolve()),
            "model_config_sha256": model_config["sha256"],
            "onnx": str(args.onnx.resolve()),
            "onnx_sha256": _file_sha256(args.onnx),
        },
        "generator": {
            "onnx_version": importlib.metadata.version("onnx"),
            "python_version": sys.version.split()[0],
        },
        "model_config": model_config,
        "action_head": action_head_inventory(checkpoint_tensors),
        "dit_refit": parameter_map,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.output}")
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": value["status"],
                "initializer_count": parameter_map["initializer_count"],
                "parameter_count": parameter_map["parameter_count"],
                "byte_count": parameter_map["byte_count"],
                "mapping_sha256": parameter_map["mapping_sha256"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
