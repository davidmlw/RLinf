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

"""Revision-zero TensorRT DiT executor for the W83 PPO identity gate."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from rlinf.hybrid_engines.tensorrt import PersistentEngine

_RECEIPT_SCHEMA = "rlinf.gr00t-n1d7-trocar-true-b8-refittable-dit-engine.v1"
_EXPECTED_INPUTS = {
    "sa_embs": ([8, 41, 1536], "BF16"),
    "vl_embs": ([8, -1, 2048], "BF16"),
    "timestep": ([8], "INT64"),
    "image_mask": ([8, -1], "BOOL"),
    "backbone_attention_mask": ([8, -1], "BOOL"),
}
_EXPECTED_OUTPUT = ([8, 41, 1024], "BF16")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(config: Mapping[str, Any], name: str) -> Any:
    value = config.get(name)
    if value is None or value == "":
        raise ValueError(f"rollout.model.tensorrt_dit_diagnostic.{name} is required")
    return value


def _load_artifacts(config: Mapping[str, Any]) -> dict[str, Any]:
    engine_path = Path(str(_required(config, "engine_path"))).resolve(strict=True)
    receipt_path = Path(str(_required(config, "receipt_path"))).resolve(strict=True)
    map_path = Path(str(_required(config, "parameter_map_path"))).resolve(strict=True)
    expected_receipt_sha = str(_required(config, "receipt_sha256"))
    expected_map_sha = str(_required(config, "parameter_map_sha256"))
    if _sha256(receipt_path) != expected_receipt_sha:
        raise RuntimeError("refittable DiT receipt SHA-256 mismatch")
    if _sha256(map_path) != expected_map_sha:
        raise RuntimeError("refittable DiT parameter-map SHA-256 mismatch")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    parameter_map = json.loads(map_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != _RECEIPT_SCHEMA or receipt.get("status") != "passed":
        raise RuntimeError("refittable DiT engine receipt is not qualified")
    if receipt["engine"]["sha256"] != _sha256(engine_path):
        raise RuntimeError("refittable DiT plan SHA-256 mismatch")
    if receipt["parameter_map"]["sha256"] != expected_map_sha:
        raise RuntimeError("engine receipt and parameter map disagree")
    entries = parameter_map.get("dit_refit", {}).get("entries", [])
    if len(entries) != 456:
        raise RuntimeError("refittable DiT map must contain exactly 456 tensors")
    return {
        "engine_path": engine_path,
        "receipt_path": receipt_path,
        "receipt_sha256": expected_receipt_sha,
        "receipt": receipt,
        "parameter_map_path": map_path,
        "parameter_map_sha256": expected_map_sha,
        "entries": entries,
    }


def _validate_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    import tensorrt as trt

    expected_version = str(_required(config, "runtime_version"))
    distribution = str(config.get("runtime_distribution", "tensorrt-cu12"))
    installed_distribution = importlib.metadata.version(distribution)
    if (
        trt.__version__ != expected_version
        or installed_distribution != expected_version
    ):
        raise RuntimeError(
            "TensorRT version mismatch: "
            f"module={trt.__version__}, distribution={installed_distribution}, "
            f"expected={expected_version}"
        )
    capability = list(torch.cuda.get_device_capability(torch.cuda.current_device()))
    expected_capability = [
        int(item) for item in _required(config, "compute_capability")
    ]
    if capability != expected_capability:
        raise RuntimeError(
            f"refittable DiT requires SM{expected_capability}, found SM{capability}"
        )
    return {
        "tensorrt": trt.__version__,
        "distribution": installed_distribution,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "compute_capability": capability,
    }


def _normalized_dtype(value: str) -> str:
    return value.rsplit(".", 1)[-1]


def _validate_bindings(engine: PersistentEngine, receipt: Mapping[str, Any]) -> None:
    actual = engine.binding_manifest()
    expected = receipt["engine"]["bindings"]
    if len(actual) != 6 or len(expected) != 6:
        raise RuntimeError("refittable DiT engine must expose six bindings")
    for actual_item, expected_item in zip(actual, expected, strict=True):
        comparable = dict(actual_item)
        comparable["dtype"] = _normalized_dtype(str(comparable["dtype"]))
        if comparable != expected_item:
            raise RuntimeError(
                "refittable DiT binding differs from its build receipt: "
                f"{comparable} != {expected_item}"
            )
    inputs = {
        item["name"]: (item["shape"], item["dtype"])
        for item in expected
        if item["mode"] == "input"
    }
    outputs = [
        (item["shape"], item["dtype"]) for item in expected if item["mode"] == "output"
    ]
    if inputs != _EXPECTED_INPUTS or outputs != [_EXPECTED_OUTPUT]:
        raise RuntimeError("refittable DiT engine does not implement the W83 B8 ABI")


def _ordered_source_digest(
    action_model: torch.nn.Module, entries: list[dict[str, Any]]
) -> str:
    state = action_model.state_dict()
    prefix = "action_head.model."
    digest = hashlib.sha256()
    seen = set()
    for entry in entries:
        source_fqn = entry["source_fqn"]
        if not source_fqn.startswith(prefix):
            raise RuntimeError(f"refit source is outside DiT: {source_fqn}")
        name = source_fqn[len(prefix) :]
        if name not in state:
            raise RuntimeError(f"Rollout DiT is missing {source_fqn}")
        value = state[name]
        if list(value.shape) != entry["source_shape"] or value.dtype != torch.bfloat16:
            raise RuntimeError(f"Rollout DiT tensor ABI mismatch: {source_fqn}")
        metadata = json.dumps(
            {"name": source_fqn, "shape": list(value.shape), "dtype": str(value.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        payload = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
        seen.add(name)
    if seen != set(state):
        raise RuntimeError("Rollout DiT keyspace differs from the refit parameter map")
    return digest.hexdigest()


class TensorRTDiTRevisionZeroDiagnostic:
    """Execute DiT with TensorRT only for the pre-update revision-zero gate."""

    def __init__(
        self,
        action_model: torch.nn.Module,
        config: Mapping[str, Any],
    ) -> None:
        self.config = dict(config)
        self.artifacts = _load_artifacts(config)
        self.runtime = _validate_runtime(config)
        self.expected_source_digest = str(_required(config, "source_digest_revision_0"))
        self.expected_revision = int(config.get("revision", 0))
        if self.expected_revision != 0:
            raise ValueError("W83 diagnostic supports only the initial revision zero")
        self.engine = PersistentEngine(str(self.artifacts["engine_path"]))
        try:
            _validate_bindings(self.engine, self.artifacts["receipt"])
        except Exception:
            self.engine.close()
            raise
        self.action_model = action_model
        self.active_revision: int | None = None
        self.observed_source_digest: str | None = None
        self.closed = False

    def verify_revision(self, revision: int) -> None:
        """Admit only the exact initial Actor DiT bytes used to build the plan."""

        if revision != self.expected_revision:
            raise RuntimeError(
                "W83 revision-zero diagnostic refuses online updates: "
                f"expected={self.expected_revision}, found={revision}"
            )
        observed = _ordered_source_digest(self.action_model, self.artifacts["entries"])
        if observed != self.expected_source_digest:
            raise RuntimeError(
                "Rollout DiT revision-zero bytes differ from the qualified plan source"
            )
        self.observed_source_digest = observed
        self.active_revision = revision

    def __call__(
        self,
        sa_embs: torch.Tensor,
        vl_embs: torch.Tensor,
        timestep: torch.Tensor,
        *,
        image_mask: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("TensorRT DiT diagnostic is closed")
        if self.active_revision != self.expected_revision:
            raise RuntimeError(
                "TensorRT DiT inference attempted before revision admission"
            )
        inputs = {
            "sa_embs": sa_embs,
            "vl_embs": vl_embs,
            "timestep": timestep,
            "image_mask": image_mask,
            "backbone_attention_mask": backbone_attention_mask,
        }
        expected_shapes = {
            "sa_embs": (8, 41, 1536),
            "vl_embs": (8, 208, 2048),
            "timestep": (8,),
            "image_mask": (8, 208),
            "backbone_attention_mask": (8, 208),
        }
        actual_shapes = {name: tuple(value.shape) for name, value in inputs.items()}
        if actual_shapes != expected_shapes:
            raise RuntimeError(
                f"TensorRT DiT live ABI mismatch: {actual_shapes} != {expected_shapes}"
            )
        for name in ("vl_embs", "image_mask", "backbone_attention_mask"):
            self.engine.set_runtime_tensor_shape(name, inputs[name].shape)
        return self.engine(**inputs)["output"]

    def telemetry(self) -> dict[str, Any]:
        """Return artifact, revision, and runtime evidence for worker receipts."""

        return {
            "scope": "revision_zero_PPO_identity_diagnostic_no_online_refit",
            "receipt_sha256": self.artifacts["receipt_sha256"],
            "parameter_map_sha256": self.artifacts["parameter_map_sha256"],
            "expected_source_digest": self.expected_source_digest,
            "observed_source_digest": self.observed_source_digest,
            "active_revision": self.active_revision,
            "runtime": self.runtime,
            "engine": self.engine.telemetry(),
            "closed": self.closed,
        }

    def close(self) -> None:
        """Wait for the final execution and release the TensorRT resources."""

        if self.closed:
            return
        self.engine.close()
        self.closed = True
