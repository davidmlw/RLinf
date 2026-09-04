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

"""Refittable TensorRT DiT executor for GR00T N1.7 rollout workers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import statistics
import time
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
_MIN_PROBE_COSINE = 0.999
_MAX_PROBE_RELATIVE_L2 = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(config: Mapping[str, Any], name: str) -> Any:
    value = config.get(name)
    if value is None or value == "":
        raise ValueError(f"TensorRT DiT config field {name!r} is required")
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


def _compare_outputs(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    lhs = reference.detach().float().flatten()
    rhs = candidate.detach().float().flatten()
    difference = rhs - lhs
    lhs_norm = torch.linalg.vector_norm(lhs)
    relative_l2 = torch.linalg.vector_norm(difference) / torch.clamp(
        lhs_norm, min=1e-12
    )
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(lhs, rhs, dim=0)),
        "mean_abs": float(difference.abs().mean()),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(relative_l2),
        "finite": bool(torch.isfinite(rhs).all()),
    }


def _timing_summary(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"count": 0}
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    return {
        "count": len(samples),
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile(0.95),
        "max_ms": max(samples),
    }


class CudaTimedDiTForward:
    """Measure a DiT callable with CUDA events without synchronizing each call."""

    def __init__(self, forward: Any) -> None:
        self.forward = forward
        self.revision: int | None = None
        self._events: list[tuple[int | None, torch.cuda.Event, torch.cuda.Event]] = []
        self._samples: list[float] = []
        self._samples_by_revision: dict[int, list[float]] = {}
        self._recent: dict[str, Any] = {"count": 0}

    def set_revision(self, revision: int) -> None:
        self.revision = revision

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not torch.cuda.is_available():
            return self.forward(*args, **kwargs)
        stream = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        output = self.forward(*args, **kwargs)
        end.record(stream)
        self._events.append((self.revision, start, end))
        return output

    def telemetry(self) -> dict[str, Any]:
        recent = []
        if self._events:
            self._events[-1][1].synchronize()
            recent = [
                float(start.elapsed_time(end)) for _revision, start, end in self._events
            ]
            for (revision, _start, _end), sample in zip(
                self._events, recent, strict=True
            ):
                if revision is not None:
                    self._samples_by_revision.setdefault(revision, []).append(sample)
            self._samples.extend(recent)
            self._events.clear()
        self._recent = _timing_summary(recent)
        return {
            "enabled": True,
            "executor": "pytorch_eager",
            "revision": self.revision,
            "recent": self._recent,
            "cumulative": _timing_summary(self._samples),
            "by_revision": {
                str(revision): _timing_summary(samples)
                for revision, samples in sorted(self._samples_by_revision.items())
            },
        }


class RefittableTensorRTDiT:
    """Execute DiT with TensorRT and optionally refit it at revision boundaries."""

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
        self.online_refit = bool(config.get("online_refit", False))
        self.probe_each_revision = bool(config.get("probe_each_revision", True))
        self.probe_seed = int(config.get("probe_seed", 83001))
        self.minimum_probe_cosine = float(
            config.get("minimum_probe_cosine", _MIN_PROBE_COSINE)
        )
        self.maximum_probe_relative_l2 = float(
            config.get("maximum_probe_relative_l2", _MAX_PROBE_RELATIVE_L2)
        )
        if self.minimum_probe_cosine < _MIN_PROBE_COSINE:
            raise ValueError("TensorRT DiT probe cosine gate cannot be weakened")
        if self.maximum_probe_relative_l2 > _MAX_PROBE_RELATIVE_L2:
            raise ValueError("TensorRT DiT probe relative-L2 gate cannot be weakened")

        free_before_engines, total_device_bytes = torch.cuda.mem_get_info()
        self.engines = [PersistentEngine(str(self.artifacts["engine_path"]))]
        if self.online_refit:
            self.engines.append(PersistentEngine(str(self.artifacts["engine_path"])))
        try:
            for engine in self.engines:
                _validate_bindings(engine, self.artifacts["receipt"])
        except Exception:
            for engine in self.engines:
                engine.close()
            raise
        self.engine = self.engines[0]
        self.active_slot = 0
        self.action_model = action_model
        self.shadow_eager = bool(config.get("shadow_eager", False))
        self.eager_forward = (
            action_model.forward if self.shadow_eager or self.online_refit else None
        )
        self.shadow_calls = 0
        self.shadow_by_timestep: dict[int, dict[str, Any]] = {}
        self.active_revision: int | None = None
        self.observed_source_digest: str | None = None
        self.refit_records: list[dict[str, Any]] = []
        self._timing_events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []
        self._timing_samples: list[float] = []
        self._timing_samples_by_revision: dict[int, list[float]] = {}
        self._recent_timing: dict[str, Any] = {"count": 0}
        self._phase = "idle"
        self._probe = None
        self._transpose_staging: dict[str, torch.Tensor] = {}
        self._refitters: list[Any] = []
        free_after_engines, _ = torch.cuda.mem_get_info()
        self._memory: dict[str, Any] = {
            "free_before_engines_bytes": free_before_engines,
            "free_after_engines_bytes": free_after_engines,
            "engine_retained_bytes": free_before_engines - free_after_engines,
            "total_device_bytes": total_device_bytes,
        }
        self.closed = False
        if self.online_refit:
            try:
                self._initialize_online_refit()
            except Exception:
                for engine in self.engines:
                    engine.close()
                self.closed = True
                raise

    def _initialize_online_refit(self) -> None:
        import tensorrt as trt

        if len(self.engines) != 2:
            raise RuntimeError("online TensorRT DiT requires exactly two engine slots")
        device = next(self.action_model.parameters()).device
        generator = torch.Generator(device="cpu").manual_seed(self.probe_seed)

        def random_bf16(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.randn(shape, generator=generator).to(
                device=device, dtype=torch.bfloat16
            )

        self._probe = {
            "sa_embs": random_bf16((8, 41, 1536)),
            "vl_embs": random_bf16((8, 208, 2048)),
            "timestep": torch.full((8,), 500, dtype=torch.int64, device=device),
            "image_mask": torch.ones((8, 208), dtype=torch.bool, device=device),
            "backbone_attention_mask": torch.ones(
                (8, 208), dtype=torch.bool, device=device
            ),
        }
        for engine in self.engines:
            for name in ("vl_embs", "image_mask", "backbone_attention_mask"):
                engine.set_runtime_tensor_shape(name, self._probe[name].shape)
        state = self.action_model.state_dict()
        for entry in self.artifacts["entries"]:
            if entry["transform"] != "transpose_2d":
                continue
            name = entry["source_fqn"].removeprefix("action_head.model.")
            value = state[name]
            self._transpose_staging[entry["initializer"]] = torch.empty(
                tuple(entry["initializer_shape"]),
                dtype=value.dtype,
                device=value.device,
            )
        self._refitters = []
        for engine in self.engines:
            refitter = trt.Refitter(engine.handle, engine.logger)
            refitter.weights_validation = True
            self._refitters.append(refitter)
        free, total = torch.cuda.mem_get_info(device)
        minimum_headroom = int(self.config.get("minimum_free_device_bytes", 8 << 30))
        if free < minimum_headroom:
            raise RuntimeError(
                "online TensorRT DiT did not retain the configured device headroom"
            )
        self._memory.update(
            {
                "free_after_initialization_bytes": free,
                "total_device_bytes": total,
                "minimum_free_device_bytes": minimum_headroom,
                "transpose_staging_bytes": sum(
                    value.numel() * value.element_size()
                    for value in self._transpose_staging.values()
                ),
            }
        )

    def _source_state(self) -> dict[str, torch.Tensor]:
        state = self.action_model.state_dict()
        result = {}
        prefix = "action_head.model."
        for entry in self.artifacts["entries"]:
            source_fqn = entry["source_fqn"]
            name = source_fqn.removeprefix(prefix)
            if not source_fqn.startswith(prefix) or name not in state:
                raise RuntimeError(f"TensorRT DiT source is missing {source_fqn}")
            value = state[name]
            if list(value.shape) != entry["source_shape"]:
                raise RuntimeError(
                    f"TensorRT DiT source shape changed for {source_fqn}"
                )
            if value.dtype != torch.bfloat16 or not value.is_cuda:
                raise RuntimeError(f"TensorRT DiT source ABI changed for {source_fqn}")
            result[source_fqn] = value
        return result

    def _stage_weights(
        self, source: Mapping[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        stream = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start.record(stream)
        staged = {}
        for entry in self.artifacts["entries"]:
            value = source[entry["source_fqn"]]
            if entry["transform"] == "identity":
                if not value.is_contiguous():
                    raise RuntimeError(
                        f"identity TensorRT refit source is not contiguous: "
                        f"{entry['source_fqn']}"
                    )
                transformed = value
            elif entry["transform"] == "transpose_2d":
                transformed = self._transpose_staging[entry["initializer"]]
                transformed.copy_(value.transpose(0, 1))
            else:
                raise RuntimeError(
                    f"unsupported TensorRT refit transform: {entry['transform']}"
                )
            if list(transformed.shape) != entry["initializer_shape"]:
                raise RuntimeError(
                    f"TensorRT refit staging shape changed: {entry['initializer']}"
                )
            staged[entry["initializer"]] = transformed
        end.record(stream)
        end.synchronize()
        return staged, {
            "staging_device_ms": float(start.elapsed_time(end)),
            "staging_wall_ms": (time.perf_counter() - wall_started) * 1000,
        }

    def _refit_slot(
        self, slot: int, staged: Mapping[str, torch.Tensor]
    ) -> dict[str, Any]:
        import tensorrt as trt

        refitter = self._refitters[slot]
        references = []
        set_started = time.perf_counter()
        for entry in self.artifacts["entries"]:
            name = entry["initializer"]
            value = staged[name]
            weights = trt.Weights(trt.bfloat16, value.data_ptr(), value.numel())
            if not refitter.set_named_weights(name, weights, trt.TensorLocation.DEVICE):
                raise RuntimeError(f"TensorRT rejected refit weight {name}")
            references.append(weights)
        missing_before = sorted(refitter.get_missing_weights())
        if missing_before:
            raise RuntimeError(
                f"TensorRT DiT refit has missing weights: {missing_before[:8]}"
            )
        set_wall_ms = (time.perf_counter() - set_started) * 1000
        stream = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start.record(stream)
        if not refitter.refit_cuda_engine_async(stream.cuda_stream):
            raise RuntimeError("TensorRT DiT async refit returned false")
        end.record(stream)
        end.synchronize()
        wall_ms = (time.perf_counter() - wall_started) * 1000
        missing_after = sorted(refitter.get_missing_weights())
        if missing_after:
            raise RuntimeError(
                f"TensorRT DiT refit left missing weights: {missing_after[:8]}"
            )
        del references
        return {
            "weight_count": len(staged),
            "set_weights_wall_ms": set_wall_ms,
            "refit_wall_ms": wall_ms,
            "refit_device_ms": float(start.elapsed_time(end)),
        }

    def _probe_engine(self, engine: PersistentEngine) -> torch.Tensor:
        output = engine(**self._probe)["output"]
        torch.cuda.current_stream().synchronize()
        return output.detach().clone()

    def _verify_probe(self, engine: PersistentEngine) -> dict[str, Any]:
        with torch.inference_mode():
            eager = self.eager_forward(
                hidden_states=self._probe["sa_embs"],
                encoder_hidden_states=self._probe["vl_embs"],
                timestep=self._probe["timestep"],
                image_mask=self._probe["image_mask"],
                backbone_attention_mask=self._probe["backbone_attention_mask"],
            ).detach()
            candidate = self._probe_engine(engine)
        metrics = _compare_outputs(eager, candidate)
        if not (
            metrics["finite"]
            and metrics["cosine"] >= self.minimum_probe_cosine
            and metrics["relative_l2"] <= self.maximum_probe_relative_l2
        ):
            raise RuntimeError(f"TensorRT DiT revision probe failed: {metrics}")
        return metrics

    def _adopt_online_revision(self, revision: int) -> None:
        if self.active_revision is None or revision != self.active_revision + 1:
            raise RuntimeError(
                "TensorRT DiT revisions must be adopted monotonically: "
                f"active={self.active_revision}, requested={revision}"
            )
        if self._phase != "idle":
            raise RuntimeError(f"TensorRT DiT cannot refit while phase={self._phase}")
        self._phase = "refitting"
        wall_started = time.perf_counter()
        inactive_slot = 1 - self.active_slot
        try:
            source = self._source_state()
            staged, staging = self._stage_weights(source)
            refit = self._refit_slot(inactive_slot, staged)
            probe = (
                self._verify_probe(self.engines[inactive_slot])
                if self.probe_each_revision
                else None
            )
            free, _ = torch.cuda.mem_get_info()
            minimum_headroom = self._memory["minimum_free_device_bytes"]
            if free < minimum_headroom:
                raise RuntimeError(
                    "TensorRT DiT refit violated configured device headroom"
                )
            self.active_slot = inactive_slot
            self.engine = self.engines[inactive_slot]
            self.active_revision = revision
            self.refit_records.append(
                {
                    "revision": revision,
                    "active_slot": inactive_slot,
                    **staging,
                    **refit,
                    "probe": probe,
                    "adoption_wall_ms": (time.perf_counter() - wall_started) * 1000,
                    "free_after_adoption_bytes": free,
                }
            )
            self._phase = "idle"
        except Exception:
            self._phase = "failed_stopped"
            raise

    def verify_revision(self, revision: int) -> None:
        """Verify revision zero or refit and atomically adopt an online revision."""

        if revision == self.active_revision:
            return
        if revision != self.expected_revision and not self.online_refit:
            raise RuntimeError(
                "W83 revision-zero diagnostic refuses online updates: "
                f"expected={self.expected_revision}, found={revision}"
            )
        if revision != self.expected_revision:
            self._adopt_online_revision(revision)
            return
        observed = _ordered_source_digest(self.action_model, self.artifacts["entries"])
        if observed != self.expected_source_digest:
            raise RuntimeError(
                "Rollout DiT revision-zero bytes differ from the qualified plan source"
            )
        if self.online_refit:
            probe = self._verify_probe(self.engines[self.active_slot])
            self.refit_records.append(
                {
                    "revision": revision,
                    "active_slot": self.active_slot,
                    "initial_probe": probe,
                }
            )
        self.observed_source_digest = observed
        self.active_revision = revision

    def __call__(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        image_mask: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("TensorRT DiT diagnostic is closed")
        if getattr(self, "_phase", "idle") != "idle" or self.active_revision is None:
            raise RuntimeError(
                "TensorRT DiT inference attempted before revision admission"
            )
        inputs = {
            "sa_embs": hidden_states,
            "vl_embs": encoder_hidden_states,
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
        timing = hidden_states.is_cuda
        if timing:
            stream = torch.cuda.current_stream()
            timing_start = torch.cuda.Event(enable_timing=True)
            timing_end = torch.cuda.Event(enable_timing=True)
            timing_start.record(stream)
        output = self.engine(**inputs)["output"]
        if timing:
            timing_end.record(stream)
            if not hasattr(self, "_timing_events"):
                self._timing_events = []
            self._timing_events.append((self.active_revision, timing_start, timing_end))
        if self.shadow_eager:
            with torch.no_grad():
                eager_output = self.eager_forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    image_mask=image_mask,
                    backbone_attention_mask=backbone_attention_mask,
                )
            self._record_shadow(timestep, eager_output, output)
        return output

    def _record_shadow(
        self,
        timestep: torch.Tensor,
        eager_output: torch.Tensor,
        tensorrt_output: torch.Tensor,
    ) -> None:
        """Accumulate same-input eager/TensorRT errors without changing rollout."""

        timestep_values = set(timestep.detach().cpu().tolist())
        if len(timestep_values) != 1:
            raise RuntimeError(
                f"TensorRT DiT shadow expected one timestep, found {timestep_values}"
            )
        if eager_output.shape != tensorrt_output.shape:
            raise RuntimeError(
                "TensorRT DiT shadow output shape mismatch: "
                f"{tuple(eager_output.shape)} != {tuple(tensorrt_output.shape)}"
            )
        if eager_output.dtype != tensorrt_output.dtype:
            raise RuntimeError(
                "TensorRT DiT shadow output dtype mismatch: "
                f"{eager_output.dtype} != {tensorrt_output.dtype}"
            )
        reference = eager_output.detach().float()
        candidate = tensorrt_output.detach().float()
        if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
            raise RuntimeError("TensorRT DiT shadow comparison found non-finite output")
        difference = candidate - reference
        absolute = difference.abs()
        terms = torch.stack(
            (
                absolute.sum(),
                difference.square().sum(),
                reference.square().sum(),
                candidate.square().sum(),
                (reference * candidate).sum(),
                absolute.max(),
            )
        ).double()
        sum_abs, error_sq, reference_sq, candidate_sq, dot, max_abs = (
            terms.cpu().tolist()
        )
        timestep_bucket = int(timestep_values.pop())
        elements = eager_output.numel()
        call = {
            "elements": elements,
            "mean_abs": sum_abs / elements,
            "max_abs": max_abs,
            "relative_l2": math.sqrt(error_sq / reference_sq) if reference_sq else None,
            "cosine": dot / math.sqrt(reference_sq * candidate_sq)
            if reference_sq and candidate_sq
            else None,
        }
        aggregate = self.shadow_by_timestep.setdefault(
            timestep_bucket,
            {
                "calls": 0,
                "elements": 0,
                "sum_abs": 0.0,
                "error_sq": 0.0,
                "reference_sq": 0.0,
                "candidate_sq": 0.0,
                "dot": 0.0,
                "max_abs": 0.0,
                "first_call": call,
            },
        )
        aggregate["calls"] += 1
        aggregate["elements"] += elements
        aggregate["sum_abs"] += sum_abs
        aggregate["error_sq"] += error_sq
        aggregate["reference_sq"] += reference_sq
        aggregate["candidate_sq"] += candidate_sq
        aggregate["dot"] += dot
        aggregate["max_abs"] = max(aggregate["max_abs"], max_abs)
        self.shadow_calls += 1

    def _shadow_summary(self) -> dict[str, Any]:
        per_timestep = []
        for timestep, aggregate in sorted(self.shadow_by_timestep.items()):
            reference_sq = aggregate["reference_sq"]
            candidate_sq = aggregate["candidate_sq"]
            per_timestep.append(
                {
                    "timestep_bucket": timestep,
                    "calls": aggregate["calls"],
                    "elements": aggregate["elements"],
                    "mean_abs": aggregate["sum_abs"] / aggregate["elements"],
                    "max_abs": aggregate["max_abs"],
                    "relative_l2": math.sqrt(aggregate["error_sq"] / reference_sq)
                    if reference_sq
                    else None,
                    "cosine": aggregate["dot"] / math.sqrt(reference_sq * candidate_sq)
                    if reference_sq and candidate_sq
                    else None,
                    "first_call": aggregate["first_call"],
                }
            )
        return {
            "enabled": self.shadow_eager,
            "calls": self.shadow_calls,
            "trajectory_executor": "tensorrt",
            "shadow_executor": "pytorch_eager",
            "per_timestep": per_timestep,
        }

    def telemetry(self) -> dict[str, Any]:
        """Return artifact, revision, and runtime evidence for worker receipts."""

        recent = []
        timing_events = getattr(self, "_timing_events", [])
        timing_samples = getattr(self, "_timing_samples", [])
        timing_samples_by_revision = getattr(self, "_timing_samples_by_revision", {})
        if timing_events:
            timing_events[-1][2].synchronize()
            recent = [
                float(start.elapsed_time(end))
                for _revision, start, end in timing_events
            ]
            for (revision, _start, _end), sample in zip(
                timing_events, recent, strict=True
            ):
                timing_samples_by_revision.setdefault(revision, []).append(sample)
            timing_samples.extend(recent)
            timing_events.clear()
        self._recent_timing = _timing_summary(recent)

        return {
            "scope": (
                "online_double_slot_refittable_tensorrt_dit"
                if self.online_refit
                else "revision_zero_PPO_identity_diagnostic_no_online_refit"
            ),
            "receipt_sha256": self.artifacts["receipt_sha256"],
            "parameter_map_sha256": self.artifacts["parameter_map_sha256"],
            "expected_source_digest": self.expected_source_digest,
            "observed_source_digest": self.observed_source_digest,
            "active_revision": self.active_revision,
            "active_slot": self.active_slot,
            "phase": self._phase,
            "online_refit": self.online_refit,
            "memory": self._memory,
            "refit_records": self.refit_records,
            "inference_timing_recent": self._recent_timing,
            "inference_timing_cumulative": _timing_summary(timing_samples),
            "inference_timing_by_revision": {
                str(revision): _timing_summary(samples)
                for revision, samples in sorted(timing_samples_by_revision.items())
            },
            "shadow": self._shadow_summary(),
            "runtime": self.runtime,
            "engine": self.engine.telemetry(),
            "engines": [engine.telemetry() for engine in self.engines],
            "closed": self.closed,
        }

    def close(self) -> None:
        """Wait for the final execution and release the TensorRT resources."""

        if self.closed:
            return
        for engine in self.engines:
            engine.close()
        self.closed = True


# Compatibility alias for revision-zero W83 diagnostic configs and receipts.
TensorRTDiTRevisionZeroDiagnostic = RefittableTensorRTDiT
