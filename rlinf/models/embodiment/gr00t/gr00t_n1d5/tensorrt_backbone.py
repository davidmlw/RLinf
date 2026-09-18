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

"""Persistent TensorRT ViT+LLM backend for the frozen GR00T N1.5 Eagle."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

from rlinf.hybrid_engines.tensorrt import PersistentEngine

_ENGINE_FILES = ("llm_bf16.engine", "vit_bf16.engine")
_INPUT_KEYS = (
    "eagle_input_ids",
    "eagle_attention_mask",
    "eagle_pixel_values",
    "eagle_image_sizes",
)
_RECEIPT_SCHEMA = "rlinf.gr00t-n1d5-stack-cube-true-b8-engines.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(config: Mapping[str, Any], name: str) -> Any:
    value = config.get(name)
    if value is None or value == "":
        raise ValueError(f"rollout.model.tensorrt_backbone.{name} is required")
    return value


def _validate_artifacts(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(_required(config, "engine_dir"))).expanduser().resolve(strict=True)
    receipt_path = Path(
        str(config.get("receipt_path", root / "rlinf-engine-receipt.json"))
    ).expanduser().resolve(strict=True)
    expected_receipt_sha256 = str(_required(config, "receipt_sha256"))
    actual_receipt_sha256 = _sha256(receipt_path)
    if actual_receipt_sha256 != expected_receipt_sha256:
        raise RuntimeError(
            "TensorRT engine receipt SHA-256 mismatch: "
            f"{actual_receipt_sha256} != {expected_receipt_sha256}"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != _RECEIPT_SCHEMA or receipt.get("status") != "passed":
        raise RuntimeError(f"TensorRT engine receipt is not qualified: {receipt_path}")
    expected = {
        "static_batch": int(_required(config, "static_batch_size")),
        "image_views": int(_required(config, "image_views")),
        "image_batch": int(_required(config, "image_batch_size")),
        "sequence_length": int(_required(config, "sequence_length")),
        "precision": "bfloat16",
        "silent_fallback": False,
    }
    actual = {name: receipt.get(name) for name in expected}
    if actual != expected:
        raise RuntimeError(
            f"TensorRT engine receipt shape/precision mismatch: {actual} != {expected}"
        )
    if set(receipt.get("engines", {})) != set(_ENGINE_FILES):
        raise RuntimeError("TensorRT receipt must contain exactly ViT and LLM engines")
    actual_plans = {path.name for path in root.glob("*.engine")}
    if actual_plans != set(_ENGINE_FILES):
        raise RuntimeError(
            "TensorRT runtime directory must contain exactly the qualified plans: "
            f"{sorted(actual_plans)}"
        )
    files = {}
    for name in _ENGINE_FILES:
        path = root / name
        expected_file = receipt["engines"][name]
        actual_sha256 = _sha256(path)
        if (
            actual_sha256 != expected_file.get("sha256")
            or path.stat().st_size != expected_file.get("bytes")
        ):
            raise RuntimeError(f"TensorRT plan does not match its receipt: {name}")
        files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_sha256,
        }
    metadata_path = root / "export_metadata.json"
    metadata_sha256 = _sha256(metadata_path)
    if metadata_sha256 != receipt.get("export_metadata_sha256"):
        raise RuntimeError("TensorRT export metadata does not match its receipt")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "schema": "rlinf.gr00t-n1d5-stack-cube-true-b8-onnx.v1",
        "status": "passed",
        "model_version": "n1d5",
        "batch_size": expected["static_batch"],
        "image_views": expected["image_views"],
        "image_batch": expected["image_batch"],
        "sequence_length": expected["sequence_length"],
        "precision": expected["precision"],
    }
    actual_metadata = {name: metadata.get(name) for name in expected_metadata}
    if actual_metadata != expected_metadata:
        raise RuntimeError(
            "TensorRT export metadata shape/precision mismatch: "
            f"{actual_metadata} != {expected_metadata}"
        )
    return {
        "root": root,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_sha256": actual_receipt_sha256,
        "metadata_path": metadata_path,
        "metadata_sha256": metadata_sha256,
        "metadata": metadata,
        "files": files,
    }


def _validate_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    import tensorrt as trt

    expected_version = str(_required(config, "runtime_version"))
    distribution = str(config.get("runtime_distribution", "tensorrt-cu12"))
    distribution_version = importlib.metadata.version(distribution)
    if trt.__version__ != expected_version or distribution_version != expected_version:
        raise RuntimeError(
            "TensorRT runtime version mismatch: "
            f"module={trt.__version__} distribution={distribution_version} "
            f"expected={expected_version}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT backbone requires a CUDA device")
    device = torch.cuda.current_device()
    capability = list(torch.cuda.get_device_capability(device))
    expected_capability = [
        int(value) for value in _required(config, "compute_capability")
    ]
    if capability != expected_capability:
        raise RuntimeError(
            f"TensorRT engine requires SM{expected_capability}, found SM{capability}"
        )
    return {
        "tensorrt_module": trt.__version__,
        "tensorrt_distribution_name": distribution,
        "tensorrt_distribution": distribution_version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": capability,
    }


class TensorRTFrozenEagleBackbone:
    """Own two qualified static-B8 engines and their persistent CUDA buffers."""

    reuses_output_buffers = True

    def __init__(self, backbone: torch.nn.Module, config: Mapping[str, Any]):
        self.backbone = backbone
        self.config = dict(config)
        self.artifacts = _validate_artifacts(config)
        self.runtime = _validate_runtime(config)
        receipt = self.artifacts["receipt"]
        self.expected_batch = int(receipt["static_batch"])
        self.expected_views = int(receipt["image_views"])
        self.expected_image_batch = int(receipt["image_batch"])
        self.expected_sequence = int(receipt["sequence_length"])

        eagle = backbone.eagle_model
        vision_model = getattr(eagle, "vision_model", None)
        language_model = getattr(eagle, "language_model", None)
        mlp = getattr(eagle, "mlp1", None)
        if vision_model is None or language_model is None or mlp is None:
            raise RuntimeError("unsupported or already-consumed N1.5 Eagle layout")
        self._embedding_layer = language_model.get_input_embeddings()
        self._image_token_index = int(eagle.image_token_index)

        root = self.artifacts["root"]
        vit_engine = None
        llm_engine = None
        try:
            vit_engine = PersistentEngine(str(root / "vit_bf16.engine"))
            llm_engine = PersistentEngine(str(root / "llm_bf16.engine"))
            self._validate_bindings("vit_bf16.engine", vit_engine)
            self._validate_bindings("llm_bf16.engine", llm_engine)
        except Exception:
            if llm_engine is not None:
                llm_engine.close()
            if vit_engine is not None:
                vit_engine.close()
            raise
        self.vit_engine = vit_engine
        self.llm_engine = llm_engine
        device = next(self._embedding_layer.parameters()).device
        self._position_ids = torch.arange(
            256, device=device, dtype=torch.int64
        ).expand(self.expected_image_batch, -1).contiguous()

        del eagle.vision_model
        del eagle.language_model
        del eagle.mlp1
        torch.cuda.empty_cache()
        self.closed = False

    def _validate_bindings(self, name: str, engine: PersistentEngine) -> None:
        expected = self.artifacts["receipt"]["engines"][name]["bindings"]
        actual = engine.binding_manifest()
        if actual != expected:
            raise RuntimeError(
                f"deserialized TensorRT bindings differ from receipt for {name}"
            )

    def _validate_input(self, values: Mapping[str, torch.Tensor]) -> None:
        missing = [name for name in _INPUT_KEYS if name not in values]
        if missing:
            raise KeyError(f"TensorRT Eagle inputs are incomplete: {missing}")
        expected_shapes = {
            "eagle_input_ids": (self.expected_batch, self.expected_sequence),
            "eagle_attention_mask": (self.expected_batch, self.expected_sequence),
            "eagle_pixel_values": (self.expected_image_batch, 3, 224, 224),
            "eagle_image_sizes": (self.expected_image_batch, 2),
        }
        actual_shapes = {name: tuple(values[name].shape) for name in _INPUT_KEYS}
        if actual_shapes != expected_shapes:
            raise ValueError(
                "TensorRT Eagle input shapes differ: "
                f"{actual_shapes} != {expected_shapes}"
            )
        if self.expected_image_batch != self.expected_batch * self.expected_views:
            raise RuntimeError(
                "qualified image-batch contract is internally inconsistent"
            )
        for name in ("eagle_input_ids", "eagle_attention_mask", "eagle_image_sizes"):
            if values[name].dtype != torch.int64:
                raise TypeError(f"TensorRT Eagle input {name} must be int64")
        if values["eagle_pixel_values"].dtype != torch.bfloat16:
            raise TypeError("TensorRT Eagle pixels must be bfloat16")
        if any(not values[name].is_cuda for name in _INPUT_KEYS):
            raise TypeError("TensorRT Eagle inputs must remain CUDA-resident")

    def __call__(self, values: Mapping[str, torch.Tensor]) -> BatchFeature:
        if self.closed:
            raise RuntimeError("TensorRT Eagle backbone is closed")
        self._validate_input(values)
        pixel_values = values["eagle_pixel_values"]
        vit_outputs = self.vit_engine(pixel_values, self._position_ids)
        image_features = vit_outputs["image_features"]

        input_ids = values["eagle_input_ids"]
        inputs_embeds = self._embedding_layer(input_ids)
        if inputs_embeds.dtype != torch.bfloat16:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)
        selected = input_ids == self._image_token_index
        flattened = image_features.reshape(-1, image_features.shape[-1])
        torch._assert_async(
            selected.sum() == flattened.shape[0],
            "Eagle image token count does not match the TensorRT ViT output",
        )
        inputs_embeds[selected] = flattened
        attention_mask = values["eagle_attention_mask"]
        backbone_features = self.llm_engine(inputs_embeds, attention_mask)[
            "backbone_features"
        ]
        return BatchFeature(
            data={
                "backbone_features": backbone_features,
                "backbone_attention_mask": attention_mask,
            }
        )

    def structure(self) -> dict[str, bool]:
        eagle = self.backbone.eagle_model
        return {
            "vit_engine_loaded": self.vit_engine is not None,
            "llm_engine_loaded": self.llm_engine is not None,
            "pytorch_vision_removed": not hasattr(eagle, "vision_model"),
            "pytorch_llm_removed": not hasattr(eagle, "language_model"),
            "pytorch_visual_projector_removed": not hasattr(eagle, "mlp1"),
            "embedding_layer_retained": self._embedding_layer is not None,
        }

    def telemetry(self) -> dict[str, Any]:
        return {
            "artifacts": {
                "receipt_path": str(self.artifacts["receipt_path"]),
                "receipt_sha256": self.artifacts["receipt_sha256"],
                "metadata_path": str(self.artifacts["metadata_path"]),
                "metadata_sha256": self.artifacts["metadata_sha256"],
                "files": self.artifacts["files"],
            },
            "runtime": self.runtime,
            "structure": self.structure(),
            "vit": self.vit_engine.telemetry(),
            "llm": self.llm_engine.telemetry(),
            "closed": self.closed,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.llm_engine.close()
        self.vit_engine.close()
        self.closed = True
