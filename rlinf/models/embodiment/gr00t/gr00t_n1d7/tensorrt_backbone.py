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

"""Persistent TensorRT ViT+LLM backend for the frozen GR00T N1.7 backbone."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from rlinf.hybrid_engines.tensorrt import PersistentEngine

_INPUT_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
_ENGINE_FILES = ("llm_bf16.engine", "vit.engine")
_RECEIPT_SCHEMA = "rlinf.gr00t-n1d7-trocar-true-b8-engines.v1"


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
    expected_batch = int(_required(config, "static_batch_size"))
    expected_sequence_opt = int(_required(config, "sequence_opt"))
    if (
        receipt.get("static_batch") != expected_batch
        or receipt.get("sequence_opt") != expected_sequence_opt
        or receipt.get("silent_fallback") is not False
    ):
        raise RuntimeError("TensorRT engine receipt batch/profile contract mismatch")
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
        expected = receipt["engines"][name]
        actual = _sha256(path)
        if actual != expected.get("sha256") or path.stat().st_size != expected.get(
            "bytes"
        ):
            raise RuntimeError(f"TensorRT plan does not match its receipt: {name}")
        files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
        }

    metadata_path = root / "export_metadata.json"
    metadata_sha256 = _sha256(metadata_path)
    if metadata_sha256 != receipt.get("export_metadata_sha256"):
        raise RuntimeError("TensorRT export metadata does not match its receipt")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("model_version") != "n1d7"
        or metadata.get("batch_size") != expected_batch
        or metadata.get("llm_seq_len") != expected_sequence_opt
    ):
        raise RuntimeError("TensorRT export metadata model/shape contract mismatch")
    return {
        "root": root,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_sha256": actual_receipt_sha256,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "metadata_sha256": metadata_sha256,
        "files": files,
    }


def _validate_runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    import tensorrt as trt

    expected_version = str(_required(config, "runtime_version"))
    distribution_version = importlib.metadata.version("tensorrt")
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
    expected_capability = [int(value) for value in _required(config, "compute_capability")]
    if capability != expected_capability:
        raise RuntimeError(
            f"TensorRT engine requires SM{expected_capability}, found SM{capability}"
        )
    return {
        "tensorrt_module": trt.__version__,
        "tensorrt_distribution": distribution_version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": capability,
    }


class TensorRTFrozenBackbone:
    """Own a qualified two-engine backbone and its live CUDA resources."""

    def __init__(self, backbone: torch.nn.Module, config: Mapping[str, Any]):
        self.backbone = backbone
        self.config = dict(config)
        self.artifacts = _validate_artifacts(config)
        self.runtime = _validate_runtime(config)
        self.expected_batch_size = int(config["static_batch_size"])
        self.expected_sequence_opt = int(config["sequence_opt"])
        metadata = self.artifacts["metadata"]
        self._baked_grid_thw = {
            tuple(int(item) for item in row)
            for row in metadata.get("vit_grid_thw", ())
        }
        if not self._baked_grid_thw:
            raise RuntimeError("TensorRT metadata must record the baked ViT grid")
        if len(self._baked_grid_thw) != 1:
            raise RuntimeError(
                "W81 TensorRT runtime requires one fixed camera grid geometry"
            )
        self._baked_grid_row = next(iter(self._baked_grid_thw))

        qwen_model = getattr(backbone, "model", None)
        inner_model = getattr(qwen_model, "model", None)
        language_model = getattr(inner_model, "language_model", None)
        visual = getattr(inner_model, "visual", None)
        final_norm = getattr(language_model, "norm", None)
        layers = getattr(language_model, "layers", None)
        if any(value is None for value in (qwen_model, inner_model, language_model)):
            raise RuntimeError("unsupported GR00T N1.7 Qwen3-VL backbone layout")
        if visual is None or final_norm is None or layers is None:
            raise RuntimeError("eager ViT/LLM modules are missing before TRT adoption")
        self._embedding_layer = language_model.get_input_embeddings()
        self._image_token_id = int(qwen_model.config.image_token_id)

        root = self.artifacts["root"]
        vit_engine = None
        llm_engine = None
        try:
            vit_engine = PersistentEngine(str(root / "vit.engine"))
            llm_engine = PersistentEngine(str(root / "llm_bf16.engine"))
            self._validate_bindings("vit.engine", vit_engine)
            self._validate_bindings("llm_bf16.engine", llm_engine)
        except Exception:
            if llm_engine is not None:
                llm_engine.close()
            if vit_engine is not None:
                vit_engine.close()
            raise
        self.vit_engine = vit_engine
        self.llm_engine = llm_engine

        del inner_model.visual
        del language_model.layers
        del language_model.norm
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
            raise KeyError(f"TensorRT backbone inputs are incomplete: {missing}")
        input_ids = values["input_ids"]
        attention_mask = values["attention_mask"]
        if input_ids.shape[0] != self.expected_batch_size:
            raise ValueError(
                "TensorRT backbone requires genuine static batch "
                f"{self.expected_batch_size}, found {input_ids.shape[0]}"
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError("TensorRT input_ids and attention_mask shapes differ")
        if input_ids.shape[1] != self.expected_sequence_opt:
            raise ValueError(
                "TensorRT production path requires the qualified sequence length "
                f"{self.expected_sequence_opt}, found {input_ids.shape[1]}"
            )
        torch._assert_async(
            (attention_mask == 1).all(),
            "TensorRT backbone requires an unpadded fixed-length B8 prompt",
        )
        grid = values["image_grid_thw"]
        expected_grid_row = torch.tensor(
            self._baked_grid_row,
            device=grid.device,
            dtype=grid.dtype,
        )
        torch._assert_async(
            (grid == expected_grid_row).all(),
            "TensorRT ViT received an unqualified camera grid",
        )

    def __call__(
        self, values: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.closed:
            raise RuntimeError("TensorRT backbone is closed")
        self._validate_input(values)
        self.backbone.set_frozen_modules_to_eval_mode()

        pixel_values = values["pixel_values"]
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = torch.cat(pixel_values, dim=0)
        vit_dtype = self.vit_engine.dtype_of("pixel_values")
        if pixel_values.dtype != vit_dtype:
            pixel_values = pixel_values.to(vit_dtype)
        self.vit_engine.set_runtime_tensor_shape("pixel_values", pixel_values.shape)
        vit_output = self.vit_engine(pixel_values)
        image_embeds = vit_output["image_embeds"]
        deepstack = list(vit_output["deepstack_features"].unbind(0))

        input_ids = values["input_ids"]
        inputs_embeds = self._embedding_layer(input_ids)
        if inputs_embeds.dtype != torch.bfloat16:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)
        if image_embeds.dtype != torch.bfloat16:
            image_embeds = image_embeds.to(torch.bfloat16)

        inner_model = self.backbone.model.model
        image_mask, _ = inner_model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]
        attention_mask = values["attention_mask"]
        position_ids, rope_deltas = inner_model.get_rope_index(
            input_ids,
            values["image_grid_thw"],
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        inner_model.rope_deltas = rope_deltas
        output_image_mask = input_ids == self._image_token_id
        output_attention_mask = attention_mask == 1

        llm_dtype = self.llm_engine.dtype_of("inputs_embeds")
        inputs_embeds = inputs_embeds.to(llm_dtype)
        attention_mask = attention_mask.to(torch.int64)
        position_ids = position_ids.to(torch.int64)
        visual_pos_masks = visual_pos_masks.to(torch.bool)
        self.llm_engine.set_runtime_tensor_shape("inputs_embeds", inputs_embeds.shape)
        self.llm_engine.set_runtime_tensor_shape("attention_mask", attention_mask.shape)
        self.llm_engine.set_runtime_tensor_shape("position_ids", position_ids.shape)
        self.llm_engine.set_runtime_tensor_shape(
            "visual_pos_masks", visual_pos_masks.shape
        )
        llm_inputs: dict[str, torch.Tensor] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "visual_pos_masks": visual_pos_masks,
        }
        for index, value in enumerate(deepstack):
            name = f"deepstack_{index}"
            value = value.to(llm_dtype)
            self.llm_engine.set_runtime_tensor_shape(name, value.shape)
            llm_inputs[name] = value
        backbone_features = self.llm_engine(**llm_inputs)["embeddings"].to(
            torch.bfloat16
        )
        return {
            "backbone_features": backbone_features,
            "backbone_attention_mask": output_attention_mask,
            "image_mask": output_image_mask,
        }

    def structure(self) -> dict[str, bool]:
        inner_model = self.backbone.model.model
        language_model = inner_model.language_model
        return {
            "vit_engine_loaded": self.vit_engine is not None,
            "llm_engine_loaded": self.llm_engine is not None,
            "pytorch_vit_removed": not hasattr(inner_model, "visual"),
            "pytorch_llm_layers_removed": not hasattr(language_model, "layers"),
            "pytorch_llm_norm_removed": not hasattr(language_model, "norm"),
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
