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

"""Export the exact GR00T N1.5 Stack Cube true-B8 frozen backbone."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer


SCHEMA = "rlinf.gr00t-n1d5-stack-cube-true-b8-onnx.v1"
EXPECTED_INPUTS = {
    "eagle_pixel_values": ([16, 3, 224, 224], torch.bfloat16),
    "eagle_input_ids": ([8, 570], torch.int64),
    "eagle_attention_mask": ([8, 570], torch.int64),
}


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
    left_norm = torch.linalg.vector_norm(left)
    return {
        "finite": bool(torch.isfinite(right).all()),
        "cosine": float(torch.nn.functional.cosine_similarity(left, right, dim=0)),
        "mean_abs": float(difference.mean()),
        "max_abs": float(difference.max()),
        "relative_l2": float(torch.linalg.vector_norm(left - right) / left_norm),
    }


class StaticB8VisionProjection(torch.nn.Module):
    """Exportable SigLIP plus the frozen Eagle visual projector."""

    def __init__(
        self,
        source: SiglipVisionTransformer,
        projection: torch.nn.Module,
        attention_backend: str,
    ) -> None:
        super().__init__()
        config = copy.deepcopy(source.config)
        config._attn_implementation = attention_backend
        self.vision = SiglipVisionTransformer(config)
        self.vision.load_state_dict(source.state_dict(), strict=True)
        self.projection = projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.vision(
            pixel_values=pixel_values,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
        return self.projection(hidden)


class StaticB8PartialQwen(torch.nn.Module):
    """Return the exact selected Qwen hidden state used by N1.5."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        projection: torch.nn.Module,
        select_layer: int,
        attention_backend: str,
    ) -> None:
        super().__init__()
        if select_layer < 1 or select_layer > len(decoder.layers):
            raise ValueError(
                f"unsupported Eagle select_layer={select_layer} for "
                f"{len(decoder.layers)} Qwen layers"
            )
        decoder.config._attn_implementation = attention_backend
        if select_layer < len(decoder.layers):
            decoder.layers = torch.nn.ModuleList(
                list(decoder.layers[: select_layer + 1])
            )
        self.decoder = decoder
        self.projection = projection
        self.select_layer = select_layer

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return self.projection(outputs.hidden_states[self.select_layer])


def _onnx_inventory(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(path, load_external_data=False)
    dtype_name = onnx.TensorProto.DataType.Name

    def tensor_shape(value: Any) -> list[int | str]:
        result = []
        for dimension in value.type.tensor_type.shape.dim:
            if dimension.dim_param:
                result.append(dimension.dim_param)
            else:
                result.append(dimension.dim_value)
        return result

    initializers: dict[str, int] = {}
    external_locations = set()
    for value in model.graph.initializer:
        name = dtype_name(value.data_type)
        initializers[name] = initializers.get(name, 0) + 1
        for item in value.external_data:
            if item.key == "location":
                external_locations.add(item.value)
    return {
        "ir_version": model.ir_version,
        "opsets": [
            {"domain": item.domain, "version": item.version}
            for item in model.opset_import
        ],
        "inputs": [
            {
                "name": value.name,
                "dtype": dtype_name(value.type.tensor_type.elem_type),
                "shape": tensor_shape(value),
            }
            for value in model.graph.input
        ],
        "outputs": [
            {
                "name": value.name,
                "dtype": dtype_name(value.type.tensor_type.elem_type),
                "shape": tensor_shape(value),
            }
            for value in model.graph.output
        ],
        "initializer_dtypes": initializers,
        "external_locations": sorted(external_locations),
        "node_count": len(model.graph.node),
    }


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    fixture_root = args.fixture.resolve(strict=True)
    fixture_receipt = json.loads(
        (fixture_root / "backbone-abi.json").read_text(encoding="utf-8")
    )
    if fixture_receipt.get("status") != "passed":
        raise RuntimeError("backbone ABI fixture is not qualified")
    fixture = torch.load(
        fixture_root / "backbone-inputs.pt", map_location="cpu", weights_only=True
    )
    expected_output = torch.load(
        fixture_root / "backbone-outputs.pt",
        map_location="cpu",
        weights_only=True,
    )["backbone_features"].cuda()
    for name, (shape, dtype) in EXPECTED_INPUTS.items():
        value = fixture[name]
        if list(value.shape) != shape or value.dtype != dtype:
            raise RuntimeError(
                f"fixture {name} differs: shape={list(value.shape)} dtype={value.dtype}"
            )
    fixture = {name: value.cuda() for name, value in fixture.items()}

    cfg = _model_config(
        args.config_root.resolve(strict=True), args.model.resolve(strict=True)
    )
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    model.cuda()
    model.eval()
    backbone = model.backbone
    eagle = backbone.eagle_model
    with torch.inference_mode():
        live_backbone = backbone(
            {
                name: fixture[name]
                for name in (
                    "eagle_input_ids",
                    "eagle_attention_mask",
                    "eagle_pixel_values",
                    "eagle_image_sizes",
                )
            }
        )["backbone_features"]
    fixture_replay = _metrics(expected_output, live_backbone)
    if not fixture_replay["finite"] or fixture_replay["cosine"] < 0.99999:
        raise RuntimeError(f"live backbone fixture replay failed: {fixture_replay}")
    source_vision = eagle.vision_model.vision_model
    vision = StaticB8VisionProjection(
        source_vision, eagle.mlp1, args.attention_backend
    ).cuda().bfloat16()
    vision.eval()

    with torch.inference_mode():
        eager_image_features = eagle.extract_feature(fixture["eagle_pixel_values"])
        export_image_features = vision(fixture["eagle_pixel_values"])
    vision_parity = _metrics(eager_image_features, export_image_features)
    if not vision_parity["finite"] or vision_parity["cosine"] < 0.999:
        raise RuntimeError(f"eager-attention vision parity failed: {vision_parity}")

    embedding = eagle.language_model.get_input_embeddings()
    with torch.inference_mode():
        selected = fixture["eagle_input_ids"] == eagle.image_token_index
        eager_inputs_embeds = embedding(fixture["eagle_input_ids"])
        eager_flattened = eager_image_features.reshape(
            -1, eager_image_features.shape[-1]
        )
        eager_inputs_embeds[selected] = eager_flattened
        source_outputs = eagle.language_model(
            inputs_embeds=eager_inputs_embeds,
            attention_mask=fixture["eagle_attention_mask"],
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        source_features = backbone.eagle_linear(
            source_outputs.hidden_states[backbone.select_layer]
        )
        inputs_embeds = embedding(fixture["eagle_input_ids"])
        flattened = export_image_features.reshape(-1, export_image_features.shape[-1])
        if int(selected.sum()) != flattened.shape[0]:
            raise RuntimeError("fixture image token count does not match ViT output")
        inputs_embeds[selected] = flattened

    decoder = eagle.language_model.model
    loaded_llm_layers = len(decoder.layers)
    language = StaticB8PartialQwen(
        decoder,
        backbone.eagle_linear,
        backbone.select_layer,
        args.attention_backend,
    ).cuda().bfloat16()
    language.eval()
    with torch.inference_mode():
        source_vision_features = language(
            eager_inputs_embeds, fixture["eagle_attention_mask"]
        )
        export_features = language(
            inputs_embeds, fixture["eagle_attention_mask"]
        )
    source_language_parity = _metrics(expected_output, source_features)
    eager_attention_parity = _metrics(expected_output, source_vision_features)
    feature_parity = _metrics(expected_output, export_features)
    if not feature_parity["finite"] or feature_parity["cosine"] < 0.999:
        diagnostics = {
            "fixture_replay": fixture_replay,
            "vision": vision_parity,
            "source_language": source_language_parity,
            "export_llm_source_vision": eager_attention_parity,
            "combined_export": feature_parity,
        }
        raise RuntimeError(f"partial-Qwen export parity failed: {diagnostics}")

    del model
    torch.cuda.empty_cache()
    vit_path = output / "vit_bf16.onnx"
    llm_path = output / "llm_bf16.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            vision,
            (fixture["eagle_pixel_values"],),
            vit_path,
            input_names=["pixel_values"],
            output_names=["image_features"],
            opset_version=19,
            dynamo=False,
            external_data=True,
            do_constant_folding=True,
        )
        torch.onnx.export(
            language,
            (inputs_embeds, fixture["eagle_attention_mask"]),
            llm_path,
            input_names=["inputs_embeds", "attention_mask"],
            output_names=["backbone_features"],
            opset_version=19,
            dynamo=False,
            external_data=True,
            do_constant_folding=True,
        )

    inventories = {
        "vit_bf16.onnx": _onnx_inventory(vit_path),
        "llm_bf16.onnx": _onnx_inventory(llm_path),
    }
    expected_bindings = {
        "vit_bf16.onnx": {
            "inputs": [("pixel_values", "BFLOAT16", [16, 3, 224, 224])],
            "outputs": [("image_features", "BFLOAT16", [16, 256, 2048])],
        },
        "llm_bf16.onnx": {
            "inputs": [
                ("inputs_embeds", "BFLOAT16", [8, 570, 2048]),
                ("attention_mask", "INT64", [8, 570]),
            ],
            "outputs": [("backbone_features", "BFLOAT16", [8, 570, 2048])],
        },
    }
    for name, expected in expected_bindings.items():
        inventory = inventories[name]
        for direction in ("inputs", "outputs"):
            actual = [
                (item["name"], item["dtype"], item["shape"])
                for item in inventory[direction]
            ]
            if actual != expected[direction]:
                raise RuntimeError(
                    f"{name} {direction} differ: {actual} != {expected[direction]}"
                )

    files = sorted(path for path in output.iterdir() if path.is_file())
    metadata = {
        "schema": SCHEMA,
        "status": "passed",
        "model_version": "n1d5",
        "model_path": str(args.model.resolve(strict=True)),
        "batch_size": 8,
        "image_views": 2,
        "image_batch": 16,
        "sequence_length": 570,
        "precision": "bfloat16",
        "export_attention_backend": args.attention_backend,
        "vision_tokens_per_image": 256,
        "vision_pixel_shuffle": False,
        "llm_loaded_layers": loaded_llm_layers,
        "llm_selected_hidden_state": 12,
        "fixture": {
            "receipt": _artifact(fixture_root / "backbone-abi.json", fixture_root),
            "inputs": _artifact(fixture_root / "backbone-inputs.pt", fixture_root),
            "outputs": _artifact(fixture_root / "backbone-outputs.pt", fixture_root),
        },
        "export_parity": {
            "fixture_replay": fixture_replay,
            "vision_export_attention_vs_runtime_flash": vision_parity,
            "source_language_reconstruction": source_language_parity,
            "llm_export_attention_with_source_vision": eager_attention_parity,
            "partial_qwen_vs_backbone_fixture": feature_parity,
        },
        "onnx": {
            name: {
                **_artifact(output / name, output),
                "inventory": inventory,
            }
            for name, inventory in inventories.items()
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    metadata_path = output / "export_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata["artifact_files"] = [_artifact(path, output) for path in files]
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--attention-backend", choices=("eager", "sdpa"), default="sdpa"
    )
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W06 N1.5 ONNX export failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
