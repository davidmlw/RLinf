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

"""Capture the exact GR00T N1.5 Stack Cube frozen-backbone ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


TASK = (
    "Stack the red block on the blue block, then stack the green block "
    "on the red block."
)
EAGLE_KEYS = (
    "eagle_input_ids",
    "eagle_attention_mask",
    "eagle_pixel_values",
    "eagle_image_sizes",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_metadata(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "stride": list(value.stride()),
        "contiguous": value.is_contiguous(),
        "bytes": value.numel() * value.element_size(),
    }


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


def _deterministic_images(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(256 * 256 * 3, dtype=torch.int64).reshape(256, 256, 3)
    main = torch.stack(
        [((base + rank * 17) % 256).to(torch.uint8) for rank in range(batch_size)]
    )
    wrist = torch.stack(
        [((base.flip(1) + rank * 29 + 7) % 256).to(torch.uint8) for rank in range(batch_size)]
    )
    return main, wrist


def capture(args: argparse.Namespace) -> dict[str, Any]:
    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = _model_config(args.config_root.resolve(strict=True), args.model.resolve(strict=True))
    model = get_model(cfg, torch_dtype=torch.bfloat16).cuda().eval()

    main, wrist = _deterministic_images(args.batch_size)
    env_observation = {
        "main_images": main,
        "wrist_images": wrist,
        "states": torch.linspace(
            -0.25,
            0.25,
            steps=args.batch_size * 7,
            dtype=torch.float32,
        ).reshape(args.batch_size, 7),
        "task_descriptions": [TASK] * args.batch_size,
    }
    observations = model.obs_convert_fn(env_observation)
    normalized = model.apply_transforms(observations)
    for name, value in normalized.items():
        if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
            normalized[name] = value.to(torch.bfloat16)
    for name in ("eagle_input_ids", "eagle_attention_mask"):
        value = normalized[name]
        normalized[name] = torch.nn.functional.pad(
            value,
            (0, model.padding_value - value.shape[-1]),
            mode="constant",
            value=0,
        )

    backbone_inputs, _ = model.prepare_input(normalized)
    with torch.inference_mode():
        backbone_outputs = model.backbone(backbone_inputs)
    input_fixture = {name: backbone_inputs[name].detach().cpu() for name in EAGLE_KEYS}
    output_fixture = {
        name: backbone_outputs[name].detach().cpu()
        for name in ("backbone_features", "backbone_attention_mask")
    }
    input_path = output / "backbone-inputs.pt"
    output_path = output / "backbone-outputs.pt"
    torch.save(input_fixture, input_path)
    torch.save(output_fixture, output_path)

    backbone = model.backbone
    eagle = backbone.eagle_model
    result = {
        "schema": "rlinf.gr00t-n1d5-stack-cube-backbone-abi.v1",
        "status": "passed",
        "model_path": str(args.model.resolve(strict=True)),
        "batch_size": args.batch_size,
        "image_views": model.image_nums,
        "padding_value": model.padding_value,
        "task": TASK,
        "input_tensors": {
            name: _tensor_metadata(value) for name, value in backbone_inputs.items()
        },
        "output_tensors": {
            name: _tensor_metadata(value) for name, value in backbone_outputs.items()
        },
        "eagle": {
            "select_layer": backbone.select_layer,
            "tune_visual": backbone.tune_visual,
            "tune_llm": backbone.tune_llm,
            "vision_num_patches": eagle.vision_model.vision_model.embeddings.num_patches,
            "image_token_index": eagle.image_token_index,
            "use_pixel_shuffle": eagle.use_pixel_shuffle,
            "downsample_ratio": backbone.downsample_ratio,
            "backbone_parameter_count": sum(
                parameter.numel() for parameter in backbone.parameters()
            ),
            "backbone_trainable_parameter_count": sum(
                parameter.numel()
                for parameter in backbone.parameters()
                if parameter.requires_grad
            ),
        },
        "fixtures": {
            "inputs": {
                "path": input_path.name,
                "bytes": input_path.stat().st_size,
                "sha256": _sha256(input_path),
            },
            "outputs": {
                "path": output_path.name,
                "bytes": output_path.stat().st_size,
                "sha256": _sha256(output_path),
            },
        },
    }
    receipt = output / "backbone-abi.json"
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=260918)
    args = parser.parse_args()
    if args.batch_size != 8:
        parser.error("W06 qualifies only the true-B8 production ABI")
    capture(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
