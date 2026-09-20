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

"""Export the GR00T N1.5 Stack Cube true-B8 DiT for TensorRT refit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from qualify_n15_backbone import _clone_feature, _model_config, _seed
from transformers import BatchFeature

EXPECTED_INPUTS = {
    "sa_embs": ([8, 49, 1536], torch.bfloat16),
    "vl_embs": ([8, 570, 2048], torch.bfloat16),
    "timestep": ([8], torch.int64),
}
EXPECTED_OUTPUT = ([8, 49, 1024], torch.bfloat16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_spec(value: torch.Tensor) -> dict[str, Any]:
    payload = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _materialize_export_inputs(
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    materialized = {
        name: value.detach().clone().contiguous() for name, value in inputs.items()
    }
    inference_inputs = [
        name for name, value in materialized.items() if torch.is_inference(value)
    ]
    if inference_inputs:
        raise RuntimeError(
            "DiT export inputs remain inference tensors: "
            f"{sorted(inference_inputs)}"
        )
    return materialized


class _Capture:
    def __init__(self) -> None:
        self.inputs: dict[str, torch.Tensor] | None = None

    def hook(
        self,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if self.inputs is not None:
            return
        self.inputs = {
            "sa_embs": kwargs["hidden_states"].detach().clone(),
            "vl_embs": kwargs["encoder_hidden_states"].detach().clone(),
            "timestep": kwargs["timestep"].detach().clone(),
        }


class _ExportWrapper(torch.nn.Module):
    def __init__(self, dit: torch.nn.Module) -> None:
        super().__init__()
        self.dit = dit

    def forward(
        self,
        sa_embs: torch.Tensor,
        vl_embs: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.dit(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timestep,
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    fixture_root = args.fixture.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    fixture_path = fixture_root / "backbone-inputs.pt"
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    fixture = {name: value.cuda() for name, value in fixture.items()}
    backbone_input = BatchFeature(
        data={
            name: fixture[name]
            for name in (
                "eagle_input_ids",
                "eagle_attention_mask",
                "eagle_pixel_values",
                "eagle_image_sizes",
            )
        }
    )
    action_input = BatchFeature(
        data={name: fixture[name] for name in ("state", "state_mask", "embodiment_id")}
    )
    cfg = _model_config(args.config_root.resolve(strict=True), model_path)
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    model.cuda()
    model.eval()
    capture = _Capture()
    hook = model.action_head.model.register_forward_pre_hook(
        capture.hook, with_kwargs=True
    )
    try:
        _seed(args.seed)
        with torch.inference_mode():
            backbone = model.backbone(backbone_input)
            model.action_head.get_rl_action(
                _clone_feature(backbone), action_input, mode="train"
            )
    finally:
        hook.remove()
    if capture.inputs is None:
        raise RuntimeError("N1.5 DiT pre-forward hook captured no input")
    # The hook runs inside inference_mode, whose tensors cannot participate in
    # the autograd-backed legacy ONNX tracer. Clone after leaving that scope.
    capture.inputs = _materialize_export_inputs(capture.inputs)
    for name, (shape, dtype) in EXPECTED_INPUTS.items():
        value = capture.inputs[name]
        if list(value.shape) != shape or value.dtype != dtype:
            raise RuntimeError(
                f"captured {name} differs: shape={list(value.shape)} "
                f"dtype={value.dtype}"
            )

    wrapper = _ExportWrapper(model.action_head.model).eval()
    with torch.inference_mode():
        expected = wrapper(**capture.inputs)
    if (
        list(expected.shape) != EXPECTED_OUTPUT[0]
        or expected.dtype != EXPECTED_OUTPUT[1]
    ):
        raise RuntimeError(
            f"N1.5 DiT output differs: shape={list(expected.shape)} "
            f"dtype={expected.dtype}"
        )

    onnx_path = output / "dit_bf16_refit.onnx"
    torch.onnx.export(
        wrapper,
        tuple(capture.inputs.values()),
        onnx_path,
        export_params=True,
        external_data=True,
        do_constant_folding=False,
        input_names=list(capture.inputs),
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    files = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    receipt = {
        "schema": "rlinf.gr00t-n1d5-stack-cube-true-b8-dit-onnx.v1",
        "status": "passed",
        "model": str(model_path),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "fixture": str(fixture_path),
        "fixture_sha256": _sha256(fixture_path),
        "seed": args.seed,
        "batch_size": 8,
        "precision": "bfloat16",
        "inputs": {name: _tensor_spec(value) for name, value in capture.inputs.items()},
        "output": _tensor_spec(expected),
        "files": files,
    }
    receipt_path = output / "rlinf-dit-export-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W06 N1.5 DiT export failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
