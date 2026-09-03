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

"""Export the official N1.7 ViT+LLM ONNX pair for one true-B8 Trocar call."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

BATCH_SIZE = 8
CAMERA_COUNT = 3
PATCHES_PER_ROW = 768
VISUAL_TOKENS_PER_ROW = 192
HIDDEN_SIZE = 2048
EXPECTED_SEQUENCE_LENGTH = 208


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, dict[str, int | str]]:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def normalize_visual_capture(
    vit_capture: Any, llm_capture: Any, batch_size: int
) -> dict[str, Any]:
    """Normalize an already-collated B8 capture to the official per-row ABI."""
    import torch

    grid = vit_capture.grid_thw
    if tuple(grid.shape) != (batch_size * CAMERA_COUNT, 3):
        raise ValueError(f"unexpected true-B visual grid shape: {tuple(grid.shape)}")
    first_row = grid[:CAMERA_COUNT].clone()
    for row in range(1, batch_size):
        start = row * CAMERA_COUNT
        if not torch.equal(grid[start : start + CAMERA_COUNT], first_row):
            raise ValueError("true-B rows have different static visual geometry")
    if first_row.tolist() != [[1, 16, 16]] * CAMERA_COUNT:
        raise ValueError(f"unexpected per-row visual grid: {first_row.tolist()}")

    if tuple(vit_capture.pixel_values_shape) != (
        batch_size * PATCHES_PER_ROW,
        1536,
    ):
        raise ValueError(
            f"unexpected captured pixel shape: {vit_capture.pixel_values_shape}"
        )
    if tuple(vit_capture.output_shape) != (
        batch_size * VISUAL_TOKENS_PER_ROW,
        HIDDEN_SIZE,
    ):
        raise ValueError(f"unexpected captured ViT output: {vit_capture.output_shape}")
    if any(
        tuple(shape) != (batch_size * VISUAL_TOKENS_PER_ROW, HIDDEN_SIZE)
        for shape in vit_capture.deepstack_shapes
    ):
        raise ValueError(
            f"unexpected captured ViT deepstack shapes: {vit_capture.deepstack_shapes}"
        )

    deepstack = list(llm_capture.deepstack_visual_embeds or [])
    if len(deepstack) != 3 or any(
        tuple(value.shape) != (batch_size * VISUAL_TOKENS_PER_ROW, HIDDEN_SIZE)
        for value in deepstack
    ):
        raise ValueError(
            "unexpected captured LLM deepstack shapes: "
            f"{[tuple(value.shape) for value in deepstack]}"
        )

    before = {
        "grid_thw": list(grid.shape),
        "pixel_values": list(vit_capture.pixel_values_shape),
        "vit_output": list(vit_capture.output_shape),
        "vit_deepstack": [list(shape) for shape in vit_capture.deepstack_shapes],
        "llm_deepstack": [list(value.shape) for value in deepstack],
    }
    vit_capture.grid_thw = first_row
    vit_capture.pixel_values_shape = (PATCHES_PER_ROW, 1536)
    vit_capture.output_shape = (VISUAL_TOKENS_PER_ROW, HIDDEN_SIZE)
    vit_capture.deepstack_shapes = [
        (VISUAL_TOKENS_PER_ROW, HIDDEN_SIZE) for _ in vit_capture.deepstack_shapes
    ]
    llm_capture.deepstack_visual_embeds = [
        value[:VISUAL_TOKENS_PER_ROW].clone() for value in deepstack
    ]
    after = {
        "grid_thw": list(vit_capture.grid_thw.shape),
        "pixel_values": list(vit_capture.pixel_values_shape),
        "vit_output": list(vit_capture.output_shape),
        "vit_deepstack": [list(shape) for shape in vit_capture.deepstack_shapes],
        "llm_deepstack": [
            list(value.shape) for value in llm_capture.deepstack_visual_embeds
        ],
    }
    return {"captured_true_b8": before, "official_per_row_export_abi": after}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    source = args.source.resolve(strict=True)
    model = args.model.resolve(strict=True)
    collated_path = args.collated.resolve(strict=True)
    fixture_receipt = args.fixture_receipt.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.mkdir(parents=True)

    deployment = source / "scripts/deployment"
    sys.path.insert(0, str(deployment))
    from export_onnx_n1d7 import (  # noqa: PLC0415
        EXPORT_METADATA_SCHEMA_VERSION,
        LLMInputCapture,
        ViTInputCapture,
        export_llm_to_onnx,
        export_vit_to_onnx,
    )
    from gr00t.policy.gr00t_policy import (  # noqa: PLC0415
        Gr00tPolicy,
        _rec_to_dtype,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy = Gr00tPolicy(
        embodiment_tag="NEW_EMBODIMENT",
        model_path=str(model),
        device="cuda",
    )
    modality = policy.get_modality_config()
    expected_cameras = ("left_wrist_view", "right_wrist_view", "room_view")
    if tuple(modality["video"].modality_keys) != expected_cameras:
        raise RuntimeError("loaded processor camera order differs from W77")
    if tuple(modality["action"].delta_indices) != tuple(range(16)):
        raise RuntimeError("loaded processor action horizon differs from W77")

    collated = torch.load(collated_path, map_location="cpu", weights_only=False)
    collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
    model_inputs = collated["inputs"] if "inputs" in collated else collated
    qwen_model = policy.model.backbone.model
    vit_capture = ViTInputCapture()
    llm_capture = LLMInputCapture()
    vit_hook = qwen_model.model.visual.register_forward_hook(
        vit_capture.hook_fn, with_kwargs=True
    )
    llm_hook = qwen_model.model.language_model.register_forward_pre_hook(
        llm_capture.hook_fn, with_kwargs=True
    )
    try:
        with torch.inference_mode():
            policy.model.get_action(model_inputs)
        torch.cuda.synchronize()
    finally:
        vit_hook.remove()
        llm_hook.remove()
    if not vit_capture.captured or not llm_capture.captured:
        raise RuntimeError("official input hooks did not capture both ViT and LLM")
    if tuple(llm_capture.inputs_embeds.shape) != (
        BATCH_SIZE,
        EXPECTED_SEQUENCE_LENGTH,
        HIDDEN_SIZE,
    ):
        raise RuntimeError(
            f"unexpected LLM input shape: {tuple(llm_capture.inputs_embeds.shape)}"
        )
    visual_geometry = normalize_visual_capture(vit_capture, llm_capture, BATCH_SIZE)

    config = policy.model.action_head.config
    metadata = {
        "schema_version": EXPORT_METADATA_SCHEMA_VERSION,
        "model_version": "n1d7",
        "sa_seq_len": int(1 + config.action_horizon),
        "vl_seq_len": EXPECTED_SEQUENCE_LENGTH,
        "llm_seq_len": EXPECTED_SEQUENCE_LENGTH,
        "llm_hidden_size": HIDDEN_SIZE,
        "num_deepstack": 3,
        "num_vis_tokens_per_row": VISUAL_TOKENS_PER_ROW,
        "num_vis_tokens": VISUAL_TOKENS_PER_ROW,
        "num_patches_per_row": PATCHES_PER_ROW,
        "num_merged_patches_per_row": VISUAL_TOKENS_PER_ROW,
        "num_patches": PATCHES_PER_ROW,
        "num_merged_patches": VISUAL_TOKENS_PER_ROW,
        "action_horizon": int(config.action_horizon),
        "max_action_dim": int(config.max_action_dim),
        "max_state_dim": int(config.max_state_dim),
        "state_history_length": int(config.state_history_length),
        "hidden_size": int(config.hidden_size),
        "input_embedding_dim": int(config.input_embedding_dim),
        "backbone_embedding_dim": int(config.backbone_embedding_dim),
        "embodiment_tag": "new_embodiment",
        "export_mode": "full_pipeline",
        "precision": "bf16",
        "batch_size": BATCH_SIZE,
        "visual_batch_expansion": "per-row-capture-times-static-batch",
        "vit_grid_thw": [[1, 16, 16]] * CAMERA_COUNT,
        "input_adapter": "rlinf export_true_b8.py",
        "collated_sha256": _sha256(collated_path),
        "fixture_receipt_sha256": _sha256(fixture_receipt),
        "model_view_receipt_sha256": _sha256(model / "rlinf-model-view.json"),
    }
    (output / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    export_vit_to_onnx(
        policy, str(output), vit_capture, use_bf16=False, batch_size=BATCH_SIZE
    )
    export_llm_to_onnx(
        policy, llm_capture, str(output), use_bf16=True, batch_size=BATCH_SIZE
    )
    expected = {"vit_fp32.onnx", "llm_bf16.onnx"}
    actual = {path.name for path in output.glob("*.onnx")}
    if actual != expected:
        raise RuntimeError(f"unexpected ONNX set: {actual} != {expected}")

    receipt = {
        "schema": "rlinf.gr00t-n1d7-trocar-true-b8-onnx.v1",
        "status": "passed",
        "source_revision": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_view": str(model),
        "collated": str(collated_path),
        "batch_size": BATCH_SIZE,
        "single_policy_call": True,
        "b1x8": False,
        "seed": args.seed,
        "visual_geometry": visual_geometry,
        "metadata": metadata,
        "files": _inventory(output),
    }
    (output / "rlinf-export-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--collated", type=Path, required=True)
    parser.add_argument("--fixture-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W80 true-B8 export failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
