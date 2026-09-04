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

"""Capture and export the true-static-B8 GR00T N1.7 DiT ONNX graph."""

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
EXPECTED_ACTION_SEQUENCE = 41
EXPECTED_VL_SEQUENCE = 208
EXPECTED_INPUT_EMBEDDING = 1536
EXPECTED_BACKBONE_EMBEDDING = 2048


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_spec(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    contiguous = value.contiguous()
    try:
        payload = contiguous.numpy().tobytes()
    except TypeError:
        import torch  # noqa: PLC0415

        payload = contiguous.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _inventory(root: Path) -> dict[str, dict[str, int | str]]:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def validate_capture(capture: Any) -> dict[str, Any]:
    """Validate the production DiT ABI before exporting an artifact."""

    required = {
        "sa_embs": (BATCH_SIZE, EXPECTED_ACTION_SEQUENCE, EXPECTED_INPUT_EMBEDDING),
        "vl_embs": (BATCH_SIZE, EXPECTED_VL_SEQUENCE, EXPECTED_BACKBONE_EMBEDDING),
        "timestep": (BATCH_SIZE,),
        "image_mask": (BATCH_SIZE, EXPECTED_VL_SEQUENCE),
        "backbone_attention_mask": (BATCH_SIZE, EXPECTED_VL_SEQUENCE),
    }
    actual = {
        name: None
        if getattr(capture, name) is None
        else tuple(getattr(capture, name).shape)
        for name in required
    }
    if actual != required:
        raise RuntimeError(f"captured DiT ABI mismatch: {actual} != {required}")
    return {name: _tensor_spec(getattr(capture, name)) for name in required}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    source = args.source.resolve(strict=True)
    model = args.model.resolve(strict=True)
    collated = args.collated.resolve(strict=True)
    fixture_receipt = args.fixture_receipt.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    output.mkdir(parents=True)

    deployment = source / "scripts/deployment"
    sys.path.insert(0, str(deployment))
    from export_onnx_n1d7 import DiTInputCapture, export_dit_to_onnx  # noqa: PLC0415
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype  # noqa: PLC0415

    source_revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_revision != args.expected_source_revision:
        raise RuntimeError(
            f"Isaac-GR00T revision mismatch: {source_revision} != "
            f"{args.expected_source_revision}"
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
        raise RuntimeError("loaded processor camera order differs from W80")
    if tuple(modality["action"].delta_indices) != tuple(range(16)):
        raise RuntimeError("loaded processor action horizon differs from W80")

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
        raise RuntimeError("DiT pre-forward hook did not capture any input")
    capture_spec = validate_capture(capture)

    onnx_path = output / "dit_bf16.onnx"
    export_dit_to_onnx(
        policy,
        capture,
        str(onnx_path),
        use_bf16=True,
        batch_size=BATCH_SIZE,
    )
    expected_files = {"dit_bf16.onnx", "dit_bf16.onnx.data"}
    actual_files = {path.name for path in output.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(
            f"unexpected DiT export set: {actual_files} != {expected_files}"
        )

    receipt = {
        "schema": "rlinf.gr00t-n1d7-trocar-true-b8-dit-onnx.v1",
        "status": "passed",
        "source_revision": source_revision,
        "model": str(model),
        "model_config_sha256": _sha256(model / "config.json"),
        "collated": str(collated),
        "collated_sha256": _sha256(collated),
        "fixture_receipt": str(fixture_receipt),
        "fixture_receipt_sha256": _sha256(fixture_receipt),
        "batch_size": BATCH_SIZE,
        "b1x8": False,
        "action_sequence": EXPECTED_ACTION_SEQUENCE,
        "vl_sequence": EXPECTED_VL_SEQUENCE,
        "precision": "bf16",
        "seed": args.seed,
        "capture": capture_spec,
        "files": _inventory(output),
    }
    (output / "rlinf-dit-export-receipt.json").write_text(
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
    parser.add_argument(
        "--expected-source-revision",
        default="51d4c89f72fda44cbf77285c6a8114b52676b8a1",
    )
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W83 true-B8 DiT export failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
