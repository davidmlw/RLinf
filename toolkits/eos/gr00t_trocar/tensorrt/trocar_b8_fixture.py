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

"""Capture a deterministic, distinct-row true-B8 Trocar model-input fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from fixture_b1 import _aggregate_hash, _rng_manifest, _tree_manifest
from resident_b1 import _load_api
from trocar_b8_model_view import (
    CAMERA_ORDER,
    EMBODIMENT,
    LANGUAGE_KEY,
    STATE_ACTION_ORDER,
)

BATCH_SIZE = 8
CAMERA_HEIGHT = 224
CAMERA_WIDTH = 224
EXPECTED_PIXEL_VALUES = (6144, 1536)
EXPECTED_GRID_THW = (24, 3)
EXPECTED_STATE = (8, 1, 132)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distinct_image(row: int, camera: int) -> np.ndarray:
    """Return a reproducible uint8 image unique to one row and camera."""
    y, x, channel = np.indices((CAMERA_HEIGHT, CAMERA_WIDTH, 3))
    return ((row * 31 + camera * 53 + y * 3 + x * 5 + channel * 17) % 256).astype(
        np.uint8
    )


def _state_rows(metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    statistics = metadata[EMBODIMENT]["statistics"]["state"]
    result = {}
    for key in STATE_ACTION_ORDER:
        low = np.asarray(statistics[key]["q01"], dtype=np.float32)
        high = np.asarray(statistics[key]["q99"], dtype=np.float32)
        if low.shape != (7,) or high.shape != (7,):
            raise ValueError(f"state statistics for {key} must be 7-D")
        rows = []
        for row in range(BATCH_SIZE):
            fraction = np.float32((row + 1) / (BATCH_SIZE + 1))
            rows.append(low + fraction * (high - low))
        result[key] = np.stack(rows)[:, None, :]
    return result


def raw_observation(metadata: dict[str, Any], prompt: str) -> dict[str, Any]:
    videos = {}
    for camera_index, key in enumerate(CAMERA_ORDER):
        videos[key] = np.stack(
            [distinct_image(row, camera_index) for row in range(BATCH_SIZE)]
        )[:, None, ...]
    return {
        "video": videos,
        "state": _state_rows(metadata),
        "language": {LANGUAGE_KEY: [[prompt] for _ in range(BATCH_SIZE)]},
    }


def _raw_manifest(observation: dict[str, Any]) -> dict[str, Any]:
    cameras = {}
    all_hashes = []
    for key in CAMERA_ORDER:
        value = observation["video"][key]
        rows = [_sha256_bytes(value[row].tobytes()) for row in range(BATCH_SIZE)]
        cameras[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "row_sha256": rows,
        }
        all_hashes.extend(rows)
    if len(set(all_hashes)) != BATCH_SIZE * len(CAMERA_ORDER):
        raise RuntimeError("fixture camera/row hashes are not all distinct")
    return {
        "camera_order": list(CAMERA_ORDER),
        "cameras": cameras,
        "states": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _sha256_bytes(value.tobytes()),
            }
            for key, value in observation["state"].items()
        },
        "language_key": LANGUAGE_KEY,
        "prompt": observation["language"][LANGUAGE_KEY][0][0],
    }


def _find_tensor(inputs: Any, suffix: str) -> Any:
    if isinstance(inputs, dict) or hasattr(inputs, "items"):
        for key, value in inputs.items():
            if str(key).endswith(suffix):
                return value
            try:
                return _find_tensor(value, suffix)
            except KeyError:
                pass
    raise KeyError(suffix)


def _assert_collated_contract(collated: Any) -> dict[str, Any]:
    pixel_values = _find_tensor(collated, "pixel_values")
    grid_thw = _find_tensor(collated, "image_grid_thw")
    state = _find_tensor(collated, "state")
    actual = {
        "pixel_values": list(pixel_values.shape),
        "image_grid_thw": list(grid_thw.shape),
        "state": list(state.shape),
    }
    expected = {
        "pixel_values": list(EXPECTED_PIXEL_VALUES),
        "image_grid_thw": list(EXPECTED_GRID_THW),
        "state": list(EXPECTED_STATE),
    }
    if actual != expected:
        raise RuntimeError(f"true-B8 collated shape mismatch: {actual} != {expected}")
    if grid_thw.tolist() != [[1, 16, 16]] * 24:
        raise RuntimeError("fixture does not contain exactly 3x8 visual grids")
    return actual


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    source = args.source.resolve(strict=True)
    model = args.model.resolve(strict=True)
    metadata_path = args.metadata.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observation = raw_observation(metadata, args.prompt)
    raw_manifest = _raw_manifest(observation)

    api = _load_api(source)
    api["set_seed"](args.seed)
    policy = api["Gr00tPolicy"](
        model_path=str(model),
        embodiment_tag=api["EmbodimentTag"].NEW_EMBODIMENT,
        device="cuda",
        strict=True,
    )
    rng_before = _rng_manifest()
    collated, states = api["prepare_model_inputs"](
        policy, observation, return_states=True
    )
    torch.cuda.synchronize()
    collated_shapes = _assert_collated_contract(collated)
    tensor_manifest = _tree_manifest(collated)

    output.mkdir(parents=True)
    torch.save(collated, output / "collated-inputs.pt")
    np.savez_compressed(
        output / "raw-observation.npz",
        **{
            **{f"video.{key}": value for key, value in observation["video"].items()},
            **{f"state.{key}": value for key, value in observation["state"].items()},
        },
    )
    model_view_receipt = model / "rlinf-model-view.json"
    if not model_view_receipt.is_file():
        raise RuntimeError("model view omits rlinf-model-view.json")
    receipt = {
        "schema": "rlinf.gr00t-n1d7-trocar-true-b8-fixture.v1",
        "status": "passed",
        "source_revision": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "model_view": str(model),
        "model_view_receipt_sha256": _sha256(model_view_receipt),
        "metadata": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "seed": args.seed,
        "batch_size": BATCH_SIZE,
        "single_policy_call": True,
        "b1x8": False,
        "raw_observation": raw_manifest,
        "collated_shapes": collated_shapes,
        "collated_tensors": tensor_manifest,
        "collated_aggregate_sha256": _aggregate_hash(tensor_manifest),
        "states_count": len(states),
        "rng_before_preprocess": rng_before,
        "artifacts": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in ("collated-inputs.pt", "raw-observation.npz")
        },
    }
    (output / "fixture.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--prompt", default="assemble trocar from tray")
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as error:
        traceback.print_exc()
        print(f"W80 fixture capture failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
