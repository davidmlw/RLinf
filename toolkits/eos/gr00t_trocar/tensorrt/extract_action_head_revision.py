#!/usr/bin/env python3
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

"""Extract the refittable GR00T DiT tensors from an Actor checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PREFIX = "action_head.model."
EXPECTED_TENSORS = 456


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    manifest = args.manifest.resolve()
    if output.exists() or manifest.exists():
        raise RuntimeError("output and manifest must both be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    state = torch.load(source, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise RuntimeError("input is not a string-keyed state dict")
    selected = {key: value for key, value in state.items() if key.startswith(PREFIX)}
    if len(selected) != EXPECTED_TENSORS:
        raise RuntimeError(
            f"expected {EXPECTED_TENSORS} DiT tensors, found {len(selected)}"
        )
    if any(not isinstance(value, torch.Tensor) for value in selected.values()):
        raise RuntimeError("selected action-head entries are not all tensors")
    dtypes = sorted({str(value.dtype) for value in selected.values()})
    if dtypes != ["torch.bfloat16"]:
        raise RuntimeError(f"expected only BF16 DiT tensors, found {dtypes}")
    tensor_bytes = sum(
        value.numel() * value.element_size() for value in selected.values()
    )
    torch.save(selected, output)
    receipt = {
        "schema": "rlinf.w98.action-head-revision-sidecar/v1",
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "revision": args.source_revision,
            "total_state_dict_entries": len(state),
        },
        "sidecar": {
            "path": str(output),
            "sha256": _sha256(output),
            "bytes": output.stat().st_size,
            "tensor_count": len(selected),
            "tensor_bytes": tensor_bytes,
            "dtypes": dtypes,
            "ordered_keys_sha256": hashlib.sha256(
                ("\n".join(selected) + "\n").encode("utf-8")
            ).hexdigest(),
        },
    }
    manifest.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-revision", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
