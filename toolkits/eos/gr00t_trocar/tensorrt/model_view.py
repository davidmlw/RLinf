# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Create an immutable local view of a GR00T N1.7 checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

GENERATED_FILES = frozenset({"config.json", "processor_config.json"})
BACKBONE_REPOSITORY_SUFFIX = "nvidia/Cosmos-Reason2-2B"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_object(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_inventory(root: Path) -> dict[str, dict[str, int | str]]:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    }


def materialize_local_model_view(
    model_root: Path, backbone_root: Path, output_root: Path
) -> dict[str, Any]:
    """Build a local-only model view without copying checkpoint weights."""
    model_root = model_root.resolve(strict=True)
    backbone_root = backbone_root.resolve(strict=True)
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"model view output must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    for name in (*GENERATED_FILES, "model.safetensors.index.json"):
        if not (model_root / name).is_file():
            raise ValueError(f"model input omits {name}")
    if not (backbone_root / "config.json").is_file():
        raise ValueError("backbone input is not a local Hugging Face model")

    model_config = _read_object(model_root / "config.json")
    processor_config = _read_object(model_root / "processor_config.json")
    original_model_name = model_config.get("model_name")
    if not isinstance(original_model_name, str):
        raise ValueError("GR00T model config omits model_name")

    selector = output_root / "local-hf" / BACKBONE_REPOSITORY_SUFFIX
    selector.parent.mkdir(parents=True)
    os.symlink(backbone_root, selector, target_is_directory=True)
    model_config["model_name"] = str(selector)
    processor_kwargs = processor_config.get("processor_kwargs")
    if not isinstance(processor_kwargs, dict):
        raise ValueError("GR00T processor config omits processor_kwargs")
    processor_kwargs["model_name"] = str(selector)

    symlinked = []
    for source in sorted(model_root.iterdir(), key=lambda path: path.name):
        if source.name in GENERATED_FILES or source.name == ".cache":
            continue
        os.symlink(
            source,
            output_root / source.name,
            target_is_directory=source.is_dir(),
        )
        symlinked.append(source.name)

    _write_object(output_root / "config.json", model_config)
    _write_object(output_root / "processor_config.json", processor_config)
    source_files = _file_inventory(model_root)
    backbone_files = _file_inventory(backbone_root)
    receipt = {
        "schema": "rlinf.gr00t-n1d7-local-model-view.v1",
        "model_root": str(model_root),
        "backbone_root": str(backbone_root),
        "resolved_backbone_root": str(backbone_root.resolve(strict=True)),
        "backbone_selector": str(selector),
        "selector_preserves_repository_suffix": str(selector).endswith(
            BACKBONE_REPOSITORY_SUFFIX
        ),
        "original_model_name": original_model_name,
        "source_files": source_files,
        "backbone_files": backbone_files,
        "generated_hashes": {
            name: _sha256(output_root / name) for name in sorted(GENERATED_FILES)
        },
        "symlinked_entries": symlinked,
    }
    _write_object(output_root / "rlinf-model-view.json", receipt)
    return receipt
