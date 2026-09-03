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

"""Materialize an immutable TensorRT-only Python path overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

SCHEMA = "rlinf.eos.tensorrt-runtime-overlay.v1"
VERSION = "10.15.1.29"
PACKAGE_PATHS = (
    "tensorrt",
    "tensorrt_bindings",
    "tensorrt_libs",
    f"tensorrt_cu12-{VERSION}.dist-info",
    f"tensorrt_cu12_bindings-{VERSION}.dist-info",
    f"tensorrt_cu12_libs-{VERSION}.dist-info",
)
MANIFEST = "rlinf-tensorrt-overlay.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.name == MANIFEST or not path.is_file():
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return files


def _manifest(root: Path, source: Path) -> dict[str, Any]:
    files = _inventory(root)
    return {
        "schema": SCHEMA,
        "status": "passed",
        "version": VERSION,
        "source_site_packages": str(source),
        "package_paths": list(PACKAGE_PATHS),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
    }


def _write_manifest(root: Path, value: dict[str, Any]) -> None:
    (root / MANIFEST).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    root.chmod(root.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def materialize(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"TensorRT overlay already exists: {output}")
    for name in PACKAGE_PATHS:
        if not (source / name).exists():
            raise FileNotFoundError(f"TensorRT package path is missing: {source / name}")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.stage-{os.getpid()}"
    if stage.exists():
        raise FileExistsError(f"TensorRT overlay stage already exists: {stage}")
    stage.mkdir()
    try:
        for name in PACKAGE_PATHS:
            source_path = source / name
            destination = stage / name
            if source_path.is_dir():
                shutil.copytree(
                    source_path,
                    destination,
                    copy_function=os.link,
                    symlinks=True,
                )
            else:
                os.link(source_path, destination)
        value = _manifest(stage, source)
        _write_manifest(stage, value)
        _make_read_only(stage)
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return value


def verify(output: Path) -> dict[str, Any]:
    output = output.resolve(strict=True)
    manifest_path = output / MANIFEST
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected.get("schema") != SCHEMA or expected.get("status") != "passed":
        raise RuntimeError("TensorRT overlay manifest is not qualified")
    actual = _manifest(output, Path(expected["source_site_packages"]))
    if actual != expected:
        raise RuntimeError("TensorRT overlay inventory differs from its manifest")
    writable = [
        path.relative_to(output).as_posix()
        for path in (output, *output.rglob("*"))
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    ]
    if writable:
        raise RuntimeError(f"TensorRT overlay contains writable paths: {writable[:8]}")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("materialize")
    prepare_parser.add_argument("--source-site-packages", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "materialize":
        value = materialize(args.source_site_packages, args.output)
    else:
        value = verify(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
