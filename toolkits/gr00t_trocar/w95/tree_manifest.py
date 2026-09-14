#!/usr/bin/env python3
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

"""Create and verify relocatable manifests for immutable runtime trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

SCHEMA = "rlinf.immutable-tree-manifest/v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(root: Path) -> Iterator[Path]:
    yield root
    if root.is_symlink() or not root.is_dir():
        return
    for entry in sorted(root.iterdir(), key=lambda path: os.fsencode(path.name)):
        yield from _paths(entry)


def _entry(root: Path, path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    relative = "." if path == root else path.relative_to(root).as_posix()
    mode = f"{stat.S_IMODE(metadata.st_mode):04o}"
    if stat.S_ISREG(metadata.st_mode):
        kind = "file"
        target = None
        digest = _sha256_file(path)
    elif stat.S_ISDIR(metadata.st_mode):
        kind = "directory"
        target = None
        digest = _sha256_bytes(b"")
    elif stat.S_ISLNK(metadata.st_mode):
        kind = "symlink"
        target = os.readlink(path)
        digest = _sha256_bytes(os.fsencode(target))
    else:
        raise ValueError(f"unsupported special file in immutable tree: {relative}")
    return {
        "relative_path": relative,
        "type": kind,
        "symlink_target": target,
        "size": metadata.st_size,
        "mode": mode,
        "sha256": digest,
    }


def create_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"manifest root must be a directory: {root}")
    entries = [_entry(root, path) for path in _paths(root)]
    canonical = json.dumps(
        entries, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return {
        "schema": SCHEMA,
        "root_name": root.name,
        "entry_count": len(entries),
        "tree_sha256": _sha256_bytes(canonical),
        "entries": entries,
    }


def verify_manifest(root: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema") != SCHEMA:
        return [f"unsupported manifest schema: {manifest.get('schema')!r}"]
    try:
        current = create_manifest(root)
    except (OSError, ValueError) as error:
        return [str(error)]
    for field in ("root_name", "entry_count", "tree_sha256", "entries"):
        if current[field] != manifest.get(field):
            errors.append(f"immutable tree differs at manifest field: {field}")
    return errors


def _relative_path(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or value in {"", "."}:
        raise ValueError(f"invalid manifest relative path: {value!r}")
    return relative


def materialize_manifest(
    source: Path, destination: Path, manifest: dict[str, Any]
) -> None:
    source = source.resolve(strict=True)
    if verify_manifest(source, manifest):
        raise ValueError("source differs from the immutable tree manifest")
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"destination already exists: {destination}")
    if destination.name != manifest["root_name"]:
        raise ValueError(
            "destination basename must match manifest root_name: "
            f"{destination.name!r} != {manifest['root_name']!r}"
        )

    entries = manifest["entries"]
    root_entries = [entry for entry in entries if entry["relative_path"] == "."]
    if len(root_entries) != 1 or root_entries[0]["type"] != "directory":
        raise ValueError("manifest must contain exactly one directory root entry")

    destination.mkdir(parents=True, mode=0o700)
    directories: list[tuple[Path, int]] = [
        (destination, int(root_entries[0]["mode"], 8))
    ]
    for entry in entries:
        if entry["relative_path"] == ".":
            continue
        relative = _relative_path(entry["relative_path"])
        source_path = source.joinpath(*relative.parts)
        destination_path = destination.joinpath(*relative.parts)
        kind = entry["type"]
        mode = int(entry["mode"], 8)
        if kind == "directory":
            destination_path.mkdir(mode=0o700)
            directories.append((destination_path, mode))
        elif kind == "file":
            shutil.copyfile(source_path, destination_path, follow_symlinks=False)
            destination_path.chmod(mode)
        elif kind == "symlink":
            destination_path.symlink_to(entry["symlink_target"])
        else:
            raise ValueError(f"unsupported manifest entry type: {kind!r}")

    for path, mode in reversed(directories):
        path.chmod(mode)
    errors = verify_manifest(destination, manifest)
    if errors:
        raise ValueError("materialized tree failed verification: " + "; ".join(errors))


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--source", type=Path, required=True)
    materialize.add_argument("--destination", type=Path, required=True)
    materialize.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "build":
        if _is_within(args.output, args.root):
            raise ValueError("manifest output must be outside the immutable tree")
        manifest = create_manifest(args.root)
        _write_manifest(args.output, manifest)
        print(json.dumps({key: manifest[key] for key in manifest if key != "entries"}))
        return 0

    manifest = json.loads(args.manifest.read_text(encoding="ascii"))
    if args.command == "materialize":
        materialize_manifest(args.source, args.destination, manifest)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "destination": str(args.destination),
                    "tree_sha256": manifest["tree_sha256"],
                }
            )
        )
        return 0

    errors = verify_manifest(args.root, manifest)
    print(json.dumps({"status": "failed" if errors else "passed", "errors": errors}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
