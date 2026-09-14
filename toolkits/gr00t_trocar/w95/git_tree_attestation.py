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

"""Bind a Git revision directly to a materialized source tree.

Immutable source bundles intentionally omit ``.git``. Running ``git -C`` on
such a bundle can therefore walk into an unrelated parent repository. This
tool records Git blob identities from the authoritative repository and verifies
the materialized files without invoking Git below the bundle root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "rlinf.git-tree-attestation/v1"


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        input=input_bytes,
        stdout=subprocess.PIPE,
    ).stdout


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return _sha256(payload)


def create_attestation(repo: Path, revision: str) -> dict[str, Any]:
    """Create an attestation for every blob in ``revision``."""
    repo = repo.resolve(strict=True)
    commit = _git(repo, "rev-parse", f"{revision}^{{commit}}").decode().strip()
    tree = _git(repo, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    raw = _git(repo, "ls-tree", "-r", "-z", "--full-tree", commit)
    entries = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, path_bytes = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        if object_type != "blob":
            raise ValueError(f"unsupported Git object at {path_bytes!r}: {object_type}")
        path = path_bytes.decode("utf-8", errors="strict")
        blob = _git(repo, "cat-file", "blob", object_id)
        entries.append(
            {
                "path": path,
                "git_mode": mode,
                "git_blob": object_id,
                "size": len(blob),
                "sha256": _sha256(blob),
            }
        )
    entries.sort(key=lambda item: item["path"])
    return {
        "schema": SCHEMA,
        "revision": commit,
        "git_tree": tree,
        "entry_count": len(entries),
        "entries_sha256": _canonical_sha256(entries),
        "entries": entries,
    }


def _bundle_entries(root: Path) -> set[str]:
    result = set()
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        for name in list(directory_names):
            path = directory_path / name
            if path.is_symlink():
                result.add(path.relative_to(root).as_posix())
                directory_names.remove(name)
        for name in file_names:
            result.add((directory_path / name).relative_to(root).as_posix())
    return result


def verify_attestation(root: Path, attestation: dict[str, Any]) -> dict[str, Any]:
    """Compare one materialized source tree with a Git-tree attestation."""
    root = root.resolve(strict=True)
    if attestation.get("schema") != SCHEMA:
        raise ValueError(f"unsupported attestation schema: {attestation.get('schema')}")
    entries = attestation.get("entries")
    if not isinstance(entries, list):
        raise ValueError("attestation entries must be a list")
    if attestation.get("entry_count") != len(entries):
        raise ValueError("attestation entry count mismatch")
    if attestation.get("entries_sha256") != _canonical_sha256(entries):
        raise ValueError("attestation entry digest mismatch")

    expected_paths = {entry["path"] for entry in entries}
    observed_paths = _bundle_entries(root)
    missing = sorted(expected_paths - observed_paths)
    extra = sorted(observed_paths - expected_paths)
    mismatched = []
    for entry in entries:
        relative = entry["path"]
        if relative in missing:
            continue
        path = root / relative
        mode = entry["git_mode"]
        if mode == "120000":
            if not path.is_symlink():
                mismatched.append({"path": relative, "reason": "expected symlink"})
                continue
            content = os.readlink(path).encode("utf-8")
        else:
            if not path.is_file() or path.is_symlink():
                mismatched.append({"path": relative, "reason": "expected file"})
                continue
            content = path.read_bytes()
            executable = bool(path.stat().st_mode & stat.S_IXUSR)
            if executable != (mode == "100755"):
                mismatched.append(
                    {
                        "path": relative,
                        "reason": "executable bit mismatch",
                        "expected_git_mode": mode,
                    }
                )
                continue
        digest = _sha256(content)
        if len(content) != entry["size"] or digest != entry["sha256"]:
            mismatched.append(
                {
                    "path": relative,
                    "reason": "content mismatch",
                    "expected_size": entry["size"],
                    "observed_size": len(content),
                    "expected_sha256": entry["sha256"],
                    "observed_sha256": digest,
                }
            )
    return {
        "schema": "rlinf.git-tree-verification/v1",
        "revision": attestation["revision"],
        "git_tree": attestation["git_tree"],
        "expected_entry_count": len(entries),
        "observed_entry_count": len(observed_paths),
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "status": "passed" if not (missing or extra or mismatched) else "failed",
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--repo", type=Path, required=True)
    build.add_argument("--revision", required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--attestation", type=Path, required=True)
    verify.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "build":
        value = create_attestation(args.repo, args.revision)
        _write_json(args.output, value)
    else:
        attestation = json.loads(args.attestation.read_text(encoding="ascii"))
        value = verify_attestation(args.root, attestation)
        if args.output is not None:
            _write_json(args.output, value)
        if value["status"] != "passed":
            print(json.dumps(value, sort_keys=True))
            return 1
    print(json.dumps({key: value[key] for key in value if key != "entries"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
