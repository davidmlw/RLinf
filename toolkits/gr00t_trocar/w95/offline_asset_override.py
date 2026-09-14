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

"""Build a fail-closed IsaacLab assets override for offline asset mirrors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

PATCH_MARKER = "RLINF_OFFLINE_ASSET_MIRROR_V1"

_HELPER_ANCHOR = 'logger = logging.getLogger(__name__)\n'
_HELPER = '''

# RLINF_OFFLINE_ASSET_MIRROR_V1
def _cached_mirror_path(path: str, download_dir: str | None = None) -> str | None:
    """Return a complete local mirror entry for a remote asset URL."""
    parsed = urlparse(path)
    if parsed.scheme not in {"http", "https", "omniverse"}:
        return None
    root = os.path.abspath(download_dir or tempfile.gettempdir())
    candidate = os.path.abspath(os.path.join(root, parsed.path.lstrip("/")))
    try:
        if os.path.commonpath((root, candidate)) != root:
            return None
    except ValueError:
        return None
    return candidate if os.path.isfile(candidate) else None
'''
_CHECK_ANCHOR = '''    if os.path.isfile(path):
        return 1
'''
_CHECK_REPLACEMENT = '''    if os.path.isfile(path):
        return 1
    # Report a mirrored remote URL as downloadable so the spawner calls
    # retrieve_file_path(), which converts it to the local path below.
    if _cached_mirror_path(path) is not None:
        return 2
'''
_RETRIEVE_ANCHOR = '''    # check file status
    file_status = check_file_path(path)
'''
_RETRIEVE_REPLACEMENT = '''    cached_path = _cached_mirror_path(path, download_dir)
    if cached_path is not None and not force_download:
        return cached_path

    # check file status
    file_status = check_file_path(path)
'''


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_in_function(
    source: str, function: str, anchor: str, replacement: str
) -> str:
    marker = f"def {function}("
    if source.count(marker) != 1:
        raise ValueError(f"expected exactly one function: {function}")
    prefix, body = source.split(marker, 1)
    next_function = body.find("\ndef ")
    if next_function < 0:
        function_body, suffix = body, ""
    else:
        function_body, suffix = body[:next_function], body[next_function:]
    if function_body.count(anchor) != 1:
        raise ValueError(
            f"expected exactly one patch anchor in {function}: {anchor!r}"
        )
    function_body = function_body.replace(anchor, replacement, 1)
    return prefix + marker + function_body + suffix


def build_override(source: str) -> str:
    """Apply the offline-mirror change to an exact upstream assets module."""
    if PATCH_MARKER in source:
        raise ValueError("source already contains the offline asset mirror patch")
    if source.count(_HELPER_ANCHOR) != 1:
        raise ValueError(f"expected exactly one patch anchor: {_HELPER_ANCHOR!r}")
    result = source.replace(_HELPER_ANCHOR, _HELPER_ANCHOR + _HELPER, 1)
    result = _replace_in_function(
        result, "check_file_path", _CHECK_ANCHOR, _CHECK_REPLACEMENT
    )
    result = _replace_in_function(
        result,
        "retrieve_file_path",
        _RETRIEVE_ANCHOR,
        _RETRIEVE_REPLACEMENT,
    )
    return result


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    args = parser.parse_args()

    source_path = args.input.resolve(strict=True)
    output_path = args.output.resolve()
    if source_path == output_path:
        raise ValueError("input and output paths must differ")
    source = source_path.read_bytes()
    source_sha256 = _sha256(source)
    if source_sha256 != args.expected_input_sha256:
        raise ValueError(
            "input assets.py SHA256 differs from the frozen contract: "
            f"expected {args.expected_input_sha256}, got {source_sha256}"
        )

    output = build_override(source.decode("utf-8")).encode("utf-8")
    _write_atomic(output_path, output)
    receipt = {
        "schema": "rlinf.offline-asset-override/v1",
        "status": "passed",
        "input": {"path": str(source_path), "sha256": source_sha256},
        "output": {"path": str(output_path), "sha256": _sha256(output)},
        "patch_marker": PATCH_MARKER,
    }
    _write_atomic(
        args.receipt.resolve(),
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
