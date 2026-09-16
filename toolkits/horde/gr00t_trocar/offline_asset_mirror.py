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

"""Route remote Isaac asset URLs to a qualified local mirror."""

from __future__ import annotations

import io
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from urllib.parse import urlparse

_REMOTE_SCHEMES = frozenset({"http", "https", "omniverse"})


def resolve_mirror_path(
    path: str,
    mirror_root: Path,
    *,
    url_prefix: str = "/Assets",
) -> Path | None:
    """Resolve a remote asset URL inside a mirror of ``url_prefix``."""
    parsed = urlparse(path)
    if parsed.scheme not in _REMOTE_SCHEMES:
        return None

    prefix = f"/{url_prefix.strip('/')}"
    if parsed.path != prefix and not parsed.path.startswith(f"{prefix}/"):
        return None
    relative_path = parsed.path[len(prefix) :].lstrip("/")

    root = mirror_root.expanduser().resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def install_offline_asset_mirror(
    mirror_root: Path,
    *,
    url_prefix: str = "/Assets",
    assets_module: ModuleType | None = None,
) -> dict[str, Callable[..., object]]:
    """Patch IsaacLab asset accessors to prefer an immutable local mirror.

    The original accessors remain the fallback for local paths and remote URLs
    not present in the mirror. The returned mapping can restore the originals.
    """
    if assets_module is None:
        from isaaclab.utils import assets as assets_module  # noqa: PLC0415

    root = mirror_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"asset mirror is not a directory: {root}")

    original_check = assets_module.check_file_path
    original_retrieve = assets_module.retrieve_file_path
    original_read = assets_module.read_file

    def check_file_path(path: str) -> int:
        if resolve_mirror_path(path, root, url_prefix=url_prefix) is not None:
            # Preserve the remote-file status so callers invoke retrieve_file_path.
            return 2
        return original_check(path)

    def retrieve_file_path(
        path: str,
        download_dir: str | None = None,
        force_download: bool = False,
    ) -> str:
        mirrored = resolve_mirror_path(path, root, url_prefix=url_prefix)
        if mirrored is not None and not force_download:
            return os.fspath(mirrored)
        return original_retrieve(
            path,
            download_dir=download_dir,
            force_download=force_download,
        )

    def read_file(path: str) -> io.BytesIO:
        mirrored = resolve_mirror_path(path, root, url_prefix=url_prefix)
        if mirrored is not None:
            return io.BytesIO(mirrored.read_bytes())
        return original_read(path)

    assets_module.check_file_path = check_file_path
    assets_module.retrieve_file_path = retrieve_file_path
    assets_module.read_file = read_file
    return {
        "check_file_path": original_check,
        "retrieve_file_path": original_retrieve,
        "read_file": original_read,
    }
