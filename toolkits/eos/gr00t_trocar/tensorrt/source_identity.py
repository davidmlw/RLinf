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

"""Resolve a source revision from Git or an externally attested source tree."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def resolve_source_revision(
    source: Path, expected_revision: str | None = None
) -> dict[str, Any]:
    """Return the source revision while failing closed on identity mismatches."""
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        revision = result.stdout.strip()
        authority = "git"
    elif expected_revision:
        revision = expected_revision
        authority = "external_tree_attestation"
    else:
        raise RuntimeError(
            "source has no readable Git revision and no externally attested "
            f"revision was provided: {result.stderr.strip()}"
        )
    if expected_revision and revision != expected_revision:
        raise RuntimeError(
            f"source revision mismatch: {revision} != {expected_revision}"
        )
    return {
        "revision": revision,
        "authority": authority,
        "git_returncode": result.returncode,
    }
