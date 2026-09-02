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

"""Materialize the exact Isaac-GR00T TensorRT builder environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

EXPECTED = {
    "torch": "2.9.0+cu128",
    "transformers": "4.57.3",
    "flash_attn": "2.8.3",
    "tensorrt": "10.15.1.29",
    "torchcodec": "0.8.1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(python: Path) -> dict[str, str]:
    script = (
        "import json, torch, transformers, flash_attn, tensorrt, torchcodec; "
        "print(json.dumps({'torch': torch.__version__, "
        "'transformers': transformers.__version__, "
        "'flash_attn': flash_attn.__version__, "
        "'tensorrt': tensorrt.__version__, "
        "'torchcodec': torchcodec.__version__}, sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python), "-c", script], check=True, capture_output=True, text=True
    )
    return json.loads(completed.stdout)


def prepare(
    source: Path,
    env_root: Path,
    uv: Path,
    uv_cache: Path,
    torchcodec_wheel: Path,
) -> dict[str, object]:
    source = source.resolve(strict=True)
    uv = uv.resolve(strict=True)
    for name in ("pyproject.toml", "uv.lock"):
        if not (source / name).is_file():
            raise ValueError(f"Isaac-GR00T source omits {name}")
    torchcodec_wheel = torchcodec_wheel.resolve(strict=True)
    inputs = {name: _sha256(source / name) for name in ("pyproject.toml", "uv.lock")}
    inputs["torchcodec_wheel"] = _sha256(torchcodec_wheel)
    manifest_path = env_root / "rlinf-trt-builder-manifest.json"
    python = env_root / "bin" / "python"
    if manifest_path.is_file() and python.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_hashes") != inputs:
            raise RuntimeError("existing TensorRT builder input hashes differ")
        if _probe(python) != EXPECTED:
            raise RuntimeError("existing TensorRT builder package versions differ")
        return manifest
    if env_root.exists():
        raise RuntimeError(
            f"unqualified TensorRT builder environment exists: {env_root}"
        )

    staging = env_root.with_name(f"{env_root.name}.building-{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"builder staging path already exists: {staging}")
    uv_cache.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "UV_PROJECT_ENVIRONMENT": str(staging),
            "UV_CACHE_DIR": str(uv_cache),
            "UV_LINK_MODE": "copy",
        }
    )
    try:
        subprocess.run(
            [str(uv), "sync", "--frozen", "--no-dev", "--project", str(source)],
            check=True,
            env=environment,
        )
        # TorchCodec 0.8.1 is the Torch 2.9-compatible bugfix that removes the
        # 0.8.0 hard dependency on libnvcuvid. The offline LIBERO fixture uses
        # its CPU fallback on H100 compute containers.
        subprocess.run(
            [
                str(uv),
                "pip",
                "install",
                "--python",
                str(staging / "bin" / "python"),
                "--reinstall",
                str(torchcodec_wheel),
            ],
            check=True,
            env=environment,
        )
        packages = _probe(staging / "bin" / "python")
        if packages != EXPECTED:
            raise RuntimeError(
                f"TensorRT builder package versions differ: {packages} != {EXPECTED}"
            )
        manifest = {
            "schema": "rlinf.gr00t-n1d7-trt-builder.v1",
            "source": str(source),
            "input_hashes": inputs,
            "packages": packages,
        }
        (staging / "rlinf-trt-builder-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(env_root)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--uv-cache", type=Path, required=True)
    parser.add_argument("--torchcodec-wheel", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source,
                args.env_root,
                args.uv,
                args.uv_cache,
                args.torchcodec_wheel,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
