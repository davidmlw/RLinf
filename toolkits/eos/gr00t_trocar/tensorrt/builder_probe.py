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

"""Fail-closed package and real-video probe for the official TensorRT builder."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import site
import subprocess
import sys
from pathlib import Path

EXPECTED_DISTRIBUTIONS = {
    "pip": "25.3",
    "torch": "2.9.0+cu128",
    "torchvision": "0.24.0+cu128",
    "transformers": "4.57.3",
    "flash-attn": "2.8.3",
    "onnx": "1.20.1",
    "onnx-ir": "0.2.1",
    "onnxscript": "0.7.0",
    "tensorrt-cu12": "10.15.1.29",
    "tensorrt-cu12-bindings": "10.15.1.29",
    "tensorrt-cu12-libs": "10.15.1.29",
    "torchcodec": "0.8.1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _loaded_torchcodec_libraries(package_root: Path) -> list[Path]:
    maps = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    loaded = {
        Path(line.rsplit(maxsplit=1)[-1]).resolve()
        for line in maps
        if "/" in line and "torchcodec" in line and ".so" in line
    }
    return sorted(path for path in loaded if _is_relative_to(path, package_root))


def _ldd(path: Path) -> list[str]:
    completed = subprocess.run(
        ["ldd", str(path)], check=False, capture_output=True, text=True
    )
    lines = (completed.stdout + completed.stderr).splitlines()
    if completed.returncode or any("not found" in line for line in lines):
        raise RuntimeError(f"native dependency resolution failed for {path}: {lines}")
    return lines


def probe(env_root: Path, video: Path) -> dict[str, object]:
    import numpy as np

    env_root = env_root.resolve(strict=True)
    video = video.resolve(strict=True)
    expected_python = env_root / "bin" / "python"
    if Path(sys.executable) != expected_python:
        raise RuntimeError(
            f"wrong builder interpreter: {sys.executable} != {expected_python}"
        )
    if os.environ.get("PYTHONNOUSERSITE") != "1" or "PYTHONPATH" in os.environ:
        raise RuntimeError(
            "builder probe requires PYTHONNOUSERSITE=1 and no PYTHONPATH"
        )
    if site.ENABLE_USER_SITE:
        raise RuntimeError("builder interpreter still enables the user site")

    packages = {
        name: importlib.metadata.version(name) for name in EXPECTED_DISTRIBUTIONS
    }
    if packages != EXPECTED_DISTRIBUTIONS:
        raise RuntimeError(f"builder package versions differ: {packages}")

    import torchcodec
    from gr00t.utils.video_utils import get_frames_by_indices

    module_path = Path(torchcodec.__file__).resolve(strict=True)
    if not _is_relative_to(module_path, env_root):
        raise RuntimeError(f"TorchCodec imported outside builder venv: {module_path}")
    if torchcodec.__version__ != EXPECTED_DISTRIBUTIONS["torchcodec"]:
        raise RuntimeError(
            f"TorchCodec module version differs: {torchcodec.__version__}"
        )

    distribution = importlib.metadata.distribution("torchcodec")
    distribution_files = sorted(
        str(distribution.locate_file(path).resolve(strict=True).relative_to(env_root))
        for path in distribution.files or ()
    )
    frames = np.ascontiguousarray(
        get_frames_by_indices(str(video), [0], decoder_kwargs={"device": "cpu"})
    )
    loaded_libraries = _loaded_torchcodec_libraries(env_root)
    if not loaded_libraries:
        raise RuntimeError(
            "TorchCodec decode loaded no native library from builder venv"
        )

    return {
        "schema": "rlinf.gr00t-n1d7-trt-builder-probe.v1",
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "sys_path": sys.path,
            "user_site_enabled": site.ENABLE_USER_SITE,
        },
        "environment": {
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
            "PYTHONPATH_present": "PYTHONPATH" in os.environ,
        },
        "packages": packages,
        "torchcodec": {
            "module_version": torchcodec.__version__,
            "module_path": str(module_path),
            "distribution_files": distribution_files,
            "loaded_native_libraries": {
                str(path.relative_to(env_root)): _ldd(path) for path in loaded_libraries
            },
        },
        "decode": {
            "utility": "gr00t.utils.video_utils.get_frames_by_indices",
            "device": "cpu",
            "video": str(video),
            "video_sha256": _sha256(video),
            "requested_indices": [0],
            "requested_timestamps": None,
            "shape": list(frames.shape),
            "dtype": str(frames.dtype),
            "min": float(frames.min()),
            "max": float(frames.max()),
            "frame_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = probe(args.env_root, args.video)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
