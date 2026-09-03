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

from builder_probe import (
    EXPECTED_DISTRIBUTIONS,
    qualification_contract,
    runtime_library_paths,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _isolated_environment(
    environment: dict[str, str], env_root: Path | None = None
) -> dict[str, str]:
    isolated = dict(environment)
    isolated.pop("PYTHONPATH", None)
    isolated["PYTHONNOUSERSITE"] = "1"
    isolated["PIP_NO_CACHE_DIR"] = "1"
    if env_root is not None:
        inherited = isolated.get("LD_LIBRARY_PATH")
        paths = [str(path) for path in runtime_library_paths(env_root)]
        if inherited:
            paths.append(inherited)
        isolated["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
    return isolated


def _run_probe(env_root: Path, source: Path, dataset: Path) -> dict[str, object]:
    output = env_root / "rlinf-trt-builder-probe-latest.json"
    video = (
        dataset / "videos/chunk-000/observation.images.image/episode_000000.mp4"
    ).resolve(strict=True)
    command = [
        str(env_root / "bin/python"),
        str(Path(__file__).with_name("builder_probe.py")),
        "--env-root",
        str(env_root),
        "--video",
        str(video),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=source,
        env=_isolated_environment(dict(os.environ), env_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        if output.is_file():
            diagnostic = output.read_text(encoding="utf-8")
        raise RuntimeError(f"builder package/video probe failed: {diagnostic}")
    return json.loads(output.read_text(encoding="utf-8"))


def prepare(
    source: Path,
    env_root: Path,
    uv: Path,
    uv_cache: Path,
    torchcodec_wheel: Path,
    dataset: Path,
) -> dict[str, object]:
    source = source.resolve(strict=True)
    dataset = dataset.resolve(strict=True)
    uv = uv.resolve(strict=True)
    for name in ("pyproject.toml", "uv.lock"):
        if not (source / name).is_file():
            raise ValueError(f"Isaac-GR00T source omits {name}")
    torchcodec_wheel = torchcodec_wheel.resolve(strict=True)
    inputs = {name: _sha256(source / name) for name in ("pyproject.toml", "uv.lock")}
    inputs["torchcodec_wheel"] = _sha256(torchcodec_wheel)
    inputs["builder_probe"] = _sha256(Path(__file__).with_name("builder_probe.py"))
    decode_video = (
        dataset / "videos/chunk-000/observation.images.image/episode_000000.mp4"
    ).resolve(strict=True)
    inputs["decode_video"] = _sha256(decode_video)
    manifest_path = env_root / "rlinf-trt-builder-manifest.json"
    python = env_root / "bin" / "python"
    if manifest_path.is_file() and python.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_hashes") != inputs:
            raise RuntimeError("existing TensorRT builder input hashes differ")
        probe = _run_probe(env_root, source, dataset)
        if probe.get("packages") != EXPECTED_DISTRIBUTIONS:
            raise RuntimeError("existing TensorRT builder package versions differ")
        contract = qualification_contract(probe)
        if contract != manifest.get("probe_contract"):
            raise RuntimeError("existing TensorRT builder probe contract changed")
        result = dict(manifest)
        result["revalidation_probe_sha256"] = _sha256(
            env_root / "rlinf-trt-builder-probe-latest.json"
        )
        result["revalidation_contract_sha256"] = _json_sha256(contract)
        return result
    if env_root.exists():
        raise RuntimeError(
            f"unqualified TensorRT builder environment exists: {env_root}"
        )

    staging = env_root.with_name(f"{env_root.name}.building-{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"builder staging path already exists: {staging}")
    uv_cache.mkdir(parents=True, exist_ok=True)
    environment = _isolated_environment(dict(os.environ))
    environment.update(
        {
            "UV_PROJECT_ENVIRONMENT": str(staging),
            "UV_CACHE_DIR": str(uv_cache),
            "UV_LINK_MODE": "copy",
        }
    )
    promoted = False
    try:
        subprocess.run(
            [str(uv), "sync", "--frozen", "--no-dev", "--project", str(source)],
            check=True,
            env=environment,
        )
        staging_python = staging / "bin/python"
        subprocess.run(
            [
                str(uv),
                "pip",
                "install",
                "--no-cache",
                "--python",
                str(staging_python),
                "pip==25.3",
            ],
            check=True,
            env=environment,
        )
        # Keep uv for the frozen project environment, but use the target
        # interpreter's pip for the one reviewed compatibility overlay.
        subprocess.run(
            [
                str(staging_python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                "--no-cache-dir",
                str(torchcodec_wheel),
            ],
            check=True,
            env=environment,
        )
        staging.rename(env_root)
        promoted = True
        pip_show = subprocess.run(
            [str(env_root / "bin/python"), "-m", "pip", "show", "-f", "torchcodec"],
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )
        pip_show_path = env_root / "torchcodec-pip-show.txt"
        pip_show_path.write_text(pip_show.stdout, encoding="utf-8")
        probe = _run_probe(env_root, source, dataset)
        if probe.get("packages") != EXPECTED_DISTRIBUTIONS:
            raise RuntimeError("TensorRT builder package versions differ")
        initial_probe = env_root / "rlinf-trt-builder-probe-initial.json"
        shutil.copyfile(env_root / "rlinf-trt-builder-probe-latest.json", initial_probe)
        contract = qualification_contract(probe)
        manifest = {
            "schema": "rlinf.gr00t-n1d7-trt-builder.v3",
            "source": str(source),
            "input_hashes": inputs,
            "packages": probe["packages"],
            "initial_probe_sha256": _sha256(initial_probe),
            "probe_contract": contract,
            "probe_contract_sha256": _json_sha256(contract),
            "pip_show_sha256": _sha256(pip_show_path),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if promoted and not manifest_path.is_file():
            shutil.rmtree(env_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--uv-cache", type=Path, required=True)
    parser.add_argument("--torchcodec-wheel", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source,
                args.env_root,
                args.uv,
                args.uv_cache,
                args.torchcodec_wheel,
                args.dataset,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
