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

"""Launch the finite W12 RTX 6000 runtime and workload qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

IMAGE = (
    "chenchaox72877/trocar-rlinf-bench@"
    "sha256:9f02e069ccb0e0a7e536833e789666e85f039c9024a77ac2f219c42cb1dfcf01"
)
EXPECTED_IMAGE_ID = (
    "sha256:db746c040dd15cdd68fdcda5b40f514bd2bf31d0fb462ae493fe83b7a0142bf1"
)
EXPECTED_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
EXPECTED_GPU_COUNT = 8
PYTHONPATH = "/w96-overlay:/w96-trt-runtime:/workspace/gr00t-n17:/workspace/rlinf-src"
PYTHON = "/isaac-sim/kit/python/bin/python3"
ISAAC_PYTHON = "/isaac-sim/python.sh"


class QualificationError(RuntimeError):
    """Raised when a fail-closed qualification gate does not pass."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    os.replace(temporary, path)


def _run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise QualificationError(
            f"command failed ({result.returncode}): {argv!r}\n{result.stderr}"
        )
    return result


def _preflight(docker: Path) -> dict[str, Any]:
    smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [[item.strip() for item in line.split(",")] for line in smi.stdout.splitlines()]
    gpu_gate = (
        len(rows) == EXPECTED_GPU_COUNT
        and [int(row[0]) for row in rows] == list(range(EXPECTED_GPU_COUNT))
        and all(row[1] == EXPECTED_GPU_NAME and row[2] == "12.0" for row in rows)
        and all(int(row[3]) < 1024 for row in rows)
    )
    inspect = _run([str(docker), "image", "inspect", IMAGE])
    inspected = json.loads(inspect.stdout)[0]
    image_gate = inspected.get("Id") == EXPECTED_IMAGE_ID and IMAGE.split("@", 1)[1] in {
        value.split("@", 1)[1]
        for value in inspected.get("RepoDigests", [])
        if "@" in value
    }
    if not gpu_gate or not image_gate:
        raise QualificationError("GPU inventory or immutable image identity did not pass")
    return {
        "gpu_rows": rows,
        "gpu_gate": gpu_gate,
        "image_id": inspected.get("Id"),
        "image_repo_digests": inspected.get("RepoDigests", []),
        "image_gate": image_gate,
    }


def _require_new_run_root(path: Path) -> None:
    if path.exists():
        raise QualificationError(f"run root must not already exist: {path}")
    path.mkdir(parents=True)


def _require_inputs(source: Path, bundle: Path) -> dict[str, Path]:
    inputs = {
        "source": source,
        "gr00t": bundle / "sources/isaac-gr00t",
        "overlay": bundle / "python/w96-overlay",
        "tensorrt": bundle / "runtime/tensorrt-10.15.1.29",
        "model": bundle / "model/GR00T-N1.7-3B",
        "backbone": bundle / "model/Cosmos-Reason2-2B",
        "config": bundle / "config/absolute-correctness-b8-all-off.yaml",
        "extension": bundle / "overrides/extension.py",
        "assets_override": bundle / "overrides/assets.py",
        "metadata": bundle / "overrides/trocar-metadata.json",
        "assets_seed": bundle / "assets/assets-cache",
    }
    missing = [f"{name}={path}" for name, path in inputs.items() if not path.exists()]
    if missing:
        raise QualificationError("missing immutable inputs: " + ", ".join(missing))
    return inputs


def _ensure_absent(docker: Path, container: str) -> None:
    result = _run([str(docker), "container", "inspect", container], check=False)
    if result.returncode == 0:
        raise QualificationError(f"container still exists: {container}")
    if "No such container" not in result.stderr:
        raise QualificationError(
            f"container absence check failed ({result.returncode}): {result.stderr}"
        )


def _container_args(
    docker: Path,
    container: str,
    run_root: Path,
    inputs: dict[str, Path],
    command: str,
) -> list[str]:
    mounts = [
        (inputs["source"], "/workspace/rlinf-src", "ro"),
        (inputs["gr00t"], "/workspace/gr00t-n17", "ro"),
        (inputs["overlay"], "/w96-overlay", "ro"),
        (inputs["tensorrt"], "/w96-trt-runtime", "ro"),
        (inputs["graphics"], "/w12-driver", "ro"),
        (inputs["model"], "/models/GR00T-N1.7-3B", "ro"),
        (inputs["backbone"], "/w96-model-inputs/Cosmos-Reason2-2B", "ro"),
        (
            inputs["config"],
            "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/"
            "contrib/assemble_trocar/config/"
            "isaaclab_ppo_gr00t_assemble_trocar_prod.yaml",
            "ro",
        ),
        (
            inputs["extension"],
            "/workspace/isaaclab/source/isaaclab_contrib/isaaclab_contrib/"
            "rl/rlinf/extension.py",
            "ro",
        ),
        (
            inputs["assets_override"],
            "/workspace/isaaclab/source/isaaclab/isaaclab/utils/assets.py",
            "ro",
        ),
        (run_root / "scratch/assets-cache", "/tmp/Assets", "rw"),
        (run_root / "scratch/kit-cache", "/isaac-sim/kit/cache", "rw"),
        (run_root / "scratch/kit-data", "/isaac-sim/kit/data", "rw"),
        (inputs["metadata"], "/w96-inputs/trocar/metadata.json", "ro"),
        (run_root, "/w12-run", "rw"),
        (run_root, "/w96-run", "rw"),
    ]
    args = [
        str(docker),
        "run",
        "--name",
        container,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--gpus",
        "all",
        "--network",
        "none",
        "--entrypoint",
        "/bin/bash",
        "--shm-size=64g",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--cap-add=SYS_ADMIN",
        "--cap-add=SYS_PTRACE",
        "--security-opt",
        "seccomp=unconfined",
    ]
    environment = {
        "HOME": "/w12-run/scratch/home",
        "USER": "liweim",
        "LOGNAME": "liweim",
        "XDG_CACHE_HOME": "/w12-run/scratch/cache",
        "TORCHINDUCTOR_CACHE_DIR": "/w12-run/scratch/torchinductor",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": PYTHONPATH,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NVIDIA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
        "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "ISAAC_PATH": "/isaac-sim",
        "EXP_PATH": "/isaac-sim/apps",
        "CARB_APP_PATH": "/isaac-sim/kit",
        "W77_BACKBONE_MODEL_ROOT": "/w96-model-inputs/Cosmos-Reason2-2B",
        "W77_TROCAR_METADATA": "/w96-inputs/trocar/metadata.json",
        "RLINF_EXT_MODULE": "toolkits.gr00t_trocar.vulkan_extension",
        "RLINF_CONFIG_FILE": (
            "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/"
            "contrib/assemble_trocar/config/"
            "isaaclab_ppo_gr00t_assemble_trocar_prod.yaml"
        ),
    }
    for name, value in environment.items():
        args.extend(["-e", f"{name}={value}"])
    for source, target, mode in mounts:
        args.extend(["-v", f"{source.resolve()}:{target}:{mode}"])
    return [*args, IMAGE, "-c", command]


def _command(phase: str) -> str:
    setup = (
        'set -euo pipefail; export LD_LIBRARY_PATH="/w12-driver:/w96-trt-runtime/'
        'tensorrt_libs:${LD_LIBRARY_PATH:-}"; '
    )
    probe = (
        f"{PYTHON} /workspace/rlinf-src/toolkits/gr00t_trocar/w12/"
        f"rtx6000_runtime_probe.py --output /w12-run/{phase}-runtime.json"
    )
    if phase == "q1":
        return setup + probe
    smoke = "/workspace/rlinf-src/toolkits/gr00t_trocar/w95/l20_q2_smoke.py"
    return (
        setup
        + probe
        + f"; {ISAAC_PYTHON} {smoke} bootstrap --output /w12-run/q2-bootstrap.json"
        + f"; {PYTHON} {smoke} model --model /models/GR00T-N1.7-3B "
        "--backbone-model /w96-model-inputs/Cosmos-Reason2-2B "
        "--metadata /w96-inputs/trocar/metadata.json "
        "--output /w12-run/q2-model.json"
        + f"; {ISAAC_PYTHON} {smoke} env --num-envs 1 "
        "--output /w12-run/q2-env1.json"
        + f"; {ISAAC_PYTHON} {smoke} env --num-envs 8 "
        "--output /w12-run/q2-env8.json"
    )


def launch(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve(strict=True)
    bundle = args.bundle.resolve(strict=True)
    graphics_runtime = args.graphics_runtime.resolve(strict=True)
    docker = Path(shutil.which("docker") or "")
    if not docker.is_file():
        raise QualificationError("docker executable is unavailable")
    _require_new_run_root(args.run_root)
    run_root = args.run_root.resolve()
    (run_root / "receipts").mkdir()
    inputs = _require_inputs(source, bundle)
    inputs["graphics"] = graphics_runtime
    shutil.copytree(inputs["assets_seed"], run_root / "scratch/assets-cache")
    for name in ("home", "kit-cache", "kit-data"):
        (run_root / f"scratch/{name}").mkdir()
    asset_cache = run_root / "scratch/assets-cache"
    for path in [asset_cache, *asset_cache.rglob("*")]:
        if not path.is_symlink():
            writable_bits = 0o700 if path.is_dir() else 0o600
            path.chmod(path.stat().st_mode | writable_bits)
    preflight = _preflight(docker)
    container = f"w12-{args.phase}-{int(time.time())}"
    _ensure_absent(docker, container)
    command = _command(args.phase)
    argv = _container_args(docker, container, run_root, inputs, command)
    _write(run_root / "receipts/container-command.json", argv)
    started = time.time()
    result: subprocess.CompletedProcess[str] | None = None
    cleanup: dict[str, Any] = {"status": "pending"}
    try:
        result = _run(argv, check=False)
        (run_root / "stdout.log").write_text(result.stdout, encoding="utf-8")
        (run_root / "stderr.log").write_text(result.stderr, encoding="utf-8")
    finally:
        removed = _run([str(docker), "rm", "-f", container], check=False)
        absent = _run([str(docker), "container", "inspect", container], check=False)
        cleanup = {
            "status": (
                "passed"
                if removed.returncode == 0
                and absent.returncode != 0
                and "No such container" in absent.stderr
                else "failed"
            ),
            "remove_rc": removed.returncode,
            "remove_stdout": removed.stdout,
            "remove_stderr": removed.stderr,
            "inspect_rc": absent.returncode,
            "inspect_stdout": absent.stdout,
            "inspect_stderr": absent.stderr,
        }
    expected_receipts = [run_root / f"{args.phase}-runtime.json"]
    if args.phase == "q2":
        expected_receipts.extend(
            run_root / name
            for name in (
                "q2-bootstrap.json",
                "q2-model.json",
                "q2-env1.json",
                "q2-env8.json",
            )
        )
    receipts = {}
    for path in expected_receipts:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        receipts[path.name] = {
            "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
            "status": value.get("status") if isinstance(value, dict) else None,
        }
    passed = (
        result is not None
        and result.returncode == 0
        and cleanup["status"] == "passed"
        and all(value["status"] == "passed" for value in receipts.values())
    )
    receipt = {
        "schema": f"rlinf.w12.rtx6000-{args.phase}/v1",
        "status": "passed" if passed else "failed",
        "phase": args.phase,
        "source_revision": args.source_revision,
        "source_path": str(source),
        "bundle_path": str(bundle),
        "graphics_runtime_path": str(graphics_runtime),
        "contract": {
            "path": str(source / "toolkits/gr00t_trocar/w12/contract-rtx6000-n1d7.json"),
            "sha256": _sha256(
                source / "toolkits/gr00t_trocar/w12/contract-rtx6000-n1d7.json"
            ),
        },
        "image": IMAGE,
        "preflight": preflight,
        "container_rc": result.returncode if result is not None else None,
        "elapsed_seconds": time.time() - started,
        "receipts": receipts,
        "cleanup": cleanup,
    }
    _write(run_root / "receipts/qualification.json", receipt)
    if not passed:
        raise QualificationError(f"{args.phase} qualification did not pass")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("q1", "q2"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--graphics-runtime", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = launch(args)
    except Exception as error:
        print(f"W12 qualification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
