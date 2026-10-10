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

"""Render and validate W18 configs from the retained W16 optimized endpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROFILE = {
    "backend": "nsight",
    "enabled": True,
    "worker_groups": ["ActorGroup"],
    "ranks": list(range(8)),
    "steps": [1],
    "continuous": True,
    "output_dir": "/w18-run/nsights",
    "options": {
        "trace": "cuda,nvtx,osrt",
        "sample": "none",
        "cpuctxsw": "none",
        "osrt-threshold": 1000,
        "force-overwrite": True,
    },
    "flags": [],
}

ALLOWED_DIFF_PREFIXES = (
    "actor.micro_batch_size",
    "cluster.profiling",
    "env.eval.video_cfg.video_base_dir",
    "env.train.video_cfg.video_base_dir",
    "runner.logger.experiment_name",
    "runner.logger.log_path",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_diff_paths(left[key], right[key], path))
        return paths
    if isinstance(left, list):
        return [] if left == right else [prefix]
    return [] if left == right else [prefix]


def render_variant(
    base: dict[str, Any], *, micro_batch_size: int, diagnostic: bool
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    if diagnostic:
        config["cluster"]["profiling"] = copy.deepcopy(PROFILE)
        name = "w18_rank_diagnostic_mb8"
    else:
        config["cluster"].pop("profiling", None)
        name = f"w18_performance_mb{micro_batch_size}"
    config["actor"]["micro_batch_size"] = micro_batch_size
    config["runner"]["logger"]["experiment_name"] = name
    config["runner"]["logger"]["log_path"] = "/w18-run/output"
    config["env"]["train"]["video_cfg"]["video_base_dir"] = (
        "/w18-run/output/video/train"
    )
    config["env"]["eval"]["video_cfg"]["video_base_dir"] = (
        "/w18-run/output/video/eval"
    )
    return config


def validate_variant(
    base: dict[str, Any], candidate: dict[str, Any], *, spec: dict[str, Any]
) -> list[str]:
    errors = []
    micro_batch_size = int(spec["micro_batch_size"])
    diagnostic = bool(spec["diagnostic"])
    expected = render_variant(
        base, micro_batch_size=micro_batch_size, diagnostic=diagnostic
    )
    if candidate != expected:
        errors.append("candidate does not match deterministic rendering")
    if candidate["actor"]["global_batch_size"] != 2048:
        errors.append("global_batch_size must remain 2048")
    if candidate["algorithm"]["update_epoch"] != 4:
        errors.append("update_epoch must remain 4")
    if 2048 % (8 * micro_batch_size) != 0:
        errors.append("micro_batch_size must divide the per-rank batch of 256")
    if diagnostic and micro_batch_size != 8:
        errors.append("the all-rank diagnostic must use baseline microbatch 8")

    unexpected = [
        path
        for path in _diff_paths(base, candidate)
        if not any(
            path == allowed or path.startswith(f"{allowed}.")
            for allowed in ALLOWED_DIFF_PREFIXES
        )
    ]
    if unexpected:
        errors.append(f"unexpected diff paths: {unexpected!r}")
    return errors


def render(args: argparse.Namespace) -> int:
    base = _load_yaml(args.base)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = {}
    for micro_batch_size in args.micro_batch_sizes:
        name = f"performance-mb{micro_batch_size}"
        config = render_variant(
            base, micro_batch_size=micro_batch_size, diagnostic=False
        )
        path = args.output_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        variants[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "micro_batch_size": micro_batch_size,
            "diagnostic": False,
        }

    name = "rank-diagnostic-mb8"
    config = render_variant(base, micro_batch_size=8, diagnostic=True)
    path = args.output_dir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    variants[name] = {
        "path": str(path),
        "sha256": _sha256(path),
        "micro_batch_size": 8,
        "diagnostic": True,
    }
    manifest = {
        "schema": "rlinf.w18.rtx6000-fsdp-configs/v1",
        "source_revision": args.source_revision,
        "base": {"path": str(args.base), "sha256": _sha256(args.base)},
        "variants": variants,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def validate(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rlinf.w18.rtx6000-fsdp-configs/v1":
        raise ValueError("unexpected manifest schema")
    spec = manifest["variants"][args.variant]
    if _sha256(args.config) != spec["sha256"]:
        raise ValueError("candidate config SHA256 mismatch")
    errors = validate_variant(
        _load_yaml(args.base), _load_yaml(args.config), spec=spec
    )
    if errors:
        raise ValueError("\n".join(errors))
    if args.receipt is not None:
        args.receipt.write_text(
            json.dumps(
                {
                    "schema": "rlinf.w18.config-validation/v1",
                    "status": "passed",
                    "variant": args.variant,
                    "config_sha256": spec["sha256"],
                    "micro_batch_size": spec["micro_batch_size"],
                    "diagnostic": spec["diagnostic"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--base", type=Path, required=True)
    render_parser.add_argument("--output-dir", type=Path, required=True)
    render_parser.add_argument("--source-revision", required=True)
    render_parser.add_argument(
        "--micro-batch-sizes", type=int, nargs="+", default=[8, 16, 32, 64]
    )
    render_parser.set_defaults(func=render)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--base", type=Path, required=True)
    validate_parser.add_argument("--config", type=Path, required=True)
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--variant", required=True)
    validate_parser.add_argument("--receipt", type=Path)
    validate_parser.set_defaults(func=validate)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
