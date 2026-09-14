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

"""Render and validate the W95 GR00T N1.7 L20 experiment contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONTRACT = ROOT / "contract-v1.json"
MISSING = object()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _parts(dotted: str) -> list[str]:
    parts = dotted.split(".")
    if not all(parts):
        raise ValueError(f"invalid dotted path: {dotted}")
    return parts


def _get(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in _parts(dotted):
        if not isinstance(value, dict) or part not in value:
            return MISSING
        value = value[part]
    return value


def _set(data: dict[str, Any], dotted: str, value: Any) -> None:
    target: dict[str, Any] = data
    parts = _parts(dotted)
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot descend through non-mapping path: {dotted}")
        target = child
    target[parts[-1]] = value


def _remove(data: dict[str, Any], dotted: str) -> None:
    target: Any = data
    parts = _parts(dotted)
    for part in parts[:-1]:
        if not isinstance(target, dict) or part not in target:
            return
        target = target[part]
    if isinstance(target, dict):
        target.pop(parts[-1], None)


def _contract(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("schema") != "rlinf.gr00t-n17-l20-contract/v1":
        raise ValueError(f"unsupported contract schema in {path}")
    return value


def render(
    base: dict[str, Any], contract: dict[str, Any], profile: str, arm: str
) -> dict[str, Any]:
    if profile not in contract["profiles"]:
        raise ValueError(f"unknown profile: {profile}")
    if arm not in contract["arms"]:
        raise ValueError(f"unknown arm: {arm}")

    rendered = copy.deepcopy(base)
    _set(
        rendered,
        "actor.micro_batch_size",
        contract["profiles"][profile]["actor_micro_batch_size"],
    )
    for dotted, value in contract["arms"][arm]["set"].items():
        _set(rendered, dotted, value)
    for dotted in contract["arms"][arm]["remove"]:
        _remove(rendered, dotted)
    _set(rendered, "runner.logger.experiment_name", f"w95_{profile}_{arm}")
    return rendered


def validate(
    config: dict[str, Any], contract: dict[str, Any], profile: str, arm: str
) -> list[str]:
    errors: list[str] = []
    if profile not in contract["profiles"]:
        return [f"unknown profile: {profile}"]
    if arm not in contract["arms"]:
        return [f"unknown arm: {arm}"]

    expected = dict(contract["common_config"])
    expected["actor.micro_batch_size"] = contract["profiles"][profile][
        "actor_micro_batch_size"
    ]
    expected.update(contract["arms"][arm]["set"])
    for dotted, wanted in expected.items():
        actual = _get(config, dotted)
        if actual is MISSING:
            errors.append(f"missing required config path: {dotted}")
        elif actual != wanted:
            errors.append(f"{dotted}: expected {wanted!r}, got {actual!r}")

    for dotted in contract["arms"][arm]["remove"]:
        if _get(config, dotted) is not MISSING:
            errors.append(f"config path must be absent for {arm}: {dotted}")
    for dotted in contract["forbidden_config_paths"]:
        if _get(config, dotted) is not MISSING:
            errors.append(f"executor path is forbidden in W97 arm config: {dotted}")
    return errors


def _git(source_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_root), *args], text=True
    ).strip()


def _manifest(
    source_root: Path,
    config_path: Path,
    contract_path: Path,
    profile: str,
    arm: str,
    evidence: list[Path],
) -> dict[str, Any]:
    contract = _contract(contract_path)
    config = _load_yaml(config_path)
    errors = validate(config, contract, profile, arm)
    if errors:
        raise ValueError("config validation failed:\n" + "\n".join(errors))

    status = _git(source_root, "status", "--porcelain")
    if status:
        raise ValueError("source worktree must be clean")
    head = _git(source_root, "rev-parse", "HEAD")
    for key in ("rlinf_base_sha", "required_ancestor_sha"):
        ancestor = contract["source"][key]
        result = subprocess.run(
            ["git", "-C", str(source_root), "merge-base", "--is-ancestor", ancestor, head],
            check=False,
        )
        if result.returncode != 0:
            raise ValueError(f"required source ancestor is absent: {ancestor}")

    files = [contract_path, config_path, *evidence]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError("missing provenance inputs: " + ", ".join(missing))
    return {
        "schema": "rlinf.gr00t-n17-l20-manifest/v1",
        "contract_schema": contract["schema"],
        "source": {
            "head": head,
            "base": contract["source"]["rlinf_base_sha"],
            "required_ancestor": contract["source"]["required_ancestor_sha"],
            "clean": True,
        },
        "profile": profile,
        "authority": contract["profiles"][profile]["authority"],
        "arm": arm,
        "files": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in files
        ],
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--base", type=Path, required=True)
    render_parser.add_argument("--profile", required=True)
    render_parser.add_argument("--arm", required=True)
    render_parser.add_argument("--output", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--config", type=Path, required=True)
    validate_parser.add_argument("--profile", required=True)
    validate_parser.add_argument("--arm", required=True)
    validate_parser.add_argument("--receipt", type=Path)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--source-root", type=Path, required=True)
    manifest_parser.add_argument("--config", type=Path, required=True)
    manifest_parser.add_argument("--profile", required=True)
    manifest_parser.add_argument("--arm", required=True)
    manifest_parser.add_argument("--evidence", type=Path, action="append", default=[])
    manifest_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    contract = _contract(args.contract)
    if args.command == "render":
        rendered = render(_load_yaml(args.base), contract, args.profile, args.arm)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")
        return 0
    if args.command == "validate":
        errors = validate(_load_yaml(args.config), contract, args.profile, args.arm)
        receipt = {
            "schema": "rlinf.gr00t-n17-l20-config-validation/v1",
            "contract_sha256": _sha256(args.contract),
            "config_sha256": _sha256(args.config),
            "profile": args.profile,
            "authority": contract["profiles"].get(args.profile, {}).get("authority"),
            "arm": args.arm,
            "status": "passed" if not errors else "failed",
            "errors": errors,
        }
        if args.receipt:
            _write_json(args.receipt, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 0 if not errors else 1
    if args.command == "manifest":
        value = _manifest(
            args.source_root,
            args.config,
            args.contract,
            args.profile,
            args.arm,
            args.evidence,
        )
        _write_json(args.output, value)
        print(json.dumps(value, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
