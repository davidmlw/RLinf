# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Materialize an isolated UV overlay without changing qualified dependencies."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    """Build external runtime state; never install into the inherited prefix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-python", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    root = args.state_root.resolve()
    repo = Path(__file__).resolve().parents[2]
    if root.is_relative_to(repo):
        raise ValueError("The reference runtime must be outside the source checkout")
    identity = json.loads(
        subprocess.check_output(
            [
                str(args.base_python),
                "-c",
                "import json,sys,sysconfig,torch;print(json.dumps({'python':sys.version,'site':sysconfig.get_path('purelib'),'torch':torch.__version__}))",
            ],
            text=True,
        )
    )
    if not identity["torch"].startswith("2.7.0"):
        raise ValueError("Reference models require the qualified Torch2.7 prefix")
    env = root / "uv/rlinf-torch27"
    if not (env / "bin/python").exists():
        subprocess.run(
            [args.uv, "venv", "--python", str(args.base_python), str(env)], check=True
        )
    purelib = subprocess.check_output(
        [
            str(env / "bin/python"),
            "-c",
            "import sysconfig;print(sysconfig.get_path('purelib'))",
        ],
        text=True,
    ).strip()
    # A regular site .pth is processed after this overlay's own dependencies.
    # It imports no Poiesis modules; source remains the pinned RLInf checkout.
    (Path(purelib) / "qualified-native-dependencies.pth").write_text(
        identity["site"] + "\n"
    )
    subprocess.run(
        [
            args.uv,
            "pip",
            "install",
            "--python",
            str(env / "bin/python"),
            "--no-deps",
            "-r",
            str(Path(__file__).with_name("requirements-overlay.in")),
        ],
        check=True,
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "runtime-dependencies.json").write_text(
        json.dumps(
            {
                "base_python": str(args.base_python),
                "base_identity": identity,
                "python": str(env / "bin/python"),
                "source": str(repo),
                "boundary": "third-party dependency reuse only; no Poiesis imports",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
