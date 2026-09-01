#!/usr/bin/env python3
"""Create the W68 Newton-compatible SurgicalTray USD overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pxr import Usd


DEGENERATE_COLLIDERS = (
    "/root/Box001/Collisions/Box001_Collider43",
    "/root/Box001/Collisions/Box001_Collider44",
    "/root/Box001/Collisions/Box001_Collider42",
    "/root/Box001_Prop003/Collisions/Box001_Prop003_Collider4",
    "/root/Box001_Prop002/Collisions/Box001_Prop002_Collider23",
    "/root/Box001_Prop002/Collisions/Box001_Prop002_Collider7",
    "/root/Box001_Prop002/Collisions/Box001_Prop002_Collider22",
    "/root/Box001_Prop002/Collisions/Box001_Prop002_Collider13",
    "/root/Box001_Prop002/Collisions/Box001_Prop002_Collider17",
    "/root/Box001_Prop002/Collisions/Box001_Prop002_Collider12",
    "/root/Box001_Prop002/Collisions/Box001_Prop002_Collider16",
    "/root/Box001_Prop002/Collisions/Box001_Prop002_Collider8",
    "/root/Box001_Prop002/Collisions/Box001_Prop002_Collider4",
    "/root/Box001_Prop002/Collisions/Box001_Prop002_Collider18",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(args.output))
    root = stage.DefinePrim("/root", "Xform")
    root.GetReferences().AddReference(args.reference_url, "/root")
    stage.SetDefaultPrim(root)
    for path in DEGENERATE_COLLIDERS:
        prim = stage.OverridePrim(path)
        prim.SetActive(False)
    stage.GetRootLayer().Save()

    check = Usd.Stage.Open(str(args.output))
    if not check or check.GetDefaultPrim().GetPath().pathString != "/root":
        raise RuntimeError("unexpected SurgicalTray USD default prim")
    for path in DEGENERATE_COLLIDERS:
        if check.GetPrimAtPath(path).IsActive():
            raise RuntimeError(f"collider remained active: {path}")

    receipt = {
        "schema": "rlinf.w68-sanitized-tray-usd/v1",
        "source": str(args.source.resolve()),
        "source_sha256": _sha256(args.source),
        "reference_url": args.reference_url,
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "deactivated_colliders": list(DEGENERATE_COLLIDERS),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
