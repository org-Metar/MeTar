#!/usr/bin/env python3
"""Place user-obtained public inputs at the canonical artifact paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = REPOSITORY_ROOT / "manifests/pretrained_artifact_contract.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe artifact path: {relative}")
    return root.joinpath(*pure.parts)


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported artifact contract schema")
    return payload


def materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    relative = os.path.relpath(source.resolve(), destination.parent.resolve())
    destination.symlink_to(relative)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--include-derived-source",
        action="store_true",
        help="link a prebuilt derived cache when it exists below --source-root",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = args.package.resolve()
    source_root = args.source_root.resolve()
    contract = load_contract(args.contract.resolve())
    rows = []
    failures = []
    for item in contract["external_inputs"]:
        relative = str(item["path"])
        expected = str(item["sha256"])
        destination = safe_path(package, relative)
        kind = str(item["materialization"])
        if destination.is_file():
            observed = sha256_file(destination)
            status = "present" if observed == expected else "hash_mismatch"
        elif kind == "derived" and not args.include_derived_source:
            observed = None
            status = "derive_with_build_h07_cache"
        elif args.check_only:
            observed = None
            status = "missing"
        else:
            source = safe_path(source_root, relative)
            if not source.is_file():
                observed = None
                status = "source_missing"
            else:
                observed = sha256_file(source)
                status = "source_ready" if observed == expected else "source_hash_mismatch"
                if status == "source_ready":
                    materialize(source, destination, args.mode)
                    status = "materialized"
        rows.append({"path": relative, "kind": kind, "status": status})
        if status in {"hash_mismatch", "missing", "source_missing", "source_hash_mismatch"}:
            failures.append(relative)
    report = {
        "package": str(package),
        "source_root": str(source_root),
        "mode": args.mode,
        "inputs": rows,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
