"""Shared helpers for article-exact pretrained verification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def critical_hashes(package: Path) -> dict[str, str]:
    """Read model identities from the checked-in public input contract."""
    contract = Path(__file__).resolve().parents[1] / "manifests/pretrained_artifact_contract.json"
    payload = json.loads(contract.read_text(encoding="utf-8"))
    return dict(payload["input_checksums"])


def hash_matches_manifest(args: argparse.Namespace, relative_path: str) -> bool:
    expected = args.critical_hashes.get(relative_path)
    return expected is not None and sha256_file(args.package / relative_path) == expected


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
