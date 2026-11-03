"""Shared helpers for the Round429 LINCS layer sensitivity overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def overlay_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def add_code_release_to_path(code_release: str | Path) -> Path:
    root = Path(code_release).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Round359 code release root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def first_existing_path(values: list[str | Path]) -> Path:
    paths = [Path(os.path.expandvars(str(v))).resolve() for v in values if v]
    for path in paths:
        if path.exists():
            return path
    if not paths:
        raise ValueError("No candidate paths were provided")
    return paths[0]


def code_release_root_from_config(cfg: dict[str, Any]) -> Path:
    repository = os.environ.get("METAR_REPO_ROOT")
    candidates = [Path(repository)] if repository else []
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        for vendor in (
            candidate / "vendor" / "metar_legacy",
            candidate / "claude_round359_code_availability_package_20260608" / "metar-code-release",
        ):
            if (vendor / "metar_legacy" / "__init__.py").is_file():
                return vendor
    release = cfg["code_release"]
    return first_existing_path([release.get("local_root", ""), release.get("server_root", "")])


def companion_root_from_config(cfg: dict[str, Any]) -> Path:
    companion = cfg["companion_data"]
    return first_existing_path([companion.get("server_root", ""), companion.get("local_audit_root", "")])


def read_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML config: {path}")
    return data


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    text = os.path.expandvars(str(value))
    path = Path(text)
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        return pd.read_csv(p, sep="\t")
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(p)
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported table suffix: {p}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        df.to_csv(p, sep="\t", index=False)
    elif suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        df.to_csv(p, index=False)
    elif suffixes.endswith(".parquet"):
        df.to_parquet(p, index=False)
    else:
        raise ValueError(f"Unsupported table suffix: {p}")


def first_existing_column(columns: list[str], candidates: list[str], *, required: bool = True) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    if required:
        raise ValueError(f"None of the candidate columns are present: {candidates}")
    return None


def parse_args_with_config(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Round429 YAML config")
    return parser.parse_args()
