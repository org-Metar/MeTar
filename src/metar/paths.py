"""Locate maintained sources while reading older, immutable data releases."""
from __future__ import annotations

import json
from pathlib import Path


def resolve_source(
    relative: str, repository: str | Path, artifacts: str | Path | None = None
) -> Path:
    """Prefer the checkout; accept original member names in older archives.

    This resolver applies to source files, not model weights or scientific data.
    Published data releases keep their original member names and checksums.
    """
    repo = Path(repository).resolve()
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a repository-relative source path: {relative}")
    catalog = repo / "manifests/source_paths.json"
    mapping = json.loads(catalog.read_text(encoding="utf-8")) if catalog.is_file() else {}
    current = repo / mapping.get(path.as_posix(), path.as_posix())
    if current.is_file():
        return current
    if artifacts is not None:
        archived = Path(artifacts).resolve() / path
        if archived.is_file():
            return archived
    raise FileNotFoundError(current)
