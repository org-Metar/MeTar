#!/usr/bin/env python3
"""Run an archived MeTar script against the package-local compatibility tree."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def resolve_target(value: str) -> Path:
    target = Path(value)
    if not target.is_absolute():
        target = PACKAGE_ROOT / target
    target = target.absolute()
    resolved = target.resolve()
    try:
        resolved.relative_to(PACKAGE_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"Target is outside the package: {target}") from exc
    if not target.is_file():
        raise FileNotFoundError(target)
    return target


def rewrite_paths(source: str) -> str:
    replacements = (
        (
            "/NFS2_home/NFS2_home_1/lwn",
            (PACKAGE_ROOT / "compat" / "NFS2_home" / "NFS2_home_1" / "lwn").as_posix(),
        ),
        ("/home/lwn", (PACKAGE_ROOT / "compat" / "home" / "lwn").as_posix()),
        (r"C:\Users\26328\Desktop\metar", PACKAGE_ROOT.as_posix()),
        ("C:/Users/26328/Desktop/metar", PACKAGE_ROOT.as_posix()),
    )
    # Protect already-relocated overlay prefixes so repeated checks are idempotent.
    protected: dict[str, str] = {}
    rewritten = source
    for index, value in enumerate(
        (
            str(PACKAGE_ROOT / "compat" / "NFS2_home" / "NFS2_home_1" / "lwn"),
            (PACKAGE_ROOT / "compat" / "NFS2_home" / "NFS2_home_1" / "lwn").as_posix(),
            str(PACKAGE_ROOT / "compat" / "home" / "lwn"),
            (PACKAGE_ROOT / "compat" / "home" / "lwn").as_posix(),
            str(PACKAGE_ROOT),
            PACKAGE_ROOT.as_posix(),
        )
    ):
        token = f"__METAR_PACKAGE_PATH_{index}__"
        protected[token] = value
        rewritten = rewritten.replace(value, token)
    relocation_tokens: dict[str, str] = {}
    for index, (original, replacement) in enumerate(replacements):
        token = f"__METAR_REWRITE_PATH_{index}__"
        relocation_tokens[token] = replacement
        rewritten = rewritten.replace(original, token)
    for token, value in protected.items():
        rewritten = rewritten.replace(token, value)
    for token, value in relocation_tokens.items():
        rewritten = rewritten.replace(token, value)
    return rewritten


def main() -> int:
    global PACKAGE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("target", help="Archived script path, relative to the package root")
    parser.add_argument("--check-only", action="store_true", help="Only validate path rewriting and syntax")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    PACKAGE_ROOT = args.package_root.resolve()

    target = resolve_target(args.target)
    rewritten = rewrite_paths(target.read_text(encoding="utf-8-sig"))
    forbidden = (
        "/NFS2_home/NFS2_home_1/lwn",
        "/home/lwn",
        r"C:\Users\26328\Desktop\metar",
        "C:/Users/26328/Desktop/metar",
    )
    masked = rewritten
    for mapped_root in (
        str(PACKAGE_ROOT / "compat" / "NFS2_home" / "NFS2_home_1" / "lwn"),
        (PACKAGE_ROOT / "compat" / "NFS2_home" / "NFS2_home_1" / "lwn").as_posix(),
        str(PACKAGE_ROOT / "compat" / "home" / "lwn"),
        (PACKAGE_ROOT / "compat" / "home" / "lwn").as_posix(),
        str(PACKAGE_ROOT),
        PACKAGE_ROOT.as_posix(),
    ):
        masked = masked.replace(mapped_root, "<PACKAGE_PATH>")
    remaining = [prefix for prefix in forbidden if prefix in masked]
    if remaining:
        raise RuntimeError(f"Unrewritten absolute path prefixes in {target}: {remaining}")
    ast.parse(rewritten, filename=str(target))
    if args.check_only:
        print(f"PORTABLE_CHECK_OK\t{target.relative_to(PACKAGE_ROOT)}")
        return 0

    if str(target.parent) not in sys.path:
        sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *args.script_args]
    namespace = {
        "__file__": str(target),
        "__name__": "__main__",
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(rewritten, str(target), "exec"), namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
