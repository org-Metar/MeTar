#!/usr/bin/env python3
"""Relocate absolute machine paths in the derived compatibility overlay."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


TEXT_EXTENSIONS = {".json", ".py", ".sh", ".toml", ".yaml", ".yml"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--copy-source-dir", nargs=2, metavar=("LEGACY_PATH", "DESTINATION"))
    return parser.parse_args()


def copy_source_directory(
    repository: Path, artifacts: Path, legacy: str, destination: Path
) -> None:
    """Overlay checkout sources using the basenames expected by old scripts."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    payload = json.loads((repository / "manifests/source_paths.json").read_text(encoding="utf-8"))
    mapping = payload.get("renames", payload)
    prefix = legacy.rstrip("/") + "/"
    sources = [(old, repository / new) for old, new in mapping.items() if old.startswith(prefix)]
    available = [(old, source) for old, source in sources if source.is_file()]
    archived = artifacts / legacy
    vendor = legacy == "code/runtime_dependencies/round359_metar"
    if archived.is_dir():
        # The public package now owns 'metar'; keep the vendor namespace isolated.
        ignore = shutil.ignore_patterns("metar") if vendor and available else None
        shutil.copytree(archived, destination, ignore=ignore)
    elif available:
        destination.mkdir(parents=True)
    else:
        raise FileNotFoundError(archived)
    for old, source in available:
        relative = old[len(prefix):]
        if vendor and relative.startswith("metar/"):
            relative = "metar_legacy/" + relative[len("metar/"):]
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def relocate_text(
    text: str,
    replacements: tuple[tuple[str, str], ...],
    protected_prefixes: tuple[str, ...],
) -> str:
    relocated = text
    protected_tokens: list[tuple[str, str]] = []
    for index, prefix in enumerate(protected_prefixes):
        token = f"__METAR_PROTECTED_PATH_{index}__"
        if token in relocated:
            raise RuntimeError(f"Reserved relocation token already present: {token}")
        pattern = re.compile(rf"{re.escape(prefix)}(?=$|[/\\])")
        relocated, count = pattern.subn(token, relocated)
        if count:
            protected_tokens.append((token, prefix))
    tokens: list[tuple[str, str]] = []
    token_index = 0
    for old, new in replacements:
        variants = {old, old.replace("\\", "\\\\")}
        for variant in sorted(variants, key=len, reverse=True):
            token = f"__METAR_RELOCATE_PATH_{token_index}__"
            token_index += 1
            pattern = re.compile(rf"(?<![A-Za-z0-9_./\\-]){re.escape(variant)}")
            relocated, count = pattern.subn(token, relocated)
            if count:
                tokens.append((token, new))
    for token, new in tokens:
        relocated = relocated.replace(token, new)
    for token, prefix in protected_tokens:
        relocated = relocated.replace(token, prefix)
    return relocated


def main() -> int:
    args = parse_args()
    root = args.package_root.resolve()
    if args.copy_source_dir:
        legacy, destination = args.copy_source_dir
        target = Path(destination).absolute()
        if not target.resolve().is_relative_to(root):
            raise ValueError("compatibility source destination escapes artifact root")
        copy_source_directory(args.repository.resolve(), root, legacy, target)
        return 0
    compat = root / "compat"
    if not compat.is_dir():
        raise FileNotFoundError(compat)
    target_root = (args.target_root or root).resolve()
    target_compat = target_root / "compat"
    target_root_text = target_root.as_posix()

    replacements_list: list[tuple[str, str]] = []
    current_prefixes = {
        str(root),
        root.as_posix(),
        str(args.package_root.absolute()),
        args.package_root.absolute().as_posix(),
    }
    for current_prefix in current_prefixes:
        if current_prefix not in {str(target_root), target_root_text}:
            replacements_list.append((current_prefix, target_root_text))
    replacements_list.extend(
        (
            (
                "/NFS2_home/NFS2_home_1/lwn",
                (target_compat / "NFS2_home" / "NFS2_home_1" / "lwn").as_posix(),
            ),
            ("/home/lwn", (target_compat / "home" / "lwn").as_posix()),
            (r"C:\Users\26328\Desktop\metar", target_root_text),
            ("C:/Users/26328/Desktop/metar", target_root_text),
        )
    )
    replacements = tuple(dict.fromkeys(replacements_list))
    protected_prefixes = (
        (target_compat / "NFS2_home" / "NFS2_home_1" / "lwn").as_posix(),
        (target_compat / "home" / "lwn").as_posix(),
        target_root_text,
    )

    changed = 0
    for path in compat.rglob("*"):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"Non-UTF8 compatibility text: {path}") from exc
        relocated = relocate_text(text, replacements, protected_prefixes)
        if relocated != text:
            path.write_text(relocated, encoding="utf-8")
            changed += 1

    print(f"COMPAT_TEXT_RELOCATED\t{changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
