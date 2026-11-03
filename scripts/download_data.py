#!/usr/bin/env python3
"""List, download, verify and safely extract the MeTar first-party artifacts.

This is a standard-library utility. It never imports torch or loads a model.
Public assets use anonymous HTTPS. Private downloads use a GitHub CLI login.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "manifests" / "artifact_downloads.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, entry: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact: {path}")
    if path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Size mismatch: {path.name}")
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path.name}")


def extract(archive: Path, package: Path) -> int:
    """Extract only verified regular files, refusing conflicting existing data."""
    package = package.resolve()
    package.mkdir(parents=True, exist_ok=True)
    copied = 0
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.name
                or not relative.parts
                or relative.parts[0] != "metar_artifacts"
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
            destination = package.joinpath(*relative.parts[1:])
            if not destination.resolve().is_relative_to(package):
                raise ValueError(f"Archive path escapes package: {member.name}")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            payload = source.extractfile(member)
            if payload is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            with payload:
                if destination.exists():
                    digest = hashlib.sha256()
                    for block in iter(lambda: payload.read(1024 * 1024), b""):
                        digest.update(block)
                    if (
                        not destination.is_file()
                        or destination.stat().st_size != member.size
                        or sha256(destination) != digest.hexdigest()
                    ):
                        raise FileExistsError(
                            f"Conflicting existing artifact (not overwritten): {destination}"
                        )
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as target:
                    shutil.copyfileobj(payload, target, length=1024 * 1024)
                copied += 1
    return copied


def github_cli() -> str:
    executable = shutil.which("gh")
    if executable:
        return executable
    candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "GitHub CLI/gh.exe"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError("GitHub CLI (gh) is required for private release downloads.")


def download_public(catalog: dict, entries: list[dict], directory: Path) -> None:
    """Stream public assets into temporary files before verifying and publishing."""
    tag = quote(catalog["tag"], safe="")
    base_url = f"https://github.com/{catalog['repository']}/releases/download/{tag}"
    for entry in entries:
        destination = directory / entry["name"]
        if destination.exists():
            continue
        request = Request(
            f"{base_url}/{quote(entry['name'], safe='')}",
            headers={"User-Agent": "MeTar-data-downloader"},
        )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=f".{entry['name']}.", suffix=".part", delete=False
            ) as target:
                temporary = Path(target.name)
                with urlopen(request, timeout=60) as response:
                    shutil.copyfileobj(response, target, length=1024 * 1024)
            verify(temporary, entry)
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def download(catalog: dict, entries: list[dict], directory: Path, user: str | None) -> None:
    if catalog.get("visibility") == "public" and not user:
        download_public(catalog, entries, directory)
        return
    executable = github_cli()
    environment = os.environ.copy()
    if user:
        token = subprocess.run(
            [executable, "auth", "token", "--user", user],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if not token:
            raise ValueError(f"No saved GitHub authentication for {user}")
        environment["GH_TOKEN"] = token
    command = [
        executable, "release", "download", catalog["tag"],
        "--repo", catalog["repository"], "--dir", str(directory), "--skip-existing",
    ]
    for entry in entries:
        command.extend(["--pattern", entry["name"]])
    subprocess.run(command, env=environment, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list assets without network access")
    parser.add_argument("--archive", action="append", help="select an archive (repeatable)")
    parser.add_argument("--directory", type=Path, default=Path("downloads/metar"))
    parser.add_argument("--download", action="store_true", help="download selected release archives")
    parser.add_argument("--github-user", help="use a saved gh login without changing the default")
    parser.add_argument("--package", type=Path, help="extract verified archives into this artifact root")
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    selected = set(args.archive or [])
    entries = [item for item in catalog["archives"] if not selected or item["name"] in selected]
    unknown = selected - {item["name"] for item in entries}
    if unknown:
        parser.error(f"Unknown archive(s): {', '.join(sorted(unknown))}")
    if args.list:
        for entry in entries:
            print(f"{entry['name']}\t{entry['bytes']} bytes\t{entry['purpose']}")
        return 0
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.download:
        download(catalog, entries, args.directory, args.github_user)
    for entry in entries:
        archive = args.directory / entry["name"]
        verify(archive, entry)
        print(f"PASS SHA-256: {entry['name']}")
        if args.package:
            count = extract(archive, args.package)
            print(f"Extracted {count} new files into {args.package}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError) as error:
        raise SystemExit(str(error)) from error
