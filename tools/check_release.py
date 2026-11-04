#!/usr/bin/env python3
"""Check local documentation, release contents and an installed wheel without assets.

Run from a Git checkout. Build first with ``uv build`` when using --dist-dir.
The optional wheel smoke reuses installed third-party dependencies, but installs
MeTar into a fresh environment outside the checkout and verifies its origin.
Only the standard library is required for the source and distribution checks.
"""

from __future__ import annotations

import argparse
import email
import html
import json
import os
import re
import shutil
import subprocess
import sysconfig
import tarfile
import tempfile
import venv
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

RELEASE_DIRS = {
    "src",
    "docs",
    "manifests",
    "examples",
    "scripts",
    "tools",
    "tests",
    "training",
    "analysis",
    "figures",
    "data",
    "preprocessing",
    "comparators",
    "vendor",
}
RELEASE_FILES = {
    ".github/workflows/ci.yml",
    "CITATION.cff",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "README.md",
    "uv.lock",
    "pyproject.toml",
    "MANIFEST.in",
}
CACHE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    ".git",
    "node_modules",
    ".idea",
    ".vscode",
}
INTERNAL_NOTES = {
    "task_plan.md",
    "working_notes.md",
    "session_summary.md",
    "readonly_progress.md",
    "codex_review.md",
    "claude_review.md",
}
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
TEXT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".cff", ".txt", ".sh"}


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(result.stdout.decode("utf-8").strip("\0").split("\0"))


def forbidden_path(name: str) -> str | None:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", name):
        return "unsafe archive path"
    if set(path.parts) & CACHE_DIRS or path.suffix in {".pyc", ".pyo"}:
        return "cache or environment file"
    if path.name in {".DS_Store", "Thumbs.db", ".coverage"}:
        return "local machine output"
    if path.name.lower() in INTERNAL_NOTES:
        return "internal process note"
    if CJK.search(name):
        return "CJK filename"
    return None


def check_source(root: Path, files: list[str]) -> list[str]:
    errors = []
    for name in files:
        reason = forbidden_path(name)
        if reason:
            errors.append(f"{name}: {reason}")
        path = root / name
        if not path.is_file():
            errors.append(f"{name}: missing tracked file")
            continue
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8-sig")
            if CJK.search(text):
                errors.append(f"{name}: CJK text in public source/documentation")
    return errors


def without_code(text: str, *, inline: bool = True) -> str:
    lines, fence = [], None
    for line in text.splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker and (fence is None or marker[1][0] == fence[0]):
            fence = marker[1] if fence is None else None
            lines.append("")
        else:
            lines.append("" if fence else line)
    text = re.sub(r"<!--[\s\S]*?-->", "", "\n".join(lines))
    return re.sub(r"(`+)([^`]*?)\1", "", text) if inline else text


class HtmlLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag, attrs) -> None:
        attrs = dict(attrs)
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.append(attrs[key])
        for key in ("id", "name"):
            if attrs.get(key):
                self.anchors.add(attrs[key])


def anchors(text: str) -> set[str]:
    parser = HtmlLinks()
    text = without_code(text, inline=False)
    parser.feed(text)
    found, counts = set(parser.anchors), {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        title = heading[1] if heading else None
        if (
            not title
            and index + 1 < len(lines)
            and re.fullmatch(r"\s*(?:=+|-+)\s*", lines[index + 1])
            and line.strip()
        ):
            title = line.strip()
        if title:
            title = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", title)
            title = html.unescape(re.sub(r"<[^>]*>", "", title)).lower()
            slug = "".join(c for c in title if c.isalnum() or c in "-_ ").replace(
                " ", "-"
            )
            count = counts.get(slug, 0)
            counts[slug] = count + 1
            found.add(f"{slug}-{count}" if count else slug)
    return found


def destination(text: str) -> str:
    text = text.strip()
    if text.startswith("<"):
        return text[1 : text.find(">")]
    return re.split(r"\s", text, maxsplit=1)[0]


def markdown_links(text: str) -> tuple[list[str], list[str]]:
    text = without_code(text)
    refs = {
        " ".join(match[1].lower().split()): destination(match[2])
        for match in re.finditer(r"^\s{0,3}\[([^\]]+)\]:\s*(.+)$", text, re.MULTILINE)
    }
    text = re.sub(r"^\s{0,3}\[[^\]]+\]:.*$", "", text, flags=re.MULTILINE)
    parser = HtmlLinks()
    parser.feed(text)
    links, errors = [*refs.values(), *parser.links], []
    for match in re.finditer(r"!?\[[^\]\n]*\]\(", text):
        start, pos, depth = match.end(), match.end(), 1
        while pos < len(text) and depth:
            if text[pos] == "\\":
                pos += 2
                continue
            depth += (text[pos] == "(") - (text[pos] == ")")
            pos += 1
        if depth == 0:
            links.append(destination(text[start : pos - 1]))
    for match in re.finditer(r"!?\[([^\]\n]+)\]\[([^\]\n]*)\]", text):
        label = " ".join((match[2] or match[1]).lower().split())
        if label not in refs:
            errors.append(f"undefined reference link: {label}")
        else:
            links.append(refs[label])
    for match in re.finditer(r"!?\[([^\]\n]+)\](?![\[(])", text):
        label = " ".join(match[1].lower().split())
        if label in refs:
            links.append(refs[label])
    return links, errors


def check_markdown(root: Path, files: list[str]) -> list[str]:
    errors, anchor_cache = [], {}
    for name in files:
        if not name.lower().endswith(".md"):
            continue
        source = root / name
        links, invalid = markdown_links(source.read_text(encoding="utf-8-sig"))
        errors.extend(f"{name}: {error}" for error in invalid)
        for link in links:
            link = html.unescape(re.sub(r"\\([() ])", r"\1", link))
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            target = (
                (
                    root / unquote(parsed.path).lstrip("/")
                    if parsed.path.startswith("/")
                    else source.parent / unquote(parsed.path)
                ).resolve()
                if parsed.path
                else source.resolve()
            )
            if not target.is_relative_to(root.resolve()):
                errors.append(f"{name}: link escapes repository: {link}")
            elif not target.exists():
                errors.append(f"{name}: missing local link: {link}")
            elif parsed.fragment and target.suffix.lower() in {".md", ".html"}:
                if target not in anchor_cache:
                    anchor_cache[target] = anchors(
                        target.read_text(encoding="utf-8-sig")
                    )
                if unquote(parsed.fragment) not in anchor_cache[target]:
                    errors.append(f"{name}: missing anchor: {link}")
    return errors


def archive_files(path: Path) -> dict[str, bytes]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return {
                name: archive.read(name)
                for name in archive.namelist()
                if not name.endswith("/")
            }
    with tarfile.open(path, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        if any(m.issym() or m.islnk() for m in archive.getmembers()):
            raise ValueError("sdist contains symbolic links")
        roots = {PurePosixPath(m.name).parts[0] for m in members}
        if len(roots) != 1:
            raise ValueError("sdist must contain one top-level directory")
        return {
            str(
                PurePosixPath(m.name).relative_to(next(iter(roots)))
            ): archive.extractfile(m).read()
            for m in members
        }


def check_distributions(
    root: Path, files: list[str], dist: Path
) -> tuple[list[str], list[dict]]:
    errors, reports = [], []
    wheels, sources = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        return ["dist directory must contain exactly one wheel and one sdist"], []
    expected = {
        name
        for name in files
        if name in RELEASE_FILES or PurePosixPath(name).parts[0] in RELEASE_DIRS
    }
    for artifact in [*wheels, *sources]:
        members = archive_files(artifact)
        for name in members:
            reason = forbidden_path(name)
            if reason:
                errors.append(f"{artifact.name}: {name}: {reason}")
        if artifact.suffix == ".whl":
            prefixes = {PurePosixPath(name).parts[0] for name in members}
            metadata_dirs = {p for p in prefixes if p.endswith(".dist-info")}
            if len(metadata_dirs) != 1 or prefixes - metadata_dirs != {"metar"}:
                errors.append(
                    f"{artifact.name}: wheel must contain only metar and one dist-info directory"
                )
            required = {
                name.removeprefix("src/"): name
                for name in expected
                if name.startswith("src/metar/")
            }
            for name in members:
                if name.startswith("metar/") and name not in required:
                    errors.append(
                        f"{artifact.name}: untracked or unexpected package file: {name}"
                    )
            metadata = next(iter(metadata_dirs), "missing.dist-info")
            for filename in ("METADATA", "WHEEL", "RECORD", "entry_points.txt"):
                if f"{metadata}/{filename}" not in members:
                    errors.append(f"{artifact.name}: missing {filename}")
            meta = email.message_from_bytes(members.get(f"{metadata}/METADATA", b""))
            if meta.get("Name") != "metar-reproducibility" or not meta.get("Version"):
                errors.append(f"{artifact.name}: invalid package name/version metadata")
            if meta.get("License-Expression") != "Apache-2.0":
                errors.append(f"{artifact.name}: missing Apache-2.0 license expression")
            entry = members.get(f"{metadata}/entry_points.txt", b"").decode()
            if not re.search(r"^metar\s*=\s*metar.cli:main\s*$", entry, re.MULTILINE):
                errors.append(f"{artifact.name}: missing metar console entry point")
            for name in files:
                if name in {"LICENSE", "THIRD_PARTY_NOTICES.md"} or (
                    name.startswith("comparators/") and name.endswith("/LICENSE")
                ):
                    key = f"{metadata}/licenses/{name}"
                    if members.get(key) != (root / name).read_bytes():
                        errors.append(
                            f"{artifact.name}: missing or changed license {name}"
                        )
        else:
            required = {name: name for name in expected}
            for name in members.keys() - expected:
                if name not in {"PKG-INFO", "setup.cfg"} and not name.startswith(
                    "src/metar_reproducibility.egg-info/"
                ):
                    errors.append(
                        f"{artifact.name}: untracked or unexpected file: {name}"
                    )
        for name, original in required.items():
            if name not in members:
                errors.append(f"{artifact.name}: missing release file: {name}")
            elif members[name] != (root / original).read_bytes():
                errors.append(f"{artifact.name}: content differs from source: {name}")
        reports.append({"artifact": artifact.name, "files": len(members)})
    return errors, reports


def wheel_smoke(root: Path, wheel: Path) -> list[dict]:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required for the wheel installation smoke")
    with tempfile.TemporaryDirectory(prefix="metar-wheel-") as directory:
        work = Path(directory).resolve()
        if work.is_relative_to(root.resolve()):
            raise RuntimeError(
                "wheel smoke temporary directory must be outside checkout"
            )
        env_root = work / "env"
        venv.EnvBuilder(with_pip=False).create(env_root)
        python = env_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        environment = {
            k: v
            for k, v in os.environ.items()
            if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
        }
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--offline",
                "--no-deps",
                str(wheel.resolve()),
            ],
            cwd=work,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        site = subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            cwd=work,
            env=environment,
            text=True,
        ).strip()
        dependencies = sorted(
            {sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}
        )
        (Path(site) / "metar_smoke_dependencies.pth").write_text(
            "\n".join(dependencies) + "\n", encoding="utf-8"
        )
        probe = "import pathlib,sys,metar,torch,numpy,pandas,yaml; from metar import cli,models,predict,training; p=pathlib.Path(metar.__file__).resolve(); assert p.is_relative_to(pathlib.Path(sys.prefix).resolve()), str(p); assert torch.version.cuda is None, torch.__version__; print('wheel import and CPU dependencies: OK')"
        commands = [[str(python), "-I", "-c", probe]]
        commands += [
            [str(python), "-I", "-m", "metar", *args]
            for args in (
                ["--help"],
                ["describe"],
                ["predict", "--help"],
                ["train", "--help"],
            )
        ]
        executable = env_root / (
            "Scripts/metar.exe" if os.name == "nt" else "bin/metar"
        )
        commands.append([str(executable), "describe"])
        results = []
        for command in commands:
            result = subprocess.run(
                command,
                cwd=work,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            results.append(
                {
                    "command": command[2:]
                    if command[0] == str(python)
                    else ["metar", "describe"],
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
        return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--dist-dir", type=Path, help="Check a previously built wheel and sdist"
    )
    parser.add_argument(
        "--wheel-smoke",
        action="store_true",
        help="Install the checked wheel outside the checkout, reusing third-party dependencies",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write a JSON result (CI artifact, not a source file)",
    )
    args = parser.parse_args(argv)
    report = {"status": "fail", "errors": [], "external_links_checked": False}
    try:
        root = args.root.resolve()
        files = tracked_files(root)
        report["errors"] = check_source(root, files) + check_markdown(root, files)
        report["tracked_files_checked"] = len(files)
        if args.dist_dir:
            errors, artifacts = check_distributions(
                root, files, args.dist_dir.resolve()
            )
            report["errors"].extend(errors)
            report["distributions"] = artifacts
        if args.wheel_smoke:
            if not args.dist_dir:
                raise ValueError("--wheel-smoke requires --dist-dir")
            if not report["errors"]:
                report["wheel_smoke"] = wheel_smoke(
                    root, next(args.dist_dir.glob("*.whl"))
                )
        report["status"] = "pass" if not report["errors"] else "fail"
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as error:
        report["errors"].append(str(error))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
