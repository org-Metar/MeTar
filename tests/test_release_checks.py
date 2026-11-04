from __future__ import annotations

import importlib.util
import tarfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_release", ROOT / "tools/check_release.py"
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_local_links_cover_html_references_encoded_paths_and_duplicate_anchors(
    tmp_path,
):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/with space.md").write_text(
        '# Title\n\n## Repeated title\n## Repeated title\n<a id="explicit"></a>\n',
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "[one](nested/with%20space.md#repeated-title-1)\n"
        '<a href="nested/with%20space.md#explicit">two</a>\n'
        '[three][ref]\n[ref]: <nested/with%20space.md#title> "A title"\n'
        "[ref]\n[ref][]\n"
        "```markdown\n[example](missing.md)\n```\n"
        "`[inline](missing.md)`\n[external](https://example.invalid/not-checked)\n",
        encoding="utf-8",
    )
    assert CHECK.check_markdown(tmp_path, ["README.md", "nested/with space.md"]) == []


def test_broken_file_heading_and_reference_links_are_rejected(tmp_path):
    (tmp_path / "target.md").write_text("# Present\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[missing](gone.md)\n[heading](target.md#absent)\n"
        '<img src="missing.svg">\n[reference][undefined]\n',
        encoding="utf-8",
    )
    errors = CHECK.check_markdown(tmp_path, ["README.md"])
    assert len(errors) == 4
    assert any("undefined reference" in error for error in errors)
    assert any("missing anchor" in error for error in errors)


def test_heading_anchor_keeps_inline_code_text(tmp_path):
    (tmp_path / "README.md").write_text(
        "# Use `metar describe`\n\n[command](#use-metar-describe)\n",
        encoding="utf-8",
    )
    assert CHECK.check_markdown(tmp_path, ["README.md"]) == []


def test_cache_and_internal_notes_rejected_but_scientific_audit_names_allowed():
    for name in [
        "training/__pycache__/model.pyc",
        ".venv/model.py",
        "task_plan.md",
        "docs/\u4e2d.md",
    ]:
        assert CHECK.forbidden_path(name)
    for name in [
        "training/leakage_audit.py",
        "figures/recall_preview.py",
        "data/plot_sources/audit.tsv",
    ]:
        assert CHECK.forbidden_path(name) is None


def test_source_cjk_content_is_rejected(tmp_path):
    (tmp_path / "README.md").write_text("# \u4e2d\u6587\n", encoding="utf-8")
    assert "CJK text" in CHECK.check_source(tmp_path, ["README.md"])[0]


def test_missing_distribution_pair_fails(tmp_path):
    errors, artifacts = CHECK.check_distributions(tmp_path, [], tmp_path)
    assert errors and not artifacts
    assert "one wheel and one sdist" in errors[0]


def test_archive_paths_and_cache_leakage_are_rejected():
    assert CHECK.forbidden_path("../escape.py")
    assert CHECK.forbidden_path("C:/escape.py")
    assert CHECK.forbidden_path("src/metar/__pycache__/cli.pyc")


def test_distribution_rejects_missing_runnable_source_and_ignored_cache(
    tmp_path, monkeypatch
):
    (tmp_path / "tool.py").write_text("print('source')\n", encoding="utf-8")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "sample.whl").touch()
    (dist / "sample.tar.gz").touch()
    monkeypatch.setattr(CHECK, "RELEASE_FILES", {"tool.py"})
    monkeypatch.setattr(
        CHECK,
        "archive_files",
        lambda path: (
            {
                "__pycache__/unexpected.pyc": b"cache",
                "PKG-INFO": b"Name: metar-reproducibility\n",
            }
            if path.name.endswith(".gz")
            else {}
        ),
    )
    errors, _ = CHECK.check_distributions(tmp_path, ["tool.py"], dist)
    assert any("missing release file: tool.py" in error for error in errors)
    assert any("cache or environment file" in error for error in errors)
    assert any("unexpected file" in error for error in errors)


def test_sdist_reader_preserves_files_without_extracting(tmp_path):
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        content = b"model = None\n"
        info = tarfile.TarInfo("source/training/model.py")
        info.size = len(content)
        archive.addfile(info, BytesIO(content))
    assert CHECK.archive_files(path) == {"training/model.py": content}
    assert not (tmp_path / "training").exists()
