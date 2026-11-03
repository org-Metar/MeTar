from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "relocate_paths.py"
)
SPEC = importlib.util.spec_from_file_location("relocate_paths", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RelocateTextTest(unittest.TestCase):
    def test_canonical_sources_keep_legacy_basenames_and_archived_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, artifacts, destination = root / "repo", root / "artifacts", root / "copy"
            (repo / "manifests").mkdir(parents=True)
            (repo / "analysis").mkdir()
            (repo / "analysis/train.py").write_text("maintained source")
            (repo / "manifests/source_paths.json").write_text(json.dumps({
                "old/scripts/run_round1.py": "analysis/train.py",
            }))
            archived = artifacts / "old/scripts"
            archived.mkdir(parents=True)
            (archived / "run_round1.py").write_text("outdated source")
            (archived / "locked_data.tsv").write_text("immutable data")
            MODULE.copy_source_directory(repo, artifacts, "old/scripts", destination)
            self.assertEqual((destination / "run_round1.py").read_text(), "maintained source")
            self.assertEqual((destination / "locked_data.tsv").read_text(), "immutable data")
            self.assertFalse((destination / "train.py").exists())

    def test_vendor_overlay_uses_isolated_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, artifacts, destination = root / "repo", root / "artifacts", root / "copy"
            (repo / "manifests").mkdir(parents=True)
            (repo / "vendor/metar_legacy/metar_legacy").mkdir(parents=True)
            (repo / "vendor/metar_legacy/metar_legacy/__init__.py").write_text("")
            legacy = "code/runtime_dependencies/round359_metar"
            (repo / "manifests/source_paths.json").write_text(json.dumps({
                legacy + "/metar/__init__.py": "vendor/metar_legacy/metar_legacy/__init__.py",
            }))
            archived = artifacts / legacy / "metar"
            archived.mkdir(parents=True)
            (archived / "__init__.py").write_text("old public namespace")
            MODULE.copy_source_directory(repo, artifacts, legacy, destination)
            self.assertTrue((destination / "metar_legacy/__init__.py").is_file())
            self.assertFalse((destination / "metar").exists())

    def test_posix_replacement_remains_valid_python_on_windows(self) -> None:
        source = 'ROOT = Path("/home/lwn/project")\n'
        relocated = MODULE.relocate_text(
            source,
            (("/home/lwn", "C:/checkout/compat/home/lwn"),),
            (),
        )

        self.assertEqual(
            relocated,
            'ROOT = Path("C:/checkout/compat/home/lwn/project")\n',
        )
        ast.parse(relocated)


if __name__ == "__main__":
    unittest.main()
