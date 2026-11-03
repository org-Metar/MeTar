from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_archived_script.py"
)
SPEC = importlib.util.spec_from_file_location("run_archived_script", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RewritePathsTest(unittest.TestCase):
    def test_runner_checks_an_external_artifact_root_without_executing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "entry.py").write_text("raise AssertionError('must not execute')\n")
            completed = subprocess.run(
                [sys.executable, str(MODULE_PATH), "--package-root", str(root),
                 "--check-only", "entry.py"],
                check=True, capture_output=True, text=True,
            )
            self.assertIn("PORTABLE_CHECK_OK", completed.stdout)

    def test_rewrites_forward_slash_windows_root(self) -> None:
        source = 'ROOT = "C:/Users/26328/Desktop/metar/data"\n'
        rewritten = MODULE.rewrite_paths(source)

        self.assertEqual(rewritten, f'ROOT = "{MODULE.PACKAGE_ROOT.as_posix()}/data"\n')
        ast.parse(rewritten)

    def test_rewrites_backslash_windows_root(self) -> None:
        source = r'ROOT = r"C:\Users\26328\Desktop\metar\data"' + "\n"
        rewritten = MODULE.rewrite_paths(source)

        self.assertEqual(rewritten, f'ROOT = r"{MODULE.PACKAGE_ROOT.as_posix()}\\data"\n')
        ast.parse(rewritten)


if __name__ == "__main__":
    unittest.main()
