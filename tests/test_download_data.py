from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import URLError


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_data.py"
SPEC = importlib.util.spec_from_file_location("download_data", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {SCRIPT}")
DOWNLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD)


class DownloadDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = b"model archive fixture"
        self.entry = {
            "name": "model.tar.gz",
            "bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }
        self.catalog = {
            "repository": "org-MeTar/MeTar",
            "tag": "artifacts-test",
            "visibility": "public",
        }

    def test_public_download_uses_anonymous_https_without_github_cli(self) -> None:
        with TemporaryDirectory() as directory, patch.object(
            DOWNLOAD, "github_cli", side_effect=AssertionError("gh must not be used")
        ), patch.object(
            DOWNLOAD, "urlopen", return_value=io.BytesIO(self.payload)
        ) as request:
            DOWNLOAD.download(self.catalog, [self.entry], Path(directory), None)
            self.assertEqual((Path(directory) / self.entry["name"]).read_bytes(), self.payload)
            self.assertEqual(list(Path(directory).glob("*.part")), [])
            request_data = request.call_args.args[0]
            self.assertEqual(
                request_data.full_url,
                "https://github.com/org-MeTar/MeTar/releases/download/"
                "artifacts-test/model.tar.gz",
            )
            self.assertFalse(request_data.has_header("Authorization"))

    def test_existing_files_are_skipped_then_checked_by_verifier(self) -> None:
        with TemporaryDirectory() as directory, patch.object(DOWNLOAD, "urlopen") as request:
            destination = Path(directory) / self.entry["name"]
            destination.write_bytes(self.payload)
            DOWNLOAD.download(self.catalog, [self.entry], Path(directory), None)
            request.assert_not_called()
            DOWNLOAD.verify(destination, self.entry)
            destination.write_bytes(b"x" * len(self.payload))
            DOWNLOAD.download(self.catalog, [self.entry], Path(directory), None)
            request.assert_not_called()
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                DOWNLOAD.verify(destination, self.entry)

    def test_corrupt_download_is_rejected_without_a_final_or_partial_file(self) -> None:
        corrupt = b"x" * len(self.payload)
        with TemporaryDirectory() as directory, patch.object(
            DOWNLOAD, "urlopen", return_value=io.BytesIO(corrupt)
        ):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                DOWNLOAD.download(self.catalog, [self.entry], Path(directory), None)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_network_failure_cleans_the_temporary_file(self) -> None:
        with TemporaryDirectory() as directory, patch.object(
            DOWNLOAD, "urlopen", side_effect=URLError("offline")
        ):
            with self.assertRaises(URLError):
                DOWNLOAD.download(self.catalog, [self.entry], Path(directory), None)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_private_release_retains_github_cli_skip_existing(self) -> None:
        catalog = {**self.catalog, "visibility": "private"}
        with TemporaryDirectory() as directory, patch.object(
            DOWNLOAD, "github_cli", return_value="gh"
        ), patch.object(DOWNLOAD.subprocess, "run") as run, patch.object(
            DOWNLOAD, "urlopen", side_effect=AssertionError("anonymous request forbidden")
        ):
            DOWNLOAD.download(catalog, [self.entry], Path(directory), None)
            command = run.call_args.args[0]
            self.assertEqual(command[:3], ["gh", "release", "download"])
            self.assertIn("--skip-existing", command)
            self.assertEqual(command[-2:], ["--pattern", self.entry["name"]])

    def test_explicit_account_uses_its_credential_even_for_a_public_release(self) -> None:
        token_result = subprocess.CompletedProcess([], 0, stdout="fixture-token\n")
        download_result = subprocess.CompletedProcess([], 0)
        with TemporaryDirectory() as directory, patch.object(
            DOWNLOAD, "github_cli", return_value="gh"
        ), patch.object(
            DOWNLOAD.subprocess, "run", side_effect=[token_result, download_result]
        ) as run, patch.object(DOWNLOAD, "urlopen") as request, patch.dict(
            os.environ, {"GH_TOKEN": "previous-token"}
        ):
            DOWNLOAD.download(self.catalog, [self.entry], Path(directory), "researcher")
            request.assert_not_called()
            self.assertEqual(
                run.call_args_list[0].args[0],
                ["gh", "auth", "token", "--user", "researcher"],
            )
            self.assertEqual(run.call_args_list[1].kwargs["env"]["GH_TOKEN"], "fixture-token")
            self.assertEqual(os.environ["GH_TOKEN"], "previous-token")


if __name__ == "__main__":
    unittest.main()
