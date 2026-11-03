from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    path = REPOSITORY_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HYDRATE = load_script("hydrate_public_inputs", "scripts/import_data.py")


class PublicArtifactLayoutTest(unittest.TestCase):
    def test_safe_path_rejects_parent_traversal(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                HYDRATE.safe_path(Path(directory), "../outside.bin")

    def test_copy_materialization_preserves_content_hash(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "nested/destination.bin"
            source.write_bytes(b"article-exact")

            HYDRATE.materialize(source, destination, "copy")

            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(HYDRATE.sha256_file(destination), expected)

    def test_contract_has_disjoint_portable_paths(self) -> None:
        contract = json.loads(
            (REPOSITORY_ROOT / "manifests/pretrained_artifact_contract.json")
            .read_text(encoding="utf-8")
        )
        first_party = set(contract["first_party_required_files"])
        external = {item["path"] for item in contract["external_inputs"]}

        self.assertEqual(contract["artifact_root"], "metar_artifacts")
        self.assertFalse(first_party & external)
        self.assertTrue(all(HYDRATE.safe_path(Path("/artifacts"), path) for path in first_party))
        self.assertTrue(all(HYDRATE.safe_path(Path("/artifacts"), path) for path in external))


if __name__ == "__main__":
    unittest.main()
