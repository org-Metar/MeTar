"""Small synthetic checks; no released component arrays or real matrix values."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_single_cell.py"
SPEC = importlib.util.spec_from_file_location("prepare_tahoe_x128", SCRIPT)
tahoe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tahoe)


class TahoePreprocessingTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py is an optional preprocessing dependency")
    def test_rejects_dense_input_before_matrix_values_are_read(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dense.h5ad"
            with h5py.File(path, "w") as handle:
                var = handle.create_group("var")
                var.create_dataset("_index", data=np.array(["g0", "g1"], dtype="S2"))
                handle.create_dataset("X", shape=(2, 2), dtype="float32")
            with (
                patch.object(tahoe, "validate_gene_order"),
                self.assertRaisesRegex(ValueError, "CSR sparse encoding"),
            ):
                tahoe.inspect_plate(path)

    @unittest.skipUnless(
        importlib.util.find_spec("scipy") and importlib.util.find_spec("sklearn"),
        "SciPy and scikit-learn are optional verification dependencies",
    )
    def test_sparse_dot_matches_reloaded_sklearn_transform(self) -> None:
        from scipy import sparse
        from sklearn.decomposition import TruncatedSVD

        raw = sparse.csr_matrix([[0, 4, 1, 0], [3, 0, 2, 8]], dtype=np.int32)
        saved = np.array([[0.5, 0.1, -0.2, 0.9], [-0.3, 0.2, 0.7, 0.4]], dtype=np.float32)
        model = TruncatedSVD(n_components=2)
        model.components_ = saved.astype(np.float64)
        model.n_features_in_ = 4
        expected = model.transform(raw).astype(np.float32)
        np.testing.assert_array_equal(tahoe.project_block(raw, model.components_), expected)

    def test_rejects_reordered_gene_metadata(self) -> None:
        names = [f"gene_{i}" for i in range(62710)]
        digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
        with patch.object(tahoe, "ORDERED_VAR_SHA256", digest):
            tahoe.validate_gene_order(names)
            with self.assertRaisesRegex(ValueError, "refusing to guess"):
                tahoe.validate_gene_order(names[::-1])

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py is an optional preprocessing dependency")
    def test_check_only_reads_headers_and_var_metadata_without_loading_state(self) -> None:
        import h5py

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "assets"
            assets.mkdir()
            raw = root / "raw"
            raw.mkdir()
            np.savez_compressed(
                assets / "tsvd_model.npz",
                components_=np.zeros((128, 6000), dtype=np.float32),
                hvg_names=np.array([f"g{i}" for i in range(6000)], dtype="U15"),
            )
            np.savez_compressed(assets / "hvg_selection.npz", hvg_names=np.array(["g"], dtype=object))
            hashes = {name: tahoe.sha256_file(assets / name) for name in tahoe.ASSET_SHA256}
            names = [f"gene_{i}" for i in range(62710)]
            digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
            with h5py.File(raw / "plate.h5ad", "w") as handle:
                var = handle.create_group("var")
                var.attrs["_index"] = "gene_name"
                var.create_dataset("gene_name", data=np.array(names, dtype="S16"))
                # No expression values are allocated. All CSR datasets are empty
                # except indptr; check-only must never materialize the matrix.
                matrix = handle.create_group("X")
                matrix.attrs["encoding-type"] = "csr_matrix"
                matrix.attrs["shape"] = [5, 62710]
                matrix.create_dataset("data", shape=(0,), dtype="int32")
                matrix.create_dataset("indices", shape=(0,), dtype="int32")
                matrix.create_dataset("indptr", data=np.zeros(6, dtype="int32"))
            output = root / "uncreated-output"
            with (
                patch.object(tahoe, "ASSET_SHA256", hashes),
                patch.object(tahoe, "ORDERED_VAR_SHA256", digest),
                patch.object(tahoe, "transform_plates", side_effect=AssertionError("transform forbidden")),
                patch.object(np, "load", side_effect=AssertionError("state loading forbidden")),
                contextlib.redirect_stdout(io.StringIO()) as captured,
            ):
                result = tahoe.main([
                    "--input-dir", str(raw), "--assets", str(assets),
                    "--output-dir", str(output), "--check-only",
                ])
            self.assertEqual(result, 0)
            self.assertIn('"matrix_values_read": false', captured.getvalue())
            self.assertFalse(output.exists())
            (assets / "tsvd_model.npz").write_bytes(b"corrupted")
            with (
                patch.object(tahoe, "ASSET_SHA256", hashes),
                self.assertRaisesRegex(ValueError, "SHA-256 mismatch"),
            ):
                tahoe.validate_assets(assets)


if __name__ == "__main__":
    unittest.main()
