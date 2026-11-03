from pathlib import Path

import numpy as np
import pandas as pd

from metar.cli import DEFAULTS, main
from metar.paths import resolve_source


def test_doctor_reports_missing_models(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(DEFAULTS, "closed_checkpoint", tmp_path / "missing.pth")
    assert main(["doctor", "--scenario", "bulk-668"]) == 1
    assert '"model_loaded": false' in capsys.readouterr().out


def test_doctor_checks_input_without_loading_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not-loaded-by-doctor")
    monkeypatch.setitem(DEFAULTS, "closed_checkpoint", checkpoint)
    monkeypatch.setitem(DEFAULTS, "feature_order", tmp_path / "no-schema.txt")
    features = tmp_path / "query.csv"
    pd.DataFrame(np.zeros((1, 978)), columns=[f"g{i}" for i in range(978)]).to_csv(features, index=False)
    assert main(["doctor", "--scenario", "bulk-668", "--cell-line", "PC3", "--input", str(features)]) == 0
    pd.DataFrame({"g1": [0.0]}).to_csv(features, index=False)
    assert main(["doctor", "--scenario", "bulk-668", "--input", str(features)]) == 1

    monkeypatch.setitem(DEFAULTS, "adapter_checkpoint", checkpoint)
    matrix = pd.DataFrame(np.zeros((1, 978)), columns=[f"g{i}" for i in range(978)])
    matrix.insert(0, "query_id", "query")
    matrix.to_csv(features, index=False)
    matrix["query_id"] = "reference"
    support = tmp_path / "support.csv"
    matrix.to_csv(support, index=False)
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"query_id": ["reference"], "targets": [""]}).to_csv(labels, index=False)
    assert main(["doctor", "--scenario", "cross-cell", "--input", str(features),
                 "--support-input", str(support), "--support-labels", str(labels)]) == 1


def test_source_resolver_prefers_checkout_and_reads_older_archive(tmp_path):
    import json

    repo, assets = tmp_path / "repo", tmp_path / "assets"
    (repo / "manifests").mkdir(parents=True)
    (repo / "manifests/source_paths.json").write_text(json.dumps({"old.py": "training/model.py"}))
    current = repo / "training/model.py"
    current.parent.mkdir()
    current.write_text("current")
    assets.mkdir()
    archived = assets / "old.py"
    archived.write_text("archived")
    assert resolve_source("old.py", repo, assets) == current
    current.unlink()
    assert resolve_source("old.py", repo, assets) == archived
