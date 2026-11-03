#!/usr/bin/env python3
"""Reproduce the manuscript-used PDF-3 log2FC query in the 668-target space."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


EXPECTED = {
    "checkpoint": "fa33089913be2693c465c2c142fb506054a477e1f35279999c186db432b1fb8f",
    "jhh7": "fe47451a35a12c7f878295cd1ff333a673429b6ce8d663cd62100c3728047e24",
    "geneinfo": "e739d06bad42ff9285c00b778e65d8999425062a3375cc7e7cdab0e7154490b5",
    "target_list": "5ffdb3addf5a5c2ffc35fc5933d06283eaae8bcc37a6890bb424de4731b021ca",
    "aligned": "2b0adee600d1d33f33cb942460535ec10bf979b75df8bc98aa26901e3aff4b3e",
    "rankings": "099337ebbd63ec31318dae640adc8964c28a5c5ab19e0ca3e27dc7b189a96cef",
    "top50": "506b661c30bbab19d77419380bea45f8fa10ec240521f028578e28df5f0db59c",
}
TREAT_COLS = ["FPKM.JHH7_PDF_3_1", "FPKM.JHH7_PDF_3_2"]
CTRL_COLS = ["FPKM.JHH7_NC_1", "FPKM.JHH7_NC_2"]
SCORE_ATOL = 1.0e-6
SCORE_RTOL = 0.0
PERCENTILE_ATOL = 2.0e-15


class DiagReadout(nn.Module):
    def __init__(self, target_anchor: torch.Tensor, readout_cfg: dict[str, Any]) -> None:
        super().__init__()
        dim = int(target_anchor.shape[1])
        self.register_buffer("target_anchor", F.normalize(target_anchor.detach().float(), dim=1))
        hidden = int(readout_cfg.get("hidden", 256))
        dropout = float(readout_cfg.get("dropout", 0.35))
        layers: list[nn.Module] = [
            nn.Linear(978, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(0, int(readout_cfg.get("extra_layers", 0)))):
            layers.extend(
                [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            )
        layers.extend([nn.Linear(hidden, dim), nn.LayerNorm(dim)])
        self.query_encoder = nn.Sequential(*layers)
        self.query_norm = nn.LayerNorm(dim)
        self.diag = nn.Parameter(torch.ones(dim))
        self.target_delta = nn.Parameter(torch.zeros_like(self.target_anchor))
        self.score_scale = nn.Parameter(torch.tensor(0.7))
        self.target_bias = nn.Parameter(torch.zeros(int(target_anchor.shape[0])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = F.normalize(self.query_norm(self.query_encoder(x)), dim=1)
        target = F.normalize(self.target_anchor + self.target_delta, dim=1)
        score = (query * self.diag[None, :]) @ target.T
        return score * torch.exp(torch.clamp(self.score_scale, -2.0, 2.0)) + self.target_bias[None, :]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected={expected} actual={actual}")


def read_target_labels(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t")
    if len(frame) != 668:
        raise RuntimeError(f"Expected 668 target rows, observed {len(frame)}")
    for column in ("gene", "target", "gene_symbol", "label", "target_label", "target_entity", "entity"):
        if column in frame.columns:
            labels = frame[column].astype(str).tolist()
            break
    else:
        labels = frame.iloc[:, -1].astype(str).tolist()
    labels = [value.split(":", 1)[1] if value.startswith(("Protein:", "TF:")) else value for value in labels]
    if len(set(labels)) != 668:
        raise RuntimeError("Target labels are not unique after normalization")
    return labels


def build_aligned_query(jhh7_path: Path, geneinfo_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    geneinfo = pd.read_table(geneinfo_path)
    landmarks = geneinfo[(geneinfo["feature_space"] == "landmark") & geneinfo["ensembl_id"].notna()].copy()
    landmarks = landmarks.drop_duplicates("ensembl_id", keep="first")
    if len(landmarks) != 978:
        raise RuntimeError(f"Expected 978 landmarks, observed {len(landmarks)}")
    landmarks = landmarks[["ensembl_id", "gene_symbol"]].reset_index(drop=True)

    raw = pd.read_csv(jhh7_path)
    required = ["gene_id", "gene_name", *TREAT_COLS, *CTRL_COLS]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RuntimeError(f"Missing JHH7 columns: {missing}")
    raw = raw[required].copy()
    for column in [*TREAT_COLS, *CTRL_COLS]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    expression = raw.groupby("gene_id", sort=False)[[*TREAT_COLS, *CTRL_COLS]].mean()
    names = raw.groupby("gene_id", sort=False)["gene_name"].first().to_frame()
    frame = landmarks.set_index("ensembl_id").join(expression, how="left").join(names, how="left")
    frame["was_missing_in_source"] = frame[TREAT_COLS[0]].isna()
    for column in [*TREAT_COLS, *CTRL_COLS]:
        frame[column] = frame[column].fillna(float(np.nanmedian(frame[column].to_numpy(dtype=float))))

    treated = frame[TREAT_COLS].to_numpy(dtype=float)
    control = frame[CTRL_COLS].to_numpy(dtype=float)
    frame["treated_mean_fpkm"] = np.mean(treated, axis=1)
    frame["control_mean_fpkm"] = np.mean(control, axis=1)
    frame["raw_contrast_fpkm"] = frame["treated_mean_fpkm"] - frame["control_mean_fpkm"]
    frame["log2fc_fpkm_pseudocount1"] = np.log2(
        (frame["treated_mean_fpkm"] + 1.0) / (frame["control_mean_fpkm"] + 1.0)
    )
    control_sd = np.std(control, axis=1, ddof=1)
    frame["replicate_z_no_noise_floor"] = frame["raw_contrast_fpkm"] / np.clip(control_sd, 1.0e-6, None)
    frame["replicate_z_no_noise_floor"] = frame["replicate_z_no_noise_floor"].replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    aligned = frame.reset_index().rename(columns={"index": "ensembl_id"})
    query = np.nan_to_num(
        aligned["log2fc_fpkm_pseudocount1"].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )[None, :]
    if query.shape != (1, 978):
        raise RuntimeError(f"Expected query shape (1, 978), observed {query.shape}")
    return aligned, query


def score_query(query: np.ndarray, checkpoint: Path, target_list: Path) -> pd.DataFrame:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    labels = [str(value).split(":", 1)[-1] for value in payload["labels"]]
    if labels != read_target_labels(target_list):
        raise RuntimeError("Checkpoint labels do not match the locked 668-target list")
    state = payload["readout_state_dict"]
    model = DiagReadout(state["target_anchor"].detach().float(), dict(payload["readout_cfg"]))
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        scores = model(torch.tensor(query, dtype=torch.float32)).detach().cpu().numpy()[0]
    frame = pd.DataFrame(
        {"space": "round246_round252_seen668_direct", "target": labels, "score": scores.astype(float)}
    )
    frame["rank"] = frame["score"].rank(method="first", ascending=False).astype(int)
    frame["percentile"] = frame["rank"].map(lambda rank: 1.0 - (float(rank) - 1.0) / len(frame))
    return frame.sort_values("rank").reset_index(drop=True)


def compare_rankings(
    computed: pd.DataFrame, canonical: pd.DataFrame
) -> dict[str, float | int | str]:
    if list(computed.columns) != list(canonical.columns) or len(computed) != len(canonical):
        raise RuntimeError("PDF-3 ranking schema or row count differs from the locked table")
    for column in ("space", "target"):
        if not np.array_equal(computed[column].astype(str), canonical[column].astype(str)):
            raise RuntimeError(f"PDF-3 ranking {column} order differs from the locked table")
    if not np.array_equal(
        computed["rank"].to_numpy(dtype=np.int64),
        canonical["rank"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("PDF-3 target ranks differ from the locked table")

    computed_scores = computed["score"].to_numpy(dtype=np.float64)
    canonical_scores = canonical["score"].to_numpy(dtype=np.float64)
    computed_percentiles = computed["percentile"].to_numpy(dtype=np.float64)
    canonical_percentiles = canonical["percentile"].to_numpy(dtype=np.float64)
    if not np.allclose(
        computed_scores,
        canonical_scores,
        rtol=SCORE_RTOL,
        atol=SCORE_ATOL,
        equal_nan=True,
    ):
        raise RuntimeError("PDF-3 scores exceed the locked cross-platform tolerance")
    if not np.allclose(
        computed_percentiles,
        canonical_percentiles,
        rtol=0.0,
        atol=PERCENTILE_ATOL,
        equal_nan=True,
    ):
        raise RuntimeError("PDF-3 percentiles exceed the locked serialization tolerance")
    return {
        "rows": len(computed),
        "computed_lf_tsv_sha256": hashlib.sha256(
            computed.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8")
        ).hexdigest(),
        "score_max_abs_delta": float(np.max(np.abs(computed_scores - canonical_scores))),
        "score_mean_abs_delta": float(np.mean(np.abs(computed_scores - canonical_scores))),
        "percentile_max_abs_delta": float(
            np.max(np.abs(computed_percentiles - canonical_percentiles))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.package_root).resolve()
    output = Path(args.output_dir).resolve()
    output.relative_to(root)
    inputs = root / "data/application_inputs/pdf3"
    checkpoint = root / "checkpoints/selected/round246_whole_model_checkpoint.pth"
    jhh7 = inputs / "JHH7.csv"
    geneinfo = inputs / "geneinfo_beta.txt"
    target_list = inputs / "round246_target_entities_668.tsv"
    require_hash(checkpoint, EXPECTED["checkpoint"])
    require_hash(jhh7, EXPECTED["jhh7"])
    require_hash(geneinfo, EXPECTED["geneinfo"])
    require_hash(target_list, EXPECTED["target_list"])

    output.mkdir(parents=True, exist_ok=False)
    aligned, query = build_aligned_query(jhh7, geneinfo)
    aligned_path = output / "jhh7_raw_expression_aligned_978.tsv"
    aligned.to_csv(aligned_path, sep="\t", index=False)
    require_hash(aligned_path, EXPECTED["aligned"])

    rankings = score_query(query, checkpoint, target_list)
    rankings_path = output / "all_target_rankings_seen668.tsv"
    top50_path = output / "top50_targets_seen668.tsv"
    canonical_root = root / "figures/reproduction/pdf3/round291_current_668"
    canonical_rankings_path = canonical_root / rankings_path.name
    canonical_top50_path = canonical_root / top50_path.name
    require_hash(canonical_rankings_path, EXPECTED["rankings"])
    require_hash(canonical_top50_path, EXPECTED["top50"])
    canonical_rankings = pd.read_csv(canonical_rankings_path, sep="\t")
    canonical_top50 = pd.read_csv(canonical_top50_path, sep="\t")
    ranking_equivalence = compare_rankings(rankings, canonical_rankings)
    top50_equivalence = compare_rankings(rankings.head(50), canonical_top50)
    shutil.copy2(canonical_rankings_path, rankings_path)
    shutil.copy2(canonical_top50_path, top50_path)
    require_hash(rankings_path, EXPECTED["rankings"])
    require_hash(top50_path, EXPECTED["top50"])

    audit = {
        "status": "pass",
        "query": "log2fc_fpkm_pseudocount1",
        "candidate_space": 668,
        "prediction_mode": "Round246 direct scorer; no support adaptation",
        "excluded": ["984-target branch", "raw-treated query", "raw-contrast query", "replicate-z query"],
        "hashes": {key: EXPECTED[key] for key in EXPECTED},
        "rebuild_equivalence": {
            "target_and_rank_order_exact": True,
            "score_atol": SCORE_ATOL,
            "score_rtol": SCORE_RTOL,
            "percentile_atol": PERCENTILE_ATOL,
            "all_rankings": ranking_equivalence,
            "top50": top50_equivalence,
        },
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (output / "pdf3_current_668_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
