#!/usr/bin/env python
"""Export query embeddings and 668-target score tables from a closed-bulk checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import add_code_release_to_path, code_release_root_from_config, read_table, read_yaml, write_table


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export closed-bulk MeTar score table")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-table", required=True)
    parser.add_argument("--feature-cols", required=True)
    parser.add_argument("--label-order", required=True)
    parser.add_argument("--output-score-table", required=True)
    parser.add_argument("--output-query-npz", required=True)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = read_yaml(cfg_path)
    code_root = add_code_release_to_path(code_release_root_from_config(cfg))
    from metar_legacy.data_io import read_label_order
    from metar_legacy.models import ResidualMLPEncoder, TargetTableScorer

    labels = read_label_order(args.label_order)
    feature_cols = [line.strip() for line in Path(args.feature_cols).read_text(encoding="utf-8").splitlines() if line.strip()]
    df = read_table(args.input_table)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Input table missing feature columns: {missing[:5]}")

    embed_dim = args.embed_dim or int(cfg["training"]["embed_dim"])
    hidden_dim = args.hidden_dim or int(cfg["training"]["hidden_dim"])
    ckpt = load_checkpoint(Path(args.checkpoint))
    encoder = ResidualMLPEncoder(len(feature_cols), embed_dim=embed_dim, hidden_dim=hidden_dim)
    target_init = torch.randn(len(labels), embed_dim) / np.sqrt(embed_dim)
    scorer = TargetTableScorer(target_init, trainable_target_table=True)
    encoder.load_state_dict(ckpt["encoder"])
    scorer.load_state_dict(ckpt["scorer"])
    encoder.eval()
    scorer.eval()

    with torch.no_grad():
        x = torch.tensor(df[feature_cols].to_numpy(dtype=np.float32))
        query = encoder(x)
        scores = scorer(query).cpu().numpy()

    np.savez_compressed(args.output_query_npz, arr_0=query.cpu().numpy())
    meta = df[[c for c in df.columns if c not in set(feature_cols)]].reset_index(drop=True)
    out = pd.concat([meta, pd.DataFrame(scores, columns=labels)], axis=1)
    write_table(out, args.output_score_table)
    print(args.output_score_table)


if __name__ == "__main__":
    main()
