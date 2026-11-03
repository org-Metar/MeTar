#!/usr/bin/env python
"""Export support-adapted score tables from precomputed query embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import add_code_release_to_path, code_release_root_from_config, read_table, read_yaml, write_table


def load_npz(path: str | Path) -> np.ndarray:
    return np.load(path)["arr_0"].astype(np.float32)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export support-adapted MeTar score table")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--query-npz", required=True)
    parser.add_argument("--query-rows", required=True)
    parser.add_argument("--support-query-npz", required=True)
    parser.add_argument("--support-label-npz", required=True)
    parser.add_argument("--target-table-npz", required=True)
    parser.add_argument("--label-order", required=True)
    parser.add_argument("--output-score-table", required=True)
    parser.add_argument("--array-key", default="arr_0")
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config).resolve())
    add_code_release_to_path(code_release_root_from_config(cfg))
    from metar_legacy.data_io import read_label_order
    from metar_legacy.models import QueryGatedSupportAdapter, TargetTableScorer, apply_query_shift

    labels = read_label_order(args.label_order)
    target = torch.tensor(np.load(args.target_table_npz)[args.array_key].astype(np.float32))
    adapter = QueryGatedSupportAdapter(target)
    state = load_checkpoint(Path(args.adapter_checkpoint))
    adapter.load_state_dict(state["adapter"])
    scorer = TargetTableScorer(target, trainable_target_table=False)
    adapter.eval()
    scorer.eval()

    query = torch.tensor(load_npz(args.query_npz))
    support_q = torch.tensor(load_npz(args.support_query_npz))
    support_y = torch.tensor(load_npz(args.support_label_npz))
    rows = read_table(args.query_rows)
    if len(rows) != query.shape[0]:
        raise SystemExit(f"query rows and embeddings are misaligned: {len(rows)} != {query.shape[0]}")

    with torch.no_grad():
        delta, gate = adapter(query, support_q, support_y)
        shifted = apply_query_shift(query, delta, gate, float(cfg["support_adapter"]["shift_scale"]))
        scores = scorer(shifted).cpu().numpy()

    out = pd.concat([rows.reset_index(drop=True), pd.DataFrame(scores, columns=labels)], axis=1)
    write_table(out, args.output_score_table)
    print(args.output_score_table)


if __name__ == "__main__":
    main()
