#!/usr/bin/env python
"""Train and export per-cell support-adapted scores for a Round429 MeTar run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from common import add_code_release_to_path, code_release_root_from_config, read_table, read_yaml, write_table


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def split_targets(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [x.strip() for x in str(value).replace(";", "|").replace(",", "|").split("|") if x.strip()]


def diverse_support_indices(rows: pd.DataFrame, labels: list[str], support_size: int) -> np.ndarray:
    label_pos = {g: i for i, g in enumerate(labels)}
    group = rows[["drug", "cell_line", "true_targets"]].drop_duplicates(["drug", "cell_line"]).reset_index()
    group["target_tuple"] = group["true_targets"].map(lambda v: tuple(t for t in split_targets(v) if t in label_pos))
    group = group[group["target_tuple"].map(bool)].copy()
    chosen_groups = []
    covered: set[str] = set()
    while len(chosen_groups) < support_size and len(chosen_groups) < len(group):
        best_idx = None
        best_key = None
        for idx, row in group.iterrows():
            if idx in chosen_groups:
                continue
            targets = set(row["target_tuple"])
            gain = len(targets - covered)
            key = (gain, len(targets), str(row["drug"]))
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        if best_idx is None:
            break
        chosen_groups.append(best_idx)
        covered |= set(group.loc[best_idx, "target_tuple"])
    return group.loc[chosen_groups, "index"].to_numpy(dtype=int)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train/query support adapter for Round429")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-table", required=True)
    parser.add_argument("--val-table", required=True)
    parser.add_argument("--test-table", required=True)
    parser.add_argument("--feature-cols", required=True)
    parser.add_argument("--label-order", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-score-table", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config).resolve())
    add_code_release_to_path(code_release_root_from_config(cfg))
    from metar_legacy.data_io import labels_to_matrix, read_label_order
    from metar_legacy.models import QueryGatedSupportAdapter, ResidualMLPEncoder, TargetTableScorer, apply_query_shift, label_bag_retrieval_loss

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device_name = cfg.get("support_training", {}).get("device", "cuda")
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = read_label_order(args.label_order)
    features = [line.strip() for line in Path(args.feature_cols).read_text(encoding="utf-8").splitlines() if line.strip()]
    train = read_table(args.train_table)
    val = read_table(args.val_table)
    test = read_table(args.test_table)
    ckpt = load_checkpoint(Path(args.checkpoint))
    embed_dim = int(cfg["training"]["embed_dim"])
    hidden_dim = int(cfg["training"]["hidden_dim"])

    encoder = ResidualMLPEncoder(len(features), embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    target_init = torch.randn(len(labels), embed_dim, device=device) / np.sqrt(embed_dim)
    scorer = TargetTableScorer(target_init, trainable_target_table=True).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    scorer.load_state_dict(ckpt["scorer"])
    encoder.eval()
    scorer.eval()
    target_table = scorer.normalized_targets().detach()

    def encode(df: pd.DataFrame, batch_size: int = 8192) -> torch.Tensor:
        chunks = []
        with torch.no_grad():
            for start in range(0, len(df), batch_size):
                x = torch.tensor(df.iloc[start : start + batch_size][features].to_numpy(dtype=np.float32), device=device)
                chunks.append(encoder(x).detach())
        return torch.cat(chunks, dim=0)

    q_train = encode(train)
    q_val = encode(val)
    q_test = encode(test)
    y_train = torch.tensor(labels_to_matrix(train, labels), dtype=torch.float32, device=device)
    y_val = torch.tensor(labels_to_matrix(val, labels), dtype=torch.float32, device=device)

    support_size = int(cfg["support_adapter"]["support_size"])
    support_idx_parts = []
    for cell, grp in pd.concat([train, val], ignore_index=True).groupby("cell_line", dropna=False):
        idx_local = diverse_support_indices(grp.reset_index(drop=True), labels, support_size)
        full_idx = grp.index.to_numpy()[idx_local]
        support_idx_parts.append(full_idx)
    support_idx = np.concatenate(support_idx_parts) if support_idx_parts else np.array([], dtype=int)
    support_pool = pd.concat([train, val], ignore_index=True)
    q_pool = torch.cat([q_train, q_val], dim=0)
    y_pool = torch.cat([y_train, y_val], dim=0)
    support_q = q_pool[torch.tensor(support_idx, dtype=torch.long, device=device)]
    support_y = y_pool[torch.tensor(support_idx, dtype=torch.long, device=device)]

    adapter = QueryGatedSupportAdapter(target_table).to(device)
    fixed_scorer = TargetTableScorer(target_table.detach(), trainable_target_table=False).to(device)
    fixed_scorer.logit_scale.data.copy_(scorer.logit_scale.data)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(cfg["support_training"]["lr"]), weight_decay=1e-5)
    batch_size = int(cfg["support_training"]["batch_size"])
    loader = DataLoader(TensorDataset(q_train.detach(), y_train.detach()), batch_size=batch_size, shuffle=True)
    history = []
    best = None
    best_state = None
    for epoch in range(1, int(cfg["support_training"]["epochs"]) + 1):
        adapter.train()
        losses = []
        for qb, yb in loader:
            opt.zero_grad(set_to_none=True)
            delta, gate = adapter(qb, support_q, support_y)
            logits = fixed_scorer(apply_query_shift(qb, delta, gate, float(cfg["support_adapter"]["shift_scale"])))
            loss = label_bag_retrieval_loss(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        adapter.eval()
        with torch.no_grad():
            delta_v, gate_v = adapter(q_val, support_q, support_y)
            val_scores = fixed_scorer(apply_query_shift(q_val, delta_v, gate_v, float(cfg["support_adapter"]["shift_scale"])))
            val_loss = float(label_bag_retrieval_loss(val_scores, y_val).detach().cpu())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "device": str(device)}
        history.append(row)
        score = -val_loss
        if best is None or score > best:
            best = score
            best_state = {"adapter": {k: v.detach().cpu() for k, v in adapter.state_dict().items()}, "epoch": epoch}
        pd.DataFrame(history).to_csv(out / "support_training_history.live.tsv", sep="\t", index=False)

    if best_state is None:
        raise SystemExit("No adapter state selected")
    adapter.load_state_dict(best_state["adapter"])
    adapter.eval()
    outputs = []
    with torch.no_grad():
        for cell, grp in test.groupby("cell_line", sort=False, dropna=False):
            idx = grp.index.to_numpy()
            q = q_test[torch.tensor(idx, dtype=torch.long, device=device)]
            pool_mask = support_pool["cell_line"].astype(str).eq(str(cell)).to_numpy()
            pool_idx = np.flatnonzero(pool_mask)
            if len(pool_idx) == 0:
                sq, sy = support_q, support_y
            else:
                local_support = diverse_support_indices(support_pool.iloc[pool_idx].reset_index(drop=True), labels, support_size)
                selected = pool_idx[local_support]
                sq = q_pool[torch.tensor(selected, dtype=torch.long, device=device)]
                sy = y_pool[torch.tensor(selected, dtype=torch.long, device=device)]
            delta, gate = adapter(q, sq, sy)
            score = fixed_scorer(apply_query_shift(q, delta, gate, float(cfg["support_adapter"]["shift_scale"]))).detach().cpu().numpy()
            meta = grp[[c for c in grp.columns if c not in set(features)]].reset_index(drop=True)
            outputs.append(pd.concat([meta, pd.DataFrame(score, columns=labels)], axis=1))

    write_table(pd.concat(outputs, ignore_index=True), args.output_score_table)
    torch.save(best_state, out / "support_adapter.pt")
    write_table(pd.DataFrame(history), out / "support_training_history.tsv")
    support_audit = support_pool.iloc[support_idx][["drug", "cell_line", "true_targets"]].copy()
    write_table(support_audit, out / "support_selection_audit.tsv")
    print(args.output_score_table)


if __name__ == "__main__":
    main()
