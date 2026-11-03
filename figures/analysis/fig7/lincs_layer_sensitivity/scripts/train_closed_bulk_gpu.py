#!/usr/bin/env python
"""GPU-capable Round429 wrapper around the Round359 closed-bulk MeTar modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import add_code_release_to_path, code_release_root_from_config, read_table, read_yaml, write_table


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train closed-set MeTar with CUDA when available")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-table", required=True)
    parser.add_argument("--val-table", required=True)
    parser.add_argument("--test-table", required=True)
    parser.add_argument("--feature-cols", required=True)
    parser.add_argument("--label-order", required=True)
    parser.add_argument("--target-col", default="true_targets")
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = read_yaml(Path(args.config).resolve())
    add_code_release_to_path(code_release_root_from_config(cfg))
    from metar_legacy.data_io import labels_to_matrix, read_label_order
    from metar_legacy.metrics import compute_unit_metrics, summarize_metric_table
    from metar_legacy.models import ResidualMLPEncoder, TargetTableScorer, drug_normalized_bce_loss, label_bag_retrieval_loss

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device_name = args.device or cfg["training"].get("device", "cuda")
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = read_label_order(args.label_order)
    feature_cols = [line.strip() for line in Path(args.feature_cols).read_text(encoding="utf-8").splitlines() if line.strip()]
    train = read_table(args.train_table)
    val = read_table(args.val_table)
    test = read_table(args.test_table)

    embed_dim = args.embed_dim or int(cfg["training"]["embed_dim"])
    hidden_dim = args.hidden_dim or int(cfg["training"]["hidden_dim"])
    epochs = args.epochs or int(cfg["training"]["epochs"])
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    lr = args.lr or float(cfg["training"]["lr"])

    target_init = torch.randn(len(labels), embed_dim, device=device) / np.sqrt(embed_dim)
    encoder = ResidualMLPEncoder(len(feature_cols), embed_dim, hidden_dim).to(device)
    scorer = TargetTableScorer(target_init, trainable_target_table=True).to(device)
    opt = torch.optim.AdamW([*encoder.parameters(), *scorer.parameters()], lr=lr, weight_decay=1e-4)

    x_train = train[feature_cols].to_numpy(dtype=np.float32)
    y_train = labels_to_matrix(train, labels, args.target_col)
    loader = make_loader(x_train, y_train, batch_size, shuffle=True)
    history = []
    best = None
    best_state = None

    def eval_df(df: pd.DataFrame) -> pd.DataFrame:
        encoder.eval()
        scorer.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(df), max(batch_size * 8, 1024)):
                x = torch.tensor(df.iloc[start : start + max(batch_size * 8, 1024)][feature_cols].to_numpy(dtype=np.float32), device=device)
                chunks.append(scorer(encoder(x)).detach().cpu().numpy())
        scores = np.vstack(chunks)
        y = labels_to_matrix(df, labels, args.target_col)
        meta = df[[c for c in df.columns if c not in set(feature_cols)]].reset_index(drop=True)
        return compute_unit_metrics(scores, y, meta)

    for epoch in range(1, epochs + 1):
        encoder.train()
        scorer.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = scorer(encoder(xb))
            loss = 0.5 * drug_normalized_bce_loss(logits, yb) + 0.5 * label_bag_retrieval_loss(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *scorer.parameters()], 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_per = eval_df(val)
        val_sum = summarize_metric_table(val_per).iloc[0].to_dict()
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "device": str(device)}
        row.update({f"val_{k}": v for k, v in val_sum.items() if k != "n_units"})
        history.append(row)
        score = (float(row.get("val_Hit@10", 0.0)), float(row.get("val_MeanAP", 0.0)), float(row.get("val_NDCG@10", 0.0)))
        if best is None or score > best:
            best = score
            best_state = {
                "encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                "scorer": {k: v.detach().cpu() for k, v in scorer.state_dict().items()},
                "epoch": epoch,
                "device": str(device),
            }
        pd.DataFrame(history).to_csv(out / "training_history.live.tsv", sep="\t", index=False)

    if best_state is None:
        raise SystemExit("Training produced no checkpoint")
    encoder.load_state_dict(best_state["encoder"])
    scorer.load_state_dict(best_state["scorer"])
    test_per = eval_df(test)
    test_summary = summarize_metric_table(test_per)
    write_table(pd.DataFrame(history), out / "training_history.tsv")
    write_table(test_per, out / "test_per_unit_metrics.tsv")
    write_table(test_summary, out / "test_metric_summary.tsv")
    torch.save(best_state, out / "checkpoint.pt")
    print(out)


if __name__ == "__main__":
    main()
