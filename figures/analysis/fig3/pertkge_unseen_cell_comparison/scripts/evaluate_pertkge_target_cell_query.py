#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE = SCRIPT_PATH.parents[1]
DEFAULT_REMOTE_ROOT = Path("/home/lwn/claude")
LOCAL_ROOT = WORKSPACE.parent

BASELINE_DIR = Path(os.environ.get("METAR_BASELINE_DIR", str(DEFAULT_REMOTE_ROOT / "baseline")))
PERTKGE_REPO = Path(os.environ.get("PERTKGE_REPO", str(BASELINE_DIR / "pertkge_code" / "PertKGE-main")))
DATA_DIR = Path(os.environ.get("METAR_BULK_DATA_DIR", str(DEFAULT_REMOTE_ROOT / "bulk" / "data")))
METAR_RUN = Path(os.environ.get("METAR_RUN_DIR", str(DEFAULT_REMOTE_ROOT / "manuscript" / "write" / "bulk_sharedsample_indunseenlabel_20260330_h07_w060_u8e4_med")))
LOCK_PATH = METAR_RUN / "manifests" / "bulk_protocol_lock_v2.json"
EXPR_PATH = DATA_DIR / "large_data.parquet"
ROUND252_SOURCE = Path(os.environ.get("ROUND252_SOURCE", str(LOCAL_ROOT / "claude_round252_epiqgate_pseudoval_sweep_20260520" / "source_data")))

HIT_KS = [1, 5, 10, 20, 30, 50]
TOP_GENES_EFFECT = 200
TRAIN_EFFECT_CELL = "PC3"
ROUND88_MAX_EPOCHS = 80
ROUND88_PATIENCE = 5
ROUND88_EVAL_EVERY = 5
ROUND88_WARMUP = 10
ROUND88_BATCH_SIZE = 4096
ROUND88_SEED = 43
ROUND246_NOADAPT_VARIANT = "r246_joint_lm0p5_sc0p15_mn2p5_official_unseen_no_adapt"
ROUND252_MAIN_VARIANT = (
    "r252_epiqgate_lm1_lr0p00015_sc0p18_mn2p5_ba0p1_s32_q32_"
    "diverse_label_seed20260524_official_unseen_diverse_label_s64"
)
_PROTOCOL_CONTEXT_CACHE: dict | None = None
_OFFICIAL_UNITS_CACHE: pd.DataFrame | None = None


def split_targets(value: str) -> Iterable[str]:
    for g in str(value).split("|"):
        g = g.strip()
        if g and g not in {"0", "-666", "nan", "None"}:
            yield g


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_info(path: Path) -> dict:
    info = {"path": str(path), "exists": path.exists()}
    if path.exists() and path.is_file():
        info["sha256"] = sha256_file(path)
        info["size_bytes"] = path.stat().st_size
    return info


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics_per_unit(scores_df: pd.DataFrame, targets_map: Dict[str, List[str]], labels: List[str], method: str) -> pd.DataFrame:
    records = []
    label_set = set(labels)
    for _, row in scores_df.iterrows():
        true_tgts = sorted(set(targets_map.get(row["drug"], [])) & label_set)
        if not true_tgts:
            continue
        scores = np.array([float(row[g]) if g in row else 0.0 for g in labels], dtype=np.float64)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(scores)[::-1]
        ranked = [labels[i] for i in order]
        true_set = set(true_tgts)
        rec = {
            "method": method,
            "variant": "round307_pc3train_targetcell_queryeffect",
            "split": "official_test",
            "evaluation_unit": "drug-mean",
            "drug": row["drug"],
            "cell_line": row["cell_line"],
            "n_profiles": int(row["n_profiles"]),
            "n_true_targets": len(true_tgts),
        }
        for k in HIT_KS:
            rec[f"Hit@{k}"] = int(bool(set(ranked[:k]) & true_set))
        y = np.array([1 if g in true_set else 0 for g in labels], dtype=np.int8)
        rec["AUROC"] = float(roc_auc_score(y, scores)) if 0 < int(y.sum()) < len(y) else 0.5
        rec["NDCG@10"] = float(ndcg_score([y], [scores], k=10))
        rec["MeanAP"] = float(average_precision_score(y, scores))
        rec["best_rank"] = min(ranked.index(t) + 1 for t in true_tgts)
        rec["best_true_target_percentile"] = 100.0 * (1.0 - rec["best_rank"] / len(labels))
        records.append(rec)
    return pd.DataFrame(records)


def summarize_units(metrics: pd.DataFrame, method: str, aggregation_scope: str) -> dict:
    row = {
        "method": method,
        "variant": "round307_pc3train_targetcell_queryeffect",
        "split": "official_test",
        "evaluation_unit": "drug-mean",
        "candidate_space": 668,
        "n_cells": int(metrics["cell_line"].nunique()) if len(metrics) else 0,
        "n_groups": int(len(metrics)),
        "n_drugs": int(metrics["drug"].nunique()) if len(metrics) else 0,
        "n_profiles": int(metrics["n_profiles"].sum()) if "n_profiles" in metrics else 0,
        "aggregation_scope": aggregation_scope,
    }
    for k in HIT_KS:
        row[f"Hit@{k}"] = float(metrics[f"Hit@{k}"].mean()) if len(metrics) else float("nan")
    for m in ["AUROC", "NDCG@10", "MeanAP", "best_true_target_percentile"]:
        row[m] = float(metrics[m].mean()) if m in metrics and len(metrics) else float("nan")
    return row


def summarize_cell_unweighted(per_cell: pd.DataFrame, method: str, variant: str = "round307_pc3train_targetcell_queryeffect") -> dict:
    row = {
        "method": method,
        "variant": variant,
        "split": "official_test",
        "evaluation_unit": "drug-mean",
        "candidate_space": 668,
        "n_cells": int(per_cell["cell_line"].nunique()) if len(per_cell) else 0,
        "n_groups": int(per_cell["n_groups"].sum()) if "n_groups" in per_cell else 0,
        "n_drugs": int(per_cell["n_drugs"].sum()) if "n_drugs" in per_cell else 0,
        "n_profiles": int(per_cell["n_profiles"].sum()) if "n_profiles" in per_cell else 0,
        "aggregation_scope": "cell_unweighted_mean",
    }
    for k in HIT_KS:
        row[f"Hit@{k}"] = float(per_cell[f"Hit@{k}"].mean()) if len(per_cell) else float("nan")
    for m in ["AUROC", "NDCG@10", "MeanAP", "best_true_target_percentile"]:
        row[m] = float(per_cell[m].mean()) if m in per_cell and len(per_cell) else float("nan")
    return row


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_protocol_context() -> dict:
    global _PROTOCOL_CONTEXT_CACHE
    if _PROTOCOL_CONTEXT_CACHE is not None:
        return _PROTOCOL_CONTEXT_CACHE
    protocol = json.loads(LOCK_PATH.read_text())
    inner_train = set(protocol["drug_split"]["inner_train_drugs"])
    val_drugs = set(protocol["drug_split"]["val_drugs"])
    test_drugs = set(protocol["drug_split"]["test_drugs"])

    cinfo = pd.read_csv(DATA_DIR / "compoundinfo_beta.txt", sep="\t")
    cinfo = cinfo[cinfo["target"].notna() & ~cinfo["target"].isin(["0", "", "-666"])]
    inner_cinfo = cinfo[cinfo["pert_id"].isin(inner_train)]
    labels = sorted({g for t in inner_cinfo["target"].dropna() for g in split_targets(t)})
    if len(labels) != 668:
        raise ValueError(f"Expected 668 labels, observed {len(labels)}")
    label_set = set(labels)

    drug_targets: Dict[str, List[str]] = {}
    for d, grp in cinfo.groupby("pert_id"):
        tgts = sorted({g for t in grp["target"].astype(str) for g in split_targets(t)})
        if tgts:
            drug_targets[d] = tgts

    seen_clean_val = sorted(d for d in val_drugs if d in drug_targets and set(drug_targets[d]).issubset(label_set))
    seen_clean_test = sorted(d for d in test_drugs if d in drug_targets and set(drug_targets[d]).issubset(label_set))

    expr = pd.read_parquet(EXPR_PATH).reset_index()
    expr = expr.rename(columns={expr.columns[0]: "profile_id"})
    gene_cols = [c for c in expr.columns if c not in ("profile_id", "drug", "cell_line", "method")]
    expr_cp = expr[expr["method"] == 0].copy()
    geneinfo = pd.read_csv(BASELINE_DIR / "geneinfo_beta.txt", sep="\t", usecols=["gene_id", "gene_symbol"], dtype=str)
    entrez2sym = dict(zip(geneinfo["gene_id"].str.strip(), geneinfo["gene_symbol"].str.strip()))

    _PROTOCOL_CONTEXT_CACHE = {
        "protocol": protocol,
        "inner_train_drugs": inner_train,
        "val_drugs": val_drugs,
        "test_drugs": test_drugs,
        "labels": labels,
        "drug_targets": drug_targets,
        "seen_clean_val_drugs": seen_clean_val,
        "seen_clean_test_drugs": seen_clean_test,
        "expr_cp": expr_cp,
        "gene_cols": gene_cols,
        "entrez2sym": entrez2sym,
    }
    return _PROTOCOL_CONTEXT_CACHE


def load_official_units() -> pd.DataFrame:
    global _OFFICIAL_UNITS_CACHE
    if _OFFICIAL_UNITS_CACHE is not None:
        return _OFFICIAL_UNITS_CACHE.copy()
    path = ROUND252_SOURCE / "per_unit_metrics.tsv"
    df = pd.read_csv(path, sep="\t", low_memory=False)
    units = df[
        (df["split"] == "official_test")
        & (df["evaluation_unit"] == "drug-mean")
        & (df["variant"] == ROUND246_NOADAPT_VARIANT)
    ][["drug", "cell_line", "n_profiles", "n_true_targets"]].copy()
    _OFFICIAL_UNITS_CACHE = units.drop_duplicates(["drug", "cell_line"]).sort_values(["cell_line", "drug"]).reset_index(drop=True)
    return _OFFICIAL_UNITS_CACHE.copy()


def official_cells() -> List[str]:
    return sorted(load_official_units()["cell_line"].unique())


def effect_rows_for_drug_means(drug_mean: pd.DataFrame, gene_cols: List[str], entrez2sym: Dict[str, str]) -> List[str]:
    rows = []
    for drug_id, row_vals in drug_mean.iterrows():
        vals = row_vals.to_numpy(dtype=np.float32)
        order = np.argsort(vals)
        top_up = order[-TOP_GENES_EFFECT:]
        top_dn = order[:TOP_GENES_EFFECT]
        for idx in top_up:
            sym = entrez2sym.get(str(gene_cols[idx]).strip())
            if sym:
                rows.append(f"CID:{drug_id}\tUpregulates\tmRNA:{sym}")
        for idx in top_dn:
            sym = entrez2sym.get(str(gene_cols[idx]).strip())
            if sym:
                rows.append(f"CID:{drug_id}\tDownregulates\tmRNA:{sym}")
    return rows


def build_cell_context(cell_line: str) -> dict:
    base = load_protocol_context()
    official_units = load_official_units()
    cell_units = official_units[official_units["cell_line"] == cell_line].copy()
    if cell_units.empty:
        raise ValueError(f"No Round252 official units for {cell_line}")

    expr_cp = base["expr_cp"]
    gene_cols = base["gene_cols"]
    train_effect_drugs = sorted(base["inner_train_drugs"])
    val_drugs = sorted(set(base["seen_clean_val_drugs"]))
    test_drugs = sorted(set(cell_units["drug"]))
    query_effect_drugs = sorted(set(val_drugs) | set(test_drugs))

    train_expr = expr_cp[(expr_cp["drug"].isin(train_effect_drugs)) & (expr_cp["cell_line"] == TRAIN_EFFECT_CELL)]
    query_expr = expr_cp[(expr_cp["drug"].isin(query_effect_drugs)) & (expr_cp["cell_line"] == cell_line)]
    train_drug_mean = train_expr.groupby("drug")[gene_cols].mean()
    query_drug_mean = query_expr.groupby("drug")[gene_cols].mean()

    # If a drug appears both in the PC3 training-effect branch and the target-cell
    # query branch, keep the target-cell query representation for val/test use.
    all_effect_mean = pd.concat(
        [
            train_drug_mean.loc[~train_drug_mean.index.isin(query_drug_mean.index)],
            query_drug_mean,
        ],
        axis=0,
    )
    missing_query = sorted(set(query_effect_drugs) - set(query_drug_mean.index))
    usable_test_drugs = sorted(set(test_drugs) & set(query_drug_mean.index))
    usable_units = cell_units[cell_units["drug"].isin(usable_test_drugs)].copy()
    val_drugs_with_effect = sorted(set(val_drugs) & set(query_drug_mean.index))

    return {
        **base,
        "cell_line": cell_line,
        "official_units": cell_units,
        "usable_units": usable_units,
        "train_effect_drugs": train_effect_drugs,
        "val_drugs": val_drugs,
        "test_drugs": test_drugs,
        "query_effect_drugs": query_effect_drugs,
        "train_drug_mean": train_drug_mean,
        "query_drug_mean": query_drug_mean,
        "all_effect_mean": all_effect_mean,
        "missing_query_effect_drugs": missing_query,
        "usable_test_drugs": usable_test_drugs,
        "val_drugs_with_effect": val_drugs_with_effect,
    }


def write_pertkge_files(ctx: dict, run_dir: Path) -> dict:
    ti = run_dir / "pertkge" / "target_inference"
    kg = run_dir / "pertkge" / "knowledge_graph"
    ti.mkdir(parents=True, exist_ok=True)
    kg.mkdir(parents=True, exist_ok=True)

    labels = set(ctx["labels"])
    drug_targets = ctx["drug_targets"]

    cause_rows = []
    for drug in sorted(ctx["inner_train_drugs"]):
        for tgt in drug_targets.get(drug, []):
            if tgt in labels:
                cause_rows.append(f"CID:{drug}\tHAS_BINDING_TO\tProtein:{tgt}")
    (ti / "cause.txt").write_text("\n".join(sorted(set(cause_rows))), encoding="utf-8")

    eval_drugs = sorted(set(ctx["val_drugs"]) | set(ctx["test_drugs"]))
    test_rows = []
    for drug in eval_drugs:
        for tgt in drug_targets.get(drug, []):
            if tgt in labels:
                test_rows.append(f"CID:{drug}\tHAS_BINDING_TO\tProtein:{tgt}")
    (ti / "test.txt").write_text("\n".join(sorted(set(test_rows))), encoding="utf-8")

    process_src = PERTKGE_REPO / "processed_data" / "knowledge_graph" / "process.txt"
    shutil.copy2(process_src, kg / "process.txt")

    effect_rows = effect_rows_for_drug_means(ctx["all_effect_mean"], ctx["gene_cols"], ctx["entrez2sym"])
    (ti / "effect.txt").write_text("\n".join(sorted(set(effect_rows))), encoding="utf-8")

    cause_e = set(cause_rows)
    test_e = set(test_rows)
    process_df = pd.read_csv(kg / "process.txt", sep="\t", names=["from", "rel", "to"])
    process_e = set("\t".join(x) for x in process_df[["from", "rel", "to"]].astype(str).itertuples(index=False, name=None))
    effect_df = pd.DataFrame([x.split("\t") for x in sorted(set(effect_rows))], columns=["from", "rel", "to"]) if effect_rows else pd.DataFrame(columns=["from", "rel", "to"])

    query_cids = {f"CID:{d}" for d in set(ctx["val_drugs"]) | set(ctx["test_drugs"])}
    effect_query_cids = set(effect_df["from"]) & query_cids if len(effect_df) else set()
    process_has_binding = int((process_df["rel"] == "HAS_BINDING_TO").sum())
    process_cid_edges = int(process_df["from"].astype(str).str.startswith("CID:").sum() + process_df["to"].astype(str).str.startswith("CID:").sum())
    target_edge_leakage = len(test_e & (cause_e | process_e))
    if target_edge_leakage:
        raise RuntimeError(f"query target edges leaked into training graph: {target_edge_leakage}")
    if process_has_binding or process_cid_edges:
        raise RuntimeError(f"process graph contains forbidden target/CID edges: HAS_BINDING_TO={process_has_binding}, CID_edges={process_cid_edges}")

    train_query_overlap = sorted(set(ctx["inner_train_drugs"]) & (set(ctx["val_drugs"]) | set(ctx["test_drugs"])))
    audit = {
        "round": "round307_pertkge_pc3train_targetcell_queryeffect_crosscell",
        "cell_line": ctx["cell_line"],
        "candidate_space": 668,
        "protocol": "PertKGE target inference: train drugs use PC3 effect plus DTI; query drugs use target-cell effect with DTI withheld.",
        "train_effect_cell": TRAIN_EFFECT_CELL,
        "query_effect_cell": ctx["cell_line"],
        "cause_edges": len(set(cause_rows)),
        "test_edges_for_eval_only": len(set(test_rows)),
        "effect_edges": len(set(effect_rows)),
        "train_effect_drugs_requested": len(ctx["train_effect_drugs"]),
        "train_effect_drugs_with_pc3_signature": int(len(ctx["train_drug_mean"])),
        "query_effect_drugs_requested": len(ctx["query_effect_drugs"]),
        "query_effect_drugs_with_targetcell_signature": int(len(ctx["query_drug_mean"])),
        "missing_query_effect_drugs_count": len(ctx["missing_query_effect_drugs"]),
        "missing_query_effect_drugs_examples": ctx["missing_query_effect_drugs"][:20],
        "official_test_units": int(len(ctx["official_units"])),
        "usable_test_units_with_query_effect": int(len(ctx["usable_units"])),
        "official_test_drugs": int(ctx["official_units"]["drug"].nunique()),
        "usable_test_drugs_with_query_effect": int(len(ctx["usable_test_drugs"])),
        "val_drugs_with_targetcell_effect": int(len(ctx["val_drugs_with_effect"])),
        "allowed_query_effect_cids": len(effect_query_cids),
        "query_dti_edges_in_training_cause_or_process": target_edge_leakage,
        "cause_query_target_edge_overlap": len(cause_e & test_e),
        "process_has_binding_to_edges": process_has_binding,
        "process_cid_edges": process_cid_edges,
        "train_query_drug_id_overlap": len(train_query_overlap),
        "train_query_drug_id_overlap_examples": train_query_overlap[:20],
        "interpretation": "Query CID effect triples are required PertKGE query information; query HAS_BINDING_TO edges remain withheld and are used only for validation/test scoring.",
    }
    (run_dir / "leakage_audit_pre_read.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def train_and_score(ctx: dict, run_dir: Path, max_epochs: int, patience_limit: int, eval_every: int, warmup: int, device: torch.device, seed: int) -> dict:
    sys.path.insert(0, str(PERTKGE_REPO / "src"))
    from utils import read_files, set_seeds  # type: ignore
    from torchkge import DistMultModel, KnowledgeGraph  # type: ignore
    from torchkge.sampling import BernoulliNegativeSampler  # type: ignore
    from torchkge.utils import DataLoader as KGEDataLoader, MarginLoss  # type: ignore

    args = SimpleNamespace(
        cause_file=str(run_dir / "pertkge" / "target_inference" / "cause.txt"),
        process_file=str(run_dir / "pertkge" / "knowledge_graph" / "process.txt"),
        effect_file=str(run_dir / "pertkge" / "target_inference" / "effect.txt"),
        test_file=str(run_dir / "pertkge" / "target_inference" / "test.txt"),
        load_processed_data=False,
        processed_data_file=str(run_dir / "pertkge" / "target_inference") + "/",
        seed=seed,
        h_dim=300,
        margin=1.0,
        lr=1e-4,
        wd=1e-5,
        n_neg=100,
        batch_size=ROUND88_BATCH_SIZE,
        warm_up=warmup,
        patients=patience_limit,
        use_cuda="cuda" if device.type == "cuda" else "",
        nepoch=max_epochs,
        save_model=True,
        save_model_path=str(run_dir / "pertkge" / "best_model") + "/",
        task="target_inference",
        run_name=f"round307_{ctx['cell_line']}",
        mode="reproduce",
        skip_train=False,
    )
    Path(args.save_model_path).mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)
    cause, pertkg_wo_cause, _test_df, ent2id, rel2id, _pro2nc, _h_cand, _t_cand = read_files(args)

    eval_cids = {f"CID:{d}" for d in set(ctx["val_drugs"]) | set(ctx["test_drugs"])}
    eval_cids_in_ent2id = sorted(eval_cids & set(ent2id))
    post_read_audit = {
        "round": "round307_pertkge_pc3train_targetcell_queryeffect_crosscell",
        "cell_line": ctx["cell_line"],
        "eval_cids_in_ent2id_count": len(eval_cids_in_ent2id),
        "val_drugs_in_ent2id": sum(1 for d in ctx["val_drugs"] if f"CID:{d}" in ent2id),
        "test_drugs_in_ent2id": sum(1 for d in ctx["test_drugs"] if f"CID:{d}" in ent2id),
        "target_label_leakage_rule": "Only query expression effect triples are allowed for val/test CIDs; val/test HAS_BINDING_TO target edges are not training facts.",
    }
    (run_dir / "leakage_audit_post_read.json").write_text(json.dumps(post_read_audit, indent=2), encoding="utf-8")

    full_kg = pd.concat([cause, pertkg_wo_cause], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    kg = KnowledgeGraph(df=full_kg, ent2ix=ent2id, rel2ix=rel2id)
    model = DistMultModel(args.h_dim, kg.n_ent, kg.n_rel).to(device)
    criterion = MarginLoss(args.margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sampler = BernoulliNegativeSampler(kg, n_neg=args.n_neg)
    loader = KGEDataLoader(kg, batch_size=args.batch_size, use_cuda=args.use_cuda)

    labels = ctx["labels"]
    target_entities = [f"Protein:{g}" for g in labels]
    missing_targets = [x for x in target_entities if x not in ent2id]
    if missing_targets:
        raise ValueError(f"Missing target entities: {len(missing_targets)}")

    val_drugs = sorted([d for d in ctx["val_drugs_with_effect"] if f"CID:{d}" in ent2id])
    test_drugs = sorted([d for d in ctx["usable_test_drugs"] if f"CID:{d}" in ent2id])
    if not val_drugs:
        logging.warning("No validation drugs with target-cell effect for %s; fallback to final epoch if needed", ctx["cell_line"])
    if not test_drugs:
        raise ValueError(f"No official test drugs in ent2id for {ctx['cell_line']}")

    def score_drug_table(model_obj, drugs: List[str]) -> pd.DataFrame:
        model_obj.normalize_parameters()
        model_obj.eval()
        with torch.no_grad():
            ent_emb, rel_emb = model_obj.get_embeddings()
            r_emb = rel_emb[rel2id["HAS_BINDING_TO"]].to(device)
            target_idx = torch.tensor([ent2id[x] for x in target_entities], dtype=torch.long, device=device)
            t_emb = ent_emb[target_idx]
            rows = []
            for drug in drugs:
                h_emb = ent_emb[ent2id[f"CID:{drug}"]].to(device)
                score = torch.sigmoid((h_emb * r_emb * t_emb).sum(dim=1)).detach().cpu().numpy()
                rec = {"drug": drug}
                rec.update({g: float(score[i]) for i, g in enumerate(labels)})
                rows.append(rec)
        return pd.DataFrame(rows)

    def val_mean_ap(model_obj) -> float:
        if not val_drugs:
            return float("-inf")
        scores = score_drug_table(model_obj, val_drugs)
        scores["cell_line"] = ctx["cell_line"]
        scores["n_profiles"] = 1
        m = compute_metrics_per_unit(scores[["drug", "cell_line", "n_profiles", *labels]], ctx["drug_targets"], labels, "PertKGE_val")
        return float(m["MeanAP"].mean()) if len(m) else float("-inf")

    best_val = -1.0
    best_epoch = 0
    best_state = None
    patience = 0
    history = []
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            h_cpu, t_cpu, r_cpu = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h_cpu, t_cpu, r_cpu)
            h, t, r = h_cpu.to(device), t_cpu.to(device), r_cpu.to(device)
            n_h, n_t = n_h.to(device), n_t.to(device)
            optimizer.zero_grad()
            pos, neg = model(h, t, r, n_h, n_t)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        row = {"epoch": epoch, "loss": total_loss / max(n_batches, 1), "n_batches": n_batches, "elapsed_sec": time.time() - start}
        if epoch >= warmup and epoch % eval_every == 0 and val_drugs:
            cur = val_mean_ap(model)
            row["val_MeanAP"] = cur
            improved = cur > best_val + 1e-8
            if improved:
                best_val = cur
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            row["best_val_MeanAP"] = best_val
            row["patience"] = patience
            logging.info("epoch=%d loss=%.5f val_MeanAP=%.5f best=%.5f patience=%d", epoch, row["loss"], cur, best_val, patience)
            history.append(row)
            if patience >= patience_limit:
                logging.info("early stop at epoch=%d best_epoch=%d best_val=%.5f", epoch, best_epoch, best_val)
                break
        else:
            row["best_val_MeanAP"] = best_val
            row["patience"] = patience
            logging.info("epoch=%d loss=%.5f", epoch, row["loss"])
            history.append(row)

    if best_state is None:
        best_val = val_mean_ap(model) if val_drugs else float("nan")
        best_epoch = int(history[-1]["epoch"])
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_state, run_dir / "checkpoint_best.pth")
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(run_dir / "training_history.tsv", sep="\t", index=False)

    drug_scores = score_drug_table(model, test_drugs)
    score_units = ctx["usable_units"][["drug", "cell_line", "n_profiles", "n_true_targets"]].merge(drug_scores, on="drug", how="inner", validate="many_to_one")
    score_units.to_parquet(run_dir / "pertkge_scores.parquet", index=False)
    metrics = compute_metrics_per_unit(score_units[["drug", "cell_line", "n_profiles", *labels]], ctx["drug_targets"], labels, "Round307_PertKGE")
    metrics.to_csv(run_dir / "per_unit_metrics.tsv", sep="\t", index=False)
    pd.DataFrame([summarize_units(metrics, "Round307 PertKGE", "drugcell_weighted_mean")]).to_csv(run_dir / "metric_summary.tsv", sep="\t", index=False)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "best_epoch": best_epoch,
        "best_val_MeanAP": best_val,
        "last_epoch": int(pd.DataFrame(history)["epoch"].max()),
        "kg_entities": len(ent2id),
        "kg_relations": len(rel2id),
        "kg_facts": int(kg.n_facts),
        "val_drugs_in_ent2id": len(val_drugs),
        "test_drugs_in_ent2id": len(test_drugs),
        "n_scored_units": len(metrics),
        "n_scored_drugs": metrics["drug"].nunique() if len(metrics) else 0,
    }


def preflight() -> None:
    cells = official_cells()
    rows = []
    for cell in cells:
        ctx = build_cell_context(cell)
        rows.append(
            {
                "cell_line": cell,
                "official_units": len(ctx["official_units"]),
                "official_test_drugs": ctx["official_units"]["drug"].nunique(),
                "usable_units_with_targetcell_query_effect": len(ctx["usable_units"]),
                "usable_test_drugs_with_targetcell_query_effect": len(ctx["usable_test_drugs"]),
                "missing_query_effect_drugs": len(ctx["missing_query_effect_drugs"]),
                "val_drugs_with_targetcell_effect": len(ctx["val_drugs_with_effect"]),
                "train_effect_drugs_requested": len(ctx["train_effect_drugs"]),
                "train_effect_drugs_with_pc3_signature": len(ctx["train_drug_mean"]),
                "query_effect_cell": cell,
                "train_effect_cell": TRAIN_EFFECT_CELL,
            }
        )
    out = WORKSPACE / "source_data" / "round307_preflight_official_cells.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"wrote {out}")


def summarize_runs() -> None:
    source_dir = WORKSPACE / "source_data"
    qa_dir = WORKSPACE / "qa"
    source_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    all_units = []
    per_cell_rows = []
    denom_rows = []
    completed = []
    missing = []
    for cell in official_cells():
        run_dir = WORKSPACE / "official_cells" / f"cell_{cell}"
        unit_path = run_dir / "per_unit_metrics.tsv"
        pre_path = run_dir / "leakage_audit_pre_read.json"
        post_path = run_dir / "leakage_audit_post_read.json"
        lock_path = run_dir / "method_lock.json"
        if not unit_path.exists():
            missing.append(cell)
            denom_rows.append({"cell_line": cell, "status": "missing", "per_unit_exists": False})
            continue
        completed.append(cell)
        unit = pd.read_csv(unit_path, sep="\t")
        all_units.append(unit)
        per_cell_row = summarize_units(unit, "Round307 PertKGE", "drugcell_weighted_mean")
        per_cell_row["cell_line"] = cell
        per_cell_rows.append(per_cell_row)
        pre = json.loads(pre_path.read_text()) if pre_path.exists() else {}
        post = json.loads(post_path.read_text()) if post_path.exists() else {}
        denom_rows.append(
            {
                "cell_line": cell,
                "status": "complete",
                "official_test_units": pre.get("official_test_units"),
                "usable_test_units_with_query_effect": pre.get("usable_test_units_with_query_effect"),
                "official_test_drugs": pre.get("official_test_drugs"),
                "usable_test_drugs_with_query_effect": pre.get("usable_test_drugs_with_query_effect"),
                "missing_query_effect_drugs_count": pre.get("missing_query_effect_drugs_count"),
                "query_dti_edges_in_training_cause_or_process": pre.get("query_dti_edges_in_training_cause_or_process"),
                "process_has_binding_to_edges": pre.get("process_has_binding_to_edges"),
                "process_cid_edges": pre.get("process_cid_edges"),
                "eval_cids_in_ent2id_count": post.get("eval_cids_in_ent2id_count"),
                "method_lock_sha256": sha256_file(lock_path) if lock_path.exists() else "",
            }
        )

    per_cell = pd.DataFrame(per_cell_rows)
    if not per_cell.empty:
        per_cell.to_csv(source_dir / "round307_pertkge_official_per_cell_metrics.tsv", sep="\t", index=False)
    denom = pd.DataFrame(denom_rows)
    denom.to_csv(source_dir / "round307_pertkge_denominator_and_leakage_audit.tsv", sep="\t", index=False)

    combined = pd.concat(all_units, ignore_index=True) if all_units else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(source_dir / "round307_pertkge_official_per_unit_metrics.tsv", sep="\t", index=False)
        summary_rows = [
            summarize_units(combined, "Round307 PertKGE", "drugcell_weighted_mean"),
            summarize_cell_unweighted(per_cell, "Round307 PertKGE"),
        ]
        pd.DataFrame(summary_rows).to_csv(source_dir / "round307_pertkge_metric_summary.tsv", sep="\t", index=False)

    compare_rows = []
    round252_metric = pd.read_csv(ROUND252_SOURCE / "metric_summary.tsv", sep="\t")
    for variant, name in [(ROUND246_NOADAPT_VARIANT, "Round246 no-adapt"), (ROUND252_MAIN_VARIANT, "Round252 EpiQGate diverse_label s64")]:
        sub = round252_metric[
            (round252_metric["split"] == "official_test")
            & (round252_metric["evaluation_unit"] == "drug-mean")
            & (round252_metric["variant"] == variant)
        ]
        if not sub.empty:
            r = sub.iloc[0]
            row = {
                "method": name,
                "variant": variant,
                "split": "official_test",
                "evaluation_unit": "drug-mean",
                "candidate_space": r.get("candidate_space", 668),
                "n_cells": r["n_cells"],
                "n_groups": r["n_groups"],
                "n_drugs": r["n_drugs"],
                "n_profiles": r["n_profiles"],
                "aggregation_scope": "cell_unweighted_mean",
                "denominator_scope": "full_round252_official",
            }
            for k in HIT_KS:
                row[f"Hit@{k}"] = r[f"Hit@{k}"]
            for m in ["AUROC", "NDCG@10", "MeanAP"]:
                row[m] = r[m]
            compare_rows.append(row)
    if not combined.empty:
        r307 = summarize_cell_unweighted(per_cell, "Round307 PertKGE")
        r307["denominator_scope"] = "scoreable_units_with_targetcell_query_effect"
        compare_rows.append(r307)
        r307_weighted = summarize_units(combined, "Round307 PertKGE", "drugcell_weighted_mean")
        r307_weighted["denominator_scope"] = "scoreable_units_with_targetcell_query_effect"
        compare_rows.append(r307_weighted)
    comp = pd.DataFrame(compare_rows)
    comp.to_csv(source_dir / "round307_vs_round252_crosscell_summary.tsv", sep="\t", index=False)

    lock = {
        "round": "round307_pertkge_pc3train_targetcell_queryeffect_crosscell",
        "created_at": datetime.now().isoformat(),
        "completed_cells": completed,
        "missing_cells": missing,
        "protocol": {
            "train_effect_cell": TRAIN_EFFECT_CELL,
            "query_effect_cell": "each official held-out cell",
            "cause": "inner-train drug HAS_BINDING_TO target edges only",
            "effect": "inner-train drug PC3 expression plus validation/test query drug target-cell expression",
            "withheld": "validation/test query drug HAS_BINDING_TO target edges",
            "selection": "validation MeanAP when validation query drugs have target-cell expression; otherwise final epoch fallback",
            "evaluation": "Round252 official unseen drug-cell denominator, 668 labels",
        },
        "training_defaults": {
            "max_epochs": ROUND88_MAX_EPOCHS,
            "patience": ROUND88_PATIENCE,
            "eval_every": ROUND88_EVAL_EVERY,
            "warmup": ROUND88_WARMUP,
            "batch_size": ROUND88_BATCH_SIZE,
            "seed": ROUND88_SEED,
        },
        "inputs": {
            "expr_path": path_info(EXPR_PATH),
            "lock_path": path_info(LOCK_PATH),
            "round252_source": str(ROUND252_SOURCE),
            "round252_metric_summary": path_info(ROUND252_SOURCE / "metric_summary.tsv"),
            "round252_per_unit": path_info(ROUND252_SOURCE / "per_unit_metrics.tsv"),
            "process_path": path_info(PERTKGE_REPO / "processed_data" / "knowledge_graph" / "process.txt"),
            "pertkge_repo": str(PERTKGE_REPO),
        },
    }
    (WORKSPACE / "method_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")

    audit_lines = [
        "# Leakage and denominator audit",
        "",
        "- PertKGE target-inference protocol: training drugs use PC3 effect plus DTI; query drugs use target-cell effect with query DTI withheld.",
        "- Candidate space: 668 seen targets.",
        "- Official cells are taken from Round252 `official_test` drug-mean denominator.",
        "- Validation/test query `CID:` entities may appear through expression-derived effect triples only.",
        "- Validation/test `HAS_BINDING_TO` target edges are not training facts.",
        "",
        "## Per-cell audit",
        "",
        md_table(denom.fillna("")),
    ]
    (qa_dir / "leakage_and_denominator_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")

    report_lines = [
        "# Cross-cell PertKGE results",
        "",
        "## Comparison setup",
        "",
        "Evaluate PertKGE in held-out cell lines. Training compounds use PC3 expression-derived effect triples and known DTI labels; query compounds use expression-derived effect triples from the evaluation cell line with their DTI labels reserved for evaluation.",
        "",
        "## Results",
        "",
        md_table(comp.fillna("")) if not comp.empty else "No completed training results are available.",
        "",
        "## Evaluation protocol",
        "",
        "- `round307_pertkge_denominator_and_leakage_audit.tsv` records query coverage and checks the separation of training and evaluation DTI labels for each cell line.",
        "- `cell_unweighted_mean` is the primary cross-cell summary; `drugcell_weighted_mean` is an additional diagnostic.",
    ]
    (WORKSPACE / "comparison_summary.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"completed={completed} missing={missing}")
    print(f"wrote {source_dir}")


def run_cell(cell_line: str, args: argparse.Namespace) -> None:
    if cell_line not in official_cells():
        raise SystemExit(f"{cell_line!r} is not in Round252 official held-out cells")
    run_dir = WORKSPACE / "official_cells" / f"cell_{cell_line}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(run_dir / "train.log", mode="w"), logging.StreamHandler(sys.stdout)],
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    logging.info("Round307 start cell=%s device=%s cuda_visible=%s", cell_line, device, os.environ.get("CUDA_VISIBLE_DEVICES"))
    ctx = build_cell_context(cell_line)
    file_info = write_pertkge_files(ctx, run_dir)
    result = train_and_score(ctx, run_dir, args.max_epochs, args.patience, args.eval_every, args.warmup, device, args.seed)
    lock = {
        "round": "round307_pertkge_pc3train_targetcell_queryeffect_crosscell",
        "created_at": datetime.now().isoformat(),
        "cell_line": cell_line,
        "candidate_space": 668,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "eval_every": args.eval_every,
        "warmup": args.warmup,
        "top_genes_effect": TOP_GENES_EFFECT,
        "batch_size": ROUND88_BATCH_SIZE,
        "protocol": {
            "name": "PertKGE cross-cell target inference with PC3 training-drug effect and target-cell query-drug effect",
            "cause_edges": "inner-train drug HAS_BINDING_TO target edges only",
            "effect_edges": "inner-train drugs use PC3 expression; validation/test query drugs use the target held-out cell expression",
            "withheld_edges": "validation/test query drug HAS_BINDING_TO target edges are excluded from training and used only for validation/test evaluation",
            "scoring": "score CID:drug against Protein targets with DistMult HAS_BINDING_TO relation after training",
            "primary_evaluation_unit": "drug-mean / drug-cell",
        },
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "inputs": {
            "expr_path": path_info(EXPR_PATH),
            "lock_path": path_info(LOCK_PATH),
            "round252_source": str(ROUND252_SOURCE),
            "process_path": path_info(PERTKGE_REPO / "processed_data" / "knowledge_graph" / "process.txt"),
            "pertkge_repo": str(PERTKGE_REPO),
        },
        "file_info": file_info,
        "result": result,
    }
    (run_dir / "method_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    logging.info("complete %s", json.dumps(result, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-line", default=None)
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=ROUND88_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=ROUND88_PATIENCE)
    parser.add_argument("--eval-every", type=int, default=ROUND88_EVAL_EVERY)
    parser.add_argument("--warmup", type=int, default=ROUND88_WARMUP)
    parser.add_argument("--seed", type=int, default=ROUND88_SEED)
    args = parser.parse_args()
    if args.preflight:
        preflight()
        return
    if args.summarize:
        summarize_runs()
        return
    cells = official_cells() if args.all_cells else ([args.cell_line] if args.cell_line else [])
    if not cells:
        raise SystemExit("--cell-line or --all-cells required unless --preflight or --summarize")
    for cell in cells:
        run_cell(cell, args)


if __name__ == "__main__":
    main()
