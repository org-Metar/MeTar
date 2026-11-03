#!/usr/bin/env python3
"""Score original-inductive PertKGE on Round259 full-space holdout rows.

This is a Round264-only scorer. It reuses the original PertKGE inductive
prediction rule from the old H07 comparison, but changes the evaluation
denominator to the Round259/Round252 cross-cell protocol:

- pseudo-unseen cells: model-selection denominator, reported for audit only;
- official unseen cells: final Fig.3 paired denominator;
- target labels are used only for evaluation.

The output intentionally stays as native-universe percentiles. Round259 ranks
are in the 16,583-gene universe, while PertKGE ranks are in its own Protein:
candidate universe.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REMOTE_WRITE_ROOT = Path("/home/lwn/claude/manuscript/write")
ROUND264_ROOT = REMOTE_WRITE_ROOT / "claude_round264_fig2_fig3_baseline_completion_20260522"
SOURCE = ROUND264_ROOT / "source_data"
QA = ROUND264_ROOT / "qa"
LOGS = ROUND264_ROOT / "logs"

ROUND259_SCRIPT = (
    REMOTE_WRITE_ROOT
    / "claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522"
    / "scripts"
    / "run_round259_crosscell_fullspace_round252query.py"
)
ROUND259_SOURCE = (
    REMOTE_WRITE_ROOT
    / "claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522"
    / "source_data"
)
ROUND263_SOURCE = (
    REMOTE_WRITE_ROOT
    / "claude_round263_fig3_fig5_latest_source_data_completion_20260522"
    / "source_data"
)
BASELINE_DIR = Path("/home/lwn/claude/baseline")
PERTKGE_ROOT = BASELINE_DIR / "pertkge_code" / "PertKGE-main"
PERTKGE_MODEL = PERTKGE_ROOT / "best_model" / "ours" / "best_model.pth"
GENEINFO_PATH = BASELINE_DIR / "geneinfo_beta.txt"

THRESHOLDS = [0.01, 0.05, 0.10, 0.20, 0.50]


def ensure_dirs() -> None:
    for path in [SOURCE, QA, LOGS]:
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_round259_holdouts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    round259 = load_module(ROUND259_SCRIPT, "round259_crosscell")
    round230 = round259.load_round230_module()
    round230.ROUND_ROOT = ROUND264_ROOT / "round230_import_tmp"
    round230.SOURCE = ROUND264_ROOT / "round230_import_tmp" / "source_data"
    round230.QA = ROUND264_ROOT / "round230_import_tmp" / "qa"
    os.environ.setdefault("METAR_BULK_PROJECT_ROOT", str(ROUND264_ROOT / "h07_import_tmp"))

    _wrapper, data, protocol = round230.load_h07_data()
    protocol_audit = round259.load_round252_protocol_audit()
    _splits, holdouts, split_counts = round259.build_round259_splits(round230, data, protocol_audit)
    return data, holdouts, split_counts, protocol


def load_pertkge(device: torch.device):
    sys.path.insert(0, str(PERTKGE_ROOT / "src"))
    from torchkge import DistMultModel, KnowledgeGraph
    from utils import generate_five_fold_files, read_files, set_seeds

    class PKGEArgs:
        cause_file = str(PERTKGE_ROOT / "processed_data" / "target_inference" / "cause.txt")
        process_file = str(PERTKGE_ROOT / "processed_data" / "knowledge_graph" / "process.txt")
        effect_file = str(PERTKGE_ROOT / "processed_data" / "target_inference" / "effect.txt")
        test_file = str(PERTKGE_ROOT / "processed_data" / "target_inference" / "test.txt")
        load_processed_data = False
        processed_data_file = str(PERTKGE_ROOT / "processed_data" / "target_inference") + "/"
        seed = 43
        h_dim = 300
        margin = 1.0
        lr = 1e-4
        wd = 1e-5
        n_neg = 100
        batch_size = 2048
        warm_up = 10
        patients = 5
        nepoch = 100
        save_model = True
        use_cuda = str()
        save_model_path = str(PERTKGE_ROOT / "best_model" / "ours") + "/"
        task = "target_inference"
        run_name = "ours"
        mode = "reproduce"
        skip_train = True

    set_seeds(PKGEArgs.seed)
    args = PKGEArgs()
    cause, pertkg_wo_cause, _test, ent2id, rel2id, _pro2nc, _h_cand, _t_cand = read_files(args)
    fft, _ffv = generate_five_fold_files(args, cause)
    full_kg = pd.concat([fft[0], pertkg_wo_cause], ignore_index=True)
    kg = KnowledgeGraph(df=full_kg, ent2ix=ent2id, rel2ix=rel2id)
    model = DistMultModel(args.h_dim, kg.n_ent, kg.n_rel).to(device)
    model.load_state_dict(torch.load(PERTKGE_MODEL, map_location=device))
    model.normalize_parameters()
    model.eval()
    with torch.no_grad():
        ent_emb, rel_emb = model.get_embeddings()
    return ent2id, rel2id, ent_emb, rel_emb


def build_expr_mapping(expr_cols: list[str], ent2id: dict[str, int]) -> tuple[np.ndarray, torch.Tensor, list[str]]:
    geneinfo = pd.read_csv(GENEINFO_PATH, sep="\t", usecols=["gene_id", "gene_symbol"], dtype=str)
    entrez2sym = dict(zip(geneinfo["gene_id"].str.strip(), geneinfo["gene_symbol"].str.strip()))
    positions: list[int] = []
    entity_ids: list[int] = []
    symbols: list[str] = []
    for idx, col in enumerate(expr_cols):
        sym = entrez2sym.get(str(col).strip())
        if not sym:
            continue
        entity = f"mRNA:{sym}"
        if entity in ent2id:
            positions.append(idx)
            entity_ids.append(ent2id[entity])
            symbols.append(sym)
    return np.asarray(positions, dtype=np.int64), torch.tensor(entity_ids, dtype=torch.long), symbols


def build_protein_mapping(ent2id: dict[str, int]) -> tuple[list[str], torch.Tensor, dict[str, int]]:
    protein_items = [(entity[8:], idx) for entity, idx in ent2id.items() if entity.startswith("Protein:")]
    protein_genes = [gene for gene, _idx in protein_items]
    protein_entity_idx = torch.tensor([idx for _gene, idx in protein_items], dtype=torch.long)
    protein2idx = {gene: idx for idx, gene in enumerate(protein_genes)}
    return protein_genes, protein_entity_idx, protein2idx


def topk_hit_columns(rank: int, n_candidates: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_space": int(n_candidates),
        "best_percentile": float(rank / n_candidates),
    }
    for pct in [1, 5, 10, 20, 50]:
        cutoff = max(1, int(math.floor(n_candidates * (pct / 100.0))))
        row[f"hit_top{pct}pct"] = int(rank <= cutoff)
        row[f"top{pct}pct_cutoff_rank"] = cutoff
    return row


@torch.no_grad()
def score_bundle(
    bundle: dict[str, Any],
    split_name: str,
    expr_positions: np.ndarray,
    mrna_embs: torch.Tensor,
    r_up: torch.Tensor,
    r_dn: torch.Tensor,
    r_bind: torch.Tensor,
    target_embs: torch.Tensor,
    gene_list: list[str],
    protein2idx: dict[str, int],
    device: torch.device,
    batch_size: int,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = bundle["df"].reset_index(drop=True)
    x = bundle["x"]
    true_indices = bundle["true_indices"]
    n_targets = int(target_embs.shape[0])

    profile_rows: list[dict[str, Any]] = []
    group_emb_sum: dict[tuple[str, str], np.ndarray] = {}
    group_n: dict[tuple[str, str], int] = defaultdict(int)
    group_true: dict[tuple[str, str], set[int]] = defaultdict(set)
    drug_emb_sum: dict[str, np.ndarray] = {}
    drug_n: dict[str, int] = defaultdict(int)
    drug_true: dict[str, set[int]] = defaultdict(set)
    profile_percentiles_by_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    profile_percentiles_by_drug: dict[str, list[float]] = defaultdict(list)

    skipped_no_pertkge_target = 0
    target_count_dist: dict[int, int] = defaultdict(int)
    k = min(top_n, len(expr_positions))
    if k <= 0:
        raise RuntimeError("No PertKGE-mapped expression features available")

    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        xb_np = x[start:end, :][:, expr_positions].astype(np.float32, copy=False)
        xb = torch.from_numpy(xb_np).to(device)
        top_idx = torch.topk(xb, k, dim=1).indices
        bot_idx = torch.topk(-xb, k, dim=1).indices
        up_embs = mrna_embs[top_idx]
        dn_embs = mrna_embs[bot_idx]
        pert_emb = (r_up.view(1, 1, -1) * up_embs).mean(dim=1) - (
            r_dn.view(1, 1, -1) * dn_embs
        ).mean(dim=1)
        logits = (pert_emb * r_bind.view(1, -1)) @ target_embs.t()
        pert_emb_np = pert_emb.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()

        for j, row_idx in enumerate(range(start, end)):
            true_genes = [gene_list[int(t)] for t in true_indices[row_idx]]
            true_pkge = sorted({protein2idx[g] for g in true_genes if g in protein2idx})
            target_count_dist[len(true_pkge)] += 1
            if not true_pkge:
                skipped_no_pertkge_target += 1
                continue

            true_scores = logits_np[j, true_pkge]
            best_true_score = float(np.max(true_scores))
            rank = int(np.sum(logits_np[j] > best_true_score) + 1)
            pct = rank / n_targets
            drug = str(df.iloc[row_idx]["drug"])
            cell_line = str(df.iloc[row_idx].get("cell_line", ""))

            row = {
                "split": split_name,
                "row_index": int(row_idx),
                "drug": drug,
                "cell_line": cell_line,
                "n_true_targets": int(len(true_indices[row_idx])),
                "n_true_targets_in_pertkge": int(len(true_pkge)),
                "true_targets_in_pertkge": "|".join([g for g in true_genes if g in protein2idx]),
                "best_rank": rank,
                "all_true_native_indices": "|".join(map(str, true_pkge)),
            }
            row.update(topk_hit_columns(rank, n_targets))
            profile_rows.append(row)

            group_key = (drug, cell_line)
            if group_key not in group_emb_sum:
                group_emb_sum[group_key] = np.zeros(pert_emb_np.shape[1], dtype=np.float64)
            group_emb_sum[group_key] += pert_emb_np[j].astype(np.float64)
            group_n[group_key] += 1
            group_true[group_key].update(true_pkge)
            profile_percentiles_by_group[group_key].append(pct)

            if drug not in drug_emb_sum:
                drug_emb_sum[drug] = np.zeros(pert_emb_np.shape[1], dtype=np.float64)
            drug_emb_sum[drug] += pert_emb_np[j].astype(np.float64)
            drug_n[drug] += 1
            drug_true[drug].update(true_pkge)
            profile_percentiles_by_drug[drug].append(pct)

    def score_mean_group(
        group_type: str,
        drug: str,
        cell_line: str,
        emb_sum: np.ndarray,
        n_profiles: int,
        true_idx: set[int],
        pcts: list[float],
    ) -> dict[str, Any]:
        mean_emb = emb_sum / max(1, n_profiles)
        mean_logits = (
            torch.tensor(mean_emb, dtype=torch.float32, device=device).unsqueeze(0) * r_bind.view(1, -1)
        ) @ target_embs.t()
        mean_logits_np = mean_logits.squeeze(0).detach().cpu().numpy()
        true_sorted = sorted(true_idx)
        best_true_score = float(np.max(mean_logits_np[true_sorted]))
        rank = int(np.sum(mean_logits_np > best_true_score) + 1)
        out = {
            "split": split_name,
            "group_type": group_type,
            "drug": drug,
            "cell_line": cell_line,
            "n_profiles_in_group": int(n_profiles),
            "n_true_targets_union_in_pertkge": int(len(true_sorted)),
            "best_rank": rank,
            "median_profile_best_percentile": float(np.median(np.asarray(pcts, dtype=float))) if pcts else np.nan,
            "best_profile_best_percentile": float(np.min(np.asarray(pcts, dtype=float))) if pcts else np.nan,
            "frac_profiles_top5pct": float(np.mean(np.asarray(pcts, dtype=float) <= 0.05)) if pcts else np.nan,
        }
        out.update(topk_hit_columns(rank, n_targets))
        return out

    group_rows: list[dict[str, Any]] = []
    for (drug, cell_line), emb_sum in group_emb_sum.items():
        group_rows.append(
            score_mean_group(
                "drug_cell_mean",
                drug,
                cell_line,
                emb_sum,
                group_n[(drug, cell_line)],
                group_true[(drug, cell_line)],
                profile_percentiles_by_group[(drug, cell_line)],
            )
        )
    for drug, emb_sum in drug_emb_sum.items():
        group_rows.append(
            score_mean_group(
                "drug_mean_across_cells",
                drug,
                "",
                emb_sum,
                drug_n[drug],
                drug_true[drug],
                profile_percentiles_by_drug[drug],
            )
        )

    meta = {
        "split": split_name,
        "input_profile_rows": int(len(x)),
        "evaluable_profile_rows": int(len(profile_rows)),
        "skipped_no_pertkge_target": int(skipped_no_pertkge_target),
        "target_count_dist_in_pertkge": {str(k): int(v) for k, v in sorted(target_count_dist.items())},
        "evaluable_drug_cell_groups": int(sum(1 for row in group_rows if row["group_type"] == "drug_cell_mean")),
        "evaluable_drugs": int(sum(1 for row in group_rows if row["group_type"] == "drug_mean_across_cells")),
    }
    return pd.DataFrame(profile_rows), pd.DataFrame(group_rows), meta


def summarize_thresholds(name: str, df: pd.DataFrame, n_candidates: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        hits = int((df["best_percentile"] <= threshold).sum())
        rows.append(
            {
                "metric_set": name,
                "threshold": float(threshold),
                "hits": hits,
                "n": int(len(df)),
                "rate": float(hits / len(df)) if len(df) else np.nan,
                "rank_cutoff": max(1, int(math.floor(n_candidates * threshold))),
                "candidate_universe": "PertKGE Protein native universe",
                "n_candidates": int(n_candidates),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[round264-pertkge] device={device}", flush=True)

    data, holdouts, split_counts, protocol = load_round259_holdouts()
    gene_list = [str(g) for g in data["gene_list"]]
    expr_cols = [str(c) for c in data["expr_cols"]]

    print("[round264-pertkge] loading PertKGE", flush=True)
    ent2id, rel2id, ent_emb, rel_emb = load_pertkge(device)
    expr_positions, expr_entity_ids, _expr_symbols = build_expr_mapping(expr_cols, ent2id)
    protein_genes, protein_entity_idx, protein2idx = build_protein_mapping(ent2id)
    if len(expr_positions) == 0:
        raise RuntimeError("No expression features mapped to PertKGE mRNA entities")
    print(
        f"[round264-pertkge] mapped expression features={len(expr_positions)}/{len(expr_cols)}; "
        f"proteins={len(protein_genes)}",
        flush=True,
    )

    r_bind = rel_emb[rel2id["HAS_BINDING_TO"]].to(device)
    r_up = rel_emb[rel2id["Upregulates"]].to(device)
    r_dn = rel_emb[rel2id["Downregulates"]].to(device)
    mrna_embs = ent_emb[expr_entity_ids.to(device)].to(device)
    target_embs = ent_emb[protein_entity_idx.to(device)].to(device)

    split_map = {
        "holdout_val": "pseudo_unseen_selection",
        "holdout_test": "official_unseen_final",
    }
    profile_parts: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    for split_name, role in split_map.items():
        print(f"[round264-pertkge] scoring {split_name}", flush=True)
        profiles, groups, meta = score_bundle(
            holdouts[split_name],
            split_name,
            expr_positions,
            mrna_embs,
            r_up,
            r_dn,
            r_bind,
            target_embs,
            gene_list,
            protein2idx,
            device,
            batch_size=int(args.batch_size),
            top_n=int(args.top_n),
        )
        profiles.insert(1, "cell_split_role", role)
        groups.insert(1, "cell_split_role", role)
        profile_parts.append(profiles)
        group_parts.append(groups)
        meta["cell_split_role"] = role
        metas.append(meta)
        summary_rows.extend(
            summarize_thresholds(
                f"PertKGE_original_inductive_{split_name}_profile",
                profiles,
                len(protein_genes),
            )
        )
        for group_type, gdf in groups.groupby("group_type", sort=False):
            summary_rows.extend(
                summarize_thresholds(
                    f"PertKGE_original_inductive_{split_name}_{group_type}",
                    gdf,
                    len(protein_genes),
                )
            )

    profile_all = pd.concat(profile_parts, ignore_index=True)
    group_all = pd.concat(group_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    profile_all.to_csv(SOURCE / "round264_pertkge_on_round259_holdout_profile_percentiles.tsv.gz", sep="\t", index=False, compression="gzip")
    group_all.to_csv(SOURCE / "round264_pertkge_on_round259_holdout_group_percentiles.tsv", sep="\t", index=False)
    summary.to_csv(SOURCE / "round264_pertkge_on_round259_threshold_summary.tsv", sep="\t", index=False)
    audit = {
        "round": "Round264 PertKGE original-inductive scoring on Round259 rows",
        "round259_script": str(ROUND259_SCRIPT),
        "round259_script_sha256": sha256_file(ROUND259_SCRIPT),
        "round259_source": str(ROUND259_SOURCE),
        "round263_source": str(ROUND263_SOURCE),
        "pertkge_root": str(PERTKGE_ROOT),
        "pertkge_model": str(PERTKGE_MODEL),
        "pertkge_model_sha256": sha256_file(PERTKGE_MODEL),
        "geneinfo": str(GENEINFO_PATH),
        "geneinfo_sha256": sha256_file(GENEINFO_PATH),
        "expression_features_total": int(len(expr_cols)),
        "expression_features_mapped_to_pertkge": int(len(expr_positions)),
        "pertkge_candidate_universe": "Protein entities from PertKGE ent2id",
        "pertkge_n_candidates": int(len(protein_genes)),
        "scoring_rule": "Original PertKGE inductive rule: top/bottom expression features build a temporary perturbation embedding with Upregulates/Downregulates; protein targets are scored by HAS_BINDING_TO DistMult.",
        "top_n": int(args.top_n),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "split_counts": split_counts,
        "protocol_keys": sorted(protocol.keys()) if isinstance(protocol, dict) else [],
        "split_meta": metas,
    }
    (SOURCE / "round264_pertkge_on_round259_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print("[round264-pertkge] done", flush=True)


if __name__ == "__main__":
    main()
