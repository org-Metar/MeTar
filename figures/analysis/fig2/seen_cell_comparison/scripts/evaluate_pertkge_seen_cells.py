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
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
import torch

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')

SCRIPT_PATH = Path(__file__).resolve()
WORKSPACE = SCRIPT_PATH.parents[1]
DEFAULT_REMOTE_ROOT = Path('/home/lwn/claude')
LOCAL_ROOT = WORKSPACE.parent

BASELINE_DIR = Path(os.environ.get('METAR_BASELINE_DIR', str(DEFAULT_REMOTE_ROOT / 'baseline')))
PERTKGE_REPO = Path(os.environ.get('PERTKGE_REPO', str(BASELINE_DIR / 'pertkge_code' / 'PertKGE-main')))
DATA_DIR = Path(os.environ.get('METAR_BULK_DATA_DIR', str(DEFAULT_REMOTE_ROOT / 'bulk' / 'data')))
ROUND87 = Path(os.environ.get('METAR_ROUND87_DIR', str(DEFAULT_REMOTE_ROOT / 'manuscript' / 'write' / 'unifiedlabel_figures_20260331' / 'Figure2_r1')))
METAR_RUN = Path(os.environ.get('METAR_RUN_DIR', str(DEFAULT_REMOTE_ROOT / 'manuscript' / 'write' / 'bulk_sharedsample_indunseenlabel_20260330_h07_w060_u8e4_med')))
LOCK_PATH = METAR_RUN / 'manifests' / 'bulk_protocol_lock_v2.json'
EXPR_PATH = DATA_DIR / 'large_data.parquet'
METAR_SCORE_PATH = ROUND87 / 'metar_scores_v4.parquet'
ROUND252_SOURCE = Path(os.environ.get('ROUND252_SOURCE', str(LOCAL_ROOT / 'claude_round252_epiqgate_pseudoval_sweep_20260520' / 'source_data')))
ROUND88_SOURCE = Path(os.environ.get('ROUND88_SOURCE', str(LOCAL_ROOT / 'claude_round88_pertkge_fixed_common_intersection_20260506' / 'source_data')))
ROUND91_SOURCE = Path(os.environ.get('ROUND91_SOURCE', str(LOCAL_ROOT / 'claude_round91_pertkge_inductive_single_cellline_20260507')))

HIT_KS = [1, 5, 10, 20, 30, 50]
TOP_GENES_EFFECT = 200
DEFAULT_SELECTED_CELL_LINES = ['PC3', 'MCF7', 'A375', 'A549', 'HEPG2', 'NPC', 'U2OS', 'YAPC', 'JURKAT', 'ASC', 'HEC108', 'MDAMB468']
SELECTED_CELL_LINES = DEFAULT_SELECTED_CELL_LINES.copy()
ROUND88_MAX_EPOCHS = 80
ROUND88_PATIENCE = 5
ROUND88_EVAL_EVERY = 5
ROUND88_WARMUP = 10
ROUND88_BATCH_SIZE = 4096
ROUND88_SEED = 43
ROUND252_SELECTED_SEEN_VARIANT = 'r252_epiqgate_lm1_lr0p00015_sc0p18_mn2p5_ba0p1_s32_q32_diverse_label_seed20260524_seen_diagnostic_diverse_label_s64'
EFFECT_SCOPES = ['same_cell']


def load_selected_cell_lines(cells_file: str | None) -> List[str]:
    if cells_file is None:
        default_path = WORKSPACE / 'selected_seen_cells.txt'
        if default_path.exists():
            cells_file = str(default_path)
        else:
            return DEFAULT_SELECTED_CELL_LINES.copy()
    cells = [line.strip() for line in Path(cells_file).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not cells:
        raise ValueError(f'No cells found in {cells_file}')
    return cells


def split_targets(value: str) -> Iterable[str]:
    for g in str(value).split('|'):
        g = g.strip()
        if g and g not in {'0', '-666', 'nan', 'None'}:
            yield g


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics_per_sample(scores_df: pd.DataFrame, targets_map: Dict[str, List[str]], labels: List[str], method: str) -> pd.DataFrame:
    records = []
    label_set = set(labels)
    for _, row in scores_df.iterrows():
        true_tgts = set(targets_map.get(row['drug'], [])) & label_set
        if not true_tgts:
            continue
        scores = np.array([float(row[g]) if g in row else 0.0 for g in labels], dtype=np.float64)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(scores)[::-1]
        ranked = [labels[i] for i in order]
        rec = {'method': method, 'profile_id': row['profile_id'], 'drug': row['drug'], 'cell_line': row['cell_line']}
        for k in HIT_KS:
            rec[f'Hit@{k}'] = int(bool(set(ranked[:k]) & true_tgts))
        y = np.array([1 if g in true_tgts else 0 for g in labels], dtype=np.int8)
        rec['AUROC'] = float(roc_auc_score(y, scores)) if 0 < int(y.sum()) < len(y) else 0.5
        rec['NDCG@10'] = float(ndcg_score([y], [scores], k=10))
        rec['MeanAP'] = float(average_precision_score(y, scores))
        best_rank = min(ranked.index(t) + 1 for t in true_tgts)
        rec['best_true_target_percentile'] = 100.0 * (1.0 - best_rank / len(labels))
        records.append(rec)
    return pd.DataFrame(records)


def summarize_metric_frame(metrics: pd.DataFrame, method: str, setting: str, unit: str, cell_line: str) -> dict:
    row = {
        'cell_line': cell_line,
        'setting': setting,
        'evaluation_unit': unit,
        'method': method,
        'n_profiles': int(metrics['profile_id'].nunique()) if 'profile_id' in metrics.columns else 0,
        'n_drugs': int(metrics['drug'].nunique()) if 'drug' in metrics.columns else 0,
    }
    for k in HIT_KS:
        row[f'Hit@{k}'] = float(metrics[f'Hit@{k}'].mean()) if f'Hit@{k}' in metrics else float('nan')
    for m in ['AUROC', 'NDCG@10', 'MeanAP', 'best_true_target_percentile']:
        row[m] = float(metrics[m].mean()) if m in metrics else float('nan')
    return row


def collapse_profile_metrics_to_drug_mean(profile_metrics: pd.DataFrame, method: str) -> pd.DataFrame:
    metric_cols = [f'Hit@{k}' for k in HIT_KS] + ['AUROC', 'NDCG@10', 'MeanAP', 'best_true_target_percentile']
    rows = []
    for (drug, cell), grp in profile_metrics.groupby(['drug', 'cell_line'], sort=False):
        rec = {
            'method': method,
            'profile_id': drug,
            'drug': drug,
            'cell_line': cell,
            'n_profiles': int(len(grp)),
        }
        for col in metric_cols:
            rec[col] = float(grp[col].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f'{float(val):.6g}')
            else:
                vals.append(str(val))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def path_info(path: Path) -> dict:
    info = {'path': str(path), 'exists': path.exists()}
    if path.exists() and path.is_file():
        info['sha256'] = sha256_file(path)
        info['size_bytes'] = path.stat().st_size
    return info


def add_random_summary(rows: List[dict], cell_line: str, n_drugs: int, n_profiles: int, setting: str, unit: str) -> None:
    row = {
        'cell_line': cell_line,
        'setting': setting,
        'evaluation_unit': unit,
        'method': 'Random',
        'n_profiles': int(n_profiles),
        'n_drugs': int(n_drugs),
    }
    for k in HIT_KS:
        row[f'Hit@{k}'] = min(float(k) / 668.0, 1.0)
    row['AUROC'] = 0.5
    row['NDCG@10'] = float('nan')
    row['MeanAP'] = float('nan')
    row['best_true_target_percentile'] = 50.0
    rows.append(row)


def safe_read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep='\t')


def build_context(cell_line: str) -> dict:
    protocol = json.loads(LOCK_PATH.read_text())
    inner_train = set(protocol['drug_split']['inner_train_drugs'])
    val_drugs = set(protocol['drug_split']['val_drugs'])
    test_drugs = set(protocol['drug_split']['test_drugs'])

    cinfo = pd.read_csv(DATA_DIR / 'compoundinfo_beta.txt', sep='\t')
    cinfo = cinfo[cinfo['target'].notna() & ~cinfo['target'].isin(['0', '', '-666'])]
    inner_cinfo = cinfo[cinfo['pert_id'].isin(inner_train)]
    labels = sorted({g for t in inner_cinfo['target'].dropna() for g in split_targets(t)})
    if len(labels) != 668:
        raise ValueError(f'Expected 668 labels, observed {len(labels)}')
    label_set = set(labels)

    drug_targets = {}
    for d, grp in cinfo.groupby('pert_id'):
        tgts = sorted({g for t in grp['target'].astype(str) for g in split_targets(t)})
        if tgts:
            drug_targets[d] = tgts

    seen_clean_test = sorted(d for d in test_drugs if d in drug_targets and set(drug_targets[d]) and set(drug_targets[d]).issubset(label_set))
    seen_clean_val = sorted(d for d in val_drugs if d in drug_targets and set(drug_targets[d]) and set(drug_targets[d]).issubset(label_set))

    expr = pd.read_parquet(EXPR_PATH).reset_index()
    expr = expr.rename(columns={expr.columns[0]: 'profile_id'})
    gene_cols = [c for c in expr.columns if c not in ('profile_id', 'drug', 'cell_line', 'method')]
    expr_cp = expr[expr['method'] == 0].copy()
    canonical = expr_cp[(expr_cp['drug'].isin(seen_clean_test)) & (expr_cp['cell_line'] == cell_line)][['profile_id', 'drug', 'cell_line', *gene_cols]].copy()
    canonical = canonical.sort_values('profile_id').reset_index(drop=True)
    if canonical.empty:
        raise ValueError(f'No canonical test profiles for {cell_line}')

    geneinfo = pd.read_csv(BASELINE_DIR / 'geneinfo_beta.txt', sep='\t', usecols=['gene_id', 'gene_symbol'], dtype=str)
    entrez2sym = dict(zip(geneinfo['gene_id'].str.strip(), geneinfo['gene_symbol'].str.strip()))
    return {
        'protocol': protocol,
        'cell_line': cell_line,
        'inner_train_drugs': inner_train,
        'val_drugs': val_drugs,
        'test_drugs': test_drugs,
        'labels': labels,
        'drug_targets': drug_targets,
        'seen_clean_test_drugs': seen_clean_test,
        'seen_clean_val_drugs': seen_clean_val,
        'expr_cp': expr_cp,
        'canonical': canonical,
        'gene_cols': gene_cols,
        'entrez2sym': entrez2sym,
    }


def write_pertkge_files(ctx: dict, run_dir: Path, effect_scope: str) -> dict:
    if effect_scope not in EFFECT_SCOPES:
        raise ValueError(f'unknown effect_scope={effect_scope!r}')
    ti = run_dir / 'pertkge' / 'target_inference'
    kg = run_dir / 'pertkge' / 'knowledge_graph'
    ti.mkdir(parents=True, exist_ok=True)
    kg.mkdir(parents=True, exist_ok=True)
    labels = set(ctx['labels'])
    drug_targets = ctx['drug_targets']

    cause_rows = []
    for drug in sorted(ctx['inner_train_drugs']):
        for tgt in drug_targets.get(drug, []):
            if tgt in labels:
                cause_rows.append(f'CID:{drug}\tHAS_BINDING_TO\tProtein:{tgt}')
    (ti / 'cause.txt').write_text('\n'.join(sorted(set(cause_rows))), encoding='utf-8')

    eval_drugs = sorted(set(ctx['seen_clean_val_drugs']) | set(ctx['seen_clean_test_drugs']))
    test_rows = []
    for drug in eval_drugs:
        for tgt in drug_targets.get(drug, []):
            if tgt in labels:
                test_rows.append(f'CID:{drug}\tHAS_BINDING_TO\tProtein:{tgt}')
    (ti / 'test.txt').write_text('\n'.join(sorted(set(test_rows))), encoding='utf-8')

    process_src = PERTKGE_REPO / 'processed_data' / 'knowledge_graph' / 'process.txt'
    shutil.copy2(process_src, kg / 'process.txt')

    # PertKGE target inference represents query compounds by adding their
    # perturbation-derived effect triples to effect.txt. These effect edges are
    # query information, not target-label edges; only HAS_BINDING_TO test edges
    # must be excluded from the training graph.
    effect_drugs = sorted(set(ctx['inner_train_drugs']) | set(ctx['seen_clean_val_drugs']) | set(ctx['seen_clean_test_drugs']))
    if effect_scope == 'same_cell':
        expr_for_effect = ctx['expr_cp'][(ctx['expr_cp']['drug'].isin(effect_drugs)) & (ctx['expr_cp']['cell_line'] == ctx['cell_line'])]
        effect_cell_line = ctx['cell_line']
        effect_scope_desc = 'same-cell; expression-derived effect triples use only the same cell line as evaluation'
    else:
        expr_for_effect = ctx['expr_cp'][
            (ctx['expr_cp']['drug'].isin(effect_drugs)) & (ctx['expr_cp']['cell_line'].isin(set(ctx['protocol']['cell_split']['train_cells'])))
        ]
        effect_cell_line = 'pooled_train_cells'
        effect_scope_desc = 'pooled; expression-derived effect triples use Round88 pooled train-cell expression'
    drug_mean = expr_for_effect.groupby('drug')[ctx['gene_cols']].mean()
    missing = sorted(set(effect_drugs) - set(drug_mean.index))
    # Single-cell-line PertKGE can only score query drugs with a signature in that cell line.
    usable_eval_drugs = sorted(set(drug_mean.index) & (set(ctx['seen_clean_val_drugs']) | set(ctx['seen_clean_test_drugs'])))
    effect_rows = []
    gene_cols = ctx['gene_cols']
    entrez2sym = ctx['entrez2sym']
    for drug_id, row_vals in drug_mean.iterrows():
        vals = row_vals.to_numpy(dtype=np.float32)
        order = np.argsort(vals)
        top_up = order[-TOP_GENES_EFFECT:]
        top_dn = order[:TOP_GENES_EFFECT]
        for idx in top_up:
            sym = entrez2sym.get(str(gene_cols[idx]).strip())
            if sym:
                effect_rows.append(f'CID:{drug_id}\tUpregulates\tmRNA:{sym}')
        for idx in top_dn:
            sym = entrez2sym.get(str(gene_cols[idx]).strip())
            if sym:
                effect_rows.append(f'CID:{drug_id}\tDownregulates\tmRNA:{sym}')
    (ti / 'effect.txt').write_text('\n'.join(sorted(set(effect_rows))), encoding='utf-8')

    cause_e = set(cause_rows)
    test_e = set(test_rows)
    process_df = pd.read_csv(kg / 'process.txt', sep='\t', names=['from', 'rel', 'to'])
    process_e = set(
        '\t'.join(x)
        for x in process_df[['from', 'rel', 'to']].astype(str).itertuples(index=False, name=None)
    )
    effect_df = pd.DataFrame(
        [x.split('\t') for x in sorted(set(effect_rows))],
        columns=['from', 'rel', 'to'],
    ) if effect_rows else pd.DataFrame(columns=['from', 'rel', 'to'])
    eval_cids = {f'CID:{d}' for d in set(ctx['seen_clean_val_drugs']) | set(ctx['seen_clean_test_drugs'])}
    effect_eval_cids = set(effect_df['from']) & eval_cids if len(effect_df) else set()
    process_has_binding = int((process_df['rel'] == 'HAS_BINDING_TO').sum())
    process_cid_edges = int(
        process_df['from'].astype(str).str.startswith('CID:').sum()
        + process_df['to'].astype(str).str.startswith('CID:').sum()
    )
    target_edge_leakage = len(test_e & (cause_e | process_e))
    if target_edge_leakage:
        raise RuntimeError(f'test target edges leaked into training graph: {target_edge_leakage}')
    if process_has_binding or process_cid_edges:
        raise RuntimeError(f'process graph contains forbidden target/CID edges: HAS_BINDING_TO={process_has_binding}, CID_edges={process_cid_edges}')

    audit = {
        'round': 'round450_fig2_seen_cellline_extension',
        'cell_line': ctx['cell_line'],
        'effect_scope': effect_scope,
        'protocol': 'round88_compatible_pertkge_effect_scope_singlevariable_protocol',
        'candidate_space': 668,
        'effect_cell_line': effect_cell_line,
        'allowed_query_effect_val_test_cids': len(effect_eval_cids),
        'forbidden_test_target_edges_in_cause_or_process': target_edge_leakage,
        'cause_test_target_edge_overlap': len(cause_e & test_e),
        'process_has_binding_to_edges': process_has_binding,
        'process_cid_edges': process_cid_edges,
        'effect_contains_eval_query_cids': bool(effect_eval_cids),
        'effect_drugs_requested': len(effect_drugs),
        'effect_drugs_with_cell_line_signature': int(len(drug_mean)),
        'missing_effect_drugs_count': len(missing),
        'missing_effect_drugs_examples': missing[:20],
        'usable_eval_drugs': len(usable_eval_drugs),
        'effect_scope_description': effect_scope_desc,
        'interpretation': 'val/test CID entities are expected because PertKGE target inference adds query effect triples; HAS_BINDING_TO target edges remain withheld.',
    }
    (run_dir / 'leakage_audit_pre_read.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')

    return {
        'cause_edges': len(set(cause_rows)),
        'test_edges': len(set(test_rows)),
        'effect_edges': len(set(effect_rows)),
        'effect_drugs_requested': len(effect_drugs),
        'effect_drugs_with_cell_line_signature': int(len(drug_mean)),
        'missing_effect_drugs': len(missing),
        'usable_eval_drugs': len(usable_eval_drugs),
        'allowed_query_effect_val_test_cids': len(effect_eval_cids),
        'forbidden_test_target_edges_in_cause_or_process': target_edge_leakage,
        'process_sha256': sha256_file(kg / 'process.txt'),
    }


def train_score(ctx: dict, run_dir: Path, effect_scope: str, max_epochs: int, patience_limit: int, eval_every: int, warmup: int, device: torch.device, seed: int) -> dict:
    sys.path.insert(0, str(PERTKGE_REPO / 'src'))
    from utils import read_files, set_seeds  # type: ignore
    from torchkge import DistMultModel, KnowledgeGraph  # type: ignore
    from torchkge.sampling import BernoulliNegativeSampler  # type: ignore
    from torchkge.utils import DataLoader as KGEDataLoader, MarginLoss  # type: ignore

    args = SimpleNamespace(
        cause_file=str(run_dir / 'pertkge' / 'target_inference' / 'cause.txt'),
        process_file=str(run_dir / 'pertkge' / 'knowledge_graph' / 'process.txt'),
        effect_file=str(run_dir / 'pertkge' / 'target_inference' / 'effect.txt'),
        test_file=str(run_dir / 'pertkge' / 'target_inference' / 'test.txt'),
        load_processed_data=False,
        processed_data_file=str(run_dir / 'pertkge' / 'target_inference') + '/',
        seed=seed,
        h_dim=300,
        margin=1.0,
        lr=1e-4,
        wd=1e-5,
        n_neg=100,
        batch_size=ROUND88_BATCH_SIZE,
        warm_up=warmup,
        patients=patience_limit,
        use_cuda='cuda' if device.type == 'cuda' else '',
        nepoch=max_epochs,
        save_model=True,
        save_model_path=str(run_dir / 'pertkge' / 'best_model') + '/',
        task='target_inference',
        run_name=f"round450_{effect_scope}_{ctx['cell_line']}",
        mode='reproduce',
        skip_train=False,
    )
    Path(args.save_model_path).mkdir(parents=True, exist_ok=True)
    set_seeds(args.seed)
    cause, pertkg_wo_cause, _test_df, ent2id, rel2id, _pro2nc, _h_cand, _t_cand = read_files(args)
    eval_cids = {f'CID:{d}' for d in set(ctx['seen_clean_val_drugs']) | set(ctx['seen_clean_test_drugs'])}
    eval_cids_in_ent2id = sorted(eval_cids & set(ent2id))
    post_read_audit = {
        'round': 'round450_fig2_seen_cellline_extension',
        'cell_line': ctx['cell_line'],
        'effect_scope': effect_scope,
        'protocol': 'round88_compatible_pertkge_effect_scope_singlevariable_protocol',
        'eval_cids_in_ent2id_count': len(eval_cids_in_ent2id),
        'eval_cids_in_ent2id_expected': True,
        'eval_cids_in_ent2id_examples': eval_cids_in_ent2id[:20],
        'target_label_leakage_rule': 'Only val/test HAS_BINDING_TO edges are forbidden; expression-derived effect edges for query drugs are allowed.',
    }
    (run_dir / 'leakage_audit_post_read.json').write_text(json.dumps(post_read_audit, indent=2), encoding='utf-8')
    full_kg = pd.concat([cause, pertkg_wo_cause], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    kg = KnowledgeGraph(df=full_kg, ent2ix=ent2id, rel2ix=rel2id)
    model = DistMultModel(args.h_dim, kg.n_ent, kg.n_rel).to(device)
    criterion = MarginLoss(args.margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sampler = BernoulliNegativeSampler(kg, n_neg=args.n_neg)
    loader = KGEDataLoader(kg, batch_size=args.batch_size, use_cuda=args.use_cuda)

    labels = ctx['labels']
    target_entities = [f'Protein:{g}' for g in labels]
    missing_targets = [x for x in target_entities if x not in ent2id]
    if missing_targets:
        raise ValueError(f'Missing target entities: {len(missing_targets)}')
    val_drugs = sorted([d for d in ctx['seen_clean_val_drugs'] if f'CID:{d}' in ent2id])
    test_drugs = sorted([d for d in ctx['canonical']['drug'].unique() if f'CID:{d}' in ent2id])
    if not val_drugs or not test_drugs:
        raise ValueError(f'No val/test drugs in ent2id for {ctx["cell_line"]}: val={len(val_drugs)} test={len(test_drugs)}')

    def score_drug_table(model_obj, drugs: List[str]) -> pd.DataFrame:
        model_obj.normalize_parameters()
        model_obj.eval()
        with torch.no_grad():
            ent_emb, rel_emb = model_obj.get_embeddings()
            r_emb = rel_emb[rel2id['HAS_BINDING_TO']].to(device)
            target_idx = torch.tensor([ent2id[x] for x in target_entities], dtype=torch.long, device=device)
            t_emb = ent_emb[target_idx]
            rows = []
            for drug in drugs:
                h_emb = ent_emb[ent2id[f'CID:{drug}']].to(device)
                score = torch.sigmoid((h_emb * r_emb * t_emb).sum(dim=1)).detach().cpu().numpy()
                rec = {'drug': drug}
                rec.update({g: float(score[i]) for i, g in enumerate(labels)})
                rows.append(rec)
        return pd.DataFrame(rows)

    def val_mean_ap(model_obj) -> float:
        scores = score_drug_table(model_obj, val_drugs)
        scores['profile_id'] = scores['drug']
        scores['cell_line'] = f'{ctx["cell_line"]}_VAL'
        scores = scores[['profile_id', 'drug', 'cell_line', *labels]]
        m = compute_metrics_per_sample(scores, ctx['drug_targets'], labels, 'PertKGE_val')
        return float(m['MeanAP'].mean())

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
        row = {'epoch': epoch, 'loss': total_loss / max(n_batches, 1), 'n_batches': n_batches, 'elapsed_sec': time.time() - start}
        if epoch >= warmup and epoch % eval_every == 0:
            cur = val_mean_ap(model)
            row['val_MeanAP'] = cur
            improved = cur > best_val + 1e-8
            if improved:
                best_val = cur
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            row['best_val_MeanAP'] = best_val
            row['patience'] = patience
            logging.info('epoch=%d loss=%.5f val_MeanAP=%.5f best=%.5f patience=%d', epoch, row['loss'], cur, best_val, patience)
            history.append(row)
            if patience >= patience_limit:
                logging.info('early stop at epoch=%d best_epoch=%d best_val=%.5f', epoch, best_epoch, best_val)
                break
        else:
            row['best_val_MeanAP'] = best_val
            row['patience'] = patience
            logging.info('epoch=%d loss=%.5f', epoch, row['loss'])
            history.append(row)
    if best_state is None:
        best_val = val_mean_ap(model)
        best_epoch = epoch
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, run_dir / 'checkpoint_best.pth')
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(run_dir / 'training_history.tsv', sep='\t', index=False)

    drug_scores = score_drug_table(model, test_drugs)
    canonical = ctx['canonical'][['profile_id', 'drug', 'cell_line']].copy()
    canonical = canonical[canonical['drug'].isin(test_drugs)].copy()
    pertkge_scores = canonical.merge(drug_scores, on='drug', how='left', validate='many_to_one')
    pertkge_scores.to_parquet(run_dir / 'pertkge_scores.parquet', index=False)

    metar = pd.read_parquet(METAR_SCORE_PATH)
    metar = metar[(metar['cell_line'] == ctx['cell_line']) & (metar['profile_id'].isin(set(canonical['profile_id'])))].copy()
    metar = metar[['profile_id', 'drug', 'cell_line', *labels]]
    pertkge_scores = pertkge_scores[['profile_id', 'drug', 'cell_line', *labels]]

    metar_metrics = compute_metrics_per_sample(metar, ctx['drug_targets'], labels, 'MeTar')
    pertkge_metrics = compute_metrics_per_sample(pertkge_scores, ctx['drug_targets'], labels, 'PertKGE')
    keys = ['profile_id', 'drug', 'cell_line']
    metric_cols = [f'Hit@{k}' for k in HIT_KS] + ['AUROC', 'NDCG@10', 'MeanAP', 'best_true_target_percentile']
    matched = metar_metrics[keys + metric_cols].rename(columns={c: f'{c}_metar' for c in metric_cols}).merge(
        pertkge_metrics[keys + metric_cols].rename(columns={c: f'{c}_pertkge' for c in metric_cols}),
        on=keys, how='inner', validate='one_to_one')
    matched.to_csv(run_dir / 'matched_profile_metrics.tsv', sep='\t', index=False)

    drug_rows = []
    for (drug, cell), grp in matched.groupby(['drug', 'cell_line'], sort=False):
        rec = {'profile_id': drug, 'drug': drug, 'cell_line': cell, 'n_profiles': int(len(grp))}
        for method in ['metar', 'pertkge']:
            for col in metric_cols:
                rec[f'{col}_{method}'] = float(grp[f'{col}_{method}'].mean())
        drug_rows.append(rec)
    matched_drug = pd.DataFrame(drug_rows)
    matched_drug.to_csv(run_dir / 'matched_drug_mean_metrics.tsv', sep='\t', index=False)

    rows = []
    for matched_df, setting, unit in [
        (matched_drug, 'drug_mean_primary', 'drug-mean'),
        (matched, 'profile_replicated_diagnostic', 'profile-replicated'),
    ]:
        for method in ['metar', 'pertkge']:
            row = {
                'cell_line': ctx['cell_line'],
                'setting': setting,
                'evaluation_unit': unit,
                'method': 'MeTar' if method == 'metar' else 'PertKGE',
                'n_profiles': int(matched_df['n_profiles'].sum()) if 'n_profiles' in matched_df.columns else len(matched_df),
                'n_drugs': int(matched_df['drug'].nunique()),
            }
            for k in HIT_KS:
                row[f'Hit@{k}'] = float(matched_df[f'Hit@{k}_{method}'].mean())
            for m in ['AUROC', 'NDCG@10', 'MeanAP', 'best_true_target_percentile']:
                row[m] = float(matched_df[f'{m}_{method}'].mean())
            rows.append(row)
        add_random_summary(
            rows,
            ctx['cell_line'],
            int(matched_df['drug'].nunique()),
            int(matched_df['n_profiles'].sum()) if 'n_profiles' in matched_df.columns else len(matched_df),
            setting,
            unit,
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(run_dir / 'metric_summary.tsv', sep='\t', index=False)

    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {
        'best_epoch': best_epoch,
        'best_val_MeanAP': best_val,
        'last_epoch': int(pd.DataFrame(history)['epoch'].max()),
        'kg_entities': len(ent2id),
        'kg_relations': len(rel2id),
        'kg_facts': int(kg.n_facts),
        'val_drugs_in_ent2id': len(val_drugs),
        'test_drugs_in_ent2id': len(test_drugs),
        'n_matched_profiles': len(matched),
        'n_matched_drugs': matched['drug'].nunique(),
        'n_matched_drug_mean_units': len(matched_drug),
    }


def preflight() -> None:
    rows = []
    for c in SELECTED_CELL_LINES:
        ctx = build_context(c)
        effect_drugs = set(ctx['inner_train_drugs']) | set(ctx['seen_clean_val_drugs']) | set(ctx['seen_clean_test_drugs'])
        for effect_scope in EFFECT_SCOPES:
            if effect_scope == 'same_cell':
                expr_for_effect = ctx['expr_cp'][(ctx['expr_cp']['drug'].isin(effect_drugs)) & (ctx['expr_cp']['cell_line'] == c)]
            else:
                expr_for_effect = ctx['expr_cp'][
                    (ctx['expr_cp']['drug'].isin(effect_drugs)) & (ctx['expr_cp']['cell_line'].isin(set(ctx['protocol']['cell_split']['train_cells'])))
                ]
            effect_drug_count = int(expr_for_effect['drug'].nunique())
            rows.append({
                'cell_line': c,
                'effect_scope': effect_scope,
                'selected_seen_cell': True,
                'n_test_profiles': len(ctx['canonical']),
                'n_test_drugs': ctx['canonical']['drug'].nunique(),
                'n_val_drugs': len(ctx['seen_clean_val_drugs']),
                'n_labels': len(ctx['labels']),
                'effect_drugs_requested': len(effect_drugs),
                'effect_drugs_with_signature': effect_drug_count,
                'missing_effect_drugs': len(effect_drugs) - effect_drug_count,
            })
    out = WORKSPACE / 'preflight_cellline_selection.tsv'
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, sep='\t', index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'wrote {out}')


def load_round252_seen_rows() -> pd.DataFrame:
    candidates = [
        ROUND252_SOURCE / 'per_cell_metrics.tsv',
        LOCAL_ROOT / 'claude_metar_article_results_archive_20260521' / 'claude_round252_epiqgate_pseudoval_sweep_20260520_archive' / 'source_data' / 'per_cell_metrics.tsv',
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, sep='\t')
            mask = (
                df['cell_line'].isin(SELECTED_CELL_LINES)
                & (df['split'] == 'seen_test')
                & (df['evaluation_unit'] == 'drug-mean')
                & (df['variant'] == ROUND252_SELECTED_SEEN_VARIANT)
            )
            sub = df.loc[mask].copy()
            if not sub.empty:
                sub['method'] = 'Round252_EpiQGate'
                sub['setting'] = 'round252_seen_diagnostic_reference'
                return sub
    return pd.DataFrame()


def load_round88_pooled_seen_rows() -> pd.DataFrame:
    path = ROUND88_SOURCE / 'Figure2_common_intersection_percell_table.tsv'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep='\t')
    sub = df[df['cell_line'].isin(SELECTED_CELL_LINES)].copy()
    if sub.empty:
        return sub
    rows = []
    for _, r in sub.iterrows():
        rows.append({
            'cell_line': r['cell_line'],
            'setting': 'round88_pooled_cell_profile_diagnostic',
            'evaluation_unit': 'profile-level common-intersection',
            'method': 'Round88 pooled PertKGE',
            'n_profiles': int(r['n_matched']),
            'n_drugs': np.nan,
            'Hit@10': float(r['PertKGE_Hit@10']),
        })
        rows.append({
            'cell_line': r['cell_line'],
            'setting': 'round88_pooled_cell_profile_diagnostic',
            'evaluation_unit': 'profile-level common-intersection',
            'method': 'Round88 pooled MeTar',
            'n_profiles': int(r['n_matched']),
            'n_drugs': np.nan,
            'Hit@10': float(r['MeTar_Hit@10']),
        })
    return pd.DataFrame(rows)


def summarize_runs() -> None:
    source_dir = WORKSPACE / 'source_data'
    qa_dir = WORKSPACE / 'qa'
    source_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    summary_frames = []
    per_cell_frames = []
    hitk_rows = []
    denom_rows = []
    lock_paths = []
    completed_runs = []
    missing_runs = []
    for effect_scope in EFFECT_SCOPES:
        for cell in SELECTED_CELL_LINES:
            run_dir = WORKSPACE / effect_scope / f'cell_{cell}'
            metric_path = run_dir / 'metric_summary.tsv'
            drug_path = run_dir / 'matched_drug_mean_metrics.tsv'
            profile_path = run_dir / 'matched_profile_metrics.tsv'
            lock_path = run_dir / 'method_lock.json'
            pre_audit_path = run_dir / 'leakage_audit_pre_read.json'
            post_audit_path = run_dir / 'leakage_audit_post_read.json'
            run_id = f'{effect_scope}:{cell}'
            if not metric_path.exists() or not drug_path.exists():
                missing_runs.append(run_id)
                denom_rows.append({
                    'effect_scope': effect_scope,
                    'cell_line': cell,
                    'status': 'missing_or_incomplete',
                    'metric_summary_exists': metric_path.exists(),
                    'matched_drug_mean_exists': drug_path.exists(),
                    'matched_profile_exists': profile_path.exists(),
                })
                continue

            completed_runs.append(run_id)
            metrics = pd.read_csv(metric_path, sep='\t')
            metrics.insert(0, 'effect_scope', effect_scope)
            drug = pd.read_csv(drug_path, sep='\t')
            profile = safe_read_tsv(profile_path)
            summary_frames.append(metrics)
            primary = metrics[metrics['setting'] == 'drug_mean_primary'].copy()
            per_cell_frames.append(primary)
            for _, row in primary.iterrows():
                for k in HIT_KS:
                    hitk_rows.append({
                        'effect_scope': effect_scope,
                        'cell_line': cell,
                        'method': row['method'],
                        'K': k,
                        'Hit@K': row[f'Hit@{k}'],
                        'evaluation_unit': row['evaluation_unit'],
                        'n_drugs': row['n_drugs'],
                        'n_profiles': row['n_profiles'],
                    })
            audit_pre = json.loads(pre_audit_path.read_text()) if pre_audit_path.exists() else {}
            audit_post = json.loads(post_audit_path.read_text()) if post_audit_path.exists() else {}
            denom_rows.append({
                'effect_scope': effect_scope,
                'cell_line': cell,
                'status': 'complete',
                'candidate_space': 668,
                'n_drug_mean_units': len(drug),
                'n_profile_rows': len(profile),
                'n_unique_drugs': int(drug['drug'].nunique()) if 'drug' in drug.columns else np.nan,
                'n_unique_profiles': int(profile['profile_id'].nunique()) if 'profile_id' in profile.columns else np.nan,
                'effect_cell_line': audit_pre.get('effect_cell_line'),
                'effect_drugs_requested': audit_pre.get('effect_drugs_requested'),
                'effect_drugs_with_signature': audit_pre.get('effect_drugs_with_cell_line_signature'),
                'missing_effect_drugs_count': audit_pre.get('missing_effect_drugs_count'),
                'usable_eval_drugs': audit_pre.get('usable_eval_drugs'),
                'allowed_query_effect_val_test_cids': audit_pre.get('allowed_query_effect_val_test_cids'),
                'forbidden_test_target_edges_in_cause_or_process': audit_pre.get('forbidden_test_target_edges_in_cause_or_process'),
                'process_has_binding_to_edges': audit_pre.get('process_has_binding_to_edges'),
                'process_cid_edges': audit_pre.get('process_cid_edges'),
                'eval_cids_in_ent2id_count': audit_post.get('eval_cids_in_ent2id_count'),
            })
            if lock_path.exists():
                lock_paths.append(lock_path)

    all_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    if not all_summary.empty:
        all_summary.to_csv(source_dir / 'round292_pertkge_effect_scope_metric_summary.tsv', sep='\t', index=False)
        all_summary.to_csv(WORKSPACE / 'round292_pertkge_effect_scope_metric_summary.tsv', sep='\t', index=False)

    per_cell = pd.concat(per_cell_frames, ignore_index=True) if per_cell_frames else pd.DataFrame()
    if not per_cell.empty:
        per_cell.to_csv(source_dir / 'round292_pertkge_effect_scope_per_cell_metrics.tsv', sep='\t', index=False)
        per_cell.to_csv(WORKSPACE / 'round292_pertkge_effect_scope_per_cell_metrics.tsv', sep='\t', index=False)

    hitk_df = pd.DataFrame(hitk_rows)
    if not hitk_df.empty:
        hitk_df.to_csv(source_dir / 'round292_pertkge_effect_scope_hitk_curve.tsv', sep='\t', index=False)
        hitk_df.to_csv(WORKSPACE / 'round292_pertkge_effect_scope_hitk_curve.tsv', sep='\t', index=False)

    denom = pd.DataFrame(denom_rows)
    denom.to_csv(source_dir / 'round292_pertkge_effect_scope_denominator_audit.tsv', sep='\t', index=False)
    denom.to_csv(WORKSPACE / 'round292_pertkge_effect_scope_denominator_audit.tsv', sep='\t', index=False)

    compare_rows = []
    if not per_cell.empty:
        compare_rows.append(per_cell)
    r252 = load_round252_seen_rows()
    if not r252.empty:
        keep = ['cell_line', 'setting', 'evaluation_unit', 'method', 'n_profiles', 'n_drugs', *[f'Hit@{k}' for k in HIT_KS], 'AUROC', 'NDCG@10', 'MeanAP']
        r252_norm = r252.copy()
        if 'n_drugs' not in r252_norm.columns and 'n_groups' in r252_norm.columns:
            r252_norm['n_drugs'] = r252_norm['n_groups']
        for col in keep:
            if col not in r252_norm.columns:
                r252_norm[col] = np.nan
        compare_rows.append(r252_norm[keep])
    if compare_rows:
        comp = pd.concat(compare_rows, ignore_index=True, sort=False)
        comp.to_csv(source_dir / 'round292_pertkge_effect_scope_vs_round252_summary.tsv', sep='\t', index=False)
        comp.to_csv(WORKSPACE / 'round292_pertkge_effect_scope_vs_round252_summary.tsv', sep='\t', index=False)

    delta_rows = []
    if not per_cell.empty:
        pert = per_cell[per_cell['method'] == 'PertKGE'].copy()
        for cell in SELECTED_CELL_LINES:
            p = pert[(pert['cell_line'] == cell) & (pert['effect_scope'] == 'pooled')]
            s = pert[(pert['cell_line'] == cell) & (pert['effect_scope'] == 'same_cell')]
            if p.empty or s.empty:
                continue
            p = p.iloc[0]
            s = s.iloc[0]
            row = {
                'cell_line': cell,
                'evaluation_unit': 'drug-mean / drug-cell',
                'variable_changed': 'PertKGE effect expression source: pooled train cells -> same cell only',
                'pooled_n_drugs': int(p['n_drugs']),
                'same_cell_n_drugs': int(s['n_drugs']),
                'pooled_n_profiles': int(p['n_profiles']),
                'same_cell_n_profiles': int(s['n_profiles']),
            }
            for metric in [f'Hit@{k}' for k in HIT_KS] + ['AUROC', 'NDCG@10', 'MeanAP', 'best_true_target_percentile']:
                row[f'pooled_{metric}'] = float(p[metric])
                row[f'same_cell_{metric}'] = float(s[metric])
                row[f'delta_same_cell_minus_pooled_{metric}'] = float(s[metric]) - float(p[metric])
            delta_rows.append(row)
    delta = pd.DataFrame(delta_rows)
    if not delta.empty:
        delta.to_csv(source_dir / 'round292_pooled_vs_samecell_pertkge_druglevel_delta.tsv', sep='\t', index=False)
        delta.to_csv(WORKSPACE / 'round292_pooled_vs_samecell_pertkge_druglevel_delta.tsv', sep='\t', index=False)

    root_lock = {
        'round': 'round450_fig2_seen_cellline_extension',
        'created_at': datetime.now().isoformat(),
        'selected_cell_lines': SELECTED_CELL_LINES,
        'effect_scopes': EFFECT_SCOPES,
        'completed_runs': completed_runs,
        'missing_or_incomplete_runs': missing_runs,
        'candidate_space': 668,
        'primary_evaluation_unit': 'drug-mean / drug-cell',
        'profile_level_note': 'PertKGE scores are drug-level within a cell and are only replicated to profiles for diagnostic output.',
        'single_variable_contract': 'pooled and same_cell branches share PertKGE architecture, split, seed, epoch settings, candidate space and evaluation unit; only expression source used to build effect.txt changes.',
        'training_defaults': {
            'max_epochs': ROUND88_MAX_EPOCHS,
            'patience': ROUND88_PATIENCE,
            'eval_every': ROUND88_EVAL_EVERY,
            'warmup': ROUND88_WARMUP,
            'batch_size': ROUND88_BATCH_SIZE,
            'seed': ROUND88_SEED,
        },
        'inputs': {
            'expr_path': path_info(EXPR_PATH),
            'metar_score_path': path_info(METAR_SCORE_PATH),
            'lock_path': path_info(LOCK_PATH),
            'pertkge_repo': str(PERTKGE_REPO),
            'round252_source': str(ROUND252_SOURCE),
            'round88_source': str(ROUND88_SOURCE),
        },
        'per_cell_method_locks': [path_info(p) for p in lock_paths],
    }
    (WORKSPACE / 'method_lock.json').write_text(json.dumps(root_lock, indent=2), encoding='utf-8')

    lines = [
        '# Leakage and denominator audit',
        '',
        '- Included cells: ' + ', '.join(f'`{cell}`' for cell in SELECTED_CELL_LINES) + '.',
        '- Effect scopes: ' + ', '.join(f'`{scope}`' for scope in EFFECT_SCOPES) + '.',
        '- Candidate space is fixed at 668 labels.',
        '- Main denominator is drug-mean / drug-cell. PertKGE profile output is a replicated diagnostic only.',
        '- Single changed variable: `effect.txt` expression source. `pooled` uses Round88 train-cell pooled signatures; `same_cell` uses the same cell line as evaluation.',
        '- Validation/test query CIDs in `effect.txt` are allowed query information under the PertKGE target-inference protocol.',
        '- Validation/test `HAS_BINDING_TO` edges are forbidden from `cause.txt` and `process.txt` and are used only for evaluation.',
        '',
        '## Per-cell audit',
        '',
        md_table(denom.fillna('')),
    ]
    (WORKSPACE / 'leakage_and_denominator_audit.md').write_text('\n'.join(lines), encoding='utf-8')
    (qa_dir / 'leakage_and_denominator_audit.md').write_text('\n'.join(lines), encoding='utf-8')

    if not per_cell.empty:
        headline = per_cell[per_cell['method'].isin(['MeTar', 'PertKGE', 'Random'])].copy()
        report_table = headline[['effect_scope', 'cell_line', 'method', 'n_drugs', 'n_profiles', 'Hit@1', 'Hit@5', 'Hit@10', 'Hit@20', 'Hit@30', 'Hit@50', 'AUROC', 'NDCG@10', 'MeanAP']]
    else:
        report_table = pd.DataFrame()
    report_lines = [
        '# Seen-cell comparator results',
        '',
        '## Comparison setup',
        '',
        'Compare pooled and same-cell PertKGE models using drug-cell mean scores. The expression source for `effect.txt` differs between branches: pooled training-cell expression or expression from the evaluated cell line.',
        '',
        '## Evaluation units',
        '',
        '- Primary unit: drug-cell mean. PertKGE produces one score vector for each compound in each cell line.',
        '- Profile-replicated metrics are reported as an additional diagnostic.',
        '- The comparison uses the same 668-target candidate set for all methods.',
        '',
        '## Results',
        '',
        md_table(report_table.fillna('')) if not report_table.empty else 'No completed cell-line results are available.',
        '',
        '## Methods',
        '',
        '- `Round252_EpiQGate` identifies the selected MeTar adapter in its seen-cell evaluation.',
        '- `pooled` and `same_cell` identify separately trained PertKGE models.',
        '- Random Hit@K is K/668.',
    ]
    if not delta.empty:
        report_lines.extend(['', '## Pooled vs same-cell delta', '', md_table(delta.fillna(''))])
    (WORKSPACE / 'comparison_summary.md').write_text('\n'.join(report_lines), encoding='utf-8')
    print(f'completed_runs={completed_runs} missing={missing_runs}')
    print(f'wrote summary files under {source_dir}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell-line', default=None)
    parser.add_argument('--effect-scope', choices=EFFECT_SCOPES, default='same_cell')
    parser.add_argument('--cells-file', default=None)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--cuda', default=None, help='Deprecated; use shell-level CUDA_VISIBLE_DEVICES before process start.')
    parser.add_argument('--max-epochs', type=int, default=ROUND88_MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=ROUND88_PATIENCE)
    parser.add_argument('--eval-every', type=int, default=ROUND88_EVAL_EVERY)
    parser.add_argument('--warmup', type=int, default=ROUND88_WARMUP)
    parser.add_argument('--seed', type=int, default=ROUND88_SEED)
    args = parser.parse_args()
    global SELECTED_CELL_LINES
    SELECTED_CELL_LINES = load_selected_cell_lines(args.cells_file)
    if args.cells_file is None:
        (WORKSPACE / 'selected_seen_cells.txt').write_text('\n'.join(SELECTED_CELL_LINES) + '\n', encoding='utf-8')
    if args.preflight:
        preflight()
        return
    if args.summarize:
        summarize_runs()
        return
    if not args.cell_line:
        raise SystemExit('--cell-line required unless --preflight or --summarize')
    if args.cell_line not in SELECTED_CELL_LINES:
        raise SystemExit(f'--cell-line must be one of {SELECTED_CELL_LINES}, got {args.cell_line!r}')
    run_dir = WORKSPACE / args.effect_scope / f'cell_{args.cell_line}'
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.FileHandler(run_dir / 'train.log', mode='w'), logging.StreamHandler(sys.stdout)])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    logging.info('round450 Round88-flow PertKGE start effect_scope=%s cell_line=%s device=%s cuda_visible=%s', args.effect_scope, args.cell_line, device, os.environ.get('CUDA_VISIBLE_DEVICES'))
    ctx = build_context(args.cell_line)
    file_info = write_pertkge_files(ctx, run_dir, args.effect_scope)
    result = train_score(ctx, run_dir, args.effect_scope, args.max_epochs, args.patience, args.eval_every, args.warmup, device, args.seed)
    lock = {
        'round': 'round450_fig2_seen_cellline_extension',
        'created_at': datetime.now().isoformat(),
        'cell_line': args.cell_line,
        'effect_scope': args.effect_scope,
        'selected_cell_lines': SELECTED_CELL_LINES,
        'effect_scopes': EFFECT_SCOPES,
        'candidate_space': 668,
        'seed': args.seed,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
        'eval_every': args.eval_every,
        'warmup': args.warmup,
        'top_genes_effect': TOP_GENES_EFFECT,
        'batch_size': ROUND88_BATCH_SIZE,
        'protocol': {
            'name': 'Round88-compatible PertKGE target-inference protocol with effect-scope single-variable switch',
            'effect_edges': 'inner-train/validation/test query-drug expression-derived effect triples are included in effect.txt; expression source is controlled by effect_scope',
            'effect_scope_contract': 'pooled uses Round88 train-cell pooled expression; same_cell uses only the current evaluation cell line',
            'withheld_edges': 'validation/test HAS_BINDING_TO drug-target edges are excluded from cause.txt and used only for evaluation',
            'scoring': 'score CID:drug against Protein targets with DistMult HAS_BINDING_TO relation after training',
            'primary_evaluation_unit': 'drug-mean / drug-cell',
            'profile_level_note': 'PertKGE scores are drug-level within a cell and are replicated to profiles only for diagnostic output.',
        },
        'device': str(device),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'inputs': {
            'expr_path': path_info(EXPR_PATH),
            'metar_score_path': path_info(METAR_SCORE_PATH),
            'lock_path': path_info(LOCK_PATH),
            'process_path': path_info(PERTKGE_REPO / 'processed_data' / 'knowledge_graph' / 'process.txt'),
            'pertkge_repo': str(PERTKGE_REPO),
        },
        'file_info': file_info,
        'result': result,
    }
    (run_dir / 'method_lock.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    logging.info('complete %s', json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
