#!/usr/bin/env python
"""
Shared-sample, independent-unseen-label bulk training script for MeTar.
"""

import argparse
import importlib.util
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from metar.runtime import report_metrics, report_progress, report_step, resolve_device
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


PROJECT_ROOT = Path(
    os.environ.get(
        "METAR_BULK_PROJECT_ROOT",
        "/home/lwn/claude/manuscript/write/bulk_sharedsample_indunseenlabel_20260330",
    )
)
DEFAULT_PROTOCOL_PATH = Path(
    os.environ.get(
        "METAR_BULK_PROTOCOL_PATH",
        "/home/lwn/claude/manuscript/write/bulk_dualbranch_20260328_b01_warm_seen_joint/manifests/bulk_protocol_lock_v2.json",
    )
)
BASE_SCRIPT_PATH = Path(__file__).resolve().parent / "train_bulk_dualbranch_v2_20260328.py"

os.environ["METAR_BULK_PROJECT_ROOT"] = str(PROJECT_ROOT)

_SPEC = importlib.util.spec_from_file_location("metar_bulk_base", BASE_SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load base bulk script from {BASE_SCRIPT_PATH}")
bulk_base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bulk_base)


logger = logging.getLogger("bulk_sharedsample_indunseenlabel_v1")


def ensure_dirs() -> None:
    for name in ["artifacts", "checkpoints", "derived_tables", "logs", "manifests", "scripts"]:
        (PROJECT_ROOT / name).mkdir(parents=True, exist_ok=True)


def copy_manifest(manifest_path: Path) -> Path:
    dst = PROJECT_ROOT / "manifests" / manifest_path.name
    if manifest_path.resolve() != dst.resolve():
        shutil.copy2(manifest_path, dst)
    return dst


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int = 256, bottleneck: int = 64, dropout: float = 0.05):
        super().__init__()
        self.dropout = dropout
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        nn.init.xavier_uniform_(self.down.weight, gain=0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = F.gelu(self.down(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        delta = self.up(h)
        return x + self.scale * delta

    def functional_forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = F.layer_norm(x, (x.shape[-1],), params["ln.weight"], params["ln.bias"])
        h = F.linear(h, params["down.weight"], params["down.bias"])
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        delta = F.linear(h, params["up.weight"], params["up.bias"])
        return x + params["scale"] * delta


class SharedSampleIndependentUnseenLabelMetaTwoTowerGNN(nn.Module):
    def __init__(
        self,
        gnn: nn.Module,
        n_expr_genes: int = 978,
        hidden: int = 512,
        out_dim: int = 256,
        shared_sample_variant: str = "residual",
        shared_label_variant: str = "residual",
        unseen_label_variant: str = "linear",
    ):
        super().__init__()
        sample_cls = bulk_base.SampleEncoderResidual if shared_sample_variant == "residual" else bulk_base.SampleEncoder
        if shared_label_variant == "residual":
            label_cls = bulk_base.LabelEncoderResidual
        elif shared_label_variant == "linear":
            label_cls = bulk_base.LabelEncoderLinear
        else:
            label_cls = bulk_base.LabelEncoder

        self.sample_shared = sample_cls(in_dim=n_expr_genes, hidden=hidden, out_dim=out_dim)
        self.gnn = gnn
        if shared_label_variant == "linear":
            self.label_shared = label_cls(in_dim=128, out_dim=out_dim)
        else:
            self.label_shared = label_cls(in_dim=128, hidden=hidden, out_dim=out_dim)

        if unseen_label_variant == "residual":
            self.label_unseen_private = bulk_base.LabelEncoderResidual(in_dim=128, hidden=hidden, out_dim=out_dim)
        elif unseen_label_variant == "default":
            self.label_unseen_private = bulk_base.LabelEncoder(in_dim=128, hidden=hidden, out_dim=out_dim)
        else:
            self.label_unseen_private = bulk_base.LabelEncoderLinear(in_dim=128, out_dim=out_dim)
        self.logit_scale_seen = nn.Parameter(torch.tensor(torch.log(torch.tensor(10.0)).item(), dtype=torch.float32))
        self.logit_scale_unseen = nn.Parameter(torch.tensor(torch.log(torch.tensor(10.0)).item(), dtype=torch.float32))

    def get_parameters_dict(self) -> Dict[str, torch.Tensor]:
        return {name: p for name, p in self.named_parameters()}

    @staticmethod
    def _subparams(params: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}

    def _head_config(self, head: str):
        if head == "seen":
            return "logit_scale_seen"
        if head == "unseen":
            return "logit_scale_unseen"
        raise ValueError(f"Unknown head: {head}")

    def sample_forward(self, x: torch.Tensor, head: str) -> torch.Tensor:
        return self.sample_shared(x)

    def label_forward(self, gene_repr: torch.Tensor, head: str) -> torch.Tensor:
        if head == "seen":
            return self.label_shared(gene_repr)
        if head == "unseen":
            return self.label_unseen_private(gene_repr)
        raise ValueError(f"Unknown head: {head}")

    def functional_logits(self, x: torch.Tensor, gene_repr: torch.Tensor, params: Dict[str, torch.Tensor], head: str) -> torch.Tensor:
        scale_key = self._head_config(head)
        sample_shared_params = self._subparams(params, "sample_shared.")
        label_shared_params = self._subparams(params, "label_shared.")
        unseen_label_params = self._subparams(params, "label_unseen_private.")
        scale = params.get(scale_key, getattr(self, scale_key)).exp().clamp(max=100.0)

        base_z = self.sample_shared.functional_forward(x, sample_shared_params)
        z = base_z
        if head == "seen":
            w = self.label_shared.functional_forward(gene_repr, label_shared_params)
        elif head == "unseen":
            w = self.label_unseen_private.functional_forward(gene_repr, unseen_label_params)
        else:
            raise ValueError(f"Unknown head: {head}")
        z = F.normalize(z, dim=1)
        w = F.normalize(w, dim=1)
        return (z @ w.t()) * scale

    def forward(self, x: torch.Tensor, gene_repr: torch.Tensor, head: str) -> torch.Tensor:
        z = F.normalize(self.sample_forward(x, head=head), dim=1)
        w = F.normalize(self.label_forward(gene_repr, head=head), dim=1)
        scale = getattr(self, f"logit_scale_{head}").exp().clamp(max=100.0)
        return (z @ w.t()) * scale


def load_checkpoint_into_shared_stem(
    model: SharedSampleIndependentUnseenLabelMetaTwoTowerGNN,
    ckpt_path: str,
    branch: str,
    device: torch.device,
    load_gnn: bool = False,
    load_branch_modules: bool = True,
    load_unseen_private: bool = False,
    initialize_scales: bool = True,
) -> None:
    if not ckpt_path:
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    src_state = ckpt.get("model_state_dict", ckpt)
    dst_state = model.state_dict()
    mapped_state = {}
    skipped_shape = []

    for key, value in src_state.items():
        dst_key = None
        if load_gnn and key.startswith("gnn."):
            dst_key = key
        elif load_unseen_private and branch == "unseen":
            if key == "logit_scale_unseen":
                dst_key = key
            elif key.startswith("label_unseen."):
                dst_key = f"label_unseen_private.{key[len('label_unseen.'):]}"
            elif key.startswith("label."):
                dst_key = f"label_unseen_private.{key[len('label.'):]}"
            else:
                continue
        elif not load_branch_modules:
            continue
        elif key.startswith("sample_shared."):
            dst_key = key
        elif key.startswith("label_shared."):
            dst_key = key
        elif key.startswith(f"sample_{branch}."):
            dst_key = f"sample_shared.{key[len(f'sample_{branch}.'):]}"
        elif key.startswith(f"label_{branch}."):
            dst_key = f"label_shared.{key[len(f'label_{branch}.'):]}"
        elif key.startswith("sample."):
            dst_key = f"sample_shared.{key[len('sample.'):]}"
        elif key.startswith("label."):
            dst_key = f"label_shared.{key[len('label.'):]}"
        elif initialize_scales and key in {"logit_scale", f"logit_scale_{branch}"}:
            for scale_key in ["logit_scale_seen", "logit_scale_unseen"]:
                if scale_key in dst_state and tuple(dst_state[scale_key].shape) == tuple(value.shape):
                    mapped_state[scale_key] = value
            continue

        if dst_key is None:
            continue
        if dst_key in dst_state and tuple(dst_state[dst_key].shape) == tuple(value.shape):
            mapped_state[dst_key] = value
        else:
            skipped_shape.append((key, dst_key, tuple(value.shape), tuple(dst_state[dst_key].shape) if dst_key in dst_state else None))

    missing, unexpected = model.load_state_dict(mapped_state, strict=False)
    logger.info(
        "Warm-loaded model state from %s | branch=%s | load_gnn=%s | load_branch_modules=%s | load_unseen_private=%s | mapped=%d | skipped_shape=%d | missing=%d | unexpected=%d",
        ckpt_path,
        branch,
        load_gnn,
        load_branch_modules,
        load_unseen_private,
        len(mapped_state),
        len(skipped_shape),
        len(missing),
        len(unexpected),
    )
    for src_key, dst_key, src_shape, dst_shape in skipped_shape[:8]:
        logger.info(
            "  Skipped incompatible tensor %s -> %s | src_shape=%s dst_shape=%s",
            src_key,
            dst_key,
            src_shape,
            dst_shape,
        )


def shared_adapter_geometry_loss(
    model: SharedSampleIndependentUnseenLabelMetaTwoTowerGNN,
    gene_repr_full: torch.Tensor,
    sample_size: int = 384,
) -> torch.Tensor:
    projected = model.label_forward(gene_repr_full, head="unseen")
    return bulk_base.geometry_preserve_loss(gene_repr_full, projected, sample_size=sample_size)


def configure_logging(variant: str) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(PROJECT_ROOT / "logs" / f"train_{variant}.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)


def build_fast_keys(params: Dict[str, torch.Tensor], head: str, unseen_fast_include_sample_shared: bool = False) -> list:
    fast_keys = []
    if head == "seen":
        prefixes = ["sample_shared.", "label_shared."]
        scale_key = "logit_scale_seen"
    elif head == "unseen":
        prefixes = ["label_unseen_private."]
        if unseen_fast_include_sample_shared:
            prefixes = ["sample_shared."] + prefixes
        scale_key = "logit_scale_unseen"
    else:
        raise ValueError(f"Unknown head: {head}")
    for key, value in params.items():
        if value.requires_grad and (any(key.startswith(prefix) for prefix in prefixes) or key == scale_key):
            fast_keys.append(key)
    return fast_keys


def export_gene_artifacts(model: SharedSampleIndependentUnseenLabelMetaTwoTowerGNN, data: dict, head: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    with torch.no_grad():
        gene_repr = model.gnn(data["adj_norm"])
        raw_repr = gene_repr.cpu().numpy()
        label_repr = F.normalize(model.label_forward(gene_repr, head=head), dim=1).cpu().numpy()
    fixed_gene_names = [bulk_base.DATE_GENE_FIX.get(g, g) for g in data["gene_list"]]
    return pd.DataFrame(raw_repr, index=fixed_gene_names), pd.DataFrame(label_repr, index=fixed_gene_names)


def snapshot_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_shared_adapter_variant(args, data: dict, device: torch.device) -> dict:
    report_step('Preparing shared bulk adapter training')
    bulk_base.set_seed(args.seed)
    n_genes = len(data["gene_list"])
    train_gene_global_idx = data["train_gene_global_idx"]

    gnn = bulk_base.choose_gnn("baseline_clean_v6", n_genes).to(device)
    bulk_base.pretrain_gnn(
        gnn=gnn,
        adj_norm=data["adj_norm"],
        ppi_edges=data["ppi_edges"],
        ppi_edges_set=data["ppi_edges_set"],
        n_genes=n_genes,
        device=device,
        epochs=args.gnn_pretrain_epochs,
        lr=args.gnn_lr_pretrain,
    )

    model = SharedSampleIndependentUnseenLabelMetaTwoTowerGNN(
        gnn=gnn,
        n_expr_genes=len(data["expr_cols"]),
        hidden=args.hidden_dim,
        out_dim=args.out_dim,
        shared_sample_variant=args.shared_sample_variant,
        shared_label_variant=args.shared_label_variant,
        unseen_label_variant=args.unseen_label_variant,
    ).to(device)

    if args.init_shared_ckpt:
        load_checkpoint_into_shared_stem(
            model=model,
            ckpt_path=args.init_shared_ckpt,
            branch=args.init_shared_branch,
            device=device,
            load_gnn=not bool(args.init_gnn_ckpt),
            load_branch_modules=True,
            load_unseen_private=False,
            initialize_scales=True,
        )
    if args.init_gnn_ckpt:
        load_checkpoint_into_shared_stem(
            model=model,
            ckpt_path=args.init_gnn_ckpt,
            branch=args.init_gnn_branch,
            device=device,
            load_gnn=True,
            load_branch_modules=False,
            load_unseen_private=False,
            initialize_scales=False,
        )
    if args.init_unseen_private_ckpt:
        load_checkpoint_into_shared_stem(
            model=model,
            ckpt_path=args.init_unseen_private_ckpt,
            branch="unseen",
            device=device,
            load_gnn=False,
            load_branch_modules=False,
            load_unseen_private=True,
            initialize_scales=False,
        )

    total_params = sum(p.numel() for p in model.parameters())
    shared_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.startswith("sample_shared.") or name.startswith("label_shared.") or name.startswith("gnn.")
    )
    specialized_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.startswith("label_unseen_private.") or name.startswith("logit_scale_")
    )
    logger.info(
        "Variant %s | total_params=%s | shared_params=%s | specialized_params=%s",
        args.variant,
        f"{total_params:,}",
        f"{shared_params:,}",
        f"{specialized_params:,}",
    )

    optimizer = torch.optim.Adam(
        [
            {"params": model.gnn.parameters(), "lr": args.gnn_lr, "weight_decay": 1e-6},
            {"params": model.sample_shared.parameters(), "lr": args.shared_lr, "weight_decay": 1e-5},
            {"params": model.label_shared.parameters(), "lr": args.shared_lr, "weight_decay": 1e-5},
            {"params": model.label_unseen_private.parameters(), "lr": args.unseen_lr, "weight_decay": 1e-5},
            {"params": [model.logit_scale_seen, model.logit_scale_unseen], "lr": args.unseen_lr, "weight_decay": 0.0},
        ]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    train_cells = [cell for cell, idxs in data["inner_train"]["pool"].items() if len(idxs) >= args.support_size + args.query_size]
    if not train_cells:
        raise RuntimeError("No train cells have enough samples for tasks.")

    best_seen_score = -1.0
    best_unseen_score = -1.0
    best_seen_epoch = 0
    best_unseen_epoch = 0
    best_seen_path = PROJECT_ROOT / "checkpoints" / f"best_seen_{args.variant}.pth"
    best_unseen_path = PROJECT_ROOT / "checkpoints" / f"best_unseen_{args.variant}.pth"
    history_rows = []
    train_start = time.time()

    report_progress(0, args.n_epochs, label='shared adapter variant epochs')
    report_metrics(vars(args), label="Training settings")
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        meta_seen_total = 0.0
        meta_unseen_total = 0.0
        geom_loss_total = 0.0
        valid_steps = 0
        epoch_start = time.time()

        for _ in range(args.steps_per_epoch):
            report_progress(_, args.steps_per_epoch, label=f"Epoch {epoch} training episodes")
            sampled_cells = bulk_base.random.sample(train_cells, min(args.meta_batch, len(train_cells)))
            optimizer.zero_grad()

            gene_repr_full = model.gnn(data["adj_norm"])
            gene_repr_train = gene_repr_full[train_gene_global_idx]
            gene_repr_full_detached = gene_repr_full.detach()
            gene_repr_train_detached = gene_repr_train.detach()
            base_params = model.get_parameters_dict()
            seen_fast_keys = build_fast_keys(base_params, head="seen")
            unseen_fast_keys = build_fast_keys(
                base_params,
                head="unseen",
                unseen_fast_include_sample_shared=args.unseen_fast_include_sample_shared,
            )

            batch_seen_loss = torch.tensor(0.0, device=device)
            batch_unseen_loss = torch.tensor(0.0, device=device)
            valid_tasks = 0

            for cell in sampled_cells:
                task = bulk_base.sample_task_from_pool(data["inner_train"], cell, args.support_size, args.query_size, device)
                if task is None:
                    continue
                x_s, y_s, x_q, y_q = task
                y_s_full = bulk_base.scatter_local_targets_to_full(y_s, train_gene_global_idx, n_genes)
                y_q_full = bulk_base.scatter_local_targets_to_full(y_q, train_gene_global_idx, n_genes)

                adapted_seen = dict(base_params)
                seen_label_mask = bulk_base.build_label_mask(y_s)
                if seen_fast_keys:
                    for _ in range(args.inner_steps):
                        logits_s_seen = model.functional_logits(x_s, gene_repr_train_detached, adapted_seen, head="seen")
                        loss_s_seen = bulk_base.masked_bce_with_logits(logits_s_seen, y_s, mask=seen_label_mask)
                        grads_seen = torch.autograd.grad(loss_s_seen, [adapted_seen[k] for k in seen_fast_keys], create_graph=False)
                        for key, grad in zip(seen_fast_keys, grads_seen):
                            adapted_seen[key] = adapted_seen[key] - args.inner_lr * grad

                adapted_unseen = dict(base_params)
                if unseen_fast_keys:
                    for _ in range(args.inner_steps):
                        logits_s_unseen = model.functional_logits(x_s, gene_repr_full_detached, adapted_unseen, head="unseen")
                        loss_s_unseen = bulk_base.compute_unseen_head_loss(logits_s_unseen, y_s_full)
                        grads_unseen = torch.autograd.grad(loss_s_unseen, [adapted_unseen[k] for k in unseen_fast_keys], create_graph=False)
                        for key, grad in zip(unseen_fast_keys, grads_unseen):
                            adapted_unseen[key] = adapted_unseen[key] - args.inner_lr * grad

                logits_q_seen = model.functional_logits(x_q, gene_repr_train, adapted_seen, head="seen")
                logits_q_unseen = model.functional_logits(x_q, gene_repr_full, adapted_unseen, head="unseen")
                batch_seen_loss = batch_seen_loss + bulk_base.compute_seen_head_loss(logits_q_seen, y_q)
                batch_unseen_loss = batch_unseen_loss + bulk_base.compute_unseen_head_loss(logits_q_unseen, y_q_full)
                valid_tasks += 1

            if valid_tasks == 0:
                continue

            batch_seen_loss = batch_seen_loss / valid_tasks
            batch_unseen_loss = batch_unseen_loss / valid_tasks
            batch_geom_loss = shared_adapter_geometry_loss(model, gene_repr_full, sample_size=384)
            total_loss = batch_seen_loss + args.unseen_loss_weight * batch_unseen_loss + args.unseen_geom_weight * batch_geom_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            meta_seen_total += batch_seen_loss.item()
            meta_unseen_total += batch_unseen_loss.item()
            geom_loss_total += batch_geom_loss.item()
            valid_steps += 1
        report_progress(args.steps_per_epoch, args.steps_per_epoch, label=f"Epoch {epoch} training episodes")

        scheduler.step()
        avg_seen_loss = meta_seen_total / max(valid_steps, 1)
        avg_unseen_loss = meta_unseen_total / max(valid_steps, 1)
        avg_geom_loss = geom_loss_total / max(valid_steps, 1)

        if epoch % args.eval_every == 0 or epoch == args.n_epochs:
            with torch.no_grad():
                gene_repr_eval = model.gnn(data["adj_norm"])
                gene_repr_train_eval = gene_repr_eval[train_gene_global_idx]

            val_seen_head = bulk_base.evaluate_seen_split_dual(model, data["val"], gene_repr_train_eval, args.eval_batch_size, device, head="seen")
            val_unseen_proxy = bulk_base.evaluate_seen_split_dual(model, data["val"], gene_repr_train_eval, args.eval_batch_size, device, head="unseen")
            val_holdout = bulk_base.evaluate_holdout_top5_dual(model, data["holdout_val"], gene_repr_eval, args.holdout_batch_size, device, head="unseen")
            seen_score = bulk_base.compute_seen_selection_score(val_seen_head)
            unseen_score = bulk_base.compute_unseen_selection_score(val_holdout, val_unseen_proxy)

            history_rows.append(
                {
                    "variant": args.variant,
                    "epoch": epoch,
                    "meta_seen_loss": avg_seen_loss,
                    "meta_unseen_loss": avg_unseen_loss,
                    "geom_loss": avg_geom_loss,
                    "val_seen_hit10": val_seen_head["Hit@10"],
                    "val_seen_auroc": val_seen_head["AUROC"],
                    "val_seen_ndcg10": val_seen_head["NDCG@10"],
                    "val_seen_map": val_seen_head["MeanAP"],
                    "val_unseen_proxy_hit10": val_unseen_proxy["Hit@10"],
                    "val_unseen_proxy_auroc": val_unseen_proxy["AUROC"],
                    "val_holdout_top5pct": val_holdout["top5pct_success"],
                    "seen_score": seen_score,
                    "unseen_score": unseen_score,
                    "epoch_time_s": time.time() - epoch_start,
                }
            )
            report_progress(epoch, args.n_epochs, label='shared adapter variant epochs')
            report_metrics(
                {key: history_rows[-1][key] for key in (
                    "meta_seen_loss", "meta_unseen_loss", "val_seen_hit10",
                    "val_seen_map", "val_holdout_top5pct",
                )},
                label=f"Epoch {epoch}/{args.n_epochs}",
            )
            logger.info(
                "Epoch %02d/%02d | seen_loss=%.4f | unseen_loss=%.4f | geom=%.4f | seen_hit10=%.4f | seen_auc=%.4f | unseen_proxy_hit10=%.4f | holdout=%.4f | seen_score=%.4f | unseen_score=%.4f",
                epoch,
                args.n_epochs,
                avg_seen_loss,
                avg_unseen_loss,
                avg_geom_loss,
                val_seen_head["Hit@10"],
                val_seen_head["AUROC"],
                val_unseen_proxy["Hit@10"],
                val_holdout["top5pct_success"],
                seen_score,
                unseen_score,
            )

            if seen_score > best_seen_score:
                best_seen_score = seen_score
                best_seen_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "variant": args.variant,
                        "head": "seen",
                        "model_state_dict": snapshot_state_dict_to_cpu(model),
                        "seen_score": seen_score,
                        "val_seen_metrics": val_seen_head,
                    },
                    best_seen_path,
                )

            if unseen_score > best_unseen_score:
                best_unseen_score = unseen_score
                best_unseen_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "variant": args.variant,
                        "head": "unseen",
                        "model_state_dict": snapshot_state_dict_to_cpu(model),
                        "unseen_score": unseen_score,
                        "val_unseen_proxy_metrics": val_unseen_proxy,
                        "val_holdout_top5pct": val_holdout["top5pct_success"],
                    },
                    best_unseen_path,
                )

    total_time = time.time() - train_start
    logger.info("Training finished in %.1f s", total_time)

    seen_ckpt = torch.load(best_seen_path, map_location=device)
    model.load_state_dict(seen_ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        gene_repr_seen_final = model.gnn(data["adj_norm"])
        gene_repr_seen_train = gene_repr_seen_final[train_gene_global_idx]
    seen_val_metrics = bulk_base.evaluate_seen_split_dual(model, data["val"], gene_repr_seen_train, args.eval_batch_size, device, head="seen")
    seen_test_metrics = bulk_base.evaluate_seen_split_dual(model, data["test"], gene_repr_seen_train, args.eval_batch_size, device, head="seen")

    unseen_ckpt = torch.load(best_unseen_path, map_location=device)
    model.load_state_dict(unseen_ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        gene_repr_unseen_final = model.gnn(data["adj_norm"])
        gene_repr_unseen_train = gene_repr_unseen_final[train_gene_global_idx]
    unseen_proxy_val = bulk_base.evaluate_seen_split_dual(model, data["val"], gene_repr_unseen_train, args.eval_batch_size, device, head="unseen")
    unseen_proxy_test = bulk_base.evaluate_seen_split_dual(model, data["test"], gene_repr_unseen_train, args.eval_batch_size, device, head="unseen")
    unseen_holdout_val = bulk_base.evaluate_holdout_top5_dual(model, data["holdout_val"], gene_repr_unseen_final, args.holdout_batch_size, device, head="unseen")
    unseen_holdout_test = bulk_base.evaluate_holdout_top5_dual(model, data["holdout_test"], gene_repr_unseen_final, args.holdout_batch_size, device, head="unseen")

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(PROJECT_ROOT / "derived_tables" / f"history_{args.variant}.tsv", sep="\t", index=False)

    seen_label_path = None
    unseen_label_path = None
    if not args.skip_artifact_export:
        model.load_state_dict(seen_ckpt["model_state_dict"])
        seen_raw, seen_label = export_gene_artifacts(model, data, head="seen")
        seen_raw.to_csv(PROJECT_ROOT / "artifacts" / f"gene_repr_gnn__{args.variant}__seenbest.csv")
        seen_label_path = PROJECT_ROOT / "artifacts" / f"gene_repr_labelproj_seen__{args.variant}.csv"
        seen_label.to_csv(seen_label_path)

        model.load_state_dict(unseen_ckpt["model_state_dict"])
        unseen_raw, unseen_label = export_gene_artifacts(model, data, head="unseen")
        unseen_raw.to_csv(PROJECT_ROOT / "artifacts" / f"gene_repr_gnn__{args.variant}__unseenbest.csv")
        unseen_label_path = PROJECT_ROOT / "artifacts" / f"gene_repr_labelproj_unseen__{args.variant}.csv"
        unseen_label.to_csv(unseen_label_path)

    results = {
        "variant": args.variant,
        "best_seen_epoch": best_seen_epoch,
        "best_seen_score": best_seen_score,
        "best_unseen_epoch": best_unseen_epoch,
        "best_unseen_score": best_unseen_score,
        "seen_head": {
            "val_metrics": seen_val_metrics,
            "test_metrics": seen_test_metrics,
            "selected_val_metrics_at_save": seen_ckpt.get("val_seen_metrics", {}),
            "checkpoint_path": str(best_seen_path),
            "label_embedding_path": str(seen_label_path) if seen_label_path else "",
        },
        "unseen_head": {
            "val_proxy_metrics": unseen_proxy_val,
            "test_proxy_metrics": unseen_proxy_test,
            "val_holdout_top5pct": unseen_holdout_val["top5pct_success"],
            "val_holdout_n_samples": unseen_holdout_val["n_samples"],
            "test_holdout_top5pct": unseen_holdout_test["top5pct_success"],
            "test_holdout_n_samples": unseen_holdout_test["n_samples"],
            "selected_val_proxy_metrics_at_save": unseen_ckpt.get("val_unseen_proxy_metrics", {}),
            "selected_val_holdout_top5pct_at_save": unseen_ckpt.get("val_holdout_top5pct", None),
            "checkpoint_path": str(best_unseen_path),
            "label_embedding_path": str(unseen_label_path) if unseen_label_path else "",
        },
        "config": {
            "n_epochs": args.n_epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "meta_batch": args.meta_batch,
            "support_size": args.support_size,
            "query_size": args.query_size,
            "inner_steps": args.inner_steps,
            "inner_lr": args.inner_lr,
            "gnn_lr": args.gnn_lr,
            "shared_lr": args.shared_lr,
            "unseen_lr": args.unseen_lr,
            "gnn_pretrain_epochs": args.gnn_pretrain_epochs,
            "eval_every": args.eval_every,
            "seed": args.seed,
            "shared_sample_variant": args.shared_sample_variant,
            "shared_label_variant": args.shared_label_variant,
            "unseen_label_variant": args.unseen_label_variant,
            "unseen_loss_weight": args.unseen_loss_weight,
            "unseen_geom_weight": args.unseen_geom_weight,
            "init_shared_ckpt": args.init_shared_ckpt,
            "init_shared_branch": args.init_shared_branch,
            "init_gnn_ckpt": args.init_gnn_ckpt,
            "init_gnn_branch": args.init_gnn_branch,
            "init_unseen_private_ckpt": args.init_unseen_private_ckpt,
            "unseen_fast_include_sample_shared": args.unseen_fast_include_sample_shared,
            "skip_artifact_export": args.skip_artifact_export,
        },
        "protocol_audit": data["audit"],
        "param_counts": {
            "total": total_params,
            "shared": shared_params,
            "specialized_and_scale": specialized_params,
        },
        "total_train_time_s": total_time,
    }
    with open(PROJECT_ROOT / "derived_tables" / f"results_{args.variant}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Shared-sample independent-unseen-label MeTar bulk training")
    parser.add_argument("--variant", type=str, default="sharedsample_indunseenlabel_v1")
    parser.add_argument("--manifest_path", type=str, default=str(DEFAULT_PROTOCOL_PATH))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int, default=8)
    parser.add_argument("--steps_per_epoch", type=int, default=40)
    parser.add_argument("--meta_batch", type=int, default=8)
    parser.add_argument("--support_size", type=int, default=32)
    parser.add_argument("--query_size", type=int, default=64)
    parser.add_argument("--inner_steps", type=int, default=3)
    parser.add_argument("--inner_lr", type=float, default=0.01)
    parser.add_argument("--gnn_lr", type=float, default=1e-4)
    parser.add_argument("--gnn_lr_pretrain", type=float, default=1e-3)
    parser.add_argument("--shared_lr", type=float, default=8e-4)
    parser.add_argument("--unseen_lr", type=float, default=1e-3)
    parser.add_argument("--gnn_pretrain_epochs", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--holdout_batch_size", type=int, default=48)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--shared_sample_variant", type=str, default="residual", choices=["default", "residual"])
    parser.add_argument("--shared_label_variant", type=str, default="residual", choices=["default", "residual", "linear"])
    parser.add_argument("--unseen_label_variant", type=str, default="linear", choices=["linear", "default", "residual"])
    parser.add_argument("--unseen_loss_weight", type=float, default=0.75)
    parser.add_argument("--unseen_geom_weight", type=float, default=0.05)
    parser.add_argument("--init_shared_ckpt", type=str, default="")
    parser.add_argument("--init_shared_branch", type=str, default="seen", choices=["seen", "unseen"])
    parser.add_argument("--init_gnn_ckpt", type=str, default="")
    parser.add_argument("--init_gnn_branch", type=str, default="unseen", choices=["seen", "unseen"])
    parser.add_argument("--init_unseen_private_ckpt", type=str, default="")
    parser.add_argument("--unseen_fast_include_sample_shared", action="store_true")
    parser.add_argument("--skip_artifact_export", action="store_true")
    return parser.parse_args()


def main():
    ensure_dirs()
    args = parse_args()
    configure_logging(args.variant)
    bulk_base.set_seed(args.seed)
    device = torch.device(resolve_device(args.device))

    manifest_path = Path(args.manifest_path)
    manifest_copy = copy_manifest(manifest_path)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Variant: %s", args.variant)
    logger.info("Device: %s", device)
    logger.info("Manifest: %s", manifest_copy)

    protocol = bulk_base.load_protocol(manifest_copy)
    data = bulk_base.prepare_data(protocol, device)
    results = train_shared_adapter_variant(args, data, device)
    logger.info("Final result summary:")
    logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
