"""Prepare an isolated scratch or stage-specific MeTar training workspace."""

from __future__ import annotations

import csv
import errno
import json
import shutil
import stat
import sys
from pathlib import Path

MARKER = "training_workspace.json"
WRITE_ROOT = "compat/home/lwn/claude/manuscript/write"
ROUND_NAMES = {
    "r176": "claude_round176_hgtlite_structural_5x_pc3_20260508",
    "r210": "claude_round210_dim256_checkpoint_sweep_20260515",
    "r211": "claude_round211_dim256_readout_capacity_sweep_20260515",
    "r215": "claude_round215_direct_query_shift_adapter_20260517",
    "r230": "claude_round230_scratch_mlp_metar_target_tower_pc3_20260518",
    "r243": "claude_round243_leakage_safe_unseen_cell_support_adapter_20260519",
    "r246": "claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519",
    "r247": "claude_round247_support_curve_query_gated_adapter_20260520",
    "r250": "claude_round250_epmeta_stable_tuning_20260520",
    "r252": "claude_round252_epiqgate_pseudoval_sweep_20260520",
    "r259_261": "claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522",
}
EXTERNAL_INPUTS = (
    "data/processed/closed/r210/pc3_landmark_response_978.tsv.gz",
    "data/processed/closed/r210/pc3_metar_drugcell_scores_668.tsv.gz",
    "data/public_raw/gene_prior/process.txt",
    "data/processed/lincs/large_data.parquet",
    "data/processed/lincs/compoundinfo_beta.txt",
    "data/processed/fullspace/pertkge_ppi_graph.csv",
    "data/processed/tahoe/tahoe_a549.h5ad",
    "data/processed/tahoe/tahoe_sw480.h5ad",
    "data/processed/tahoe/tahoe_nci_h460.h5ad",
    "data/processed/tahoe/tahoe_dti.csv",
    "data/processed/proteomics/proteomicsdb_data.csv",
)
METADATA_INPUTS = (
    "data/processed/lincs/bulk_protocol_lock_v2.json",
    "data/processed/fullspace/splits_v3.json",
    "data/processed/tahoe/sc_protocol_lock_v3_20260326.json",
    "code/original_runs/r259_261/proteomics_results/round261_proteomics_round259_embedding_method_lock.json",
)
GENE_ORDER_SOURCE = "data/processed/fullspace/round259_target_embedding_aligned_to_proteomics_gene_order.csv"
TRAINING_STAGES = (
    "all",
    "gene-prior",
    "closed-bulk",
    "support-initializer",
    "cross-cell-adapter",
    "full-space",
    "single-cell",
    "proteomics",
)
SINGLE_CELL_FILES = {
    "A549": "a549",
    "SW480": "sw480",
    "NCI-H460": "nci_h460",
}
TARGET_TABLE = "data/processed/fullspace/round259_target_embedding.csv"


def assert_private_output(root: Path, path: Path) -> None:
    """Reject output paths redirected through symlinks or Windows junctions."""
    root = root.resolve()
    absolute = path.absolute()
    if not absolute.is_relative_to(root) or not absolute.resolve().is_relative_to(root):
        raise RuntimeError(f"training output path escapes workspace: {path}")
    for component in (absolute, *absolute.parents):
        if component == root:
            break
        if component.is_symlink():
            raise RuntimeError(f"training output path contains a symlink: {component}")
        try:
            attributes = getattr(component.lstat(), "st_file_attributes", 0)
        except FileNotFoundError:
            continue
        if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise RuntimeError(
                f"training output path contains a reparse point: {component}"
            )


SOURCE_OVERLAY = {
    "training/structural_graph_encoder/structural_graph_encoder.py": "compat/home/lwn/claude/manuscript/write/claude_round176_hgtlite_structural_5x_pc3_20260508/scripts/run_round176_hgtlite_structural_5x_pc3.py",
    "training/gene_prior/pretrain_gene_prior.py": "compat/home/lwn/claude/manuscript/write/claude_round210_dim256_checkpoint_sweep_20260515/scripts/run_round210_dim256_checkpoint_sweep.py",
    "training/bulk_readout/bulk_readout.py": "compat/home/lwn/claude/manuscript/write/claude_round211_dim256_readout_capacity_sweep_20260515/scripts/run_round211_dim256_readout_capacity_sweep.py",
    "training/support_query_shift/support_query_shift.py": "compat/home/lwn/claude/manuscript/write/claude_round215_direct_query_shift_adapter_20260517/scripts/run_round208_a01_fixedtarget_maml_readout.py",
    "training/full_space_architecture/full_space_architecture.py": "compat/home/lwn/claude/manuscript/write/claude_round230_scratch_mlp_metar_target_tower_pc3_20260518/scripts/run_round230_scratch_mlp_metar_target_tower_pc3.py",
    "training/leakage_safe_support/leakage_safe_support.py": "compat/home/lwn/claude/manuscript/write/claude_round243_leakage_safe_unseen_cell_support_adapter_20260519/scripts/run_round243_leakage_safe_unseen_cell_support_adapter.py",
    "training/closed_bulk/train_closed_bulk.py": "compat/home/lwn/claude/manuscript/write/claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519/scripts/run_round246_joint_seen_target_train_pc3_a375_mcf7.py",
    "training/support_initializer/train_support_initializer.py": "compat/home/lwn/claude/manuscript/write/claude_round247_support_curve_query_gated_adapter_20260520/scripts/run_round247_support_curve_query_gated_adapter.py",
    "training/episodic_meta_learning/episodic_meta_learning.py": "compat/home/lwn/claude/manuscript/write/claude_round250_epmeta_stable_tuning_20260520/scripts/run_round250_epmeta_stable_tuning.py",
    "training/cross_cell_adapter/train_cross_cell_adapter.py": "compat/home/lwn/claude/manuscript/write/claude_round252_epiqgate_pseudoval_sweep_20260520/scripts/run_round252_epiqgate_pseudoval_sweep.py",
    "training/modality_transfer/scripts/align_proteomics_targets.py": "compat/home/lwn/claude/manuscript/write/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/scripts/align_round259_embedding_to_proteomics_gene_order.py",
    "training/modality_transfer/scripts/train_full_space.py": "compat/home/lwn/claude/manuscript/write/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/scripts/run_round259_crosscell_fullspace_round252query.py",
    "training/modality_transfer/scripts/train_single_cell.py": "compat/home/lwn/claude/manuscript/write/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/scripts/run_round260_sc_round259_embedding.py",
    "training/modality_transfer/scripts/train_proteomics.py": "compat/home/lwn/claude/manuscript/write/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/scripts/run_round261_proteomics_round259_embedding.py",
    "training/modality_transfer/scripts/single_cell_model.py": "compat/home/lwn/claude/manuscript/write/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/scripts/run_sc_label_bag_calibration_multicell_base.py",
    "training/modality_transfer/train_A549/train_single_cell.py": "compat/home/lwn/claude/manuscript/write/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/train_A549/run_round260_sc_round259_embedding.py",
    "training/modality_transfer/train_A549/single_cell_model.py": "compat/home/lwn/claude/manuscript/write/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/train_A549/run_sc_label_bag_calibration_multicell_base.py",
    "training/modality_transfer/train_NCI_H460/train_single_cell.py": "compat/home/lwn/claude/manuscript/write/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/train_NCI_H460/run_round260_sc_round259_embedding.py",
    "training/modality_transfer/train_NCI_H460/single_cell_model.py": "compat/home/lwn/claude/manuscript/write/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/train_NCI_H460/run_sc_label_bag_calibration_multicell_base.py",
    "training/modality_transfer/train_SW480/train_single_cell.py": "compat/home/lwn/claude/manuscript/write/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/train_SW480/run_round260_sc_round259_embedding.py",
    "training/modality_transfer/train_SW480/single_cell_model.py": "compat/home/lwn/claude/manuscript/write/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/train_SW480/run_sc_label_bag_calibration_multicell_base.py",
    "training/bulk_data/prepare_inductive_bulk.py": "compat/home/lwn/claude/manuscript/write/bulk_sharedsample_indunseenlabel_20260330_h07/scripts/train_bulk_sharedsample_indunseenlabel_v1_20260330.py",
    "training/bulk_data/bulk_dual_branch.py": "compat/home/lwn/claude/manuscript/write/bulk_sharedsample_indunseenlabel_20260330_h07/scripts/train_bulk_dualbranch_v2_20260328.py",
    "training/support_query_shift/fixed_cell_panel.json": "compat/home/lwn/claude/manuscript/write/claude_round215_direct_query_shift_adapter_20260517/round215_fixed_cell_panel.json",
}


def code_overlay(
    code_root: Path, stage: str = "all", cell_line: str | None = None
) -> dict[str, str]:
    """Map canonical source names to the frozen scientific execution layout."""
    if stage == "all":
        return dict(SOURCE_OVERLAY)
    if stage == "single-cell":
        directory = f"training/modality_transfer/train_{cell_line.replace('-', '_')}/"
        return {
            key: value
            for key, value in SOURCE_OVERLAY.items()
            if key.startswith(directory)
        }
    if stage == "proteomics":
        names = {"train_proteomics.py", "align_proteomics_targets.py"}
        return {
            key: value
            for key, value in SOURCE_OVERLAY.items()
            if Path(key).name in names
        }
    if stage == "gene-prior":
        groups = {"structural_graph_encoder", "gene_prior", "bulk_readout"}
    elif stage == "full-space":
        groups = {"full_space_architecture", "bulk_data"}
    else:
        groups = {
            "structural_graph_encoder",
            "bulk_readout",
            "support_query_shift",
            "leakage_safe_support",
            "closed_bulk",
            "bulk_data",
        }
        if stage in {"support-initializer", "cross-cell-adapter"}:
            groups.add("support_initializer")
        if stage == "cross-cell-adapter":
            groups.update({"episodic_meta_learning", "cross_cell_adapter"})
    return {
        key: value
        for key, value in SOURCE_OVERLAY.items()
        if key.split("/")[1] in groups
        or (stage == "full-space" and key.endswith("scripts/train_full_space.py"))
    }


def workspace_inputs(
    stage: str, cell_line: str | None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select assay inputs; unrelated assay downloads are not prerequisites."""
    if stage == "all":
        return EXTERNAL_INPUTS, METADATA_INPUTS
    if stage == "gene-prior":
        return EXTERNAL_INPUTS[:3], ()
    if stage == "single-cell":
        name = SINGLE_CELL_FILES[cell_line]
        return (
            f"data/processed/tahoe/tahoe_{name}.h5ad",
            "data/processed/tahoe/tahoe_dti.csv",
        ), (METADATA_INPUTS[2],)
    if stage == "proteomics":
        return (EXTERNAL_INPUTS[-1],), (METADATA_INPUTS[-1],)
    return EXTERNAL_INPUTS[3:6], METADATA_INPUTS[:2]


def pretrained_inputs(stage: str) -> dict[str, str]:
    """Map read-only upstream release files to private runtime destinations."""
    # Imported lazily because training imports this module's layout constants.
    from .training import LOCKED_OUTPUTS

    upstream = {
        "closed-bulk": ("gene-prior",),
        "support-initializer": ("gene-prior", "closed-bulk"),
        "cross-cell-adapter": ("gene-prior", "closed-bulk", "support-initializer"),
        "full-space": ("closed-bulk",),
    }
    names = {
        "gene-prior": "round210_process_only_epoch030.pth",
        "closed-bulk": "round246_whole_model_checkpoint.pth",
        "support-initializer": "round247_adapter_init.pth",
    }
    result = {
        f"checkpoints/selected/{names[name]}": LOCKED_OUTPUTS[name]
        for name in upstream.get(stage, ())
    }
    if stage in {"single-cell", "proteomics"}:
        result[TARGET_TABLE] = TARGET_TABLE
    return result


def _relocate(text: str, root: Path) -> str:
    # Both historical mounts refer to one isolated scratch tree.
    mapped = (root / "compat/home/lwn").as_posix()
    rewritten = text.replace("/NFS2_home/NFS2_home_1/lwn", "__METAR_TRAIN_HOME__")
    rewritten = rewritten.replace("/home/lwn", "__METAR_TRAIN_HOME__")
    rewritten = rewritten.replace(
        "C:/Users/26328/Desktop/metar", "__METAR_TRAIN_ROOT__"
    )
    rewritten = rewritten.replace(
        r"C:\Users\26328\Desktop\metar", "__METAR_TRAIN_ROOT__"
    )
    return rewritten.replace("__METAR_TRAIN_HOME__", mapped).replace(
        "__METAR_TRAIN_ROOT__", root.as_posix()
    )


def _copy(
    source: Path, destination: Path, root: Path, *, relocate: bool = False
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if relocate:
        text = source.read_text(encoding="utf-8-sig")
        destination.write_text(_relocate(text, root), encoding="utf-8")
    else:
        shutil.copy2(source, destination)


def _link(source: Path, destination: Path, *, copy_inputs: bool = False) -> bool:
    """Materialize an input, returning whether a link restriction required a copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_inputs:
        shutil.copy2(source, destination)
    else:
        try:
            destination.symlink_to(source.resolve())
        except OSError as exc:
            if getattr(exc, "winerror", None) != 1314 and exc.errno not in {
                errno.EPERM,
                errno.EACCES,
                errno.ENOTSUP,
            }:
                raise
            shutil.copy2(source, destination)
            return True
    return False


def prepare_training_workspace(
    code_root: str | Path,
    artifact_root: str | Path,
    destination: str | Path,
    *,
    check_only: bool = False,
    copy_inputs: bool = False,
    stage: str = "all",
    reuse_pretrained: bool = False,
    cell_line: str | None = None,
) -> dict[str, object]:
    """Inspect prerequisites, then create a fresh tree with private mutable outputs.

    Large inputs are symlinked and are used only for reading. Sources and mutable
    metadata are ordinary copies. A pre-existing destination is never modified.
    Scratch mode copies no selected model values. Stage-specific reuse copies
    only upstream weights or vectors, never that stage's selected output.
    """
    if stage not in TRAINING_STAGES:
        raise ValueError(f"unknown training stage: {stage}")
    if stage == "single-cell" and cell_line not in SINGLE_CELL_FILES:
        raise ValueError(
            "single-cell preparation requires --cell-line A549, SW480 or NCI-H460"
        )
    if cell_line is not None and stage != "single-cell":
        raise ValueError("--cell-line is only used with --stage single-cell")
    if reuse_pretrained and stage in {"all", "gene-prior"}:
        raise ValueError(
            "--reuse-pretrained requires a stage with upstream pretrained inputs"
        )
    code = Path(code_root).resolve()
    artifacts = Path(artifact_root).resolve()
    candidate = Path(destination).absolute()
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"training destination must not exist: {candidate}")
    root = candidate.resolve()
    for source_root in (code, artifacts):
        if root.is_relative_to(source_root) or source_root.is_relative_to(root):
            raise ValueError("training destination and source roots must be separate")
    overlay = code_overlay(code, stage, cell_line)
    external, metadata = workspace_inputs(stage, cell_line)
    warm_start = pretrained_inputs(stage) if reuse_pretrained else {}
    needs_gene_order = stage in {"all", "proteomics"}
    extra_sources: dict[str, str] = {}
    if reuse_pretrained and stage == "support-initializer":
        extra_sources["training/closed_bulk/selected_configuration.json"] = (
            f"{WRITE_ROOT}/{ROUND_NAMES['r246']}/source_data/selected_r246_joint_lm0p5_sc0p15_mn2p5.json"
        )
    if stage == "full-space":
        extra_sources["training/cross_cell_adapter/cell_splits.json"] = (
            f"{WRITE_ROOT}/{ROUND_NAMES['r252']}/source_data/cell_splits.json"
        )
    required = [(code, path) for path in overlay]
    required.extend((code, path) for path in extra_sources)
    required.extend((artifacts, path) for path in (*external, *metadata, *warm_start))
    if needs_gene_order:
        required.append((artifacts, GENE_ORDER_SOURCE))
    missing = [
        (base / path).as_posix()
        for base, path in required
        if not (base / path).is_file()
    ]
    report: dict[str, object] = {
        "status": "missing_inputs" if missing else "inputs_present",
        "workspace": str(root),
        "code_root": str(code),
        "artifact_root": str(artifacts),
        "missing_inputs": missing,
        "stage": stage,
        "cell_line": cell_line,
        "reuse_pretrained": reuse_pretrained,
        "selected_checkpoints_copied": any(key.endswith(".pth") for key in warm_start),
        "reused_inputs": list(warm_start),
        "prepared_stages": list(TRAINING_STAGES[1:]) if stage == "all" else [stage],
        "input_mode": "copy" if copy_inputs else "symlink",
        "cache_present": (
            artifacts / "data/processed/fullspace/h07_prepared_data.pt"
        ).is_file(),
        "verification": "file presence only; no training, model loading, or inference",
    }
    if check_only:
        return report
    if missing:
        raise FileNotFoundError("missing training inputs:\n" + "\n".join(missing))
    root.mkdir(parents=True)
    used_copy_fallback = False

    def materialize_input(source: Path, target: Path) -> None:
        nonlocal used_copy_fallback
        fallback = _link(source, target, copy_inputs=copy_inputs)
        if fallback and not used_copy_fallback:
            print(
                "MeTar: input symlinks are unavailable; copying affected inputs "
                "into the training workspace. Use --copy-inputs to copy all inputs.",
                file=sys.stderr,
                flush=True,
            )
            used_copy_fallback = True

    for relative, target in overlay.items():
        _copy(code / relative, root / target, root, relocate=True)
        if target.endswith("run_round210_dim256_checkpoint_sweep.py"):
            # R210 launches its own readout subprocesses. Each needs the same
            # explicit NumPy-metadata opt-in as the parent training runtime.
            path = root / target
            text = path.read_text(encoding="utf-8")
            old = "import torch.nn.functional as F\n"
            if text.count(old) != 1:
                raise RuntimeError(
                    "Round210 compatibility adaptation no longer matches source"
                )
            new = (
                old
                + "import functools\ntorch.load = functools.partial(torch.load, weights_only=False)\n"
            )
            path.write_text(text.replace(old, new), encoding="utf-8")
        if target.endswith("run_round230_scratch_mlp_metar_target_tower_pc3.py"):
            path = root / target
            text = path.read_text(encoding="utf-8")
            old = 'H07_CACHE = REMOTE_WRITE_ROOT / "claude_round209_fullspace_a01_extrapolation_20260517" / "cache" / "h07_prepared_data.pt"'
            if text.count(old) != 1:
                raise RuntimeError(
                    "h07 cache adaptation no longer matches the archived source"
                )
            path.write_text(
                text.replace(
                    old,
                    f"H07_CACHE = Path({(root / 'data/processed/fullspace/h07_prepared_data.pt').as_posix()!r})",
                ),
                encoding="utf-8",
            )
    # Preserve released cache-builder paths and sibling basenames too.
    for relative, target in overlay.items():
        if relative.startswith("training/bulk_data/"):
            legacy = "code/remote_dependencies/h07/" + Path(target).name
            _copy(code / relative, root / legacy, root, relocate=True)
    for relative, target in extra_sources.items():
        _copy(code / relative, root / target, root, relocate=True)
    for relative in metadata:
        _copy(artifacts / relative, root / relative, root, relocate=True)
    for relative in external:
        materialize_input(artifacts / relative, root / relative)
    cache = "data/processed/fullspace/h07_prepared_data.pt"
    if (
        stage not in {"gene-prior", "single-cell", "proteomics"}
        and (artifacts / cache).is_file()
    ):
        materialize_input(artifacts / cache, root / cache)
    for relative, target in warm_start.items():
        _copy(artifacts / relative, root / target, root)
    if TARGET_TABLE in warm_start:
        # The selected single-cell source reads the table in its own source_data.
        target = f"{WRITE_ROOT}/{ROUND_NAMES['r259_261']}/source_data/round259_target_embedding.csv"
        _copy(artifacts / TARGET_TABLE, root / target, root)
    aliases = {
        "data/processed/closed/r210/pc3_landmark_response_978.tsv.gz": f"{WRITE_ROOT}/claude_round93_landmark_target_relation_bridge_pc3_20260507/source_data/pc3_landmark_response_978.tsv.gz",
        "data/processed/closed/r210/pc3_metar_drugcell_scores_668.tsv.gz": f"{WRITE_ROOT}/claude_round92_metar_fixed_graph_prior_pc3_20260507/source_data/pc3_metar_drugcell_scores_668.tsv.gz",
        "data/public_raw/gene_prior/process.txt": f"{WRITE_ROOT}/claude_round91_pertkge_inductive_single_cellline_20260507/cell_PC3/pertkge/knowledge_graph/process.txt",
        "data/processed/lincs/large_data.parquet": "compat/home/lwn/claude/bulk/data/large_data.parquet",
        "data/processed/lincs/compoundinfo_beta.txt": "compat/home/lwn/claude/bulk/data/compoundinfo_beta.txt",
        "data/processed/fullspace/pertkge_ppi_graph.csv": "compat/home/lwn/claude/bulk/data/pertkge_ppi_graph.csv",
        "data/processed/fullspace/splits_v3.json": "compat/home/lwn/claude/metar/splits_v3.json",
        "data/processed/tahoe/sc_protocol_lock_v3_20260326.json": f"{WRITE_ROOT}/sc_agg_upgrade_20260321/derived_tables/sc_protocol_lock_v3_20260326.json",
    }
    for relative in EXTERNAL_INPUTS:
        if "/tahoe/" in relative:
            aliases[relative] = f"compat/home/lwn/claude/sc/data/{Path(relative).name}"
    for relative, target in aliases.items():
        if (root / relative).is_file():
            materialize_input(root / relative, root / target)
    for round_name in (
        "bulk_sharedsample_indunseenlabel_20260330_h07_w060_u8e4_med",
        "bulk_dualbranch_20260328_b01_warm_seen_joint",
    ):
        target = root / WRITE_ROOT / round_name / "manifests/bulk_protocol_lock_v2.json"
        protocol = root / "data/processed/lincs/bulk_protocol_lock_v2.json"
        if protocol.is_file():
            _copy(protocol, target, root)
    if METADATA_INPUTS[0] in metadata:
        (
            root
            / WRITE_ROOT
            / "bulk_sharedsample_indunseenlabel_20260330_h07/derived_tables"
        ).mkdir(parents=True, exist_ok=True)
    if needs_gene_order:
        gene_order = root / "data/processed/proteomics/gene_order.tsv"
        gene_order.parent.mkdir(parents=True, exist_ok=True)
        with (artifacts / GENE_ORDER_SOURCE).open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.reader(handle)
            next(reader)
            names = [row[0] for row in reader]
        if not names or len(names) != len(set(names)):
            raise ValueError("proteomics gene-order source must have unique gene names")
        gene_order.write_text("gene\n" + "\n".join(names) + "\n", encoding="utf-8")
    report["status"] = "prepared"
    if used_copy_fallback:
        report["input_mode"] = "copy_fallback"
    report["schema_version"] = 2
    (root / MARKER).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
