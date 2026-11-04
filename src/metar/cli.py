"""Command-line interface for article-aligned MeTar prediction and training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .consensus import consensus_from_long
from .io import (
    load_feature_matrix,
    load_grouped_cells,
    load_support_labels,
    load_target_table,
    read_feature_order,
    read_table,
)
from .predict import (
    predict_bulk_668,
    predict_cross_cell,
    predict_full_space,
    predict_proteomics,
    predict_single_cell,
)
from .preprocessing import ProteomicsPreprocessor, fit_round261_preprocessor
from .result import PredictionResult
from .runtime import (
    RunReporter,
    report_metrics,
    report_progress,
    report_step,
    resolve_device,
)
from .training import SINGLE_CELL_SELECTIONS, run_training_stage

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(os.environ.get("METAR_ARTIFACT_ROOT", PACKAGE_ROOT)).expanduser()
DEFAULTS = {
    "closed_checkpoint": ARTIFACT_ROOT
    / "checkpoints/selected/round246_whole_model_checkpoint.pth",
    "adapter_checkpoint": ARTIFACT_ROOT
    / "checkpoints/selected/round252_adapter_seed20260524.pth",
    "full_checkpoint": ARTIFACT_ROOT / "checkpoints/selected/round259_e13_bce001.pth",
    "target_table": ARTIFACT_ROOT
    / "data/processed/fullspace/round259_target_embedding.csv",
    "proteomics_target_table": ARTIFACT_ROOT
    / "data/processed/fullspace/round259_target_embedding_aligned_to_proteomics_gene_order.csv",
    "feature_order": ARTIFACT_ROOT / "data/application_inputs/pdf3/geneinfo_beta.txt",
    "proteomics_checkpoint": ARTIFACT_ROOT
    / "checkpoints/selected/round261_proteomics_epoch070.pth",
    "proteomics_reference": ARTIFACT_ROOT
    / "data/processed/proteomics/proteomicsdb_data.csv",
    "proteomics_preprocessor": ARTIFACT_ROOT
    / "data/processed/proteomics/round261_preprocessor.json",
}
DEFAULT_SINGLE_CELL = {
    "A549": ARTIFACT_ROOT / "checkpoints/selected/round260_A549_seed2043.pth",
    "SW480": ARTIFACT_ROOT / "checkpoints/selected/round260_SW480_seed2041.pth",
    "NCI-H460": ARTIFACT_ROOT / "checkpoints/selected/round260_NCI_H460_seed2042.pth",
}


def _existing_default(name: str) -> str | None:
    path = DEFAULTS[name]
    return str(path) if path.exists() else None


def _require_path(value: str | None, label: str) -> Path:
    if not value:
        raise ValueError(
            f"{label} is required because the hydrated default is unavailable"
        )
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(
            f"{label} is empty: {path}; restore the complete input or release asset"
        )
    return path


def _validate_prediction_output(args: argparse.Namespace) -> None:
    """Reject destinations that would replace an input or a selected model asset."""
    protected: list[Path] = []
    for name in (
        "input",
        "support_input",
        "support_labels",
        "support_weights",
        "checkpoint",
        "adapter_checkpoint",
        "target_table",
        "preprocessor",
        "reference_data",
        "target_order",
        "feature_order",
    ):
        value = getattr(args, name, None)
        if value:
            values = value if isinstance(value, list) else [value]
            protected.extend(Path(item).expanduser() for item in values)

    def select(explicit: str | None, default: str) -> str | None:
        value = explicit or _existing_default(default)
        if value:
            protected.append(Path(value).expanduser())
        return value

    scenario = args.scenario
    if scenario in {"bulk-668", "cross-cell", "full-space"}:
        select(args.feature_order, "feature_order")
    if scenario in {"bulk-668", "cross-cell"}:
        select(args.checkpoint, "closed_checkpoint")
    elif scenario == "full-space":
        select(args.checkpoint, "full_checkpoint")
    elif scenario == "proteomics":
        select(args.checkpoint, "proteomics_checkpoint")
    elif scenario == "single-cell":
        value = args.checkpoint or DEFAULT_SINGLE_CELL.get(args.cell_line)
        if value and Path(value).is_file():
            protected.append(Path(value))
    if scenario == "cross-cell" and args.mode == "adapted":
        select(args.adapter_checkpoint, "adapter_checkpoint")
    if scenario in {"single-cell", "full-space", "proteomics"}:
        select(
            args.target_table,
            "proteomics_target_table" if scenario == "proteomics" else "target_table",
        )
    outputs = [Path(args.output).expanduser()]
    if scenario == "proteomics":
        preprocessor = select(args.preprocessor, "proteomics_preprocessor")
        if preprocessor is None:
            select(args.reference_data, "proteomics_reference")
            if args.write_preprocessor:
                outputs.append(Path(args.write_preprocessor).expanduser())

    def same_path(left: Path, right: Path) -> bool:
        return left.resolve() == right.resolve() or (
            left.exists() and right.exists() and left.samefile(right)
        )

    for index, output in enumerate(outputs):
        for source in protected:
            if same_path(output, source):
                raise ValueError(
                    "prediction output would overwrite an input or model asset: "
                    f"{output} (input: {source})"
                )
        if any(same_path(output, earlier) for earlier in outputs[:index]):
            raise ValueError(
                "prediction and preprocessor outputs must use different paths"
            )


def _feature_order(args: argparse.Namespace) -> list[str] | None:
    value = args.feature_order or _existing_default("feature_order")
    if value is None:
        return None
    return read_feature_order(value, column=args.feature_column)


def _with_metadata(result: PredictionResult, **values: object) -> PredictionResult:
    metadata = result.metadata.reset_index(drop=True).copy()
    if metadata.empty:
        metadata = pd.DataFrame(index=range(len(result.query_ids)))
    for key, value in values.items():
        if value is not None:
            metadata[key] = value
    return PredictionResult(
        result.scores,
        result.targets,
        result.query_ids,
        result.scenario,
        metadata,
    )


def _write_result(result: PredictionResult, args: argparse.Namespace) -> int:
    report_step("Write ranked predictions")
    output = result.write(
        args.output, output_format=args.output_format, top_k=args.top_k
    )
    report_metrics(
        {
            "queries": len(result.query_ids),
            "targets": len(result.targets),
            "output": output.resolve(),
        },
        label="Summary",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "scenario": result.scenario,
                "queries": len(result.query_ids),
                "candidate_targets": len(result.targets),
                "output": str(output.resolve()),
            },
            indent=2,
        )
    )
    return 0


def command_bulk_668(args: argparse.Namespace) -> int:
    features, query_ids = load_feature_matrix(
        args.input,
        expected_dim=978,
        id_column=args.id_column,
        feature_order=_feature_order(args),
    )
    result = predict_bulk_668(
        features,
        query_ids,
        _require_path(
            args.checkpoint or _existing_default("closed_checkpoint"),
            "bulk checkpoint",
        ),
        device=args.device,
        batch_size=args.batch_size,
    )
    return _write_result(
        _with_metadata(result, cell_line=args.cell_line, drug=query_ids), args
    )


def command_full_space(args: argparse.Namespace) -> int:
    features, query_ids = load_feature_matrix(
        args.input,
        expected_dim=978,
        id_column=args.id_column,
        feature_order=_feature_order(args),
    )
    target_names, target_values = load_target_table(
        _require_path(
            args.target_table or _existing_default("target_table"),
            "target embedding table",
        )
    )
    result = predict_full_space(
        features,
        query_ids,
        _require_path(
            args.checkpoint or _existing_default("full_checkpoint"),
            "full-space checkpoint",
        ),
        target_names,
        target_values,
        device=args.device,
        batch_size=args.batch_size,
    )
    return _write_result(
        _with_metadata(result, cell_line=args.cell_line, drug=query_ids), args
    )


def _load_weights(path: str | None, count: int) -> np.ndarray | None:
    if path is None:
        return None
    source = Path(path)
    if source.suffix.lower() == ".npy":
        values = np.load(source, allow_pickle=False)
    else:
        frame = read_table(source)
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.shape[1] != 1:
            raise ValueError(
                "support weight table must contain exactly one numeric column"
            )
        values = numeric.iloc[:, 0].to_numpy()
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (count,):
        raise ValueError(f"expected {count} support weights, observed {values.shape}")
    return values


def command_cross_cell(args: argparse.Namespace) -> int:
    order = _feature_order(args)
    query_features, query_ids = load_feature_matrix(
        args.input, expected_dim=978, id_column=args.id_column, feature_order=order
    )
    base_path = _require_path(
        args.checkpoint or _existing_default("closed_checkpoint"), "bulk checkpoint"
    )
    if args.mode == "direct":
        result = predict_bulk_668(
            query_features,
            query_ids,
            base_path,
            device=args.device,
            batch_size=args.batch_size,
        )
        result = PredictionResult(
            result.scores,
            result.targets,
            result.query_ids,
            "cross_cell_direct",
        )
        return _write_result(
            _with_metadata(
                result,
                cell_line=args.cell_line,
                drug=query_ids,
                prediction_mode="direct",
            ),
            args,
        )
    if not args.support_input or not args.support_labels:
        raise ValueError(
            "adapted cross-cell prediction requires --support-input and --support-labels"
        )
    support_features, support_ids = load_feature_matrix(
        args.support_input,
        expected_dim=978,
        id_column=args.id_column,
        feature_order=order,
    )
    from .models import ClosedDiagReadout

    _base, targets, _payload = ClosedDiagReadout.from_checkpoint(base_path)
    support_labels = load_support_labels(
        args.support_labels,
        support_ids,
        targets,
        id_column=args.id_column,
        target_column=args.target_column,
    )
    result = predict_cross_cell(
        query_features,
        query_ids,
        support_features,
        support_labels,
        base_path,
        _require_path(
            args.adapter_checkpoint or _existing_default("adapter_checkpoint"),
            "support adapter checkpoint",
        ),
        support_ids=support_ids,
        support_weights=_load_weights(args.support_weights, len(support_ids)),
        device=args.device,
        batch_size=args.batch_size,
    )
    return _write_result(
        _with_metadata(result, cell_line=args.cell_line, drug=query_ids), args
    )


def command_single_cell(args: argparse.Namespace) -> int:
    features, groups = load_grouped_cells(
        args.input, group_column=args.id_column, expected_dim=128
    )
    target_names, target_values = load_target_table(
        _require_path(
            args.target_table or _existing_default("target_table"),
            "target embedding table",
        )
    )
    checkpoint = args.checkpoint or (
        str(DEFAULT_SINGLE_CELL[args.cell_line])
        if DEFAULT_SINGLE_CELL[args.cell_line].exists()
        else None
    )
    result = predict_single_cell(
        features,
        groups,
        _require_path(checkpoint, f"single-cell {args.cell_line} checkpoint"),
        target_names,
        target_values,
        cell_line=args.cell_line,
        max_cells=args.max_cells,
        sampling_seed=args.sampling_seed,
        device=args.device,
    )
    result = _with_metadata(result, drug=result.query_ids)
    return _write_result(result, args)


def command_proteomics(args: argparse.Namespace) -> int:
    frame = read_table(args.input)
    if args.id_column in frame.columns and (
        frame[args.id_column].isna().any()
        or frame[args.id_column].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("proteomics query identifiers must be nonempty")
    query_ids = (
        tuple(frame[args.id_column].astype(str))
        if args.id_column in frame.columns
        else tuple(f"query_{index}" for index in range(len(frame)))
    )
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("proteomics query identifiers must be unique")
    target_names, target_values = load_target_table(
        _require_path(
            args.target_table or _existing_default("proteomics_target_table"),
            "proteomics-aligned target table",
        )
    )
    preprocessor = _load_proteomics_preprocessor(args, target_names)
    result = predict_proteomics(
        frame,
        query_ids,
        _require_path(
            args.checkpoint or _existing_default("proteomics_checkpoint"),
            "proteomics checkpoint",
        ),
        target_names,
        target_values,
        preprocessor,
        device=args.device,
        batch_size=args.batch_size,
    )
    return _write_result(_with_metadata(result, drug=query_ids), args)


def _load_proteomics_preprocessor(
    args: argparse.Namespace, target_names: tuple[str, ...]
) -> ProteomicsPreprocessor:
    preprocessor_path = args.preprocessor or _existing_default(
        "proteomics_preprocessor"
    )
    if preprocessor_path:
        return ProteomicsPreprocessor.read(
            _require_path(preprocessor_path, "proteomics preprocessor")
        )
    reference = _require_path(
        args.reference_data or _existing_default("proteomics_reference"),
        "proteomics reference data or --preprocessor",
    )
    preprocessor = fit_round261_preprocessor(reference, target_names)
    if args.write_preprocessor:
        preprocessor.write(args.write_preprocessor)
    return preprocessor


def command_consensus(args: argparse.Namespace) -> int:
    frames = [read_table(path) for path in args.input]
    target_order = (
        tuple(read_feature_order(args.target_order, column=args.target_order_column))
        if args.target_order
        else None
    )
    result = consensus_from_long(
        pd.concat(frames, ignore_index=True),
        method=args.method,
        drug_column=args.drug_column,
        cell_line_column=args.cell_line_column,
        min_cell_lines=args.min_cell_lines,
        target_order=target_order,
    )
    return _write_result(result, args)


def command_train(args: argparse.Namespace) -> int:
    if args.stage != "all":
        return run_training_stage(
            args.stage,
            args.workspace,
            device=args.device,
            output_dir=args.output_dir,
            cell_line=args.cell_line,
            dry_run=args.dry_run,
            check_only=args.check_only,
        )
    sequence: list[tuple[str, str | None]] = [
        ("gene-prior", None),
        ("closed-bulk", None),
        ("support-initializer", None),
        ("cross-cell-adapter", None),
        ("full-space", None),
        *(("single-cell", cell) for cell in SINGLE_CELL_SELECTIONS),
        ("proteomics", None),
    ]
    result_code = 0
    for index, (stage, cell_line) in enumerate(sequence, 1):
        report_step(
            f"Training stage {index}/{len(sequence)}: {stage}"
            + (f" ({cell_line})" if cell_line else "")
        )
        output_dir = args.output_dir
        if output_dir and stage in {"single-cell", "proteomics"}:
            suffix = f"single_cell_{cell_line}" if cell_line else stage
            output_dir = str(Path(output_dir) / suffix)
        code = run_training_stage(
            stage,
            args.workspace,
            device=args.device,
            output_dir=output_dir,
            cell_line=cell_line,
            dry_run=args.dry_run,
            check_only=args.check_only,
        )
        if code == 0:
            report_progress(index, len(sequence), label="Training stages")
        if code != 0:
            result_code = code
            if not args.check_only:
                return code
    return result_code


def command_describe(_args: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "prediction_scenarios": {
                    "bulk-668": "Direct ranking over 668 targets",
                    "full-space": "Direct ranking over 16,583 genes",
                    "cross-cell": "Shared bulk model with support adaptation",
                    "single-cell": "Cell-line-specific model over processed X128 cells",
                    "proteomics": "Protein-abundance model with training-derived preprocessing",
                    "consensus": "post-prediction raw or candidate-wise z-score mean across cell lines",
                },
                "training_stages": [
                    "gene-prior",
                    "closed-bulk",
                    "support-initializer",
                    "cross-cell-adapter",
                    "full-space",
                    "single-cell",
                    "proteomics",
                ],
            },
            indent=2,
        )
    )
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    from .diagnostics import check_prediction

    report = check_prediction(args, DEFAULTS, DEFAULT_SINGLE_CELL)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-format", choices=("long", "wide"), default="long")
    parser.add_argument(
        "--top-k",
        type=_nonnegative_int,
        default=0,
        help="0 writes the complete candidate space",
    )
    _add_quiet_option(parser)


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return number


def _add_quiet_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Hide progress on stderr; keep result JSON on stdout",
    )


def _add_bulk_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--id-column", default="query_id")
    parser.add_argument("--feature-order")
    parser.add_argument("--feature-column")
    parser.add_argument("--cell-line")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:N or mps")
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    _add_output_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="metar", description=__doc__)
    _add_quiet_option(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)
    describe = subparsers.add_parser(
        "describe", help="List scientific scenarios and selected stages"
    )
    describe.set_defaults(handler=command_describe)

    doctor = subparsers.add_parser(
        "doctor",
        help="Check required files and input dimensions without running a model",
    )
    doctor.add_argument(
        "--scenario",
        required=True,
        choices=(
            "bulk-668",
            "full-space",
            "cross-cell",
            "single-cell",
            "proteomics",
            "consensus",
        ),
    )
    doctor.add_argument("--input", action="append")
    doctor.add_argument("--checkpoint")
    doctor.add_argument("--target-table")
    doctor.add_argument("--adapter-checkpoint")
    doctor.add_argument("--preprocessor")
    doctor.add_argument("--feature-order")
    doctor.add_argument("--feature-column")
    doctor.add_argument("--id-column", default="query_id")
    doctor.add_argument("--cell-line")
    doctor.add_argument("--mode", choices=("direct", "adapted"), default="adapted")
    doctor.add_argument("--support-input")
    doctor.add_argument("--support-labels")
    doctor.add_argument("--target-column", default="targets")
    doctor.add_argument("--drug-column", default="query_id")
    doctor.add_argument("--cell-line-column", default="cell_line")
    doctor.add_argument("--min-cell-lines", type=int, default=2)
    doctor.add_argument("--device", default="auto")
    doctor.set_defaults(handler=command_doctor)

    predict = subparsers.add_parser(
        "predict", help="Run one article-aligned prediction scenario"
    )
    scenarios = predict.add_subparsers(dest="scenario", required=True)

    bulk = scenarios.add_parser(
        "bulk-668", help="Direct 668-target prediction from 978 features"
    )
    _add_bulk_options(bulk)
    bulk.add_argument("--checkpoint")
    bulk.set_defaults(handler=command_bulk_668)

    full = scenarios.add_parser(
        "full-space", help="Direct 16,583-gene prediction from 978 features"
    )
    _add_bulk_options(full)
    full.add_argument("--checkpoint")
    full.add_argument("--target-table")
    full.set_defaults(handler=command_full_space)

    cross = scenarios.add_parser(
        "cross-cell", help="Support-conditioned held-out-cell prediction"
    )
    _add_bulk_options(cross)
    cross.add_argument("--mode", choices=("direct", "adapted"), default="adapted")
    cross.add_argument("--checkpoint")
    cross.add_argument("--adapter-checkpoint")
    cross.add_argument("--support-input")
    cross.add_argument("--support-labels")
    cross.add_argument("--support-weights")
    cross.add_argument("--target-column", default="targets")
    cross.set_defaults(handler=command_cross_cell)

    single = scenarios.add_parser(
        "single-cell", help="Full-space prediction from grouped X128 cells"
    )
    single.add_argument("--input", required=True)
    single.add_argument("--id-column", default="query_id")
    single.add_argument(
        "--cell-line", choices=tuple(SINGLE_CELL_SELECTIONS), required=True
    )
    single.add_argument("--checkpoint")
    single.add_argument("--target-table")
    single.add_argument(
        "--max-cells",
        type=_nonnegative_int,
        default=300,
        help="Maximum cells per drug; 0 uses all cells",
    )
    single.add_argument("--sampling-seed", type=int, default=20260327)
    single.add_argument("--device", default="auto")
    _add_output_options(single)
    single.set_defaults(handler=command_single_cell)

    proteomics = scenarios.add_parser(
        "proteomics", help="Full-space prediction from protein abundance"
    )
    proteomics.add_argument("--input", required=True)
    proteomics.add_argument("--id-column", default="query_id")
    proteomics.add_argument("--checkpoint")
    proteomics.add_argument("--target-table")
    proteomics.add_argument("--preprocessor")
    proteomics.add_argument("--reference-data")
    proteomics.add_argument("--write-preprocessor")
    proteomics.add_argument("--device", default="auto")
    proteomics.add_argument("--batch-size", type=_positive_int, default=64)
    _add_output_options(proteomics)
    proteomics.set_defaults(handler=command_proteomics)

    consensus = scenarios.add_parser(
        "consensus", help="Combine matched predictions across cell lines"
    )
    consensus.add_argument("--input", action="append", required=True)
    consensus.add_argument(
        "--method",
        choices=("cell_zscore_mean", "raw_score_mean"),
        default="cell_zscore_mean",
    )
    consensus.add_argument("--drug-column", default="query_id")
    consensus.add_argument("--cell-line-column", default="cell_line")
    consensus.add_argument("--min-cell-lines", type=_positive_int, default=2)
    consensus.add_argument("--target-order")
    consensus.add_argument("--target-order-column")
    _add_output_options(consensus)
    consensus.set_defaults(handler=command_consensus)

    train = subparsers.add_parser(
        "train", help="Run exact archived training stages in a separate workspace"
    )
    train.add_argument(
        "--stage",
        choices=(
            "gene-prior",
            "closed-bulk",
            "support-initializer",
            "cross-cell-adapter",
            "full-space",
            "single-cell",
            "proteomics",
            "all",
        ),
        required=True,
    )
    train.add_argument("--workspace", required=True)
    train.add_argument("--device", default="auto")
    _add_quiet_option(train)
    train.add_argument("--output-dir")
    train.add_argument("--cell-line", choices=tuple(SINGLE_CELL_SELECTIONS))
    train.add_argument("--dry-run", action="store_true")
    train.add_argument(
        "--check-only",
        action="store_true",
        help="Inspect stage input presence without loading models or training",
    )
    train.set_defaults(handler=command_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    old_progress = os.environ.get("METAR_PROGRESS")
    try:
        if args.command == "predict":
            _validate_prediction_output(args)
        if getattr(args, "quiet", False):
            os.environ["METAR_PROGRESS"] = "0"
        preview = args.command in {"describe", "doctor"} or (
            args.command == "train" and (args.dry_run or args.check_only)
        )
        if preview:
            return int(args.handler(args))
        requested_device = getattr(args, "device", None)
        if requested_device is not None:
            args.device = resolve_device(requested_device)
        settings = {
            "setting": getattr(args, "scenario", getattr(args, "stage", None)),
            "mode": getattr(args, "mode", None),
            "cell_line": getattr(args, "cell_line", None),
            "device": getattr(args, "device", None),
            "requested_device": requested_device,
            "batch_size": getattr(args, "batch_size", None),
            "input": getattr(args, "input", None),
            "workspace": getattr(args, "workspace", None),
            "output": getattr(args, "output", None),
        }
        with RunReporter(args.command.capitalize(), settings):
            code = int(args.handler(args))
            if code:
                raise RuntimeError(
                    f"{args.command} stopped with exit status {code}; see stage error above"
                )
            return code
    except ImportError as exc:
        parser.exit(
            2,
            f"ERROR: {exc}\nInstall optional dependencies in this Python environment: "
            f'{os.sys.executable} -m pip install -e ".[training]"\n',
        )
    except (OSError, RuntimeError, ValueError) as exc:
        detail = str(exc)
        if "out of memory" in detail.lower():
            detail += (
                "\nFor prediction, lower --batch-size (where supported), or select "
                "--device cpu and ensure sufficient RAM. For training, free device "
                "memory or choose a device with enough memory; training budgets were not changed."
            )
        parser.exit(2, f"ERROR: {detail}\n")
    finally:
        if old_progress is None:
            os.environ.pop("METAR_PROGRESS", None)
        else:
            os.environ["METAR_PROGRESS"] = old_progress
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
