#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?artifact root is required}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export METAR_REPO_ROOT="$REPO_ROOT"
if [[ ! -d "$ROOT" ]]; then
  printf 'ERROR\tmissing_artifact_root\t%s\n' "$ROOT" >&2
  exit 2
fi

link_file() {
  local target_rel=$1
  local link_rel=$2
  local target
  target=$(python3 "$REPO_ROOT/tools/resolve_source.py" --repository "$REPO_ROOT" --artifacts "$ROOT" "$target_rel")
  local link="$ROOT/$link_rel"
  if [[ ! -f "$target" ]]; then
    printf 'ERROR\tmissing_target\t%s\n' "$target_rel" >&2
    exit 3
  fi
  mkdir -p "$(dirname "$link")"
  if [[ -e "$link" || -L "$link" ]]; then
    printf 'ERROR\tcompat_path_exists\t%s\n' "$link_rel" >&2
    exit 3
  fi
  ln -s "$(realpath --relative-to="$(dirname "$link")" "$target")" "$link"
}

link_dir() {
  local target_rel=$1
  local link_rel=$2
  local target="$ROOT/$target_rel"
  local link="$ROOT/$link_rel"
  if [[ ! -d "$target" ]]; then
    printf 'ERROR\tmissing_target_dir\t%s\n' "$target_rel" >&2
    exit 3
  fi
  mkdir -p "$(dirname "$link")"
  if [[ -e "$link" || -L "$link" ]]; then
    printf 'ERROR\tcompat_path_exists\t%s\n' "$link_rel" >&2
    exit 3
  fi
  ln -s "$(realpath --relative-to="$(dirname "$link")" "$target")" "$link"
}

copy_code_file() {
  local target_rel=$1
  local copy_rel=$2
  local target
  target=$(python3 "$REPO_ROOT/tools/resolve_source.py" --repository "$REPO_ROOT" --artifacts "$ROOT" "$target_rel")
  local copy="$ROOT/$copy_rel"
  if [[ ! -f "$target" ]]; then
    printf 'ERROR\tmissing_code\t%s\n' "$target_rel" >&2
    exit 3
  fi
  mkdir -p "$(dirname "$copy")"
  if [[ -e "$copy" || -L "$copy" ]]; then
    printf 'ERROR\tcompat_path_exists\t%s\n' "$copy_rel" >&2
    exit 3
  fi
  cp -p "$target" "$copy"
}

C=compat/home/lwn/claude
W=$C/manuscript/write
N=compat/NFS2_home/NFS2_home_1/lwn

copy_code_file code/original_runs/r176/run_round176_hgtlite_structural_5x_pc3.py "$W/claude_round176_hgtlite_structural_5x_pc3_20260508/scripts/run_round176_hgtlite_structural_5x_pc3.py"
copy_code_file code/original_runs/r210/run_round210_dim256_checkpoint_sweep.py "$W/claude_round210_dim256_checkpoint_sweep_20260515/scripts/run_round210_dim256_checkpoint_sweep.py"
link_file checkpoints/selected/round210_process_only_epoch030.pth "$W/claude_round210_dim256_checkpoint_sweep_20260515/checkpoints/process_only/r210_a02_reladd2_d256_f16_t020_edge050/r210_a02_reladd2_d256_f16_t020_edge050_kg_checkpoint_epoch030.pth"
copy_code_file code/original_runs/r211/run_round211_dim256_readout_capacity_sweep.py "$W/claude_round211_dim256_readout_capacity_sweep_20260515/scripts/run_round211_dim256_readout_capacity_sweep.py"
copy_code_file code/original_runs/r215/run_round208_a01_fixedtarget_maml_readout.py "$W/claude_round215_direct_query_shift_adapter_20260517/scripts/run_round208_a01_fixedtarget_maml_readout.py"
link_file code/original_runs/r215/round215_fixed_cell_panel.json "$W/claude_round215_direct_query_shift_adapter_20260517/round215_fixed_cell_panel.json"
copy_code_file code/original_runs/r230/run_round230_scratch_mlp_metar_target_tower_pc3.py "$W/claude_round230_scratch_mlp_metar_target_tower_pc3_20260518/scripts/run_round230_scratch_mlp_metar_target_tower_pc3.py"
copy_code_file code/original_runs/r243/run_round243_leakage_safe_unseen_cell_support_adapter.py "$W/claude_round243_leakage_safe_unseen_cell_support_adapter_20260519/scripts/run_round243_leakage_safe_unseen_cell_support_adapter.py"
copy_code_file code/original_runs/r246/run_round246_joint_seen_target_train_pc3_a375_mcf7.py "$W/claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519/scripts/run_round246_joint_seen_target_train_pc3_a375_mcf7.py"
link_file code/original_runs/r246/selected_r246_joint_lm0p5_sc0p15_mn2p5.json "$W/claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519/source_data/selected_r246_joint_lm0p5_sc0p15_mn2p5.json"
link_file checkpoints/selected/round246_whole_model_checkpoint.pth "$W/claude_round246_joint_seen_target_train_pc3_a375_mcf7_20260519/checkpoints/r246_joint_lm0p5_sc0p15_mn2p5/whole_model_checkpoint.pth"
copy_code_file code/original_runs/r247/run_round247_support_curve_query_gated_adapter.py "$W/claude_round247_support_curve_query_gated_adapter_20260520/scripts/run_round247_support_curve_query_gated_adapter.py"
link_file checkpoints/selected/round247_adapter_init.pth "$W/claude_round247_support_curve_query_gated_adapter_20260520/checkpoints/r247_query_gated_h128_sc0p15_mn2p5/adapter_checkpoint.pth"
copy_code_file code/original_runs/r250/run_round250_epmeta_stable_tuning.py "$W/claude_round250_epmeta_stable_tuning_20260520/scripts/run_round250_epmeta_stable_tuning.py"
copy_code_file code/original_runs/r252/run_round252_epiqgate_pseudoval_sweep.py "$W/claude_round252_epiqgate_pseudoval_sweep_20260520/scripts/run_round252_epiqgate_pseudoval_sweep.py"
link_file code/original_runs/r252/round252_protocol_audit.json "$W/claude_round252_epiqgate_pseudoval_sweep_20260520/source_data/cell_splits.json"
link_file checkpoints/selected/round252_adapter_seed20260524.pth "$W/claude_round252_epiqgate_pseudoval_sweep_20260520/checkpoints/r252_epiqgate_lm1_lr0p00015_sc0p18_mn2p5_ba0p1_s32_q32_diverse_label_seed20260524/adapter_checkpoint.pth"

copy_code_file code/original_runs/r259_261/scripts/run_round259_crosscell_fullspace_round252query.py "$W/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/scripts/run_round259_crosscell_fullspace_round252query.py"
copy_code_file code/original_runs/r259_261/scripts/run_sc_label_bag_calibration_multicell_base.py "$W/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/scripts/run_sc_label_bag_calibration_multicell_base.py"
link_dir code/original_runs/r259_261/source_data "$W/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/source_data_local_archive"
link_file data/processed/fullspace/round259_target_embedding.csv "$W/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/source_data/round259_target_embedding.csv"
link_file checkpoints/selected/round259_e13_bce001.pth "$W/claude_round259_crosscell_fullspace_round252query_then_sc_proteomics_20260522/checkpoints/e13_bce001/selected_checkpoint.pth"

copy_code_file code/remote_dependencies/h07/train_bulk_sharedsample_indunseenlabel_v1_20260330.py "$W/bulk_sharedsample_indunseenlabel_20260330_h07/scripts/train_bulk_sharedsample_indunseenlabel_v1_20260330.py"
copy_code_file code/remote_dependencies/h07/train_bulk_dualbranch_v2_20260328.py "$W/bulk_sharedsample_indunseenlabel_20260330_h07/scripts/train_bulk_dualbranch_v2_20260328.py"
link_file data/processed/lincs/bulk_protocol_lock_v2.json "$W/bulk_sharedsample_indunseenlabel_20260330_h07_w060_u8e4_med/manifests/bulk_protocol_lock_v2.json"
link_file data/processed/lincs/bulk_protocol_lock_v2.json "$W/bulk_dualbranch_20260328_b01_warm_seen_joint/manifests/bulk_protocol_lock_v2.json"
link_file data/processed/fullspace/h07_prepared_data.pt "$W/claude_round209_fullspace_a01_extrapolation_20260517/cache/h07_prepared_data.pt"
link_file data/processed/lincs/large_data.parquet "$C/bulk/data/large_data.parquet"
link_file data/processed/lincs/compoundinfo_beta.txt "$C/bulk/data/compoundinfo_beta.txt"
link_file data/processed/fullspace/pertkge_ppi_graph.csv "$C/bulk/data/pertkge_ppi_graph.csv"
link_file data/processed/fullspace/splits_v3.json "$C/metar/splits_v3.json"
link_file data/processed/tahoe/sc_protocol_lock_v3_20260326.json "$W/sc_agg_upgrade_20260321/derived_tables/sc_protocol_lock_v3_20260326.json"
link_file data/processed/tahoe/sc_cell_pool_lock_v3_20260326.json.gz "$W/sc_agg_upgrade_20260321/derived_tables/sc_cell_pool_lock_v3_20260326.json.gz"
link_file data/processed/tahoe/tahoe_a549.h5ad "$C/sc/data/tahoe_a549.h5ad"
link_file data/processed/tahoe/tahoe_sw480.h5ad "$C/sc/data/tahoe_sw480.h5ad"
link_file data/processed/tahoe/tahoe_nci_h460.h5ad "$C/sc/data/tahoe_nci_h460.h5ad"
link_file data/processed/tahoe/tahoe_dti.csv "$C/sc/data/tahoe_dti.csv"
link_dir "$C/manuscript/write" "$N/claude/manuscript/write"
link_dir "$C/sc/data" "$N/claude/sc/data"
link_dir "$C/bulk/data" "$N/claude/bulk/data"

python3 "$REPO_ROOT/tools/relocate_paths.py" "$ROOT"
printf 'ARTICLE_EXACT_COMPAT_LAYOUT_OK\t%s\n' "$ROOT/compat"
