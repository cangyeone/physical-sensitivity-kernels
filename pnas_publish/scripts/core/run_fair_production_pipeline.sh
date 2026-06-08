#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

DEVICE="${DEVICE:-mps}"
LOG_DIR="$ROOT/results/fair_di_comparison/production/logs"
mkdir -p "$LOG_DIR" "$ROOT/results/fair_di_comparison/production" "$ROOT/figures/fair_di_comparison/production"

PIPELINE_LOG="$LOG_DIR/pipeline.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting fair DI production pipeline"
echo "root=$ROOT"
echo "device=$DEVICE"
python - <<'PY' || echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING torch/MPS preflight failed; continuing to training command"
import torch
print("torch", torch.__version__)
print("mps_available", torch.backends.mps.is_available())
print("mps_built", torch.backends.mps.is_built())
PY

run_step() {
  local name="$1"
  shift
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] BEGIN $name"
  "$@" 2>&1 | tee -a "$LOG_DIR/${name}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $name"
}

skip_or_run() {
  local sentinel="$1"
  local name="$2"
  shift 2
  if [[ -e "$sentinel" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP $name; found $sentinel"
  else
    run_step "$name" "$@"
  fi
}

skip_or_run "$ROOT/ckpt/fair_di_strong_full_seed642026/training_complete.json" \
  train_fair_strong_full \
  python -u scripts/train_di_fair.py --config configs/fair_di_strong_full.yaml --device "$DEVICE"

skip_or_run "$ROOT/ckpt/fair_di_weak_full_seed642026/training_complete.json" \
  train_fair_weak_full \
  python -u scripts/train_di_fair.py --config configs/fair_di_weak_full.yaml --device "$DEVICE"

skip_or_run "$ROOT/results/fair_di_comparison/production/fair_di_metrics.csv" \
  eval_fair_di_comparison \
  python -u scripts/eval_fair_di_comparison.py \
    --strong-ckpt ckpt/fair_di_strong_full_seed642026/best.pt \
    --weak-ckpt ckpt/fair_di_weak_full_seed642026/best.pt \
    --out-dir results/fair_di_comparison/production \
    --fig-dir figures/fair_di_comparison/production \
    --n-test 1024 --n-envelope 10000 \
    --posterior-samples 64 --euler-steps 24 \
    --bootstrap 2000 --batch-size 16 --device "$DEVICE"

skip_or_run "$ROOT/results/fair_di_comparison/production/calibration/calibration_metrics.csv" \
  eval_fair_calibration \
  python -u scripts/eval_fair_calibration.py \
    --strong-ckpt ckpt/fair_di_strong_full_seed642026/best.pt \
    --weak-ckpt ckpt/fair_di_weak_full_seed642026/best.pt \
    --out-dir results/fair_di_comparison/production/calibration \
    --fig-dir figures/fair_di_comparison/production/calibration \
    --n-eval 2048 --calibration-examples 1024 \
    --posterior-samples 64 --euler-steps 24 --bootstrap 2000 \
    --batch-size 16 --device "$DEVICE"

skip_or_run "$ROOT/results/fair_di_comparison/production/noise/noise_sensitivity.csv" \
  eval_fair_noise_sensitivity \
  python -u scripts/eval_fair_noise_sensitivity.py \
    --strong-ckpt ckpt/fair_di_strong_full_seed642026/best.pt \
    --weak-ckpt ckpt/fair_di_weak_full_seed642026/best.pt \
    --out-dir results/fair_di_comparison/production/noise \
    --fig-dir figures/fair_di_comparison/production/noise \
    --noise-sigma-km-s 0.00 0.02 0.05 0.10 \
    --n-eval 1024 --posterior-samples 64 --euler-steps 24 \
    --batch-size 16 --device "$DEVICE"

skip_or_run "$ROOT/results/fair_di_comparison/production/missing_band/missing_band_uncertainty.csv" \
  eval_fair_missing_band \
  python -u scripts/eval_fair_missing_band.py \
    --ckpt ckpt/fair_di_weak_full_seed642026/best.pt \
    --out-dir results/fair_di_comparison/production/missing_band \
    --fig-dir figures/fair_di_comparison/production/missing_band \
    --n-eval 1024 --posterior-samples 64 --euler-steps 24 \
    --batch-size 16 --device "$DEVICE"

skip_or_run "$ROOT/ckpt/det_di_strong_full_seed642026/training_complete.json" \
  train_det_strong_full \
  python -u scripts/train_deterministic_di_fair.py --config configs/det_di_strong_full.yaml --device "$DEVICE"

skip_or_run "$ROOT/ckpt/det_di_weak_full_seed642026/training_complete.json" \
  train_det_weak_full \
  python -u scripts/train_deterministic_di_fair.py --config configs/det_di_weak_full.yaml --device "$DEVICE"

skip_or_run "$ROOT/results/fair_di_comparison/production/baselines/baseline_metrics.csv" \
  eval_fair_baselines \
  python -u scripts/eval_fair_baselines.py \
    --fair-results results/fair_di_comparison/production/fair_di_metrics.csv \
    --det-strong ckpt/det_di_strong_full_seed642026/best.pt \
    --det-weak ckpt/det_di_weak_full_seed642026/best.pt \
    --ind-fwd ckpt/struct2disp_cpmlp.prior_boundary_v3.pt \
    --out-dir results/fair_di_comparison/production/baselines \
    --fig-dir figures/fair_di_comparison/production/baselines \
    --n-test 1024 --posterior-samples 64 --euler-steps 24 \
    --batch-size 16 --device "$DEVICE"

skip_or_run "$ROOT/field_masw_results_fair_weak/bayan_obo_masw_vs_depth_summary.csv" \
  field_masw_posterior_inversion \
  python -u scripts/field_masw_posterior_inversion.py \
    --ckpt ckpt/fair_di_weak_full_seed642026/best.pt \
    --out-dir field_masw_results_fair_weak \
    --fig-dir gji_dnn_posterior_inversion/figures \
    --period-min 2 --period-max 40 \
    --posterior-samples 64 --num-steps 24 --batch-size 16 \
    --device "$DEVICE"

skip_or_run "$ROOT/results/fair_di_comparison/production/field/field_summary.csv" \
  field_masw_compare_fair \
  python -u scripts/field_masw_compare_fair.py \
    --dnn-dir field_masw_results_fair_weak \
    --masw-dir "Bayan_Obo_Dataset/Subarray-Based MASW" \
    --out-dir results/fair_di_comparison/production/field \
    --fig-dir figures/fair_di_comparison/production/field

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Fair DI production pipeline completed"
touch "$ROOT/results/fair_di_comparison/production/PRODUCTION_PIPELINE_COMPLETE"
