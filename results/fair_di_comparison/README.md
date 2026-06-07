# Fair DI-Strong vs DI-Weak comparison

This folder is reserved for the matched-budget DI-Strong/DI-Weak comparison.
Older prior-boundary diagnostic outputs remain in `results/prior_boundary_*` and
are not overwritten by this protocol.

## Protocol

The intended difference between the two runs is only the structural prior
generator:

- DI-Strong uses `utils/generate_data.py`.
- DI-Weak uses `utils/generate_data_weak_prior.py`.

Both matched configs use the same architecture, optimizer, learning-rate
schedule, batch size, epoch count, train/validation sizes, mask augmentation,
loss weights, random seed, posterior sample count, Euler steps, and evaluation
test sets.

Both fair checkpoints are initialized from scratch. Older unequal-budget or
checkpoint-initialized results are retained only as ablations, not as the fair
comparison.

## Full production commands

```bash
python scripts/train_di_fair.py --config configs/fair_di_strong_full.yaml
python scripts/train_di_fair.py --config configs/fair_di_weak_full.yaml

python scripts/eval_fair_di_comparison.py \
  --strong-ckpt ckpt/fair_di_strong_full_seed642026/best.pt \
  --weak-ckpt ckpt/fair_di_weak_full_seed642026/best.pt \
  --out-dir results/fair_di_comparison/production \
  --fig-dir figures/fair_di_comparison/production \
  --n-test 1024 \
  --n-envelope 10000 \
  --posterior-samples 64 \
  --euler-steps 24 \
  --bootstrap 2000 \
  --batch-size 16

python scripts/eval_fair_calibration.py \
  --strong-ckpt ckpt/fair_di_strong_full_seed642026/best.pt \
  --weak-ckpt ckpt/fair_di_weak_full_seed642026/best.pt \
  --out-dir results/fair_di_comparison/production/calibration \
  --fig-dir figures/fair_di_comparison/production/calibration \
  --n-eval 2048 \
  --calibration-examples 1024 \
  --posterior-samples 64 \
  --euler-steps 24 \
  --bootstrap 2000

python scripts/eval_fair_noise_sensitivity.py \
  --strong-ckpt ckpt/fair_di_strong_full_seed642026/best.pt \
  --weak-ckpt ckpt/fair_di_weak_full_seed642026/best.pt \
  --out-dir results/fair_di_comparison/production/noise \
  --noise-sigma-km-s 0.00 0.02 0.05 0.10 \
  --n-eval 1024 \
  --posterior-samples 64 \
  --euler-steps 24

python scripts/eval_fair_missing_band.py \
  --ckpt ckpt/fair_di_weak_full_seed642026/best.pt \
  --out-dir results/fair_di_comparison/production/missing_band \
  --fig-dir figures/fair_di_comparison/production/missing_band \
  --n-eval 1024 \
  --posterior-samples 64 \
  --euler-steps 24

python scripts/train_deterministic_di_fair.py --config configs/det_di_strong_full.yaml
python scripts/train_deterministic_di_fair.py --config configs/det_di_weak_full.yaml

python scripts/eval_fair_baselines.py \
  --fair-results results/fair_di_comparison/production/fair_di_metrics.csv \
  --det-strong ckpt/det_di_strong_full_seed642026/best.pt \
  --det-weak ckpt/det_di_weak_full_seed642026/best.pt \
  --ind-fwd ckpt/struct2disp_cpmlp.prior_boundary_v3.pt \
  --out-dir results/fair_di_comparison/production/baselines \
  --fig-dir figures/fair_di_comparison/production/baselines \
  --n-test 1024

python scripts/field_masw_posterior_inversion.py \
  --ckpt ckpt/fair_di_weak_full_seed642026/best.pt \
  --out-dir field_masw_results_fair_weak \
  --fig-dir gji_dnn_posterior_inversion/figures \
  --period-min 2 \
  --period-max 40 \
  --posterior-samples 64 \
  --num-steps 24 \
  --batch-size 16

python scripts/field_masw_compare_fair.py \
  --dnn-dir field_masw_results_fair_weak \
  --masw-dir "Bayan_Obo_Dataset/Subarray-Based MASW" \
  --out-dir results/fair_di_comparison/production/field \
  --fig-dir figures/fair_di_comparison/production/field

python scripts/make_gji_review_figures.py
make gji-build
make gji-check
```

## Matched pilot commands

Use these when production GPU/MPS time is limited. Pilot outputs should not be
reported as the final method ranking.

```bash
python scripts/train_di_fair.py --config configs/fair_di_strong_pilot.yaml
python scripts/train_di_fair.py --config configs/fair_di_weak_pilot.yaml

python scripts/eval_fair_di_comparison.py \
  --strong-ckpt ckpt/fair_di_strong_pilot_seed642026/best.pt \
  --weak-ckpt ckpt/fair_di_weak_pilot_seed642026/best.pt \
  --out-dir results/fair_di_comparison/pilot \
  --fig-dir figures/fair_di_comparison/pilot \
  --n-test 24 \
  --n-envelope 512 \
  --posterior-samples 16 \
  --euler-steps 8 \
  --bootstrap 200 \
  --batch-size 8

python scripts/eval_fair_calibration.py \
  --strong-ckpt ckpt/fair_di_strong_pilot_seed642026/best.pt \
  --weak-ckpt ckpt/fair_di_weak_pilot_seed642026/best.pt \
  --out-dir results/fair_di_comparison/pilot/calibration \
  --fig-dir figures/fair_di_comparison/pilot/calibration \
  --n-eval 96 \
  --calibration-examples 48 \
  --posterior-samples 16 \
  --euler-steps 8 \
  --bootstrap 200

python scripts/eval_fair_noise_sensitivity.py \
  --strong-ckpt ckpt/fair_di_strong_pilot_seed642026/best.pt \
  --weak-ckpt ckpt/fair_di_weak_pilot_seed642026/best.pt \
  --out-dir results/fair_di_comparison/pilot/noise \
  --noise-sigma-km-s 0.00 0.02 0.05 0.10 \
  --n-eval 96 \
  --posterior-samples 16 \
  --euler-steps 8

python scripts/eval_fair_missing_band.py \
  --ckpt ckpt/fair_di_weak_pilot_seed642026/best.pt \
  --out-dir results/fair_di_comparison/pilot/missing_band \
  --fig-dir figures/fair_di_comparison/pilot/missing_band \
  --n-eval 96 \
  --posterior-samples 16 \
  --euler-steps 8
```

For a fast smoke test of the file/figure export path, add
`--skip-dispersion-residuals` and reduce `--n-test`, `--posterior-samples`,
`--euler-steps`, and `--bootstrap`. The full evaluation command above should be
used for manuscript statistics because it includes forward-solver dispersion
residuals.

## Expected artifacts

Training writes each run's `config_resolved.json`, `source_config.yaml`,
`normalization_stats.pt`, `normalization_stats_summary.json`,
`runtime_metadata.json`, `epoch_metrics.csv`, `epoch_metrics.jsonl`,
`best_selection.json`, checkpoints, and `training_complete.json`.

Core evaluation writes `fair_di_metrics.csv`, `fair_di_metrics.json`,
`fair_di_protocol.json`, `fair_di_diagnostics.npz`, and summary figures.
Calibration writes `calibration_metrics.csv`, `temperature_scaling.json`,
`depth_binned_coverage.csv`, `rank_diagnostics.csv`, and reliability figures.
Noise and missing-band diagnostics write separate CSV/JSON tables and figures.
Baselines write `baseline_metrics.csv/json`. Field comparison writes
`field_summary.csv`, `field_reference_comparison.csv` and
`field_posterior_predictive.json`.

The manuscript figures are regenerated by `scripts/make_gji_review_figures.py`
from the CSV tables. The script writes colour-blind-safe, larger-font PDF/PNG
figures to `gji_dnn_posterior_inversion/figures` and mirrors the production
figure directories where applicable.

## Current production status and known blockers

The current production chain is complete when
`results/fair_di_comparison/production/PRODUCTION_PIPELINE_COMPLETE` exists
and the CSV/JSON files listed above are newer than the matched checkpoints.
The GJI manuscript deliberately marks two remaining submission blockers:

- a public project archive DOI or approved repository URL plus archive DOI must
  be minted before journal submission;
- production posterior sample-count and Euler-step sensitivity tables are not
  yet included in the manuscript statistics, so no quantitative claim is made
  from that diagnostic.
