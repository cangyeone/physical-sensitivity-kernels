# GJI reproducibility archive manifest

This manifest lists the files that should be included in the public
manuscript-specific release archive.

## Manuscript and submission files

- `gji_dnn_posterior_inversion/gjilguid2e.tex`
- `gji_dnn_posterior_inversion/gjilguid2e.pdf`
- `gji_dnn_posterior_inversion/references.bib`
- `gji_dnn_posterior_inversion/gji.cls`
- `gji_dnn_posterior_inversion/gji.bst`
- `gji_dnn_posterior_inversion/gji_extra.sty`
- `gji_dnn_posterior_inversion/timet.sty`
- `gji_dnn_posterior_inversion/cover_letter_GJI.md`
- `pre_submission_blockers.md`

## Code, configs and build targets

- `Makefile`
- `README.md`
- `disp_inv_train.v1.3.py`
- `utils/generate_data.py`
- `utils/generate_data_weak_prior.py`
- `models/struct2disp_transformer.py`
- `configs/fair_di_strong_full.yaml`
- `configs/fair_di_weak_full.yaml`
- `configs/det_di_strong_full.yaml`
- `configs/det_di_weak_full.yaml`
- `scripts/train_di_fair.py`
- `scripts/eval_fair_di_comparison.py`
- `scripts/eval_fair_calibration.py`
- `scripts/eval_fair_noise_sensitivity.py`
- `scripts/eval_fair_missing_band.py`
- `scripts/eval_fair_sampling_sensitivity.py`
- `scripts/train_deterministic_di_fair.py`
- `scripts/eval_fair_baselines.py`
- `scripts/field_masw_posterior_inversion.py`
- `scripts/field_masw_compare_fair.py`
- `scripts/make_gji_review_figures.py`
- `scripts/update_gji_from_fair_results.py`
- `scripts/run_fair_production_pipeline.sh`

## Result tables and protocols

- `results/fair_di_comparison/README.md`
- `results/fair_di_comparison/production/fair_di_metrics.csv`
- `results/fair_di_comparison/production/fair_di_metrics.json`
- `results/fair_di_comparison/production/fair_di_protocol.json`
- `results/fair_di_comparison/production/calibration/calibration_metrics.csv`
- `results/fair_di_comparison/production/calibration/depth_binned_coverage.csv`
- `results/fair_di_comparison/production/calibration/rank_diagnostics.csv`
- `results/fair_di_comparison/production/calibration/temperature_scaling.json`
- `results/fair_di_comparison/production/noise/noise_sensitivity.csv`
- `results/fair_di_comparison/production/noise/noise_sensitivity.json`
- `results/fair_di_comparison/production/missing_band/missing_band_uncertainty.csv`
- `results/fair_di_comparison/production/missing_band/missing_band_uncertainty.json`
- `results/fair_di_comparison/production/sampling_sensitivity/sampling_sensitivity.csv`
- `results/fair_di_comparison/production/sampling_sensitivity/sampling_sensitivity.json`
- `results/fair_di_comparison/production/baselines/baseline_metrics.csv`
- `results/fair_di_comparison/production/baselines/baseline_metrics.json`
- `results/fair_di_comparison/production/field/field_summary.csv`
- `results/fair_di_comparison/production/field/field_reference_comparison.csv`
- `results/fair_di_comparison/production/field/field_posterior_predictive.json`

## Figures

- `gji_dnn_posterior_inversion/figures/fig01_dnn_posterior_workflow.pdf`
- `gji_dnn_posterior_inversion/figures/fig02_control_points.pdf`
- `gji_dnn_posterior_inversion/figures/fair_di_metric_summary.pdf`
- `gji_dnn_posterior_inversion/figures/fair_di_example_profiles.pdf`
- `gji_dnn_posterior_inversion/figures/fair_calibration_reliability.pdf`
- `gji_dnn_posterior_inversion/figures/fair_missing_band_uncertainty.pdf`
- `gji_dnn_posterior_inversion/figures/fair_missing_band_random_scatter.pdf`
- `gji_dnn_posterior_inversion/figures/fair_noise_sensitivity.pdf`
- `gji_dnn_posterior_inversion/figures/fair_baseline_metric_summary.pdf`
- `gji_dnn_posterior_inversion/figures/fair_field_dispersion_qc.pdf`
- `gji_dnn_posterior_inversion/figures/fair_field_summary_vs_depth.pdf`
- `gji_dnn_posterior_inversion/figures/fair_field_vs_median_slices.pdf`
- `gji_dnn_posterior_inversion/figures/fair_field_vs_std_slices.pdf`
- `gji_dnn_posterior_inversion/figures/fair_field_reference_difference.pdf`

## Large artifacts

Include these in the release archive if file-size limits allow; otherwise
provide explicit download or regeneration instructions in the release notes.

- `ckpt/fair_di_strong_full_seed642026/best.pt`
- `ckpt/fair_di_weak_full_seed642026/best.pt`
- `ckpt/det_di_strong_full_seed642026/best.pt`
- `ckpt/det_di_weak_full_seed642026/best.pt`
- `ckpt/struct2disp_cpmlp.prior_boundary_v3.pt`
- `field_masw_results_fair_weak/`

## External dataset

Do not re-host the Bayan Obo dataset unless its license permits redistribution.
Reference the public Zenodo record instead:

- Yang, Chen (2025), Dataset for Ambient Noise Tomography in the Bayan Obo
  Region Using a Seismic Dense Array, Zenodo,
  `https://doi.org/10.5281/zenodo.17292491`.
