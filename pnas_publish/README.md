# PNAS Publication Reproducibility Package

This directory collects the scripts, configuration files, selected production
results, and selected publication figures used for the manuscript
`Training-Prior Boundaries Shape Learned Inverse Models in Scientific Inference`.
See `REPRODUCIBILITY.md` for the table/figure manifest, rerun commands,
runtime notes, and current TODOs.

The original project-level script locations are retained for backward
compatibility with existing checkpoints and run logs. This directory is the
publication-facing consolidation point.

## Layout

- `scripts/core/`: prior-boundary evaluation, fair direct-inversion training,
  calibration, missing-band, noise, field-transfer, gravity, and indirect
  forward-surrogate scripts copied from the project `scripts/` directory.
- `scripts/manuscript/`: manuscript-side helper scripts and small diagnostics
  copied from the article repository's `pnas_publish/disp_inv_scripts/`
  directory.
- `scripts/top_level/`: top-level training and testing entry points used during
  surface-wave direct/forward development.
- `configs/`: YAML prior and training configurations.
- `utils/`: synthetic-data and prior-generation utilities.
- `results/fair_di_comparison/production/`: selected archived CSV/JSON outputs
  from the matched production comparison.
- `figures/fair_di_comparison/production/`: selected publication figures from
  the matched production comparison.

## Main Surface-Wave Evidence

The production direct-inversion comparison is defined by
`results/fair_di_comparison/production/fair_di_protocol.json`:

- 100,000 training examples and 2,048 validation examples for both DI-Strong
  and DI-Weak.
- 24 epochs for both direct models.
- 1,024 test examples per in-prior, boundary, and out-of-prior direct-evaluation
  regime.
- 64 posterior samples and 24 Euler steps for direct posterior sampling.
- The intended experimental difference is the structural prior generator.

Main metrics are stored in
`results/fair_di_comparison/production/fair_di_metrics.csv`. Indirect
forward-surrogate and deterministic baseline summaries are stored in
`results/fair_di_comparison/production/baselines/baseline_metrics.csv`.

## Reproduction Entry Points

Typical project-root commands are:

```bash
bash pnas_publish/scripts/core/run_fair_production_pipeline.sh
python pnas_publish/scripts/core/eval_fair_di_comparison.py --help
python pnas_publish/scripts/core/eval_fair_calibration.py --help
python pnas_publish/scripts/core/eval_fair_noise_sensitivity.py --help
python pnas_publish/scripts/core/eval_fair_missing_band.py --help
python pnas_publish/scripts/core/run_indirect_forward_inversion.py --help
python pnas_publish/scripts/core/eval_gravity_prior_boundary_effect.py --help
```

Some commands rely on checkpoints and cached synthetic data stored outside this
publication package. Paths should be resolved relative to the project root.
