# Simulated Gravity Prior-Boundary Summary

This companion diagnostic uses the existing strong/weak Earth-model priors but replaces the observation operator with a simulated gravity forward model. The inferred state is density rho(z).

| Method | Test set | N | Rho MAE | Gravity MAE | Coverage | Pull-in | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DI-Strong | in-prior | 96 | 0.092 | 0.214 | 0.063 | 0.612 | 0.0 |
| DI-Weak | in-prior | 96 | 0.156 | 0.277 | 0.095 | 0.567 | 0.0 |
| IND-FWD | in-prior | 30 | 0.169 | 0.463 | 0.431 | 0.969 | 39.5 |
| DI-Strong | boundary | 96 | 0.079 | 0.258 | 0.098 | 0.504 | 0.0 |
| DI-Weak | boundary | 96 | 0.099 | 0.220 | 0.124 | 0.476 | 0.0 |
| IND-FWD | boundary | 30 | 0.182 | 0.393 | 0.646 | 0.596 | 39.7 |
| DI-Strong | out-of-prior | 96 | 0.100 | 0.268 | 0.111 | 0.371 | 0.0 |
| DI-Weak | out-of-prior | 96 | 0.131 | 0.306 | 0.109 | 0.236 | 0.0 |
| IND-FWD | out-of-prior | 30 | 0.180 | 0.491 | 0.575 | 0.366 | 39.4 |

Interpretation: gravity inversion is intentionally non-unique. These rows diagnose prior-boundary behavior rather than defining a calibrated field-data posterior.
