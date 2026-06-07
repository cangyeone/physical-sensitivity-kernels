# Physical Sensitivity Kernels Can Emerge in Data-Driven Forward Models

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.04107-b31b1b.svg)](https://arxiv.org/abs/2604.04107)

Official implementation of **"Physical Sensitivity Kernels Can Emerge in Data-Driven Forward Models: Evidence From Surface-Wave Dispersion"** (arXiv:2604.04107, 2026) by Ziye Yu, Yuqi Cai, and Xin Liu.

---

## What Does This Project Do?

**Core question:** When you train a neural network to act as a surrogate forward model in geophysics (mapping Earth structure → seismic observations), does it learn *only* the input-output mapping, or does it also recover the *underlying physical sensitivity structure*?

**Answer:** Yes — physical sensitivity kernels **emerge naturally** from the trained neural network's automatic differentiation, without any explicit physics constraints during training. We demonstrate this using surface-wave dispersion as a testbed.

In practical terms, this project provides:

1. **A Transformer-based neural operator** (`Struct2DispTransformer`) that maps 1D Earth velocity models to surface-wave dispersion curves — a fast, differentiable surrogate for expensive physics-based forward modeling
2. **A Conditional Rectified Flow generative model** (`Disp2StructCRF`) that solves the inverse problem — recovering Earth structure from observed dispersion curves, with uncertainty quantification
3. **Sensitivity kernel validation** — automatic Jacobian computation shows the learned model recovers physically meaningful Fréchet derivatives that match theoretical kernels from the `disba` library

### Why This Matters

- **Differentiable surrogate**: Replace slow, non-differentiable physics solvers with a fast neural network that supports gradient-based inversion
- **Emergent physics**: The model learns real wave-propagation physics without being told about it — it's not just curve-fitting
- **Uncertainty quantification**: Both forward and inverse models output probabilistic predictions (mean + variance)
- **Practical inversion**: Build full Fisher Information matrices for resolution analysis, experimental design, and uncertainty propagation

---

## Models

### 1. Struct2DispTransformer (Forward Model)

Learns the mapping **Earth structure → dispersion curves**:

$$\mathcal{F}: m(z) = [V_p(z), V_s(z), \rho(z)] \;\longrightarrow\; c(T) = [c_R(T), c_L(T)]$$

| Component | Description |
|-----------|-------------|
| Input | `[B, 4, H]` — depth, Vp, Vs, density at H depth points |
| Depth Embedding | Linear projection → d_model, with sinusoidal positional encoding |
| Transformer Encoder | 6–8 layers of multi-head self-attention over depth tokens |
| Period Queries | 59 learnable query tokens (one per period, T=2–60s), with optional period-value injection |
| Transformer Decoder | 3–8 layers of cross-attention (queries attend to encoder memory) |
| Output Heads | Two MLP heads → μ (mean) and log σ² (log-variance) for Rayleigh + Love |
| Final Output | `[B, 2, T]` — phase velocity predictions with uncertainties |

Key design choices:
- **LayerNorm-first** for training stability
- **GELU activations** throughout
- **Log-variance clamped to [-10, 3]** for numerical stability
- **Probabilistic output** — outputs a conditional Gaussian distribution `p(c|m)`

### 2. Disp2StructCRF (Inverse Model) 🆕

Learns the inverse mapping **dispersion curves → Earth structure** using **Conditional Rectified Flow** — a generative model that learns an ODE flow from noise to data:

$$m \sim p(m | c_{\text{obs}})$$

| Component | Description |
|-----------|-------------|
| DispersionEncoder | 1D CNN with residual blocks, encodes masked/incomplete dispersion observations (5 channels: period, Rayleigh vel, Love vel, Rayleigh mask, Love mask) |
| Flow Network | 3-layer MLP predicting ODE velocity `dx/dt` from [conditioning, noisy sample, time embedding] |
| Sampling | Euler ODE integration (24–32 steps) from noise → structured profile, with ensemble support |
| Two variants | **v1.1** — full H-depth output (768 params); **v1.2** — control-point parameterization (~55 control points, linearly interpolated to full grid) |

Key features:
- Handles **missing data** naturally (masked dispersion observations)
- Produces **ensembles** for uncertainty quantification
- Control-point variant (v1.2) reduces high-frequency artifacts and improves efficiency
- Multi-component loss: flow velocity matching + reconstruction + smoothness (slope + curvature) regularization

---

## Project Structure

```
SurfFlow/
├── models/
│   └── struct2disp_transformer.py       # Struct2DispTransformer forward model
├── utils/
│   ├── generate_data.py                 # Synthetic data generator (strong tectonic priors)
│   └── generate_data_weak_prior.py      # Synthetic data generator (weak priors)
├── disp_gen_train.v1.1.py               # Train forward model
├── disp_inv_train.v1.1.py               # Train inverse model (full-depth CRF)
├── disp_inv_train.v1.2.py               # Train inverse model (control-point CRF)
├── disp_gen_test.sk.v1.1.py             # Sensitivity kernel validation (NN vs disba)
├── disp_gen_test.metrics.v1.1.py        # Systematic metric evaluation
├── disp_gen_test.fisher.v1.1.py         # Fisher information analysis (full-depth)
├── disp_gen_test.fisher.control_point.v1.1.py      # Fisher analysis (control-point)
├── disp_gen_test.fisher.control_point.v1.1.ckpt.py # Fisher analysis (checkpoint-based)
├── ckpt/
│   ├── disp2struct_crf.v1.1/            # Inverse model checkpoints
│   └── disp2struct_crf.v1.2_cp/         # Control-point inverse model checkpoints
└── README.md
```

---

## Installation

```bash
git clone https://github.com/cangyeone/SurfFlow.git
cd SurfFlow
pip install torch numpy scipy matplotlib
```

Dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥1.9 | Deep learning framework |
| `numpy` | ≥1.20 | Numerical computations |
| `scipy` | ≥1.7 | Scientific utilities (interpolation) |
| `matplotlib` | ≥3.4 | Visualization |
| `disba` | — | Surface-wave dispersion computation (for data generation & validation) |

---

## Quick Start

### 1. Train the Forward Model

```bash
python disp_gen_train.v1.1.py
```

This trains `Struct2DispTransformer` on 100,000 synthetic Earth models to predict Rayleigh and Love wave dispersion curves at periods 2–60s. Checkpoints and diagnostic plots are saved to `ckpt/` and `tfig/`.

### 2. Train the Inverse Model

```bash
# Full-depth version
python disp_inv_train.v1.1.py

# Control-point version (more efficient, smoother outputs)
python disp_inv_train.v1.2.py
```

This trains `Disp2StructCRF` using Conditional Rectified Flow — the model learns to invert dispersion observations back to Earth structure, handling masked/incomplete data.

### 2b. Fair DI-Strong vs DI-Weak posterior-inversion comparison

The GJI posterior-inversion manuscript uses a matched-budget protocol so that
DI-Strong and DI-Weak differ only in the structural prior generator. Warm-started
or unequal-budget outputs are retained only as ablations.

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/train_di_fair.py --config configs/fair_di_strong_full.yaml
python scripts/train_di_fair.py --config configs/fair_di_weak_full.yaml

python scripts/eval_fair_di_comparison.py \
  --strong-ckpt ckpt/fair_di_strong_full_seed642026/best.pt \
  --weak-ckpt ckpt/fair_di_weak_full_seed642026/best.pt \
  --out-dir results/fair_di_comparison/production \
  --fig-dir figures/fair_di_comparison/production \
  --n-test 1024 --n-envelope 10000 \
  --posterior-samples 64 --euler-steps 24 \
  --bootstrap 2000 --batch-size 16
```

Additional calibration, noise, missing-band, baseline and field-comparison
commands are documented in `results/fair_di_comparison/README.md`.

### 2c. GJI review manuscript build and checks

The GJI posterior-inversion manuscript is in
`gji_dnn_posterior_inversion/gjilguid2e.tex`. The current review build uses the
matched fair-comparison production results under
`results/fair_di_comparison/production` and the field workflow outputs under
`field_masw_results_fair_weak`.

```bash
# Regenerate table-driven manuscript figures from production CSV files.
make gji-review-figures

# Compile the referee-layout manuscript and run citation/reference/stale-text checks.
make gji-build
make gji-check

# Convenience target for figure regeneration + build + checks.
make gji-paper
```

The manuscript uses seed `642026`, `100000` training examples, `2048`
validation examples, 24 training epochs, 64 posterior samples and 24 Euler
steps for the production synthetic statistics unless a table caption states
otherwise. The production evaluation exports bootstrap confidence intervals,
depth-binned coverage, reliability/rank diagnostics, missing-band uncertainty,
additive-noise sensitivity, posterior-predictive residuals, deterministic DNN
baselines, sample-count/Euler-step sensitivity checks and the Rayleigh-only
Bayan Obo workflow demonstration. Pre-submission archive tasks are tracked in
`pre_submission_blockers.md`.

### 3. Validate Sensitivity Kernels

```bash
python disp_gen_test.sk.v1.1.py
```

Compares neural network Jacobians (automatic differentiation) against theoretical sensitivity kernels from `disba`. This is the core scientific validation — confirming the model has learned real physics.

### 4. Run Fisher Information Analysis

```bash
python disp_gen_test.fisher.control_point.v1.1.py
```

Uses the trained forward model as a differentiable operator to build Fisher Information matrices and perform gradient-based inversion with uncertainty quantification.

### 5. Use a Pre-trained Model (API)

```python
import torch
from models.struct2disp_transformer import Struct2DispTransformer

# Load model
model = Struct2DispTransformer(
    H=256, T=59, C_in=4,
    d_model=512, nhead=8,
    num_enc_layers=8, num_dec_layers=8,
    dim_ff=1024, dropout=0.1,
    use_period_values=True,
    period_minmax=(1.0, 100.0)
)
checkpoint = torch.load('ckpt/struct2disp_transformer.v1.1.pt', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# Predict dispersion from Earth model
# Input: [B, 4, H] — depth, Vp, Vs, rho
earth_model = ...  # your velocity model here
periods = torch.linspace(2, 60, 59)
with torch.no_grad():
    mu, logvar = model(earth_model, periods=periods)
    # mu: [B, 2, 59] — Rayleigh and Love phase velocities
    # logvar: [B, 2, 59] — log-variance (uncertainty)

# Compute sensitivity kernel (Fréchet derivative) via autograd
earth_model.requires_grad_(True)
mu, _ = model(earth_model, periods=periods)
kernel_vs = torch.autograd.grad(
    mu[0, 0, :].sum(), earth_model,
    create_graph=False
)[0][0, 2, :]  # ∂c_R/∂Vs at all depths
```

---

## Synthetic Data Generation

Two data generators are provided:

### Strong-Prior (tectonic-type based)
`utils/generate_data.py` — Generates realistic 1D Earth models with explicit geological priors:

| Tectonic Type | Weight | Characteristics |
|---------------|--------|-----------------|
| Oceanic | 15% | Thin crust (5–15 km), basaltic, hot mantle, common LVZ |
| Shield | 25% | Thick crust (35–55 km), cold mantle, high Vs |
| Platform | 25% | Moderate crust (28–45 km), variable sediments |
| Orogen | 20% | Thick crust (40–70 km), hot mantle, common LVZ |
| Rift | 15% | Thin crust (20–40 km), hot mantle, very common LVZ |

### Weak-Prior (unconstrained)
`utils/generate_data_weak_prior.py` — Generates models with minimal structural assumptions (no explicit Moho, tectonic classes, or LVZ definitions). Uses random control points with broad bounds. **Recommended** for evaluating whether neural surrogates learn true physics versus memorizing training distribution structure.

Both use the `disba` library for physics-based forward computation of Rayleigh and Love wave dispersion curves, and support realistic observation masks (random missing periods, missing wave types).

---

## Sensitivity Kernel Analysis

This is the **core scientific contribution**. The pipeline:

1. **NN kernels**: Compute Jacobian $K_{\text{nn}} = \partial c / \partial V_s$ via `torch.func.jacrev` + `vmap`
2. **Theoretical kernels**: Compute analytical sensitivity kernels via `disba.PhaseSensitivity`
3. **Fréchet conversion**: $K_{\text{Fréchet}} = \frac{V_s}{c} \cdot \frac{\partial c}{\partial V_s}$ (dimensionless, comparable form)
4. **Comparison**: Cosine similarity, correlation, RMSE, MAE across period bands (2–5s, 10–15s, 20–30s, 30–40s, 50–60s)

**Key result**: The NN-computed kernels match the main depth-dependent structure of theoretical kernels across a broad range of periods, confirming that physical understanding emerges from data-driven training.

---

## Mathematical Background

### Forward Problem

$$\mathcal{F}: m(z) = [V_p(z), V_s(z), \rho(z)] \rightarrow c(T) = [c_R(T), c_L(T)]$$

The Transformer learns a probabilistic mapping:

$$p(c|m) = \mathcal{N}(c; \mu_\theta(m), \sigma^2_\theta(m))$$

### Sensitivity Kernels (Fréchet Derivatives)

$$K_j(z_k; T) = \frac{\partial c(T)}{\partial m_j(z_k)}, \quad j \in \{V_p, V_s, \rho\}$$

The full Jacobian matrix has shape $[2T \times 3H]$ (118 × 768 for T=59, H=256), computed efficiently via automatic differentiation.

### Inverse Problem (Rectified Flow)

The Conditional Rectified Flow learns an ODE:

$$\frac{dx}{dt} = v_\theta(x_t, t, c_{\text{obs}})$$

that transports samples from a simple prior $x_0 \sim \mathcal{N}(0, I)$ to the target distribution $x_1 \sim p(m | c_{\text{obs}})$. Sampling uses Euler integration with 24–32 steps.

---

## Citation

```bibtex
@article{yu2026physical,
  title={Physical Sensitivity Kernels Can Emerge in Data-Driven Forward Models:
         Evidence From Surface-Wave Dispersion},
  author={Yu, Ziye and Cai, Yuqi and Liu, Xin},
  journal={arXiv preprint arXiv:2604.04107},
  year={2026},
  url={https://arxiv.org/abs/2604.04107}
}

@software{SurfFlow_code,
  author = {Yu, Ziye and Cai, Yuqi and Liu, Xin},
  title = {Physical Sensitivity Kernels: Official Implementation},
  year = {2026},
  url = {https://github.com/cangyeone/SurfFlow}
}
```

---

## Author

- **Ziye Yu** ([@cangyeone](https://github.com/cangyeone)) — cangye@hotmail.com
- **Yuqi Cai**
- **Xin Liu**

---

## License

GNU General Public License v3.0 (GPLv3). See [LICENSE](LICENSE) for details.

---

**⭐ If you find this repository useful, please consider giving it a star!**
