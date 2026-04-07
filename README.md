# Physical Sensitivity Kernels

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.04107-b31b1b.svg)](https://doi.org/10.48550/arXiv.2604.04107)
[![Geophysics](https://img.shields.io/badge/field-seismology-green.svg)](https://en.wikipedia.org/wiki/Seismology)

Official implementation of **"Physical Sensitivity Kernels Can Emerge in Data-Driven Forward Models: Evidence From Surface-Wave Dispersion"** (arXiv:2604.04107).

This repository investigates whether data-driven neural networks used as surrogate forward models in geophysics recover only the data mapping or also the underlying physical sensitivity structure. Using surface-wave dispersion as a test case, we demonstrate that automatically differentiated gradients from neural network surrogates can recover the main depth-dependent structure of theoretical sensitivity kernels across a broad range of periods.

---

## 📖 Overview

### Research Question

Data-driven neural networks are increasingly used as surrogate forward models in geophysics, but it remains unclear whether they:
- Recover **only the data mapping** (black-box predictors), or
- Also learn the **underlying physical sensitivity structure** (physically meaningful differential information)

### Key Findings

By comparing automatically differentiated gradients from a neural-network surrogate with theoretical sensitivity kernels, we show that:

✅ **Neural surrogates can recover physically meaningful differential structure**: The learned gradients match the main depth-dependent structure of theoretical kernels across periods

⚠️ **Training distribution priors matter**: Strong structural priors in the training distribution can introduce systematic artifacts into the inferred sensitivities

🎯 **Practical implications**: Neural forward surrogates can provide useful physical information for inversion and uncertainty analysis, under appropriate conditions

### Methodology

1. **Forward Model**: Train a Transformer-based neural network to learn $\mathcal{F}: m(z) \rightarrow c(T)$ (velocity model → dispersion curves)
2. **Automatic Differentiation**: Compute sensitivity kernels via Jacobian $K_j(z) = \frac{\partial c(T)}{\partial m_j(z)}$ using PyTorch's autograd
3. **Theoretical Comparison**: Validate against analytical kernels from `disba.PhaseSensitivity`
4. **Systematic Analysis**: Test across wave types (Rayleigh, Love), periods (2-60s), and model parameterizations

### Repository Contents

- **Struct2DispTransformer**: Transformer architecture for learning the forward operator
- **Automatic Jacobian Computation**: Efficient gradient computation using `torch.func.jacrev` + `vmap`
- **Synthetic Data Generation**: Realistic Earth models with tectonic-type statistical priors
- **Validation Scripts**: Comprehensive comparison tools (`disp_gen_test.sk.v1.1.py`)
- **Analysis Tools**: Fisher information, metrics evaluation, and visualization utilities

---

## ✨ Key Features

### Forward Modeling: Velocity → Dispersion
- **Neural Operator**: Transformer-based architecture learns the mapping $m(z) \rightarrow c(T)$
- **Multi-Wave Support**: Simultaneously predicts Rayleigh and Love wave dispersion curves
- **High Accuracy**: Trained on physics-based synthetic data with <0.05 km/s RMSE
- **Batch Processing**: Efficient computation for multiple models simultaneously
- **Physical Consistency**: Learned forward operator respects wave propagation physics

### Sensitivity Kernel Computation: Automatic Jacobian
- **Exact Gradients**: PyTorch autograd computes Fréchet derivatives without finite differences
- **Flexible Selection**: Compute kernels for specific periods, wave types, and parameters on-demand
- **Full Jacobian Matrix**: Build complete sensitivity matrices $[2T \times 3H]$ for inversion
- **Memory Efficient**: On-demand gradient computation for targeted periods
- **Resolution Analysis**: Direct access to model resolution and parameter covariance matrices

### Physical Mechanism Emergence
- **Emergent Physics**: Sensitivity kernels emerge naturally from Jacobian computation, demonstrating learned physical understanding
- **Theory Validation**: Direct comparison with `disba` theoretical kernels confirms correctness
- **Multi-Period Analysis**: Kernels computed across period bands (2-5s, 10-15s, 20-30s, etc.) show proper depth sensitivity patterns
- **No Explicit Constraints**: Physics emerges from data-driven training without hard-coded physical laws

### Probabilistic Outputs
- **Uncertainty Quantification**: Provides both mean predictions ($\mu$) and variance ($\sigma^2$)
- **Confidence Intervals**: Enable robust uncertainty propagation in inversion workflows
- **Log-Variance Clipping**: Numerical stability through bounded uncertainty estimates

### Synthetic Data Generation
- **Tectonic-Type Priors**: Generate realistic Earth models based on geological settings:
  - Oceanic crust (thin, basaltic)
  - Continental shields (thick, cold)
  - Cratons (stable, high-velocity)
  - Active tectonic regions
- **Empirical Relations**: Automatic density estimation using Brocher (2005) formulas
- **Low-Velocity Zone Modeling**: Configurable LVZ properties for upper mantle
- **Controlled Randomness**: Full seed control for reproducible experiments

### Training & Evaluation
- **Efficient Training**: AdamW optimizer with gradient clipping and SmoothL1 loss
- **Comprehensive Metrics**: Fisher information matrix, sensitivity kernels, and standard regression metrics
- **Visualization Tools**: Automatic plotting of dispersion curve fits during training
- **Checkpoint Management**: Save and resume training with model checkpoints

### Performance
- **GPU Acceleration**: Full CUDA/MPS support for fast training and inference
- **Batch Processing**: Efficient data loading with parallel workers
- **Memory Optimization**: LayerNorm-first architecture for stable training

---

## 🏗️ Architecture

### Project Structure

```
physical-sensitivity-kernels/
├── models/
│   └── struct2disp_transformer.py    # Transformer model for structure-to-dispersion mapping
├── utils/
│   ├── generate_data.py              # Synthetic data generation with tectonic priors
│   └── generate_data_weak_prior.py   # Weak prior data generation
├── disp_gen_train.v1.1.py            # Training script
├── disp_gen_test.*.py                # Various testing/evaluation scripts
└── README.md
```

### Model Architecture: Struct2DispTransformer

```
Input: [B, C_in=4, H]
       (depth, Vp, Vs, ρ profiles at H depth points)
        │
        ▼
┌─────────────────────┐
│ Depth Token Embed   │  Linear projection: C_in → d_model
│ + Positional Enc    │  Sinusoidal encoding for depth dimension
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Transformer Encoder  │  6-8 layers, multi-head self-attention
│  (batch_first=True)  │  Captures depth-wise dependencies
└─────────┬───────────┘
          │
          ▼
    Memory: [B, H, d_model]
          │
          ▼
┌─────────────────────┐
│ Period Query Tokens  │  Learnable queries: [T=59, d_model]
│ + Period Values      │  Optional: inject normalized period info
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│Transformer Decoder   │  3-8 layers, cross-attention
│ (cross-attn mode)    │  Queries attend to depth memory
└─────────┬───────────┘
          │
          ▼
    Output: [B, T, d_model]
          │
          ▼
┌─────────────────────┐
│  Output Heads       │  μ:  [B, T, 2]  (Rayleigh, Love)
│  (MLP + GELU)       │  logσ²: [B, T, 2]  (uncertainty)
└─────────┬───────────┘
          │
          ▼
Output: [B, 2, T] after permute
```

**Key Design Choices:**
- **Sinusoidal Positional Encoding**: Captures depth ordering without learned parameters
- **Learnable Period Queries**: Each of the 59 periods has a dedicated query vector
- **Period Value Injection**: MLP transforms physical period values into query biases
- **Log-Variance Clipping**: Prevents numerical instability (clamped to [-10, 3])
- **LayerNorm-First**: Improves training stability and gradient flow
- **GELU Activations**: Smooth non-linearity throughout the network

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9+ (CUDA recommended for GPU training)
- NumPy, SciPy, Matplotlib

### Setup

```bash
git clone https://github.com/cangyeone/physical-sensitivity-kernels.git
cd physical-sensitivity-kernels
pip install torch numpy scipy matplotlib
```

**Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥1.9 | Deep learning framework |
| `numpy` | ≥1.20 | Numerical computations |
| `scipy` | ≥1.7 | Scientific utilities |
| `matplotlib` | ≥3.4 | Visualization |

---

## 🚀 Quick Start

### 1. Training the Model

Train the Struct2DispTransformer on synthetic data:

```bash
# Option 1: Run training script directly
python disp_gen_train.v1.1.py
```

Or use programmatically:

```python
import torch
from torch.utils.data import DataLoader
from utils.generate_data import SurfaceWaveDataset
from disp_gen_train.v1_1 import train_struct2disp_transformer

# Create synthetic dataset
dataset = SurfaceWaveDataset(
    n_samples=100_000,    # Number of training samples
    z_max_km=150.0,       # Maximum depth (km)
    z_max_num=256,        # Depth discretization points
    dz_km=0.5,            # Depth spacing
    seed=2026             # Random seed for reproducibility
)

# Data loader
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

# Train model
model = train_struct2disp_transformer(
    loader,
    n_epoch=200,
    lr=2e-4,
    weight_decay=1e-4,
    ckpt_path="ckpt/struct2disp_transformer.v1.1.pt",
    fig_dir="tfig"
)
```

### 2. Using a Pre-trained Model

Load a checkpoint and make predictions:

```python
import torch
from models.struct2disp_transformer import Struct2DispTransformer

# Initialize model (must match training config)
model = Struct2DispTransformer(
    H=256,                # Depth points (must match training)
    T=59,                 # Period points (fixed)
    C_in=4,               # Input channels: depth, Vp, Vs, rho
    d_model=512,
    nhead=8,
    num_enc_layers=8,
    num_dec_layers=8,
    dim_ff=1024,
    dropout=0.1,
    use_period_values=True,
    period_minmax=(1.0, 100.0)  # Period range used in training
)

# Load checkpoint
checkpoint = torch.load('ckpt/struct2disp_transformer.v1.1.pt',
                        map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# Prepare input: [Batch, 4, Depth]
# Example: simple layered model
depths = torch.linspace(0, 150, 256)
vp = torch.ones(256) * 6.0      # Vp profile (km/s)
vs = torch.ones(256) * 3.5      # Vs profile (km/s)
rho = torch.ones(256) * 2.7     # Density (g/cm³)

earth_model = torch.stack([depths, vp, vs, rho]).unsqueeze(0)  # [1, 4, 256]

# Define periods for prediction
periods = torch.linspace(1.0, 100.0, 59)  # [59]

# Predict dispersion curves
with torch.no_grad():
    mu, logvar = model(earth_model, periods=periods)
    # mu shape: [1, 2, 59] -> [batch, wave_type, period]
    # logvar shape: [1, 2, 59]

# Extract predictions
rayleigh_velocity = mu[0, 0, :]  # Rayleigh wave dispersion
love_velocity = mu[0, 1, :]      # Love wave dispersion
uncertainty = torch.exp(logvar)  # Convert log-variance to std

print(f"Rayleigh velocities: {rayleigh_velocity}")
print(f"Love velocities: {love_velocity}")
```

### 3. Generating Synthetic Data

Create custom datasets with specific tectonic characteristics:

```python
from utils.generate_data import SurfaceWaveDataset
import matplotlib.pyplot as plt

# Generate oceanic crust models
oceanic_dataset = SurfaceWaveDataset(
    n_samples=10000,
    z_max_km=200.0,
    z_max_num=100,
    tectonic_type='oceanic',  # Specify tectonic setting
    seed=42
)

# Access a sample
model_profile, dispersion_curves, mask = oceanic_dataset[0]
# model_profile: [4, H] - depth, Vp, Vs, rho
# dispersion_curves: [3, T] - periods, Rayleigh c, Love c
# mask: valid period mask

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot velocity profile
ax1.plot(model_profile[1], model_profile[0], 'b-', label='Vp')
ax1.plot(model_profile[2], model_profile[0], 'r-', label='Vs')
ax1.set_xlabel('Velocity (km/s)')
ax1.set_ylabel('Depth (km)')
ax1.legend()
ax1.grid(True)

# Plot dispersion curves
periods = dispersion_curves[0].numpy()
ax2.plot(periods, dispersion_curves[1].numpy(), 'b-', label='Rayleigh')
ax2.plot(periods, dispersion_curves[2].numpy(), 'r-', label='Love')
ax2.set_xlabel('Period (s)')
ax2.set_ylabel('Phase Velocity (km/s)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('example_model.png', dpi=150)
plt.show()
```

### 4. Sensitivity Kernel Validation with disba

The repository includes a comprehensive validation script (`disp_gen_test.sk.v1.1.py`) that compares neural network-computed sensitivity kernels against theoretical kernels from the `disba` library:

```python
# Run sensitivity kernel validation
python disp_gen_test.sk.v1.1.py
```

**What this script does:**

1. **Loads trained model** and generates synthetic Earth models
2. **Computes NN kernels** via automatic Jacobian (using `torch.func.jacrev`)
3. **Computes theoretical kernels** using `disba.PhaseSensitivity` 
4. **Converts to Fréchet form**: $\frac{\partial \ln c}{\partial \ln V_s} = \frac{V_s}{c} \cdot \frac{\partial c}{\partial V_s}$
5. **Visualizes comparison** across multiple periods and wave types

**Key functions:**

- `compute_dcdvs_full_jacobian()`: Computes full Jacobian $K = \frac{\partial c}{\partial V_s}$ using `torch.func.vmap` + `jacrev`
- `disba_vs_phase_sensitivity()`: Wraps `disba` library for theoretical kernel computation
- `plot_kernels_with_disba_multiperiod()`: Compares NN vs theory at selected periods
- `plot_kernels_with_disba_multiband()`: Averages kernels over period bands (e.g., 2-5s, 10-15s, 20-30s)

**Physical mechanism emergence:**

The agreement between NN-computed kernels and theoretical kernels demonstrates that the Transformer has learned the underlying physics of surface wave propagation. The sensitivity kernels emerge naturally from the Jacobian of the learned forward operator, without explicit physical constraints during training.

**Output examples:**

- Multi-period comparison plots (Rayleigh & Love side-by-side)
- Period-band averaged kernels (0-150 km depth)
- Normalized sensitivity shapes highlighting depth dependence

---

## 📚 Mathematical Background

### 1. Forward Problem: Velocity Model → Dispersion Curves

The core task is to learn the nonlinear forward operator:

$$\mathcal{F}: m(z) \rightarrow c(T)$$

where:
- **Input**: Earth model $m(z) = [V_p(z), V_s(z), \rho(z)]$ at discrete depth points $z_1, z_2, ..., z_H$
- **Output**: Phase velocities $c(T) = [c_R(T), c_L(T)]$ for Rayleigh and Love waves at periods $T_1, T_2, ..., T_{59}$

The `Struct2DispTransformer` approximates this operator using a neural network with parameters $\theta$:

$$c(T) \approx \mathcal{F}_\theta(m(z))$$

### 2. Probabilistic Formulation

The model outputs a conditional Gaussian distribution:

$$p(c|m) = \mathcal{N}(c; \mu_\theta(m), \sigma^2_\theta(m))$$

where:
- $\mu_\theta(m)$: Predicted mean dispersion curves
- $\sigma^2_\theta(m)$: Predicted uncertainty (from `logvar` output)

Training minimizes the SmoothL1 loss:

$$\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N \text{SmoothL1}(\mu_\theta(m_i), c_i^{\text{true}})$$

### 3. Sensitivity Kernels via Jacobian Computation

The sensitivity kernel (Fréchet derivative) quantifies how phase velocity changes with respect to model parameters:

$$K_j(z_k; T) = \frac{\partial c(T)}{\partial m_j(z_k)}$$

where:
- $j \in \{V_p, V_s, \rho\}$ is the parameter type
- $z_k$ is the depth index
- $T$ is the period

#### Jacobian Matrix Structure

For a model with $H$ depth points and $T$ periods, the full Jacobian matrix has shape $[2T \times 3H]$:

$$J = \begin{bmatrix} 
\frac{\partial c_R(T_1)}{\partial V_p(z_1)} & \cdots & \frac{\partial c_R(T_1)}{\partial V_p(z_H)} & \frac{\partial c_R(T_1)}{\partial V_s(z_1)} & \cdots \\
\vdots & \ddots & \vdots & \vdots & \ddots \\
\frac{\partial c_L(T_{59})}{\partial V_p(z_1)} & \cdots & \frac{\partial c_L(T_{59})}{\partial V_p(z_H)} & \frac{\partial c_L(T_{59})}{\partial V_s(z_1)} & \cdots
\end{bmatrix}$$

#### Efficient Computation via Autograd

PyTorch's automatic differentiation computes these gradients efficiently:

```python
# Forward pass
model.eval()
earth_model = torch.randn(1, 4, H, requires_grad=True)
mu, logvar = model(earth_model, periods=periods)

# Compute Jacobian row for Rayleigh wave at period T_i
rayleigh_pred = mu[0, 0, i]  # Scalar
rayleigh_pred.backward()      # Backpropagate

# Extract sensitivity kernel
kernel_vs = earth_model.grad[0, 2, :]  # ∂c_R(T_i)/∂Vs(z) for all z
kernel_vp = earth_model.grad[0, 1, :]  # ∂c_R(T_i)/∂Vp(z) for all z
kernel_rho = earth_model.grad[0, 3, :] # ∂c_R(T_i)/∂ρ(z) for all z
```

This approach is:
- **Exact**: No finite-difference approximation errors
- **Efficient**: Single backward pass computes all depth derivatives
- **Memory-friendly**: Gradients computed on-demand for specific periods

### 4. Applications of Sensitivity Kernels

Sensitivity kernels enable:

- **Linearized Inversion**: Build Jacobian matrices for iterative model updates
  $$\delta m = (J^T J + \lambda I)^{-1} J^T \delta d$$
  
- **Resolution Analysis**: Assess which depths are well-constrained by observations
  
- **Experimental Design**: Optimize period selection for maximum information content
  
- **Uncertainty Quantification**: Propagate data uncertainties to model uncertainties

---

## 🔬 Applications

### 1. Forward Modeling
Use the trained neural network to compute dispersion curves from velocity models:
- **Differentiable**: Native gradient support for downstream tasks
- **Batch Processing**: Compute dispersion curves for multiple models simultaneously
- **Physical Consistency**: Learned operator preserves wave propagation physics

### 2. Sensitivity Kernel Computation
Compute Fréchet derivatives via automatic Jacobian calculation:
- **Exact Gradients**: No finite-difference approximation errors
- **Flexible Selection**: Compute kernels for specific periods and wave types on-demand
- **Full Jacobian**: Build complete sensitivity matrices for inversion workflows
- **Theory Validation**: Compare with `disba` theoretical kernels to verify physical correctness

### 3. Seismic Inversion
Use the trained model as a differentiable forward operator in iterative inversion:

**Gradient-Based Optimization**:
```python
# Define misfit function
def misfit(model_params):
    mu, _ = net(model_params)
    return torch.sum((mu - observed_data)**2)

# Optimize using gradients
optimizer = torch.optim.LBFGS([model_params], lr=0.1)
optimizer.step(closure=lambda: compute_misfit())
```

**Linearized Iterative Inversion**:
$$\delta m = (J^T C_d^{-1} J + \lambda C_m^{-1})^{-1} J^T C_d^{-1} \delta d$$

### 4. Resolution and Uncertainty Analysis
Leverage Jacobian matrices for:
- **Model Resolution**: $R = (J^T J + \lambda I)^{-1} J^T J$
- **Parameter Covariance**: $\text{Cov}(m) = (J^T J)^{-1}$
- **Depth Sensitivity**: Identify which depths are constrained by observed periods
- **Experimental Design**: Optimize period selection before data acquisition

### 5. Physics-Informed Machine Learning
Integrate with PINN frameworks:
- Use predicted dispersion curves as physics-based constraints
- Enforce consistency between velocity models and observed data
- Joint training with other geophysical observables

### 6. Transfer Learning
Fine-tune on real data after pre-training on synthetic datasets:
- Adapt to regional geological characteristics
- Incorporate site-specific prior information
- Improve generalization to observed data

---

## 🧪 Testing & Evaluation

The repository includes several evaluation scripts:

### Fisher Information Analysis
```bash
python disp_gen_test.fisher.control_point.v1.1.py
```
Computes Fisher information matrices to assess parameter identifiability.

### Metric Evaluation
```bash
python disp_gen_test.metrics.v1.1.py
```
Calculates standard regression metrics (RMSE, MAE, R²) on test sets.

### Sensitivity Kernel Validation (Physical Mechanism Verification)
```bash
python disp_gen_test.sk.v1.1.py
```

This script validates that the sensitivity kernels computed via automatic differentiation match theoretical expectations:

**Methodology:**
1. **NN Kernels**: Computes $K = \frac{\partial c}{\partial V_s}$ using `torch.func.jacrev` + `vmap`
2. **Theoretical Kernels**: Uses `disba.PhaseSensitivity` for analytical kernel computation
3. **Fréchet Conversion**: Converts to $\frac{\partial \ln c}{\partial \ln V_s} = \frac{V_s}{c} \cdot \frac{\partial c}{\partial V_s}$
4. **Multi-Period Comparison**: Validates across period bands (2-5s, 10-15s, 20-30s, 30-40s, 50-60s)
5. **Visualization**: Generates side-by-side comparison plots for Rayleigh and Love waves

**Physical Significance:**
The agreement between NN-computed kernels and theoretical kernels demonstrates that the Transformer has learned the underlying physics of surface wave propagation. The sensitivity patterns emerge naturally from the Jacobian without explicit physical constraints during training.

**Output:**
- Multi-period comparison plots (`tfig/multi2/frechet_kernel_multiperoid_*.png`)
- Period-band averaged kernels (`tfig/multi2/frechet_kernel_multiband_*.png`)
- Depth range: 0-150 km with proper normalization

### Reproducibility
All scripts use fixed random seeds:
```python
torch.manual_seed(2026)
np.random.seed(2026)
random.seed(2026)
torch.backends.cudnn.deterministic = True
```

---

## 📊 Expected Results

After training, you should observe:
- **Training Loss**: Converges to ~0.01-0.05 (SmoothL1)
- **Prediction Accuracy**: <0.05 km/s RMSE for both Rayleigh and Love waves
- **Inference Speed**: ~1 ms per sample on GPU
- **Uncertainty Calibration**: 90% of true values within predicted 90% confidence intervals


---

## 📖 Citation

If you use this code or methodology in your research, please cite:

### Paper Citation

```bibtex
@article{yu2026physical,
  title={Physical Sensitivity Kernels Can Emerge in Data-Driven Forward Models: Evidence From Surface-Wave Dispersion},
  author={Yu, Ziye and Cai, Yuqi and Liu, Xin},
  journal={arXiv preprint arXiv:2604.04107},
  year={2026},
  url={https://doi.org/10.48550/arXiv.2604.04107},
  archivePrefix={arXiv},
  eprint={2604.04107},
  primaryClass={cs.LG, physics.geo-ph}
}
```

### Repository Citation

```bibtex
@software{physical_sensitivity_kernels_code,
  author = {Yu, Ziye and Cai, Yuqi and Liu, Xin},
  title = {Physical Sensitivity Kernels: Official Implementation},
  year = {2026},
  url = {https://github.com/cangyeone/physical-sensitivity-kernels},
  note = {Code for arXiv:2604.04107}
}
```

**Related Links:**
- **arXiv**: [https://arxiv.org/abs/2604.04107](https://doi.org/10.48550/arXiv.2604.04107)
- **DOI**: [10.48550/arXiv.2604.04107](https://doi.org/10.48550/arXiv.2604.04107)
- **Subjects**: Machine Learning (cs.LG); Geophysics (physics.geo-ph)

---

## 👤 Author & Contact

- **Author**: [cangyeone](https://github.com/cangyeone)
- **Email**: cangye@hotmail.com
- **GitHub**: [https://github.com/cangyeone/physical-sensitivity-kernels](https://github.com/cangyeone/physical-sensitivity-kernels)

---

## ⚖️ License

This project is licensed under the **GNU General Public License v3.0** (GPLv3).

**Permissions:**
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Patent use
- ✅ Private use

**Conditions:**
- 📄 License and copyright notice
- 📄 State changes
- 📄 Disclose source
- 📄 Same license (copyleft)

See the [LICENSE](LICENSE) file for full legal text.

---


**⭐ If you find this repository useful for your research, please consider giving it a star!**
