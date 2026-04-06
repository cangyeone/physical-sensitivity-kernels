

# Physical Sensitivity Kernels

[](https://opensource.org/licenses/MIT)
[](https://www.python.org/downloads/)
[](https://en.wikipedia.org/wiki/Geophysics)

A high-performance Python library for calculating **Seismic Sensitivity Kernels** (Fréchet derivatives). This tool is designed for geophysical inversion, seismic tomography, and the integration of physical constraints into machine learning models (e.g., Physics-Informed Neural Networks).

-----

## 📖 Overview

In seismic inversion, sensitivity kernels $K(z)$ describe how much a specific observation (such as phase velocity or travel time) changes in response to a small perturbation in model parameters (such as $V_p, V_s,$ or density $\rho$) at a certain depth:

$$\delta d = \int_0^H K(z) \delta m(z) dz$$

This repository provides an efficient engine to compute these kernels for layered media, supporting both Rayleigh and Love waves across various frequencies.

-----

## ✨ Key Features

  * **Multi-Parameter Support**: Calculate kernels for $V_p$, $V_s$, and density $\rho$.
  * **Surface Wave Analysis**: Specialized for Phase Velocity and Group Velocity kernels.
  * **Performance Optimized**: Leverages vectorized NumPy operations for fast matrix assembly and kernel integration.
  * **AI-Ready**: Designed to produce Jacobian matrices that can be directly plugged into PyTorch/TensorFlow pipelines for geophysical deep learning.
  * **Physically Consistent**: Built on fundamental wave equations and perturbation theory (Rayleigh Principle).

-----

## 🛠️ Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/cangyeone/physical-sensitivity-kernels.git
cd physical-sensitivity-kernels
pip install -r requirements.txt
```

**Core Dependencies:**

  * `numpy` (Numerical computation)
  * `scipy` (Integration and matrix solvers)
  * `matplotlib` (Visualization)

-----

## 🚀 Quick Start

The following example demonstrates how to compute the $V_s$ sensitivity kernel for a layered earth model:

```python
from kernel_engine import SensitivityKernel
import matplotlib.pyplot as plt

# Define the Earth Model: [Thickness (km), Vp (km/s), Vs (km/s), Density (g/cm^3)]
model = [
    [10.0, 5.8, 3.4, 2.7],
    [20.0, 6.5, 3.8, 2.9],
    [0.0,  8.1, 4.5, 3.3]  # Half-space
]

# Initialize engine
engine = SensitivityKernel(model)

# Compute Rayleigh wave Vs kernel at 20s period
depths, kernel_values = engine.compute_kernel(
    period=20.0, 
    wave_type='Rayleigh', 
    target_param='Vs'
)

# Plotting
plt.figure(figsize=(4, 6))
plt.plot(kernel_values, depths)
plt.gca().invert_yaxis()
plt.title("Vs Sensitivity Kernel (T=20s)")
plt.xlabel("Sensitivity")
plt.ylabel("Depth (km)")
plt.grid(True)
plt.show()
```

-----

## 📚 Mathematical Background

The kernel calculation is based on the **Rayleigh Quotient** perturbation. For an eigenfrequency $\omega$, the sensitivity to a parameter $m$ is derived from the energy integrals of the eigenfunctions:

$$K_m(z) = \frac{\delta \omega}{\delta m(z)}$$

Specifically for shear modulus $\mu$, the kernel involves the displacement eigenfunctions $r_1(z)$ and $r_2(z)$ and their derivatives. This library handles the numerical stabilization required when dealing with high-frequency modes or complex layering.

-----

## 🤝 Contributing

Contributions are welcome\! If you have suggestions for new features (e.g., anisotropic kernels, 2D/3D kernels) or bug fixes, please open an issue or submit a pull request.

  * **Author**: [cangyeone](https://github.com/cangyeone)
  * **Contact**: cangye@hotmail.com

-----

## ⚖️ License

This project is licensed under the **GPLv3 License**.

**Citation Notice:** If you use this code in your academic research, please cite this repository and acknowledge the author's work in your publications.
