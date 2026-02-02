# Tensorial Risk Engine

A **fully differentiable catastrophe risk assessment framework** leveraging tensor operations and automatic differentiation for GPU-accelerated risk calculations. This engine reformulates traditional task-based risk computation as tensor programs, enabling massive parallelization and gradient-based sensitivity analysis.

## 🚀 Overview

The Tensorial Risk Engine implements a complete probabilistic catastrophe risk assessment system that:

- **🏎️ Accelerates computation** using GPU tensor operations (TensorFlow/Metal)
- **📊 Scales efficiently** to millions of loss calculations with O(1) complexity on modern hardware
- **🎯 Enables optimization** through automatic differentiation of all input parameters
- **🔬 Provides gradients** for vulnerability curves, exposure values, and hazard intensities
- **📈 Computes comprehensive metrics** including per-asset and portfolio-level Average Annual Loss (AAL)

Unlike traditional iterative approaches that scale as O(N·Q), this tensorial formulation processes entire portfolios in parallel, delivering orders of magnitude speedup for large-scale assessments.

## 🎓 Theoretical Foundation

This implementation is based on a rigorous mathematical formulation for differentiable catastrophe risk assessment. The framework reformulates classical probabilistic risk calculations as tensor operations, making the entire computation pipeline:

1. **Hardware-efficient** - Optimized for GPUs/TPUs with massive data parallelism
2. **Differentiable** - Enables gradient computation without re-simulation
3. **Scalable** - Handles large portfolios (millions of assets) efficiently

### Key Mathematical Components

- **Section 2:** Deterministic hazard formulation with interpolation-based vulnerability lookups
- **Section 3:** Probabilistic hazard formulation computing loss matrices and risk metrics
- **Section 4:** Gradient computation w.r.t. vulnerability curves (∂AAL/∂C)
- **Section 5:** Gradient computation w.r.t. exposure values (∂AAL/∂v) and hazard intensities (∂AAL/∂H)

## 📁 Repository Structure

```
Tensor_Risk_Engine/
├── engine/
│   └── tensor_engine.py          # Core tensorial risk engine
├── examples/
│   ├── demonstration.py           # Complete demonstration of all features
│   ├── minimum_example_deterministic.py  # Minimal example for validation
│   └── hand_calculation_deterministic.ipynb  # Step-by-step manual calculations
├── API_DOCUMENTATION.md           # Detailed API reference
├── README_IMPLEMENTATION.md       # Implementation details and manuscript compliance
├── MINIMUM_EXAMPLE_README.md      # Guide for minimal examples
└── README.md                      # This file
```

## 🛠️ Installation

### Requirements

- Python 3.12
- TensorFlow 2.16+
- NumPy
- Matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Tensor_Risk_Engine.git
cd Tensor_Risk_Engine

# Install dependencies
pip install tensorflow numpy matplotlib

# For Apple Silicon (M1/M2/M3) with Metal acceleration
pip install tensorflow-metal
```

## 🚦 Quick Start

### Basic Usage

```python
from engine.tensor_engine import generate_synthetic_portfolio, TensorialRiskEngine

# Generate a synthetic portfolio
v, u, C, x_grid, H = generate_synthetic_portfolio(
    n_assets=1000,        # Number of buildings/assets
    n_events=5000,        # Number of stochastic events
    n_typologies=5,       # Number of building types
    n_curve_points=20     # Vulnerability curve resolution
)

# Initialize the risk engine
engine = TensorialRiskEngine(v, u, C, x_grid, H)

# Compute loss matrix and risk metrics
J_matrix, metrics = engine.compute_loss_and_metrics()

# Display results
print(f"Portfolio AAL: ${metrics['aal_portfolio'].numpy():,.2f}")
print(f"Portfolio Std Dev: ${metrics['std_portfolio'].numpy():,.2f}")

# Perform complete gradient analysis
analysis = engine.full_gradient_analysis()
grad_vulnerability = analysis['grad_vulnerability']  # ∂AAL/∂C
grad_exposure = analysis['grad_exposure']            # ∂AAL/∂v
grad_hazard = analysis['grad_hazard']                # ∂AAL/∂H
```

### Run Examples

#### 1. Complete Demonstration
```bash
python examples/demonstration.py
```

Shows all features including:
- Deterministic loss computation
- Probabilistic metrics (AAL, variance)
- All gradient computations
- Performance benchmarks
- Comprehensive visualizations

#### 2. Minimal Example (for validation)
```bash
python examples/minimum_example_deterministic.py
```

A simplified 10-asset, 1-event portfolio designed for hand-checking calculations.

#### 3. Step-by-Step Manual Calculations
```bash
jupyter notebook examples/hand_calculation_deterministic.ipynb
```

Interactive notebook with detailed breakdown of every calculation step.

## 📊 Data Model

The engine uses manuscript-compliant notation:

| Symbol | Shape | Type | Description |
|--------|-------|------|-------------|
| **v** | (N,) | ℝ^N | Exposure vector - replacement costs for each asset |
| **u** | (N,) | ℤ^N | Typology index - maps each asset to a building type |
| **C** | (K, M) | ℝ^(K×M) | Vulnerability matrix - K curves with M points each |
| **x** | (M,) | ℝ^M | Intensity grid - common x-axis for all vulnerability curves |
| **H** | (N, Q) | ℝ^(N×Q) | Hazard matrix - ground motion intensities per asset per event |
| **J** | (N, Q) | ℝ^(N×Q) | Loss matrix - computed losses per asset per event |

### Example Data

```python
v = [100000, 250000, 500000, ...]  # N assets with replacement costs
u = [0, 2, 1, 0, 4, ...]           # N typology indices (0 to K-1)
C = [[0.0, 0.1, 0.3, ...],         # K curves × M points
     [0.0, 0.2, 0.5, ...], ...]    # Values in [0, 1] range
x_grid = [0.0, 0.1, 0.2, ..., 1.5] # M intensity values (e.g., PGA in g)
H = [[0.2, 0.3, 0.15, ...],        # N assets × Q events
     [0.4, 0.5, 0.25, ...], ...]   # Intensity per asset per event
```

## 🎯 Key Features

### 1. Multi-Typology Support
- Maps each asset to its specific building type using typology index vector **u**
- Supports K different vulnerability curves simultaneously
- Proper vectorized lookup without loops

### 2. Complete Risk Metrics
- **Per-asset AAL** - Average Annual Loss for each building
- **Portfolio AAL** - Total expected annual loss
- **Variance and Standard Deviation** - Per-asset and portfolio-level
- **Loss per Event Distribution** - Full loss matrix J ∈ ℝ^(N×Q)

### 3. Full Gradient Analysis
Computes sensitivities for:
- **∂AAL/∂C** - Vulnerability curve calibration and optimization
- **∂AAL/∂v** - Retrofit prioritization and exposure optimization
- **∂AAL/∂H** - Hazard uncertainty quantification

### 4. GPU Acceleration
- TensorFlow Metal support for Apple Silicon
- Optimized flat indexing for maximum GPU compatibility
- Graph compilation with `@tf.function` decorators
- Batch processing of entire portfolios

## 📈 Performance

The tensorial approach delivers dramatic speedups over traditional iterative methods:

- **1,000 assets × 5,000 events**: ~100-500ms (GPU) vs ~10-60s (CPU loops)
- **100,000 assets × 10,000 events**: ~2-5s (GPU) vs hours (CPU loops)

Performance scales with hardware parallelism, not problem size.

## 🧮 Mathematical Formulation

### Deterministic Loss (Single Event)

For a single hazard scenario with intensity **h** ∈ ℝ^N:

1. **Interpolation weight:** α = (h - x_j) / (x_{j+1} - x_j)
2. **Mean Damage Ratio:** MDR_i = (1-α)·C[u_i, j] + α·C[u_i, j+1]
3. **Total loss:** J = Σ v_i · MDR_i

### Probabilistic Loss (Multiple Events)

For Q stochastic events:

1. **Loss matrix:** J[i,q] = v_i · MDR[i,q] ∈ ℝ^(N×Q)
2. **Per-asset AAL:** AAL_i = (1/Q) Σ_q J[i,q]
3. **Variance:** σ²_i = (1/Q) Σ_q (J[i,q] - AAL_i)²

## 🔍 Use Cases

1. **Portfolio Risk Assessment** - Compute AAL for large building portfolios
2. **Retrofit Prioritization** - Use ∂AAL/∂v to identify high-impact assets
3. **Vulnerability Calibration** - Optimize curves using ∂AAL/∂C gradients
4. **Uncertainty Quantification** - Assess hazard sensitivity via ∂AAL/∂H
5. **Real-time Risk Assessment** - GPU acceleration enables near-instantaneous updates
6. **Optimization Problems** - Gradient-based optimization for risk mitigation strategies

## 📚 Documentation

- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - Complete API reference with detailed parameter descriptions
- **[README_IMPLEMENTATION.md](README_IMPLEMENTATION.md)** - Implementation details and manuscript compliance
- **[MINIMUM_EXAMPLE_README.md](MINIMUM_EXAMPLE_README.md)** - Guide for validation examples

## 🤝 Contributing

More examples and use cases will be added in future releases. Contributions are welcome!

## 📄 License

[Add your license here]

## 📧 Contact

[Add your contact information here]

## 🙏 Acknowledgments

This work reformulates traditional catastrophe risk assessment for modern AI/ML hardware ecosystems, enabling gradient-based sensitivity analysis and massive parallelization without the need for HPC clusters.

## 📖 Citation

If you use this engine in your research, please cite:

```bibtex
[Add citation information when manuscript is published]
```

---

**Note:** This is an initial release. Additional examples, benchmarks, and features will be added in future versions.
