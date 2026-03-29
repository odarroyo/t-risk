# Tensorial Risk Engine

A **fully differentiable risk engine** leveraging tensor operations and automatic differentiation for GPU-accelerated risk calculations. This engine reformulates traditional task-based risk computation as tensor programs, enabling massive parallelization and gradient-based sensitivity analysis.

Complemented by an interactive web interface available! Run `streamlit run streamlit/app.py` for a complete browser-based risk analysis platform. See [Web Application](#-web-application) section below.

## 🚀 Overview

The Tensorial Risk Engine implements a complete probabilistic risk assessment system that:

- **🏎️ Accelerates computation** using GPU tensor operations (TensorFlow/Metal)
- **📊 Scales efficiently** to millions of loss calculations with O(1) complexity on modern hardware
- **🎯 Enables optimization** through automatic differentiation of all input parameters
- **🔬 Provides gradients** for vulnerability curves, exposure values, and hazard intensities
- **📈 Computes comprehensive metrics** including per-asset and portfolio-level Average Annual Loss (AAL)
- **🌐 Web interface** with interactive visualizations and CSV/XLSX data upload (Streamlit)

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
├── streamlit/                     # 🌐 Web Application (NEW)
│   ├── app.py                     # Main Streamlit interface
│   ├── utils/                     # Application utilities
│   │   ├── data_loader.py         # CSV/XLSX file parsers
│   │   ├── validators.py          # Input data validation
│   │   ├── visualizations.py      # Plotly chart generators
│   │   └── persistence.py         # Save/load functionality
│   ├── requirements.txt           # Web app dependencies
│   └── APP_DOCUMENTATION.md       # Detailed usage guide
├── examples/
│   ├── demonstration.py           # Complete demonstration of all features
│   ├── minimum_example_deterministic.py  # Minimal example for validation
│   └── hand_calculation_deterministic.ipynb  # Step-by-step manual calculations
├── Documentation/
│   ├── API_DOCUMENTATION.md       # Detailed API reference
│   ├── README_IMPLEMENTATION.md   # Implementation details
│   └── MINIMUM_EXAMPLE_README.md  # Guide for minimal examples
└── README.md                      # This file
```

## 🛠️ Installation

### Core Engine Requirements

- Python 3.12+
- TensorFlow 2.10+
- NumPy
- Matplotlib

### Web Application Requirements (Optional)

- Python 3.12+
- Streamlit 1.31+
- Plotly 5.18+
- Pandas 2.0+
- Additional packages (see `streamlit/requirements.txt`)

### Setup

#### Option 1: Core Engine Only

```bash
# Clone the repository
git clone https://github.com/yourusername/Tensor_Risk_Engine.git
cd Tensor_Risk_Engine

# Install dependencies
pip install tensorflow numpy matplotlib

# For Apple Silicon (M1/M2/M3/M4) with Metal acceleration
pip install tensorflow-metal
```

#### Option 2: Web Application

```bash
# Clone the repository
git clone https://github.com/yourusername/Tensor_Risk_Engine.git
cd Tensor_Risk_Engine/streamlit

# Install web app dependencies
pip install -r requirements.txt

# Launch the web interface
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## 🚦 Quick Start

### Option 1: Web Application (Recommended for New Users)

**Launch the browser-based interface:**

```bash
cd streamlit
streamlit run app.py
```

**5-Minute Demo:**
1. Go to **Setup** tab → Select "Generate Synthetic Data"
2. Configure portfolio (N=1000 assets, Q=5000 events)
3. Click "Generate" → Go to **Run Analysis** tab
4. Check "Compute Gradients" → Click "Run Risk Analysis"
5. Explore **Dashboard** and **Gradients** tabs

See [streamlit/APP_DOCUMENTATION.md](streamlit/APP_DOCUMENTATION.md) for complete guide including:
- CSV/XLSX upload formats
- 15+ interactive Plotly visualizations
- Save/load analysis results
- Retrofit optimizer
- Gradient sensitivity analysis

### Option 2: Python API (Programmatic Use)

**Basic Usage:**

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
7. **Interactive Exploration** - Web interface for non-technical stakeholders
8. **Scenario Analysis** - Compare multiple portfolios or hazard scenarios
9. **Budget Allocation** - Use retrofit optimizer to maximize AAL reduction

## 🌐 Web Application

The Streamlit web interface provides a complete browser-based platform for catastrophe risk analysis:

### Key Features

- **📤 Flexible Input**: Upload CSV/XLSX files, generate synthetic data, or load saved analyses
- **✓ Real-time Validation**: Immediate feedback on data quality (shapes, monotonicity, ranges)
- **📊 Interactive Visualizations**: 15+ Plotly charts with hover, zoom, pan
- **🎯 Gradient Analysis**: Full sensitivity analysis for all input parameters
- **💾 Persistence**: Save/load complete analyses as compressed ZIP archives
- **🔧 Retrofit Optimizer**: Budget-constrained mitigation planning using exposure gradients
- **📄 Export**: Results as CSV, summary reports as TXT, complete analysis as ZIP

### Quick Start (Web App)

```bash
cd streamlit
streamlit run app.py
```

**Interface Tabs:**
1. **🏠 Setup** - Choose data source (upload, synthetic, or load saved)
2. **📥 Inputs** - Upload files and validate
3. **⚡ Run Analysis** - Execute computation with optional gradients
4. **📊 Results Dashboard** - Explore portfolio metrics and visualizations
5. **🎯 Gradients** - Sensitivity analysis and retrofit optimization

### Data Format Support

**Supported File Types:**
- CSV (comma-separated values)
- XLSX (Excel spreadsheets)

**Required Inputs:**
- **Assets**: Exposure values and typology indices
- **Vulnerability Curves**: K curves × M intensity points (values in [0, 1])
- **Hazard Matrix**: N assets × Q events (intensity values)
- **Scenario Rates** (optional): Occurrence rates (λ) per event

**Template Downloads:**
- Built-in template generators for all file types
- Example synthetic portfolio with configurable size
- Compatible with OpenQuake and other standard formats

### Visualizations Included

1. **Vulnerability Curves** - MDR vs intensity for each typology
2. **AAL vs Exposure Scatter** - Identify high-risk assets
3. **Exposure/AAL Distribution** - Portfolio composition analysis
4. **Event Loss Distribution** - Scenario loss variability
5. **Vulnerability Gradient Heatmap** - Curve sensitivity (∂AAL/∂C)
6. **Exposure Gradient Chart** - Asset-level retrofit priority (∂AAL/∂v)
7. **Hazard Sensitivity Analysis** - Critical asset-event combinations (∂AAL/∂H)
8. **Event Contribution Plot** - Which scenarios drive AAL (∂AAL/∂λ)
9. **Scenario Loss vs Rate** - Importance sampling distribution
10. **Top Assets Table** - Sortable risk rankings
11. **Retrofit Optimizer Results** - Budget-constrained recommendations

See [streamlit/APP_DOCUMENTATION.md](streamlit/APP_DOCUMENTATION.md) for comprehensive guide including:
- Detailed file format specifications
- Interpretation guide for each visualization
- Advanced features (synthetic data generator, batch processing)
- Troubleshooting common issues
- API reference for custom extensions

## 📚 Documentation

- **[streamlit/APP_DOCUMENTATION.md](streamlit/APP_DOCUMENTATION.md)** - Complete web application user guide (75+ pages)
  - Installation & setup
  - Data input formats and templates
  - Visualization interpretation
  - Save/load functionality
  - Advanced features (retrofit optimizer, gradient analysis)
  - Troubleshooting guide
  - API reference
  
- **[Documentation/API_DOCUMENTATION.md](Documentation/API_DOCUMENTATION.md)** - Core engine API reference
  - TensorialRiskEngine class methods
  - Gradient computation functions
  - Synthetic data generators
  - Performance optimization tips
  
- **[Documentation/README_IMPLEMENTATION.md](Documentation/README_IMPLEMENTATION.md)** - Implementation details and manuscript compliance
  
- **[Documentation/MINIMUM_EXAMPLE_README.md](Documentation/MINIMUM_EXAMPLE_README.md)** - Guide for validation examples

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

```

---

**Note:** This is an initial release. Additional examples, benchmarks, and features will be added in future versions.
