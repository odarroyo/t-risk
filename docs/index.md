# T-Risk: Tensorial Risk Engine

A **fully differentiable catastrophe risk assessment framework** leveraging tensor operations and automatic differentiation for GPU-accelerated seismic portfolio risk analysis.

---

## What is T-Risk?

T-Risk reformulates traditional iterative probabilistic catastrophe risk calculations as **tensor programs**, enabling:

- **GPU-accelerated computation** via TensorFlow (including Apple Silicon Metal)
- **Automatic differentiation** for gradient-based sensitivity analysis
- **Massive parallelization** — process entire portfolios in a single vectorized operation

Unlike traditional task-based approaches that iterate over assets and events, T-Risk computes the complete $N \times Q$ loss matrix in parallel, delivering orders of magnitude speedup.

## Key Capabilities

| Capability | Description |
|---|---|
| **Deterministic Loss** | Single-event portfolio loss via differentiable interpolation |
| **Probabilistic Loss** | Full $N \times Q$ loss matrix with rate-weighted risk metrics |
| **Gradient Analysis** | $\partial\text{AAL}/\partial C$, $\partial\text{AAL}/\partial v$, $\partial\text{AAL}/\partial H$, $\partial\text{AAL}/\partial \lambda$, $\partial\text{Var}/\partial \Sigma$ |
| **Vulnerability Uncertainty** | Law of total variance decomposition via $\Sigma$ matrix |
| **Classical Risk** | Hazard-curve convolution for average annual loss |
| **Fragility Analysis** | Damage-state probabilities from fragility curves |
| **Benefit-Cost Ratio** | Retrofit decision-making with present value factors |

## Quick Example

```python
from tensor_engine import generate_synthetic_portfolio, TensorialRiskEngine

# Generate synthetic portfolio
v, u, C, x_grid, H, lambdas, Sigma = generate_synthetic_portfolio(
    n_assets=1000, n_events=5000, sigma_fraction=0.3
)

# Initialize engine and compute
engine = TensorialRiskEngine(v, u, C, x_grid, H, lambdas=lambdas, Sigma=Sigma)
J_matrix, metrics = engine.compute_loss_and_metrics()

print(f"Portfolio AAL: ${metrics['aal_portfolio'].numpy():,.2f}")

# Full gradient analysis in one call
analysis = engine.full_gradient_analysis()
```

## Data Model

The engine uses manuscript-compliant notation:

| Symbol | Shape | Description |
|---|---|---|
| $v$ | $(N,)$ | Exposure vector — replacement cost per asset |
| $u$ | $(N,)$ | Typology index — maps asset to vulnerability curve |
| $C$ | $(K, M)$ | Vulnerability matrix — K curves × M intensity points |
| $x$ | $(M,)$ | Intensity grid (e.g., PGA 0.0–1.5g) |
| $H$ | $(N, Q)$ | Hazard matrix — intensity per asset per event |
| $\lambda$ | $(Q,)$ | Scenario occurrence rates |
| $\Sigma$ | $(K, M)$ | Vulnerability std dev matrix (optional) |
| $F$ | $(K, D, M)$ | Fragility tensor (optional) |
| $J$ | $(N, Q)$ | Loss matrix — computed output |

## Navigation

- **[Getting Started](getting-started/installation.md)** — Installation and quick start
- **[API Reference](api/overview.md)** — Complete function and class documentation
