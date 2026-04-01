# TensorialRiskEngine

::: tensor_engine.TensorialRiskEngine
    options:
      members:
        - __init__
        - compute_loss_and_metrics
        - gradient_wrt_vulnerability
        - gradient_wrt_exposure
        - gradient_wrt_hazard
        - gradient_wrt_lambdas
        - gradient_wrt_sigma
        - full_gradient_analysis

## Usage

### Basic Workflow

```python
from tensor_engine import generate_synthetic_portfolio, TensorialRiskEngine

v, u, C, x_grid, H, lambdas, Sigma = generate_synthetic_portfolio(
    n_assets=1000, n_events=5000,
    lambda_distribution='exponential', sigma_fraction=0.3
)

engine = TensorialRiskEngine(v, u, C, x_grid, H, lambdas=lambdas, Sigma=Sigma)
J_matrix, metrics = engine.compute_loss_and_metrics()
```

### Full Gradient Analysis

```python
analysis = engine.full_gradient_analysis()

# All gradients available:
grad_C = analysis['grad_vulnerability']   # (K, M) — ∂AAL/∂C
grad_v = analysis['grad_exposure']        # (N,)   — ∂AAL/∂v
grad_H = analysis['grad_hazard']          # (N, Q) — ∂AAL/∂H
grad_L = analysis['grad_lambdas']         # (Q,)   — ∂AAL/∂λ
grad_S = analysis.get('grad_sigma')       # (K, M) — ∂Var/∂Σ (when Sigma set)
```

### Using CoV Instead of Sigma

```python
# Constant coefficient of variation per typology
import numpy as np
CoV = np.array([0.3, 0.25, 0.35, 0.2, 0.4], dtype=np.float32)  # per typology
engine = TensorialRiskEngine(v, u, C, x_grid, H, lambdas=lambdas, CoV=CoV)
```

## Attributes

| Attribute | Type | Shape | Description |
|---|---|---|---|
| `v` | tf.Variable | (N,) | Exposure (differentiable) |
| `u` | tf.Constant | (N,) | Typology indices (not differentiable) |
| `C` | tf.Variable | (K, M) | Vulnerability (differentiable) |
| `x_grid` | tf.Constant | (M,) | Intensity grid (not differentiable) |
| `H` | tf.Variable | (N, Q) | Hazard (differentiable) |
| `lambdas` | tf.Variable | (Q,) | Scenario rates (differentiable) |
| `Sigma` | tf.Variable or None | (K, M) | Vulnerability std dev (differentiable) |
| `n_assets` | int | — | Number of assets (N) |
| `n_events` | int | — | Number of events (Q) |
| `n_typologies` | int | — | Number of typologies (K) |
