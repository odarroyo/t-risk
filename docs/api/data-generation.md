# Data Generation

## generate_synthetic_portfolio

::: tensor_engine.generate_synthetic_portfolio

### Example

```python
from tensor_engine import generate_synthetic_portfolio

# Basic usage
v, u, C, x_grid, H, lambdas, Sigma = generate_synthetic_portfolio(
    n_assets=1000, n_events=5000
)

# With vulnerability uncertainty
v, u, C, x_grid, H, lambdas, Sigma = generate_synthetic_portfolio(
    n_assets=1000,
    n_events=5000,
    n_typologies=5,
    n_curve_points=20,
    lambda_distribution='exponential',
    sigma_fraction=0.3
)

print(f"v: {v.shape}, u: {u.shape}, C: {C.shape}")
print(f"H: {H.shape}, lambdas: {lambdas.shape}")
print(f"Sigma: {Sigma.shape if Sigma is not None else None}")
```

### Notes

- Uses fixed random seed (42) for reproducibility
- Vulnerability curves modeled as sigmoid functions with varying steepness and midpoints
- Exponential lambda distribution mimics importance sampling typical in CAT modeling
- Sigma is computed as `sigma_fraction × √(C × (1 − C))`, the maximum Beta std dev
