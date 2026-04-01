# Quick Start

## Basic Usage

```python
from tensor_engine import generate_synthetic_portfolio, TensorialRiskEngine

# 1. Generate synthetic portfolio data
v, u, C, x_grid, H, lambdas, Sigma = generate_synthetic_portfolio(
    n_assets=1000,
    n_events=5000,
    n_typologies=5,
    n_curve_points=20,
    lambda_distribution='exponential',
    sigma_fraction=0.3
)

# 2. Initialize the risk engine
engine = TensorialRiskEngine(v, u, C, x_grid, H, lambdas=lambdas, Sigma=Sigma)

# 3. Compute loss matrix and risk metrics
J_matrix, metrics = engine.compute_loss_and_metrics()

print(f"Portfolio AAL: ${metrics['aal_portfolio'].numpy():,.2f}")
print(f"Total Std Dev: ${metrics['std_total_portfolio'].numpy():,.2f}")
print(f"Loss matrix shape: {J_matrix.shape}")
```

## Gradient Analysis

```python
# Full gradient analysis in a single pass
analysis = engine.full_gradient_analysis()

# Vulnerability sensitivity: which curve points matter most?
grad_C = analysis['grad_vulnerability']  # shape (K, M)

# Exposure sensitivity: which assets to retrofit?
grad_v = analysis['grad_exposure']  # shape (N,)

# Hazard sensitivity: which scenarios are critical?
grad_H = analysis['grad_hazard']  # shape (N, Q)

# Scenario rate sensitivity
grad_lambdas = analysis['grad_lambdas']  # shape (Q,)

# Vulnerability uncertainty sensitivity (only when Sigma provided)
grad_sigma = analysis.get('grad_sigma')  # shape (K, M) or None
```

## Individual Gradient Methods

```python
# Compute gradients one at a time (alternative to full_gradient_analysis)
grad_C, metrics = engine.gradient_wrt_vulnerability()
grad_v, metrics = engine.gradient_wrt_exposure()
grad_H, metrics = engine.gradient_wrt_hazard()
grad_lambdas, metrics = engine.gradient_wrt_lambdas()
grad_sigma, metrics = engine.gradient_wrt_sigma()  # requires Sigma
```

## Deterministic Loss (Single Event)

```python
from tensor_engine import deterministic_loss
import tensorflow as tf

v_tf = tf.constant(v, dtype=tf.float32)
u_tf = tf.constant(u, dtype=tf.int32)
C_tf = tf.constant(C, dtype=tf.float32)
x_tf = tf.constant(x_grid, dtype=tf.float32)
h_tf = tf.constant(H[:, 0], dtype=tf.float32)  # First event

loss = deterministic_loss(v_tf, u_tf, C_tf, x_tf, h_tf)
print(f"Scenario loss: ${loss.numpy():,.2f}")
```

## Running Example Scripts

```bash
# Deterministic example with hand-checkable results
python minimum_example_deterministic.py

# Full manuscript example with stochastic formulation
python minimum_example_manuscript.py

# Stochastic example with exponential lambda rates
python minimum_example_manuscript_stochastic_exponential_lambda.py

# Vulnerability uncertainty propagation
python minimum_example_vulnerability_uncertainty.py

# Inverse problem: calibrate curves from observed losses
python minimum_example_inverse_problem.py
```
