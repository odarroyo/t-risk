# Tensorial Risk Engine - API Documentation

**Version:** 1.1  
**Date:** March 31, 2026  
**Author:** Based on manuscript formulation for differentiable catastrophe risk assessment

---

## Table of Contents

1. [Overview](#overview)
2. [Data Generation](#data-generation)
3. [Core Functions](#core-functions)
4. [TensorialRiskEngine Class](#tensorialriskengine-class)
5. [Classical Risk Functions](#classical-risk-functions)
6. [Fragility & Damage State Functions](#fragility--damage-state-functions)
7. [Benefit-Cost Ratio](#benefit-cost-ratio)
8. [Usage Examples](#usage-examples)
9. [Mathematical Background](#mathematical-background)

---

## Overview

The Tensorial Risk Engine implements a fully differentiable catastrophe risk assessment framework based on tensor operations. Unlike traditional task-based approaches, this engine:

- **Vectorizes** all operations for GPU acceleration
- **Computes gradients** w.r.t. all input parameters automatically
- **Scales efficiently** to millions of loss calculations
- **Enables optimization** through automatic differentiation

### Key Features

✅ Multi-typology vulnerability support  
✅ Deterministic and probabilistic hazard formulations  
✅ Per-asset and portfolio-level metrics  
✅ Complete gradient computation (∂J/∂v, ∂J/∂C, ∂J/∂H, ∂J/∂λ, ∂Var/∂Σ)  
✅ GPU-accelerated via TensorFlow Metal  
✅ Vulnerability uncertainty propagation (Sigma matrix)  
✅ Classical risk via hazard-curve convolution  
✅ Fragility curves and damage-state distributions  
✅ Benefit-cost ratio for retrofit analysis  
✅ Manuscript-compliant implementation (Sections 2-5)

---

## Data Generation

### `generate_synthetic_portfolio()`

Generate synthetic portfolio data for testing and demonstration.

```python
def generate_synthetic_portfolio(
    n_assets: int,
    n_events: int,
    n_typologies: int = 5,
    n_curve_points: int = 20,
    lambdas: Optional[np.ndarray] = None,
    lambda_distribution: str = 'exponential',
    sigma_fraction: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_assets` | int | - | Number of assets in portfolio (N) |
| `n_events` | int | - | Number of stochastic event realizations (Q) |
| `n_typologies` | int | 5 | Number of building typologies/curves (K) |
| `n_curve_points` | int | 20 | Discretization points for curves (M) |
| `lambdas` | np.ndarray | None | Pre-specified scenario occurrence rates ∈ ℝ^Q. If None, will be generated |
| `lambda_distribution` | str | 'exponential' | Distribution for generated lambdas: 'uniform' or 'exponential' |
| `sigma_fraction` | float | None | Fraction of maximum Beta std dev: Sigma = sigma_fraction × √(C × (1-C)). Typical values: 0.2–0.5. If None, no Sigma is generated |

#### Returns

Returns a tuple of 7 values (NumPy arrays or None):

1. **v** `(N,) float32` - Exposure vector ∈ ℝ^N
   Replacement costs ranging from $100,000 to $1,000,000

2. **u** `(N,) int32` - Typology index vector ∈ ℤ^N
   Maps each asset to a vulnerability curve, values in {0, ..., K-1}

3. **C** `(K, M) float32` - Vulnerability matrix ∈ ℝ^(K×M)
   K vulnerability curves with M points each, values in [0, 1]

4. **x_grid** `(M,) float32` - Intensity grid vector ∈ ℝ^M
   Common intensity axis for all curves, ranging from 0.0g to 1.5g

5. **H** `(N, Q) float32` - Hazard intensity matrix ∈ ℝ^(N×Q)
   Ground motion intensities for each asset in each event

6. **lambdas** `(Q,) float32` - Scenario occurrence rate vector ∈ ℝ^Q
   λ_q = annual occurrence rate of event q (events/year)

7. **Sigma** `(K, M) float32` or `None` - Vulnerability std dev matrix ∈ ℝ^(K×M)
   Standard deviation of MDR at each curve point. None if sigma_fraction not specified

#### Example

```python
# Generate portfolio with 1000 assets, 5000 events, and vulnerability uncertainty
v, u, C, x_grid, H, lambdas, Sigma = generate_synthetic_portfolio(
    n_assets=1000,
    n_events=5000,
    n_typologies=5,
    n_curve_points=20,
    lambda_distribution='exponential',
    sigma_fraction=0.3
)

print(f"Portfolio has {len(v)} assets")
print(f"Using {C.shape[0]} vulnerability curves")
print(f"Analyzing {H.shape[1]} stochastic events")
print(f"Total occurrence rate: {lambdas.sum():.4f} events/year")
print(f"Sigma provided: {Sigma is not None}")
```

#### Notes

- Uses sigmoid functions to generate realistic vulnerability curves
- Different typologies have varying fragility (steepness) and thresholds
- Hazard intensities are uniformly distributed for demonstration purposes
- **Exponential distribution** for lambdas mimics importance sampling typical in CAT modeling
  - Higher rates for more frequent (lower intensity) events
  - Lower rates for rare (higher intensity) events
  - Normalized so sum equals 1.0 (interpretable as probabilities)
- **Uniform distribution** for lambdas sets λ_q = 1/Q for all events
- Fixed random seed (42) ensures reproducible results

- Uses fixed random seed (42) for reproducibility
- Vulnerability curves modeled as sigmoid functions
- Different typologies have varying fragility and damage thresholds
- Hazard intensities uniformly distributed (for demonstration only)

---

## Core Functions

### `deterministic_loss()`

Compute total loss for a single hazard scenario using differentiable interpolation.

```python
@tf.function
def deterministic_loss(
    v: tf.Tensor,
    u: tf.Tensor,
    C: tf.Tensor,
    x_grid: tf.Tensor,
    h: tf.Tensor
) -> tf.Tensor
```

#### Mathematical Formulation

Implements Manuscript Section 2, Equations 1-3:

1. **Interpolation weight:** α = (h - x_j) / (x_{j+1} - x_j)
2. **Mean Damage Ratio:** MDR_i = (1-α)·C[u_i,j] + α·C[u_i,j+1]
3. **Total loss:** J = Σ_i v_i · MDR_i

#### Parameters

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `v` | (N,) | float32 | Exposure vector |
| `u` | (N,) | int32 | Typology indices (0 to K-1) |
| `C` | (K, M) | float32 | Vulnerability matrix |
| `x_grid` | (M,) | float32 | Intensity grid (monotonic) |
| `h` | (N,) | float32 | Intensity vector for scenario |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `J` | scalar float32 | Total portfolio loss for this scenario |

#### Example

```python
# Create tensors
v_tf = tf.constant(v, dtype=tf.float32)
u_tf = tf.constant(u, dtype=tf.int32)
C_tf = tf.constant(C, dtype=tf.float32)
x_tf = tf.constant(x_grid, dtype=tf.float32)
h_tf = tf.constant(H[:, 0], dtype=tf.float32)  # First event

# Compute loss
loss = deterministic_loss(v_tf, u_tf, C_tf, x_tf, h_tf)
print(f"Scenario loss: ${loss.numpy():,.2f}")
```

#### Implementation Details

- **Grid lookup:** Uses `tf.searchsorted()` for efficient O(log M) search
- **Numerical stability:** Adds epsilon (1e-8) to prevent division by zero
- **GPU compatibility:** Employs flat indexing instead of `gather_nd`
- **Differentiability:** Fully differentiable w.r.t. v, C, and h

---

### `probabilistic_loss_matrix()`

Compute complete loss matrix for all assets and all stochastic events.

```python
@tf.function
def probabilistic_loss_matrix(
    v: tf.Tensor,
    u: tf.Tensor,
    C: tf.Tensor,
    x_grid: tf.Tensor,
    H: tf.Tensor
) -> tf.Tensor
```

#### Mathematical Formulation

Implements Manuscript Section 3, Equation 4:

**J[i,q] = v[i] × MDR[i,q]**

where MDR[i,q] is computed via interpolation for each asset-event pair.

#### Parameters

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `v` | (N,) | float32 | Exposure vector |
| `u` | (N,) | int32 | Typology indices |
| `C` | (K, M) | float32 | Vulnerability matrix |
| `x_grid` | (M,) | float32 | Intensity grid |
| `H` | (N, Q) | float32 | Hazard intensity matrix |

#### Returns

| Return | Shape | Type | Description |
|--------|-------|------|-------------|
| `J_matrix` | (N, Q) | float32 | Loss matrix ∈ ℝ^(N×Q) |

#### Example

```python
# Convert to tensors
v_tf = tf.constant(v, dtype=tf.float32)
u_tf = tf.constant(u, dtype=tf.int32)
C_tf = tf.constant(C, dtype=tf.float32)
x_tf = tf.constant(x_grid, dtype=tf.float32)
H_tf = tf.constant(H, dtype=tf.float32)

# Compute loss matrix
J = probabilistic_loss_matrix(v_tf, u_tf, C_tf, x_tf, H_tf)
print(f"Loss matrix shape: {J.shape}")
print(f"Total computations: {J.shape[0] * J.shape[1]:,}")
```

#### Memory Considerations

For large portfolios, the loss matrix can be substantial:

- **N=10,000, Q=100,000:** 1 billion elements (~4GB in float32)
- Consider event chunking for memory-constrained environments
- GPU memory is typically the limiting factor

#### Computational Strategy

1. Flatten H to vector (N×Q elements)
2. Vectorized grid lookups for all intensities
3. Tile typology indices Q times
4. Gather vulnerability values via flat indexing
5. Interpolate MDR for all N×Q combinations
6. Broadcast multiply with exposure
7. Reshape to (N, Q) matrix

---

### `compute_risk_metrics()`

Derive comprehensive risk metrics from the loss matrix.

```python
@tf.function
def compute_risk_metrics(
    J_matrix: tf.Tensor,
    lambdas: Optional[tf.Tensor] = None
) -> Dict[str, tf.Tensor]
```

#### Mathematical Formulation

Implements Manuscript Section 3c with rate-weighted formulation:

- **Rate-weighted AAL per asset:** AAL_i = Σ_q λ_q × J[i,q]
- **Mean per event:** μ_i = AAL_i / Λ, where Λ = Σ_q λ_q
- **Rate-weighted variance:** σ²_i = Σ_q w_q × (J[i,q] - μ_i)², where w_q = λ_q / Λ

#### Parameters

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `J_matrix` | (N, Q) | float32 | Loss matrix |
| `lambdas` | (Q,) | float32 | Scenario occurrence rates (optional, defaults to uniform 1/Q) |

#### Returns

Dictionary containing:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `aal_per_asset` | (N,) | float32 | Rate-weighted Average Annual Loss per asset |
| `aal_portfolio` | scalar | float32 | Total portfolio AAL |
| `mean_per_event_per_asset` | (N,) | float32 | Mean loss per event occurrence per asset |
| `variance_per_asset` | (N,) | float32 | Rate-weighted loss variance per asset |
| `std_per_asset` | (N,) | float32 | Loss standard deviation per asset |
| `loss_per_event` | (Q,) | float32 | Total loss per event |
| `total_rate` | scalar | float32 | Total occurrence rate Λ = Σ_q λ_q |

#### Example

```python
# Compute metrics with scenario rates
lambdas = tf.constant([0.5, 0.3, 0.2], dtype=tf.float32)  # Non-uniform rates
metrics = compute_risk_metrics(J_matrix, lambdas)

# Access results
print(f"Portfolio AAL: ${metrics['aal_portfolio'].numpy():,.2f}")
print(f"Total rate: {metrics['total_rate'].numpy():.4f} events/year")
print(f"Riskiest asset AAL: ${metrics['aal_per_asset'].numpy().max():,.2f}")
print(f"Average volatility: ${metrics['std_per_asset'].numpy().mean():,.2f}")

# Find top 10 risky assets
top_10 = tf.argsort(metrics['aal_per_asset'], direction='DESCENDING')[:10]
print(f"Top 10 assets: {top_10.numpy()}")
```

#### Applications

- **Portfolio AAL:** Capital reserve requirements
- **Per-asset AAL:** Asset ranking and prioritization  
- **Variance/Std:** Uncertainty quantification, VaR calculations
- **Loss per event:** Event importance analysis, scenario planning

---

### `compute_risk_metrics_with_uncertainty()`

Extend risk metrics with vulnerability uncertainty decomposition using the law of total variance.

```python
@tf.function
def compute_risk_metrics_with_uncertainty(
    J_matrix: tf.Tensor,
    lambdas: tf.Tensor,
    v: tf.Tensor,
    sigma_interpolated: tf.Tensor
) -> Dict[str, tf.Tensor]
```

#### Mathematical Formulation

Implements variance decomposition via the law of total variance:

- **Aleatory variance:** Var_aleatory_i = Σ_q w_q × (J[i,q] - mu_i)²
- **Vulnerability variance:** Var_vulnerability_i = Σ_q w_q × v_i² × sigma_interpolated[i,q]²
- **Total variance:** Var_total_i = Var_aleatory_i + Var_vulnerability_i

#### Parameters

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `J_matrix` | (N, Q) | float32 | Loss matrix |
| `lambdas` | (Q,) | float32 | Scenario occurrence rates |
| `v` | (N,) | float32 | Exposure vector |
| `sigma_interpolated` | (N, Q) | float32 | Interpolated vulnerability std dev at each asset-event |

#### Returns

Dictionary containing all keys from `compute_risk_metrics()` plus:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `variance_vulnerability_per_asset` | (N,) | float32 | Vulnerability variance component per asset |
| `variance_vulnerability_portfolio` | scalar | float32 | Portfolio-level vulnerability variance |
| `variance_total_per_asset` | (N,) | float32 | Total variance (aleatory + vulnerability) per asset |
| `std_total_per_asset` | (N,) | float32 | Total standard deviation per asset |
| `variance_total_portfolio` | scalar | float32 | Total portfolio variance |
| `std_total_portfolio` | scalar | float32 | Total portfolio standard deviation |

#### Example

```python
# Typically called internally by TensorialRiskEngine, but can be used directly:
metrics = compute_risk_metrics_with_uncertainty(J_matrix, lambdas_tf, v_tf, sigma_interp)

print(f"Aleatory variance: {metrics['variance_per_asset'].numpy().sum():,.2f}")
print(f"Vulnerability variance: {metrics['variance_vulnerability_portfolio'].numpy():,.2f}")
print(f"Total variance: {metrics['variance_total_portfolio'].numpy():,.2f}")
print(f"Total std dev: {metrics['std_total_portfolio'].numpy():,.2f}")
```

---

## TensorialRiskEngine Class

Main class encapsulating the complete risk engine with gradient computation.

### Constructor

```python
class TensorialRiskEngine:
    def __init__(
        self,
        v: np.ndarray,
        u: np.ndarray,
        C: np.ndarray,
        x_grid: np.ndarray,
        H: np.ndarray,
        lambdas: Optional[np.ndarray] = None,
        Sigma: Optional[np.ndarray] = None,
        CoV: Optional[float] = None
    )
```

#### Parameters

All parameters are NumPy arrays with the same specifications as `generate_synthetic_portfolio()` returns.

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `v` | (N,) | float32 | Exposure vector (replacement costs) |
| `u` | (N,) | int32 | Typology index vector |
| `C` | (K, M) | float32 | Vulnerability matrix |
| `x_grid` | (M,) | float32 | Intensity grid vector |
| `H` | (N, Q) | float32 | Hazard intensity matrix |
| `lambdas` | (Q,) | float32 | Scenario occurrence rates (optional, defaults to uniform 1/Q) |
| `Sigma` | (K, M) | float32 | Vulnerability std dev matrix (optional). Mutually exclusive with CoV |
| `CoV` | (K,) or (K, M) | float32 | Coefficient of variation array (optional). If shape (K,): scalar CoV per typology, Sigma = CoV[k] × C[k,:]. If shape (K, M): per-point CoV, Sigma = CoV × C. Mutually exclusive with Sigma |

#### Attributes

After initialization, the engine contains:

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `v` | tf.Variable | (N,) | Exposure (differentiable) |
| `u` | tf.Constant | (N,) | Typology indices (not differentiable) |
| `C` | tf.Variable | (K, M) | Vulnerability (differentiable) |
| `x_grid` | tf.Constant | (M,) | Intensity grid (not differentiable) |
| `H` | tf.Variable | (N, Q) | Hazard (differentiable) |
| `lambdas` | tf.Variable | (Q,) | Scenario occurrence rates (differentiable) |
| `Sigma` | tf.Variable or None | (K, M) | Vulnerability std dev matrix (differentiable), None if not provided |
| `n_assets` | int | - | Number of assets |
| `n_events` | int | - | Number of events |
| `n_typologies` | int | - | Number of typologies |

---

### Method: `compute_loss_and_metrics()`

Compute loss matrix and all risk metrics.

```python
def compute_loss_and_metrics(self) -> Tuple[tf.Tensor, Dict]
```

#### Returns

- **J_matrix:** `(N, Q)` tensor - Complete loss matrix
- **metrics:** dict - All risk metrics (see `compute_risk_metrics`)

#### Example

```python
engine = TensorialRiskEngine(v, u, C, x_grid, H, lambdas)
J_matrix, metrics = engine.compute_loss_and_metrics()

print(f"Computed {J_matrix.shape[0] * J_matrix.shape[1]:,} loss values")
print(f"Portfolio AAL: ${metrics['aal_portfolio'].numpy():,.2f}")
print(f"Total rate: {metrics['total_rate'].numpy():.4f} events/year")
```

---

### Method: `gradient_wrt_vulnerability()`

Compute gradient ∂(AAL)/∂C for vulnerability curve sensitivity analysis.

```python
def gradient_wrt_vulnerability(self) -> Tuple[tf.Tensor, Dict]
```

#### Returns

- **grad_C:** `(K, M)` tensor - Vulnerability gradient
- **metrics:** dict - Current risk metrics

#### Interpretation

- **grad_C[k,m]:** Change in AAL per unit change in curve point C[k,m]
- **Positive values:** Increasing curve at this point increases AAL
- **Magnitude:** Indicates sensitivity to curve calibration

#### Example

```python
grad_C, metrics = engine.gradient_wrt_vulnerability()

# Find most sensitive curve point
k, m = np.unravel_index(grad_C.numpy().argmax(), grad_C.shape)
print(f"Most sensitive: Typology {k}, Grid point {m}")
print(f"Gradient: {grad_C[k,m].numpy():.2e}")

# Identify most impactful typology
impact_per_type = tf.reduce_sum(tf.abs(grad_C), axis=1)
most_impactful = tf.argmax(impact_per_type).numpy()
print(f"Most impactful typology: {most_impactful}")
```

#### Applications

- Vulnerability curve calibration to loss data
- Identify critical damage thresholds
- Quantify impact of modeling uncertainty
- Guide curve refinement priorities

---

### Method: `gradient_wrt_exposure()`

Compute gradient ∂(AAL)/∂v for exposure sensitivity analysis.

```python
def gradient_wrt_exposure(self) -> Tuple[tf.Tensor, Dict]
```

#### Returns

- **grad_v:** `(N,)` tensor - Exposure gradient
- **metrics:** dict - Current risk metrics

#### Interpretation

- **grad_v[i]:** AAL increase per $1 of additional exposure at asset i
- **Units:** Dimensionless ($/$ of exposure)
- **grad_v = 0.5:** $1 more exposure → $0.50 more AAL

#### Example

```python
grad_v, metrics = engine.gradient_wrt_exposure()

# Find top 10 risk contributors
top_10 = tf.argsort(grad_v, direction='DESCENDING')[:10].numpy()

print("Top 10 assets for retrofit:")
for rank, idx in enumerate(top_10, 1):
    print(f"{rank}. Asset {idx}: ∂AAL/∂v = {grad_v[idx].numpy():.4f}")
    print(f"   Current exposure: ${v[idx]:,.0f}")
    print(f"   Typology: {u[idx]}")
```

#### Applications

- **Retrofit prioritization:** Target high-gradient assets
- **Portfolio optimization:** Minimize AAL subject to budget
- **Risk-based pricing:** Differential premiums by location
- **Capital allocation:** Allocate reserves proportional to gradient

---

### Method: `gradient_wrt_hazard()`

Compute gradient ∂(AAL)/∂H for hazard sensitivity analysis.

```python
def gradient_wrt_hazard(self) -> Tuple[tf.Tensor, Dict]
```

#### Returns

- **grad_H:** `(N, Q)` tensor - Hazard gradient
- **metrics:** dict - Current risk metrics

#### Interpretation

- **grad_H[i,q]:** Change in AAL per g of intensity change at asset i in event q
- **Units:** $/g (dollars per unit ground motion)
- **Large values:** Critical asset-event combinations

#### Example

```python
grad_H, metrics = engine.gradient_wrt_hazard()

# Overall hazard sensitivity
avg_sensitivity = tf.reduce_mean(tf.abs(grad_H)).numpy()
print(f"Average hazard sensitivity: ${avg_sensitivity:.2e}/g")

# Find most critical event
event_sensitivity = tf.reduce_sum(tf.abs(grad_H), axis=0)
critical_event = tf.argmax(event_sensitivity).numpy()
print(f"Most critical event: {critical_event}")

# Find most hazard-sensitive asset
asset_sensitivity = tf.reduce_sum(tf.abs(grad_H), axis=1)
sensitive_asset = tf.argmax(asset_sensitivity).numpy()
print(f"Most hazard-sensitive asset: {sensitive_asset}")
```

#### Applications

- Assess impact of hazard model uncertainty
- Identify dominant scenarios
- Guide site-specific hazard studies
- Quantify epistemic uncertainty contribution

#### Memory Warning

For large portfolios (N×Q matrix can be gigabytes), consider:

```python
# Compute aggregate statistics without storing full gradient
with tf.GradientTape() as tape:
    J_matrix, metrics = engine.compute_loss_and_metrics()
    aal = metrics['aal_portfolio']

# Per-event sensitivity
event_aal = tf.reduce_sum(J_matrix, axis=0)
grad_per_event = tape.gradient(aal, event_aal)

# This avoids materializing the full (N, Q) gradient
```

---

### Method: `gradient_wrt_lambdas()`

Compute gradient ∂(AAL)/∂λ for scenario occurrence rate sensitivity analysis.

```python
def gradient_wrt_lambdas(self) -> Tuple[tf.Tensor, Dict]
```

#### Returns

- **grad_lambdas:** `(Q,)` tensor - Scenario rate gradient
- **metrics:** dict - Current risk metrics

#### Interpretation

- **grad_lambdas[q]:** Change in AAL per unit change in occurrence rate λ_q
- **Units:** $ (dollars, total portfolio loss for event q)
- **Large values:** Scenarios that contribute most to portfolio AAL
- **Always positive:** Increasing any event rate increases AAL

#### Example

```python
grad_lambdas, metrics = engine.gradient_wrt_lambdas()

# Find most critical scenarios
critical_events = tf.argsort(grad_lambdas, direction='DESCENDING')[:10].numpy()

print("Top 10 most critical scenarios:")
for rank, event_idx in enumerate(critical_events, 1):
    print(f"{rank}. Event {event_idx}:")
    print(f"   ∂AAL/∂λ = ${grad_lambdas[event_idx].numpy():,.2f}")
    print(f"   Current rate: {lambdas[event_idx]:.6f} events/year")
    print(f"   Contribution to AAL: ${(grad_lambdas[event_idx] * lambdas[event_idx]).numpy():,.2f}")

# Analyze scenario importance distribution
total_gradient = tf.reduce_sum(grad_lambdas).numpy()
print(f"\nTotal gradient magnitude: ${total_gradient:,.2f}")
print(f"Average per scenario: ${total_gradient / len(grad_lambdas):,.2f}")
```

#### Applications

- **Event catalog analysis:** Identify which scenarios drive portfolio risk
- **Importance sampling:** Optimize event set weights for Monte Carlo
- **Catalog completeness:** Assess sensitivity to missing rare events
- **Event set optimization:** Guide catalog refinement and pruning decisions
- **Return period analysis:** Understand contribution by frequency band

#### Mathematical Note

The gradient is straightforward: ∂(∑_q λ_q × L_q)/∂λ_q = L_q, where L_q is the total portfolio loss in event q. However, this gradient provides crucial insight into which scenarios dominate the risk profile.

---

### Method: `gradient_wrt_sigma()`

Compute gradient ∂(Var_total)/∂Sigma for vulnerability uncertainty sensitivity analysis.

```python
def gradient_wrt_sigma(self) -> Tuple[tf.Tensor, Dict]
```

#### Returns

- **grad_sigma:** `(K, M)` tensor - Vulnerability std dev gradient
- **metrics:** dict - Current risk metrics including variance decomposition

#### Interpretation

- **grad_sigma[k,m]:** Change in total portfolio variance per unit change in Sigma[k,m]
- **Always non-negative:** Increasing uncertainty always increases variance
- **Targets Var_total, not AAL:** Since AAL is independent of Sigma (E[MDR] = C regardless of uncertainty), ∂(AAL)/∂Sigma = 0. The meaningful sensitivity is ∂(Var_total)/∂Sigma
- **Magnitude:** Indicates which curve points contribute most to portfolio uncertainty

#### Example

```python
# Requires Sigma to be set in the engine
engine = TensorialRiskEngine(v, u, C, x_grid, H, lambdas, Sigma=Sigma)

grad_sigma, metrics = engine.gradient_wrt_sigma()

# Find most uncertainty-sensitive curve point
k, m = np.unravel_index(grad_sigma.numpy().argmax(), grad_sigma.shape)
print(f"Most sensitive: Typology {k}, Grid point {m}")
print(f"∂(Var_total)/∂Sigma[{k},{m}] = {grad_sigma[k,m].numpy():.2e}")

# Uncertainty contribution by typology
uncertainty_per_type = tf.reduce_sum(tf.abs(grad_sigma), axis=1)
for k in range(uncertainty_per_type.shape[0]):
    print(f"Typology {k}: {uncertainty_per_type[k].numpy():.2e}")
```

#### Applications

- **Uncertainty prioritization:** Identify which vulnerability curves need better calibration
- **Research allocation:** Focus experimental efforts on high-sensitivity curve points
- **Model refinement:** Quantify impact of reducing epistemic uncertainty
- **Variance budgeting:** Understand contribution of each typology to total portfolio uncertainty

---

### Method: `full_gradient_analysis()`

Compute complete gradient ∇J = [∂J/∂H, ∂J/∂C, ∂J/∂v, ∂J/∂λ, ∂Var/∂Sigma] in single pass.

```python
def full_gradient_analysis(self) -> Dict
```

#### Returns

Dictionary containing:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `grad_hazard` | (N, Q) | float32 | ∂(AAL)/∂H |
| `grad_vulnerability` | (K, M) | float32 | ∂(AAL)/∂C |
| `grad_exposure` | (N,) | float32 | ∂(AAL)/∂v |
| `grad_lambdas` | (Q,) | float32 | ∂(AAL)/∂λ |
| `grad_sigma` | (K, M) | float32 | ∂(Var_total)/∂Sigma (only present when Sigma is set) |
| `metrics` | dict | - | All risk metrics |
| `loss_matrix` | (N, Q) | float32 | Complete loss matrix |

#### Example

```python
# Complete analysis
analysis = engine.full_gradient_analysis()

# Extract gradients
grad_H = analysis['grad_hazard']
grad_C = analysis['grad_vulnerability']
grad_v = analysis['grad_exposure']
grad_lambdas = analysis['grad_lambdas']
grad_sigma = analysis.get('grad_sigma')  # Present only when Sigma is set

# Analyze each component
print("="*60)
print("COMPLETE SENSITIVITY ANALYSIS")
print("="*60)

# Vulnerability sensitivity
vuln_sens = tf.reduce_sum(tf.abs(grad_C), axis=1).numpy()
print(f"\nVulnerability sensitivity by typology:")
for k in range(len(vuln_sens)):
    print(f"  Typology {k}: {vuln_sens[k]:.2e}")

# Exposure sensitivity
exp_top5 = tf.argsort(grad_v, direction='DESCENDING')[:5].numpy()
print(f"\nTop 5 assets for retrofit:")
for i in exp_top5:
    print(f"  Asset {i}: ∂AAL/∂v = {grad_v[i].numpy():.4f}")

# Hazard sensitivity
hazard_sens = tf.reduce_mean(tf.abs(grad_H)).numpy()
print(f"\nAverage hazard sensitivity: ${hazard_sens:.2e}/g")

# Scenario rate sensitivity
top_scenarios = tf.argsort(grad_lambdas, direction='DESCENDING')[:3].numpy()
print(f"\nTop 3 critical scenarios:")
for event_idx in top_scenarios:
    print(f"  Event {event_idx}: ∂AAL/∂λ = ${grad_lambdas[event_idx].numpy():,.2f}")

# Sigma sensitivity (only when Sigma is set)
if grad_sigma is not None:
    print(f"\nSigma gradient shape: {grad_sigma.shape}")
    print(f"Max ∂Var/∂Σ: {tf.reduce_max(grad_sigma).numpy():.2e}")

# Portfolio metrics
print(f"\nPortfolio AAL: ${analysis['metrics']['aal_portfolio'].numpy():,.2f}")
```

#### Efficiency Note

This method computes all AAL gradients (∂AAL/∂H, ∂AAL/∂C, ∂AAL/∂v, ∂AAL/∂λ) in a single `tape.gradient(aal, [list])` call, which is more efficient than calling the gradient methods separately because:
- Single forward pass through the computation graph
- Single backward pass for all four gradients (the "cheap gradient principle")
- Sigma gradient uses a separate tape targeting variance (since ∂AAL/∂Σ = 0 by design)

---

## Classical Risk Functions

### `_interpolate_matrix()`

Shared helper that implements flat-indexing linear interpolation for any (K, M) matrix at hazard intensities.

```python
@tf.function
def _interpolate_matrix(
    u: tf.Tensor,
    Matrix: tf.Tensor,
    x_grid: tf.Tensor,
    H: tf.Tensor
) -> tf.Tensor
```

#### Parameters

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `u` | (N,) | int32 | Typology index per asset, values in {0, ..., K-1} |
| `Matrix` | (K, M) | float32 | Any per-typology matrix to interpolate (e.g., C or Sigma) |
| `x_grid` | (M,) | float32 | Intensity grid vector |
| `H` | (N, Q) | float32 | Hazard intensity matrix |

#### Returns

| Return | Shape | Type | Description |
|--------|-------|------|-------------|
| `result` | (N, Q) | float32 | Interpolated values for each asset-event pair |

#### Notes

- Fully differentiable w.r.t. `Matrix` and `H`
- Uses flat indexing for Metal GPU compatibility
- Used internally by `probabilistic_loss_matrix` and uncertainty propagation

---

### `classical_loss()`

Compute average annual loss per asset from hazard curves via classical risk convolution (differentiable).

```python
@tf.function
def classical_loss(
    v: tf.Tensor,
    u: tf.Tensor,
    C: tf.Tensor,
    x_grid: tf.Tensor,
    hazard_poes: tf.Tensor,
    hazard_imls: tf.Tensor
) -> tf.Tensor
```

#### Mathematical Formulation

For each asset *i*, integrates the vulnerability mean-damage-ratio curve against the hazard curve using the trapezoidal rule:

```
AAL_i = v_i × Σ_l MDR(iml_l) × (PoE_{l-1} − PoE_l)
```

where PoE values decrease with increasing IML (exceedance probabilities), so (PoE_{l-1} − PoE_l) represents the probability mass in each IML bin.

#### Parameters

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `v` | (N,) | float32 | Exposure values per asset |
| `u` | (N,) | int32 | Typology index per asset, values in {0, ..., K-1} |
| `C` | (K, M) | float32 | Vulnerability matrix (K curves × M curve points) |
| `x_grid` | (M,) | float32 | Intensity grid for vulnerability curves |
| `hazard_poes` | (N, L) | float32 | Hazard PoE matrix — hazard_poes[i,l] = P(IML > imls[l]) at asset i. Monotonically decreasing along axis 1 |
| `hazard_imls` | (L,) | float32 | IML levels corresponding to the hazard curve columns |

#### Returns

| Return | Shape | Type | Description |
|--------|-------|------|-------------|
| `avg_loss` | (N,) | float32 | Average annual loss per asset |

#### Example

```python
# Hazard curves: 5 IML levels, 2 assets
hazard_imls = tf.constant([0.1, 0.3, 0.5, 0.7, 1.0], dtype=tf.float32)
hazard_poes = tf.constant([
    [0.8, 0.5, 0.2, 0.05, 0.01],  # Asset 0
    [0.6, 0.3, 0.1, 0.02, 0.005]  # Asset 1
], dtype=tf.float32)

aal = classical_loss(v_tf, u_tf, C_tf, x_tf, hazard_poes, hazard_imls)
print(f"AAL per asset: {aal.numpy()}")
```

#### Notes

- Fully differentiable w.r.t. `v`, `C`, `hazard_poes`
- Uses the same linear interpolation scheme as `probabilistic_loss_matrix`

---

## Fragility & Damage State Functions

### `fragility_damage_distribution()`

Compute damage-state probability distributions from fragility curves (differentiable).

```python
@tf.function
def fragility_damage_distribution(
    u: tf.Tensor,
    F: tf.Tensor,
    x_grid: tf.Tensor,
    H: tf.Tensor
) -> tf.Tensor
```

#### Mathematical Formulation

For each asset *i* and event *q*, interpolates D fragility exceedance curves at intensity H[i,q], then converts to damage-state probabilities via differencing:

```
P(ds_0) = 1 − P(exceed ds_1)
P(ds_d) = P(exceed ds_d) − P(exceed ds_{d+1})   for d = 1..D-1
P(ds_D) = P(exceed ds_D)
```

#### Parameters

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `u` | (N,) | int32 | Typology index per asset, values in {0, ..., K-1} |
| `F` | (K, D, M) | float32 | Fragility tensor — F[k,d,m] = P(exceeding limit-state d \| IML = x_grid[m]) for typology k |
| `x_grid` | (M,) | float32 | Intensity grid for fragility curves |
| `H` | (N, Q) | float32 | Hazard intensity matrix |

#### Returns

| Return | Shape | Type | Description |
|--------|-------|------|-------------|
| `damage_probs` | (N, Q, D+1) | float32 | Probability of each damage state per asset per event. Index 0 = no damage, index D = complete damage |

#### Example

```python
# 2 typologies, 3 limit states, 5 IML points
F = tf.constant([
    [[0.0, 0.1, 0.5, 0.9, 1.0],   # Typ 0, slight damage
     [0.0, 0.05, 0.3, 0.7, 0.95],  # Typ 0, moderate damage
     [0.0, 0.01, 0.1, 0.4, 0.8]],  # Typ 0, complete damage
    [[0.0, 0.2, 0.6, 0.95, 1.0],
     [0.0, 0.1, 0.4, 0.8, 0.98],
     [0.0, 0.02, 0.15, 0.5, 0.85]]
], dtype=tf.float32)  # shape (2, 3, 5)

damage_probs = fragility_damage_distribution(u_tf, F, x_tf, H_tf)
print(f"Damage probs shape: {damage_probs.shape}")  # (N, Q, 4) — 3 limit states → 4 damage states
```

#### Notes

- Fully differentiable w.r.t. `F` and `H`
- Uses vectorised flat indexing across all D limit states (no Python loops)

---

### `consequence_loss()`

Compute losses from damage-state probabilities and consequence ratios (differentiable).

```python
@tf.function
def consequence_loss(
    damage_probs: tf.Tensor,
    consequence_ratios: tf.Tensor,
    v: tf.Tensor,
    u: tf.Tensor
) -> tf.Tensor
```

#### Mathematical Formulation

For each asset *i* and event *q*:

```
loss[i,q] = v[i] × Σ_d damage_probs[i,q,d] × consequence_ratios[u[i],d]
```

#### Parameters

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `damage_probs` | (N, Q, D+1) | float32 | Damage-state probabilities (from `fragility_damage_distribution`) |
| `consequence_ratios` | (K, D+1) | float32 | Loss ratio per damage state per typology. Index 0 = no damage (typically 0), index D = complete (typically 1) |
| `v` | (N,) | float32 | Exposure values per asset |
| `u` | (N,) | int32 | Typology index per asset |

#### Returns

| Return | Shape | Type | Description |
|--------|-------|------|-------------|
| `loss_matrix` | (N, Q) | float32 | Loss per asset per event |

#### Example

```python
# Consequence ratios: 2 typologies, 4 damage states (none, slight, moderate, complete)
consequence_ratios = tf.constant([
    [0.0, 0.05, 0.25, 1.0],  # Typology 0
    [0.0, 0.08, 0.35, 1.0],  # Typology 1
], dtype=tf.float32)

loss_matrix = consequence_loss(damage_probs, consequence_ratios, v_tf, u_tf)
print(f"Loss matrix shape: {loss_matrix.shape}")
```

#### Notes

- Fully differentiable w.r.t. `damage_probs`, `consequence_ratios`, `v`
- Can be combined with `fragility_damage_distribution` for a complete fragility-based risk pipeline

---

### `classical_damage()`

Compute expected damage-state fractions per asset from hazard curves and fragility functions (differentiable).

```python
@tf.function
def classical_damage(
    u: tf.Tensor,
    F: tf.Tensor,
    x_grid: tf.Tensor,
    hazard_poes: tf.Tensor,
    hazard_imls: tf.Tensor
) -> tf.Tensor
```

#### Mathematical Formulation

Integrates fragility exceedance probabilities against the hazard curve:

```
E[P(ds_d)] = ∫ P(ds_d | iml) × |dPoE/diml| diml
```

using the same ΔPoE discretization as `classical_loss`.

#### Parameters

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `u` | (N,) | int32 | Typology index per asset |
| `F` | (K, D, M) | float32 | Fragility tensor |
| `x_grid` | (M,) | float32 | Intensity grid for fragility curves |
| `hazard_poes` | (N, L) | float32 | Hazard PoE matrix |
| `hazard_imls` | (L,) | float32 | IML levels for hazard curves |

#### Returns

| Return | Shape | Type | Description |
|--------|-------|------|-------------|
| `expected_damage` | (N, D+1) | float32 | Expected fraction in each damage state per asset |

#### Example

```python
expected_damage = classical_damage(u_tf, F, x_tf, hazard_poes, hazard_imls)
print(f"Expected damage shape: {expected_damage.shape}")  # (N, D+1)

# Check probabilities sum to ~1 per asset
print(f"Sum per asset: {tf.reduce_sum(expected_damage, axis=1).numpy()}")
```

#### Notes

- Fully differentiable w.r.t. `F`, `hazard_poes`
- Combines fragility analysis with hazard-curve convolution

---

## Benefit-Cost Ratio

### `benefit_cost_ratio()`

Compute benefit-cost ratio for retrofitting (differentiable).

```python
@tf.function
def benefit_cost_ratio(
    aal_original: tf.Tensor,
    aal_retrofitted: tf.Tensor,
    retrofit_cost: tf.Tensor,
    interest_rate: float = 0.05,
    asset_life_expectancy: float = 50.0
) -> tf.Tensor
```

#### Mathematical Formulation

```
BCR_i = (AAL_original_i − AAL_retrofitted_i) × BPVF / retrofit_cost_i
BPVF = (1 − (1+r)^(−T)) / r
```

where BPVF is the Benefit Present Value Factor, r is the discount rate, and T is the asset life expectancy.

#### Parameters

| Parameter | Shape | Type | Description |
|-----------|-------|------|-------------|
| `aal_original` | (N,) | float32 | Average annual loss per asset with original vulnerability |
| `aal_retrofitted` | (N,) | float32 | Average annual loss per asset with retrofitted vulnerability |
| `retrofit_cost` | (N,) | float32 | Cost of retrofitting each asset |
| `interest_rate` | float | - | Annual discount rate (default: 0.05 = 5%) |
| `asset_life_expectancy` | float | - | Expected remaining life of the asset in years (default: 50) |

#### Returns

| Return | Shape | Type | Description |
|--------|-------|------|-------------|
| `bcr` | (N,) | float32 | Benefit-cost ratio per asset. BCR > 1 indicates retrofit is cost-effective |

#### Example

```python
# Compute BCR for a portfolio
aal_original = metrics_original['aal_per_asset']
aal_retrofitted = metrics_retrofitted['aal_per_asset']
retrofit_cost = tf.constant(v * 0.1, dtype=tf.float32)  # 10% of exposure

bcr = benefit_cost_ratio(aal_original, aal_retrofitted, retrofit_cost,
                         interest_rate=0.05, asset_life_expectancy=50.0)

# Find cost-effective retrofits
cost_effective = tf.where(bcr > 1.0)
print(f"Assets worth retrofitting: {len(cost_effective)}")
print(f"Best BCR: {tf.reduce_max(bcr).numpy():.2f}")
```

#### Notes

- Fully differentiable w.r.t. `aal_original`, `aal_retrofitted`, `retrofit_cost`
- Standard formula used in earthquake engineering (e.g., FEMA P-58)

---

## Usage Examples

### Complete Workflow

```python
import numpy as np
import tensorflow as tf
from tensor_engine import generate_synthetic_portfolio, TensorialRiskEngine

# 1. Generate portfolio data
print("Generating portfolio...")
v, u, C, x_grid, H, lambdas, Sigma = generate_synthetic_portfolio(
    n_assets=1000,
    n_events=5000,
    n_typologies=5,
    n_curve_points=20,
    lambda_distribution='exponential',
    sigma_fraction=0.3
)

# 2. Initialize engine (with vulnerability uncertainty)
print("Initializing engine...")
engine = TensorialRiskEngine(v, u, C, x_grid, H, lambdas, Sigma=Sigma)

# 3. Compute base metrics
print("Computing risk metrics...")
J_matrix, metrics = engine.compute_loss_and_metrics()

print(f"Portfolio AAL: ${metrics['aal_portfolio'].numpy():,.2f}")
print(f"Number of loss calculations: {J_matrix.shape[0] * J_matrix.shape[1]:,}")

# 4. Full gradient analysis
print("\nComputing gradients...")
analysis = engine.full_gradient_analysis()

# 5. Extract insights
print("\n" + "="*60)
print("RISK INSIGHTS")
print("="*60)

# Top risk contributors
aal_per_asset = metrics['aal_per_asset'].numpy()
top_assets = np.argsort(aal_per_asset)[-5:][::-1]

print("\nTop 5 Risk Contributors:")
for rank, idx in enumerate(top_assets, 1):
    print(f"{rank}. Asset {idx}:")
    print(f"   AAL: ${aal_per_asset[idx]:,.2f}")
    print(f"   Exposure: ${v[idx]:,.0f}")
    print(f"   Typology: {u[idx]}")
    print(f"   ∂AAL/∂v: {analysis['grad_exposure'][idx].numpy():.4f}")

# Vulnerability insights
grad_C = analysis['grad_vulnerability'].numpy()
vuln_impact = np.sum(np.abs(grad_C), axis=1)
most_impactful = np.argmax(vuln_impact)

print(f"\nMost Impactful Typology: {most_impactful}")
print(f"Total gradient magnitude: {vuln_impact[most_impactful]:.2e}")

# Hazard sensitivity
grad_H = analysis['grad_hazard']
avg_hazard_sens = tf.reduce_mean(tf.abs(grad_H)).numpy()

print(f"\nAverage Hazard Sensitivity: ${avg_hazard_sens:.2e}/g")
```

### Portfolio Optimization

```python
# Find optimal retrofit strategy under budget constraint

RETROFIT_BUDGET = 1_000_000  # $1M budget
RETROFIT_EFFECTIVENESS = 0.3  # 30% vulnerability reduction

# Get exposure gradient
grad_v, _ = engine.gradient_wrt_exposure()

# Rank assets by gradient (bang-for-buck)
ranked_assets = tf.argsort(grad_v, direction='DESCENDING').numpy()

# Allocate budget to highest-gradient assets
allocated = 0
retrofit_list = []

for idx in ranked_assets:
    retrofit_cost = v[idx] * 0.1  # Assume 10% of exposure
    
    if allocated + retrofit_cost <= RETROFIT_BUDGET:
        allocated += retrofit_cost
        potential_reduction = grad_v[idx].numpy() * v[idx] * RETROFIT_EFFECTIVENESS
        retrofit_list.append((idx, retrofit_cost, potential_reduction))
        
print(f"Optimal Retrofit Strategy (${RETROFIT_BUDGET:,.0f} budget):")
print(f"Total allocated: ${allocated:,.2f}")
print(f"\nAssets to retrofit:")

total_reduction = 0
for idx, cost, reduction in retrofit_list:
    print(f"Asset {idx}: Cost=${cost:,.0f}, AAL Reduction~${reduction:,.0f}")
    total_reduction += reduction

print(f"\nEstimated total AAL reduction: ${total_reduction:,.2f}")
print(f"ROI: {total_reduction/allocated:.2%}")
```

### Vulnerability Calibration

```python
# Use gradients to calibrate vulnerability curves to match target AAL

TARGET_AAL = 250_000_000  # Target $250M AAL
LEARNING_RATE = 1e-8

# Current AAL
_, metrics = engine.compute_loss_and_metrics()
current_aal = metrics['aal_portfolio'].numpy()

print(f"Current AAL: ${current_aal:,.2f}")
print(f"Target AAL: ${TARGET_AAL:,.2f}")
print(f"Difference: ${current_aal - TARGET_AAL:,.2f}")

# Compute gradient w.r.t. vulnerability
grad_C, _ = engine.gradient_wrt_vulnerability()

# Gradient descent step
adjustment = -LEARNING_RATE * (current_aal - TARGET_AAL) * grad_C.numpy()

# Apply adjustment (clip to [0, 1] for valid MDR)
new_C = np.clip(engine.C.numpy() + adjustment, 0.0, 1.0)
engine.C.assign(new_C)

# Check new AAL
_, new_metrics = engine.compute_loss_and_metrics()
new_aal = new_metrics['aal_portfolio'].numpy()

print(f"\nAfter adjustment:")
print(f"New AAL: ${new_aal:,.2f}")
print(f"Improvement: ${current_aal - new_aal:,.2f}")
```

---

## Mathematical Background

### Manuscript Equations Reference

#### Section 2: Deterministic Formulation

**Equation 1** - Interpolation weight:
```
α_i = (h_i - x_{j_i}) / (x_{j_i+1} - x_{j_i})
```

**Equation 2** - Mean Damage Ratio:
```
MDR_i = (1 - α_i) · C[u_i, j_i] + α_i · C[u_i, j_i+1]
```

**Equation 3** - Total loss:
```
J = Σ_{i=1}^N v_i · MDR_i
```

#### Section 3: Probabilistic Formulation

**Equation 4** - Loss matrix:
```
J[i,q] = v_i · MDR[i,q]
```

**Equation 5** - Per-asset rate-weighted AAL:
```
AAL_i = Σ_{q=1}^Q λ_q × J[i,q]
```
where λ_q are scenario occurrence rates.

**Equation 6** - Rate-weighted variance:
```
σ²_i = Σ_{q=1}^Q w_q × (J[i,q] - μ_i)²
```
where w_q = λ_q / Λ are normalized weights and μ_i = AAL_i / Λ.

**Law of total variance** (when Sigma provided):
```
Var_total_i = Var_aleatory_i + Var_vulnerability_i
Var_vulnerability_i = Σ_q w_q × v_i² × σ²_interpolated[i,q]
```

#### Section 4-5: Gradients

**Complete gradient:**
```
∇J = [∂J/∂H, ∂J/∂C, ∂J/∂v, ∂J/∂λ]
∂(Var_total)/∂Σ  (when Sigma provided)
```

### Differentiability

The key innovation is expressing all operations as differentiable tensor operations:

**Traditional (non-differentiable):**
```python
if intensity < threshold:
    damage = 0.0
else:
    damage = interpolate(intensity)
```

**Tensorial (differentiable):**
```python
alpha = (intensity - x_lower) / (x_upper - x_lower)
damage = (1 - alpha) * damage_lower + alpha * damage_upper
```

The second formulation is a smooth function whose gradient can be computed analytically.

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Grid lookup (searchsorted) | O(N×Q×log M) | Binary search per intensity |
| Interpolation | O(N×Q) | Linear in number of evaluations |
| Loss matrix | O(N×Q) | Element-wise operations |
| AAL computation | O(N×Q) | Reduction over Q dimension |
| Gradient (backprop) | O(N×Q) | Same order as forward pass |

For N=1000, Q=5000, M=20:
- Forward pass: ~5M operations
- Backward pass: ~5M operations
- Total: **~10M operations per gradient computation**

On modern GPUs, this completes in **milliseconds**.

---

## Performance Benchmarks

Typical performance on Apple M4 Pro with TensorFlow Metal:

| Portfolio Size | N | Q | Forward Pass | Gradient Compute | Total |
|----------------|---|---|--------------|------------------|-------|
| Small | 100 | 1,000 | 5 ms | 8 ms | 13 ms |
| Medium | 1,000 | 5,000 | 50 ms | 75 ms | 125 ms |
| Large | 10,000 | 10,000 | 400 ms | 600 ms | 1,000 ms |
| Very Large | 10,000 | 100,000 | 3,500 ms | 5,000 ms | 8,500 ms |

**Scalability:**
- Linear in N (number of assets)
- Linear in Q (number of events)
- Logarithmic in M (curve discretization)

---

## Best Practices

### 1. Memory Management

For very large portfolios:

```python
# Chunk over events to manage memory
CHUNK_SIZE = 10000

total_aal = 0.0
for q_start in range(0, n_events, CHUNK_SIZE):
    q_end = min(q_start + CHUNK_SIZE, n_events)
    H_chunk = H[:, q_start:q_end]
    
    # Process chunk
    engine_chunk = TensorialRiskEngine(v, u, C, x_grid, H_chunk)
    _, metrics = engine_chunk.compute_loss_and_metrics()
    total_aal += metrics['aal_portfolio'].numpy() * (q_end - q_start) / n_events
```

### 2. Numerical Stability

Always include epsilon in divisions:

```python
# Good
alpha = (h - x_lower) / (x_upper - x_lower + 1e-8)

# Bad (can produce NaN)
alpha = (h - x_lower) / (x_upper - x_lower)
```

### 3. Gradient Checking

Validate gradients numerically:

```python
def check_gradient_exposure(engine, i, epsilon=1e-5):
    """Numerical gradient check for exposure"""
    # Analytic gradient
    grad_v, metrics = engine.gradient_wrt_exposure()
    grad_analytic = grad_v[i].numpy()
    
    # Numerical gradient
    v_original = engine.v[i].numpy()
    
    engine.v[i].assign(v_original + epsilon)
    _, metrics_plus = engine.compute_loss_and_metrics()
    
    engine.v[i].assign(v_original - epsilon)
    _, metrics_minus = engine.compute_loss_and_metrics()
    
    grad_numeric = (metrics_plus['aal_portfolio'] - metrics_minus['aal_portfolio']) / (2 * epsilon)
    
    # Restore
    engine.v[i].assign(v_original)
    
    print(f"Analytic: {grad_analytic:.6f}")
    print(f"Numeric:  {grad_numeric.numpy():.6f}")
    print(f"Relative error: {abs(grad_analytic - grad_numeric) / abs(grad_numeric):.2e}")
```

### 4. GPU Utilization

Monitor GPU usage:

```python
# Check GPU availability
print("GPUs:", tf.config.list_physical_devices('GPU'))

# Monitor memory
import nvidia_smi  # For NVIDIA GPUs
# or check Activity Monitor on macOS for Metal usage
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory**
- Reduce `n_events` or use chunking
- Decrease batch size
- Use float16 instead of float32 (with caution)

**2. NaN in Gradients**
- Check for division by zero (add epsilon)
- Verify intensity grid is strictly monotonic
- Ensure vulnerability curves are in [0, 1]

**3. Slow Performance**
- Ensure GPU is being used (`tf.config.list_physical_devices('GPU')`)
- Check if data is on correct device
- Use `@tf.function` decorator for graph compilation

**4. Type Errors**
- Ensure all numeric inputs are correct dtype (float32/int32)
- Convert NumPy arrays before passing to TensorFlow functions

---

## Version History

- **v1.1** (March 2026) - Added classical risk (hazard-curve convolution), fragility/damage-state functions, consequence loss, classical damage, and benefit-cost ratio. Fixed sigma_fraction formula, CoV type, and metrics return keys documentation.
- **v1.0** (January 2026) - Initial release with full manuscript implementation

---

## References

1. Manuscript: "Tensorial Formulation for Differentiable Catastrophe Risk Assessment"
2. TensorFlow Documentation: https://www.tensorflow.org/
3. Automatic Differentiation: Griewank & Walther (2008)

---

## License & Citation

If you use this engine in research, please cite:

```
@software{tensorial_risk_engine,
  title = {Tensorial Risk Engine: Differentiable Catastrophe Risk Assessment},
  year = {2026},
  version = {1.0}
}
```

---

*End of Documentation*
