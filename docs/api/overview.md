# API Overview

The Tensorial Risk Engine API consists of:

## Standalone Functions

All decorated with `@tf.function` for graph compilation and GPU acceleration.

| Function | Section | Description |
|---|---|---|
| [`generate_synthetic_portfolio`](data-generation.md) | — | Generate synthetic test data |
| [`deterministic_loss`](core-functions.md#deterministic_loss) | §2 | Single-event portfolio loss |
| [`probabilistic_loss_matrix`](core-functions.md#probabilistic_loss_matrix) | §3 | Full N×Q loss matrix |
| [`compute_risk_metrics`](core-functions.md#compute_risk_metrics) | §3c | Rate-weighted AAL, variance, std |
| [`compute_risk_metrics_with_uncertainty`](core-functions.md#compute_risk_metrics_with_uncertainty) | §3c | Adds vulnerability variance decomposition |
| [`_interpolate_matrix`](classical-risk.md#_interpolate_matrix) | — | Shared interpolation helper |
| [`classical_loss`](classical-risk.md#classical_loss) | §6 | Hazard-curve convolution |
| [`fragility_damage_distribution`](fragility-damage.md#fragility_damage_distribution) | §7 | Damage-state probabilities |
| [`consequence_loss`](fragility-damage.md#consequence_loss) | §7 | Loss from damage states |
| [`classical_damage`](fragility-damage.md#classical_damage) | §8 | Hazard-curve × fragility |
| [`benefit_cost_ratio`](benefit-cost.md#benefit_cost_ratio) | §9 | Retrofit BCR |

## Engine Class

| Class | Description |
|---|---|
| [`TensorialRiskEngine`](engine-class.md) | Complete risk engine with gradient computation |

### Engine Methods

| Method | Target | Description |
|---|---|---|
| `compute_loss_and_metrics()` | — | Loss matrix + all risk metrics |
| `gradient_wrt_vulnerability()` | $\partial\text{AAL}/\partial C$ | Vulnerability curve sensitivity |
| `gradient_wrt_exposure()` | $\partial\text{AAL}/\partial v$ | Exposure sensitivity |
| `gradient_wrt_hazard()` | $\partial\text{AAL}/\partial H$ | Hazard intensity sensitivity |
| `gradient_wrt_lambdas()` | $\partial\text{AAL}/\partial \lambda$ | Scenario rate sensitivity |
| `gradient_wrt_sigma()` | $\partial\text{Var}/\partial \Sigma$ | Uncertainty sensitivity |
| `full_gradient_analysis()` | All | Complete gradient analysis in one pass |

## Conventions

- **Dtypes**: `float32` for tensors, `int32` for typology indices
- **Variable naming**: follows manuscript notation ($v$, $u$, $C$, $H$, $J$, $\lambda$, $\Sigma$)
- **Dimension aliases**: N = assets, Q = events, K = typologies, M = curve points, D = limit states
- **Numerical stability**: epsilon `1e-8` for all divisions
- **GPU compatibility**: flat indexing for Metal backend (no `gather_nd`)
