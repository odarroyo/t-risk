# Core Functions

## deterministic_loss

::: tensor_engine.deterministic_loss

---

## probabilistic_loss_matrix

::: tensor_engine.probabilistic_loss_matrix

!!! warning "Memory Considerations"
    For large portfolios, the loss matrix can be substantial:
    
    - N=10,000 × Q=100,000 → 1 billion elements (~4GB in float32)
    - Consider chunking over events for memory-constrained environments

---

## compute_risk_metrics

::: tensor_engine.compute_risk_metrics

### Return Dictionary Keys

| Key | Shape | Description |
|---|---|---|
| `aal_per_asset` | (N,) | Rate-weighted Average Annual Loss per asset |
| `aal_portfolio` | scalar | Total portfolio AAL |
| `mean_per_event_per_asset` | (N,) | Mean loss per event occurrence per asset |
| `variance_per_asset` | (N,) | Rate-weighted loss variance per asset |
| `std_per_asset` | (N,) | Loss standard deviation per asset |
| `loss_per_event` | (Q,) | Total portfolio loss per event |
| `total_rate` | scalar | Total occurrence rate $\Lambda = \sum_q \lambda_q$ |

---

## compute_risk_metrics_with_uncertainty

::: tensor_engine.compute_risk_metrics_with_uncertainty

### Additional Return Keys

In addition to all keys from `compute_risk_metrics`:

| Key | Shape | Description |
|---|---|---|
| `variance_vulnerability_per_asset` | (N,) | Vulnerability variance component per asset |
| `variance_vulnerability_portfolio` | scalar | Portfolio-level vulnerability variance |
| `variance_total_per_asset` | (N,) | Total variance (aleatory + vulnerability) per asset |
| `std_total_per_asset` | (N,) | Total standard deviation per asset |
| `variance_total_portfolio` | scalar | Total portfolio variance |
| `std_total_portfolio` | scalar | Total portfolio standard deviation |
