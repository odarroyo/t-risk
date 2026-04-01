# Benefit-Cost Ratio

## benefit_cost_ratio

::: tensor_engine.benefit_cost_ratio

### Mathematical Formulation

$$\text{BCR}_i = \frac{(\text{AAL}_{\text{original},i} - \text{AAL}_{\text{retrofitted},i}) \times \text{BPVF}}{\text{retrofit\_cost}_i}$$

where the Benefit Present Value Factor is:

$$\text{BPVF} = \frac{1 - (1+r)^{-T}}{r}$$

- $r$ = annual discount rate (default 5%)
- $T$ = asset life expectancy in years (default 50)
- BCR > 1 indicates the retrofit is cost-effective

### Example

```python
from tensor_engine import benefit_cost_ratio

# Compare original vs retrofitted portfolios
bcr = benefit_cost_ratio(
    aal_original=metrics_original['aal_per_asset'],
    aal_retrofitted=metrics_retrofitted['aal_per_asset'],
    retrofit_cost=tf.constant(v * 0.1, dtype=tf.float32),
    interest_rate=0.05,
    asset_life_expectancy=50.0
)

# Find cost-effective retrofits
cost_effective = tf.where(bcr > 1.0)
print(f"Assets worth retrofitting: {len(cost_effective)}")
print(f"Best BCR: {tf.reduce_max(bcr).numpy():.2f}")
```
