# Fragility & Damage State Functions

## fragility_damage_distribution

::: tensor_engine.fragility_damage_distribution

### Damage State Conversion

The function converts exceedance probabilities to damage-state probabilities:

$$P(ds_0) = 1 - P(\text{exceed } ds_1)$$

$$P(ds_d) = P(\text{exceed } ds_d) - P(\text{exceed } ds_{d+1}) \quad \text{for } d = 1 \ldots D-1$$

$$P(ds_D) = P(\text{exceed } ds_D)$$

---

## consequence_loss

::: tensor_engine.consequence_loss

### Mathematical Formulation

For each asset $i$ and event $q$:

$$\text{loss}[i,q] = v_i \times \sum_d P(ds_d | i, q) \times \text{CR}[u_i, d]$$

where CR is the consequence ratio matrix.

### Complete Fragility Pipeline

```python
from tensor_engine import fragility_damage_distribution, consequence_loss

# 1. Compute damage-state probabilities
damage_probs = fragility_damage_distribution(u_tf, F, x_tf, H_tf)

# 2. Convert to losses via consequence ratios
consequence_ratios = tf.constant([
    [0.0, 0.02, 0.10, 0.50, 1.0],  # Typology 0
    [0.0, 0.05, 0.20, 0.60, 1.0],  # Typology 1
], dtype=tf.float32)

loss_matrix = consequence_loss(damage_probs, consequence_ratios, v_tf, u_tf)
```

---

## classical_damage

::: tensor_engine.classical_damage

### Mathematical Formulation

Integrates fragility against the hazard curve:

$$E[P(ds_d)] = \int P(ds_d | \text{iml}) \times \left|\frac{d\text{PoE}}{d\text{iml}}\right| d\text{iml}$$
