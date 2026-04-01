# Classical Risk Functions

## _interpolate_matrix

::: tensor_engine._interpolate_matrix

---

## classical_loss

::: tensor_engine.classical_loss

### Mathematical Formulation

For each asset $i$, integrates the vulnerability curve against the hazard curve:

$$\text{AAL}_i = v_i \times \sum_l \text{MDR}(\text{iml}_l) \times (\text{PoE}_{l-1} - \text{PoE}_l)$$

where $\text{PoE}_{l-1} - \text{PoE}_l$ represents the probability mass in each IML bin.

### Example

```python
import tensorflow as tf
from tensor_engine import classical_loss

# Hazard curves: monotonically decreasing PoE
hazard_imls = tf.constant([0.1, 0.3, 0.5, 0.7, 1.0], dtype=tf.float32)
hazard_poes = tf.constant([
    [0.8, 0.5, 0.2, 0.05, 0.01],   # Asset 0
    [0.6, 0.3, 0.1, 0.02, 0.005],  # Asset 1
], dtype=tf.float32)

aal = classical_loss(v_tf, u_tf, C_tf, x_tf, hazard_poes, hazard_imls)
print(f"AAL per asset: {aal.numpy()}")
```
