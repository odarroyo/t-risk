# Installation

## Requirements

- Python 3.9+
- TensorFlow 2.15+
- NumPy
- Matplotlib

## Core Engine

```bash
git clone https://github.com/odarroyo/t-risk.git
cd t-risk

pip install tensorflow numpy matplotlib
```

### Apple Silicon (M1/M2/M3/M4)

For Metal GPU acceleration:

```bash
pip install tensorflow-metal
```

## Web Application (Optional)

The Streamlit web interface provides a browser-based risk analysis platform:

```bash
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```

## Verify Installation

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

from tensor_engine import generate_synthetic_portfolio, TensorialRiskEngine
v, u, C, x_grid, H, lambdas, _ = generate_synthetic_portfolio(100, 500)
engine = TensorialRiskEngine(v, u, C, x_grid, H, lambdas=lambdas)
_, metrics = engine.compute_loss_and_metrics()
print(f"Portfolio AAL: ${metrics['aal_portfolio'].numpy():,.2f}")
print("Installation OK!")
```
