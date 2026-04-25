# T-Risk: Tensorial Risk Engine

T-Risk is a differentiable disaster risk engine that expresses portfolio
risk calculations as tensors. The core implementation uses TensorFlow
to compute event-based losses, risk metrics, and automatic-differentiation
gradients for large asset portfolios.

The project is designed around one central idea: risk calculations that are
traditionally implemented as task-based loops can be reformulated as tensor
operations. This enables hardware acceleration and makes risk sensitivities
available directly through automatic differentiation.

Repository: <https://github.com/odarroyo/t-risk>

## Main Capabilities

- Event-based portfolio loss calculation.
- Multi-typology vulnerability-curve lookup.
- Rate-weighted Average Annual Loss (AAL) metrics.
- Per-asset and per-event risk outputs.
- Automatic gradients with respect to:
  - vulnerability curves, `dAAL/dC`
  - exposure values, `dAAL/dv`
  - hazard intensities, `dAAL/dH`
  - event occurrence rates, `dAAL/dlambda`
- TensorFlow execution with optional Apple Metal acceleration.
- Streamlit interface for interactive exploration.
- OpenQuake-oriented tests and comparison material.
- Experimental PyTorch and JAX backend-validation material.

## Repository Structure

The public repository currently contains the following main components:

```text
t-risk/
├── Documentation/
├── engine/
├── examples/
├── oq_tests/
├── streamlit/
├── API_DOCUMENTATION.md
├── LICENSE
├── MINIMUM_EXAMPLE_README.md
├── README.md
└── README_IMPLEMENTATION.md
```

If included in this branch or release, the backend-validation bundle is:

```text
JAX_Pytorch_Implementation_Validation/
├── tensor_engine_jax.py
├── tensor_engine_pytorch.py
├── validate_bogota_jax_backend.py
├── tensor_backend_compatibility_report.pdf
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/odarroyo/t-risk.git
cd t-risk
```

Install the core dependencies:

```bash
pip install tensorflow numpy matplotlib
```

For Apple Silicon TensorFlow acceleration:

```bash
pip install tensorflow-metal
```

For the Streamlit application:

```bash
cd streamlit
pip install -r requirements.txt
```

## Quick Start: Python API

The canonical TensorFlow implementation is exposed through the tensorial risk
engine. Depending on the repository layout in your checkout, import from the
`engine` package:

```python
from engine.tensor_engine import generate_synthetic_portfolio, TensorialRiskEngine
```

Create a synthetic portfolio and compute risk metrics:

```python
v, u, C, x_grid, H, lambdas = generate_synthetic_portfolio(
    n_assets=1000,
    n_events=5000,
    n_typologies=5,
    n_curve_points=20,
)

engine = TensorialRiskEngine(v, u, C, x_grid, H, lambdas)
j_matrix, metrics = engine.compute_loss_and_metrics()

print(f"Portfolio AAL: {metrics['aal_portfolio'].numpy():,.2f}")
```

Compute gradients:

```python
analysis = engine.full_gradient_analysis()

grad_vulnerability = analysis["grad_vulnerability"]
grad_exposure = analysis["grad_exposure"]
grad_hazard = analysis["grad_hazard"]
grad_lambdas = analysis["grad_lambdas"]
```

## Quick Start: Streamlit App

Launch the web interface:

```bash
cd streamlit
streamlit run app.py
```

The app provides a browser-based workflow for loading or generating inputs,
running risk calculations, inspecting outputs, and visualizing gradients.

## Data Model

The event-based engine uses the following tensor inputs:

| Symbol | Shape | Description |
|---|---:|---|
| `v` | `(N,)` | Exposure or replacement value per asset |
| `u` | `(N,)` | Integer typology index per asset |
| `C` | `(K, M)` | Vulnerability matrix with `K` curves and `M` intensity points |
| `x_grid` | `(M,)` | Intensity grid for vulnerability interpolation |
| `H` | `(N, Q)` | Hazard intensity matrix for `N` assets and `Q` events |
| `lambdas` | `(Q,)` | Event occurrence-rate vector |
| `J` | `(N, Q)` | Computed loss matrix |

For each asset-event pair, the engine interpolates the appropriate
vulnerability curve and computes:

```text
J[i, q] = v[i] * MDR(u[i], H[i, q])
```

The portfolio AAL is:

```text
AAL = sum_i sum_q lambda[q] * J[i, q]
```

## Differentiability

The differentiable quantities are:

- `v`: exposure vector
- `C`: vulnerability matrix
- `H`: hazard matrix
- `lambdas`: event-rate vector

The typology index vector `u` is integer-valued and is therefore not
differentiable. The interpolation bin selected by `searchsorted` is also
non-differentiable, so gradients with respect to `H` are piecewise-defined
within vulnerability-grid intervals.

## Examples And Documentation

Use the examples folder for executable demonstrations:

```bash
python examples/demonstration.py
python examples/minimum_example_deterministic.py
```

Additional documentation is available in:

- `API_DOCUMENTATION.md`
- `README_IMPLEMENTATION.md`
- `MINIMUM_EXAMPLE_README.md`
- `Documentation/`
- `streamlit/APP_DOCUMENTATION.md`, if present in your checkout

## OpenQuake Comparison Material

The repository includes two OpenQuake-oriented comparison areas:

- `oq_tests/`: validation material based on the Nepal example distributed with
  the OpenQuake engine repository.
- `Bogota_benchmark/`: a Bogotá scenario benchmark for an Mw 5.95 earthquake
  affecting Bogotá, the capital city of Colombia.

The broader benchmark workflow compares T-Risk with OpenQuake for event-based
risk scenarios, including agreement of loss outputs and timing comparisons for
gradient calculations.

The key distinction is that T-Risk computes gradients through automatic
differentiation, while OpenQuake-style gradient experiments require finite
differences or external perturbation workflows.

## Backend Compatibility: TensorFlow, PyTorch, And JAX

TensorFlow remains the canonical backend. Experimental PyTorch and JAX ports
were implemented to test whether the formulation is portable across tensor
libraries.

The validation bundle, if included, is in:

```text
JAX_Pytorch_Implementation_Validation/
```

The compiled report in that folder summarizes full-portfolio compatibility
tests. In brief:

- PyTorch and JAX reproduce TensorFlow forward losses at approximately float32
  precision.
- Exposure, hazard, and event-rate gradients agree closely with TensorFlow.
- Vulnerability gradients show larger relative L2 differences because many
  indexed asset-event contributions accumulate into a compact vulnerability
  matrix, but correlations remain very high.
- TensorFlow Metal is currently the most balanced Apple Silicon backend.
- JAX Metal can accelerate the full backward pass, but it is experimental and
  should be validated carefully on each machine.
- PyTorch MPS was not performant for the indexing-heavy workload tested here;
  CUDA testing is the next meaningful PyTorch accelerator benchmark.

## Running The Backend Validation Bundle

If `JAX_Pytorch_Implementation_Validation/` is present and the Bogotá input
files are available in the repository root, run:

```bash
PYTHONPATH=".:JAX_Pytorch_Implementation_Validation" \
python JAX_Pytorch_Implementation_Validation/validate_bogota_jax_backend.py \
  --inputs bogota_trisk_inputs.npz \
  --hazard bogota_hazard_chia.npz \
  --max-assets 2000 \
  --max-events 200 \
  --repeat 3 \
  --out JAX_Pytorch_Implementation_Validation/bogota_jax_validation.json
```

For full validation, use:

```bash
PYTHONPATH=".:JAX_Pytorch_Implementation_Validation" \
python JAX_Pytorch_Implementation_Validation/validate_bogota_jax_backend.py \
  --modes all \
  --out JAX_Pytorch_Implementation_Validation/bogota_jax_validation_all.json
```

## Performance Notes

This workload is not dominated by dense matrix multiplication. The critical
operations are:

- clipping
- `searchsorted`
- flattened gather/indexing
- reshape and broadcast operations
- indexed gradient accumulation
- reductions over assets and events

As a result, accelerator performance depends strongly on backend kernel
coverage and compiler behavior. GPU acceleration should always be reported with
the hardware, backend version, and whether timings include compilation.

For JAX, report first-JIT and steady-state timings separately.

## License

This project is distributed under the GPL-3.0 license. See `LICENSE`.

## Citation

If you use T-Risk in a manuscript or technical report, cite the repository and
include the commit hash or release tag used for the analysis. This is important
because backend behavior and acceleration results depend on library versions
and hardware.
