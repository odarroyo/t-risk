# JAX and PyTorch Backend Validation

This folder contains independent JAX and PyTorch implementations of the
tensorial risk formulation used by T-Risk, plus a compiled report summarizing
cross-backend validation.

The purpose of this folder is to support reproducibility of the backend
compatibility tests. The canonical implementation remains the TensorFlow engine
in the main T-Risk repository.

## Contents

```text
JAX_Pytorch_Implementation_Validation/
├── tensor_engine_jax.py
├── tensor_engine_pytorch.py
├── validate_bogota_jax_backend.py
├── tensor_backend_compatibility_report.pdf
└── README.md
```

## Files

- `tensor_engine_jax.py`: JAX implementation of the Tensorial Risk Engine API.
- `tensor_engine_pytorch.py`: PyTorch implementation of the Tensorial Risk Engine API.
- `validate_bogota_jax_backend.py`: JAX-vs-TensorFlow validator for the Bogotá benchmark.
- `tensor_backend_compatibility_report.pdf`: compiled report comparing TensorFlow, PyTorch, and JAX.

## Relationship To The Main Repository

This folder is designed to be placed inside the main `t-risk` repository:

```text
t-risk/
├── tensor_engine.py
├── bogota_trisk_inputs.npz
├── bogota_hazard_chia.npz
└── JAX_Pytorch_Implementation_Validation/
```

The validator imports:

- `tensor_engine` from the main repository root.
- `tensor_engine_jax` from this validation folder.

Therefore, run the validator from the repository root and include this folder
on `PYTHONPATH`.

## Required Inputs

For the Bogotá validation, the following files must be available in the
repository root:

- `bogota_trisk_inputs.npz`
- `bogota_hazard_chia.npz`

These are the same input files used in the T-Risk/OpenQuake Bogotá benchmark.
If large files are not committed to GitHub, download them from the project data
archive before running the validation.

## Environment

The validator compares JAX against TensorFlow, so the environment must include:

```bash
pip install tensorflow numpy jax jaxlib
```

For Apple Silicon TensorFlow acceleration:

```bash
pip install tensorflow-metal
```

For PyTorch experiments:

```bash
pip install torch
```

For JAX Metal, use a separate environment. The tested Apple Silicon stack was:

```bash
pip install --force-reinstall jax-metal==0.1.1 jax==0.5.0 jaxlib==0.5.0 numpy
```

Metal runs also require:

```bash
export ENABLE_PJRT_COMPATIBILITY=1
```

JAX Metal is experimental. Treat Metal results as exploratory unless the
numerical checks are repeated on the target machine.

## Run The JAX Validator

From the main repository root:

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

To run all available modes explicitly:

```bash
PYTHONPATH=".:JAX_Pytorch_Implementation_Validation" \
python JAX_Pytorch_Implementation_Validation/validate_bogota_jax_backend.py \
  --modes all \
  --out JAX_Pytorch_Implementation_Validation/bogota_jax_validation_all.json
```

The output JSON records:

- TensorFlow and JAX device visibility.
- Forward loss-matrix agreement.
- Per-asset AAL and event-loss agreement.
- Gradients with respect to vulnerability, exposure, hazard, and event rates.
- JAX eager timing.
- First JIT-call timing, including compilation.
- Steady-state JIT timing after compilation.

## Using The Backend Modules

Both backend modules mirror the public risk API of the TensorFlow engine:

```python
from tensor_engine_jax import TensorialRiskEngine as JAXRiskEngine
from tensor_engine_pytorch import TensorialRiskEngine as PyTorchRiskEngine
```

The expected inputs are NumPy arrays:

- `v`: exposure vector, shape `(N,)`
- `u`: typology index vector, shape `(N,)`
- `C`: vulnerability matrix, shape `(K, M)`
- `x_grid`: vulnerability intensity grid, shape `(M,)`
- `H`: hazard intensity matrix, shape `(N, Q)`
- `lambdas`: event-rate vector, shape `(Q,)`

Example:

```python
engine = JAXRiskEngine(v, u, C, x_grid, H, lambdas)
j_matrix, metrics = engine.compute_loss_and_metrics()
analysis = engine.full_gradient_analysis()
```

The same pattern works with `PyTorchRiskEngine`.

## Validation Findings

The compiled report in this folder summarizes the full validation. In brief:

- Forward loss matrices match TensorFlow at approximately float32 precision.
- Exposure, hazard, and event-rate gradients agree closely with TensorFlow.
- Vulnerability-gradient relative L2 errors are larger because many indexed
  asset-event contributions accumulate into a compact vulnerability matrix, but
  correlations remain above 0.999997.
- TensorFlow remains the canonical implementation and the most balanced Apple
  Silicon backend in the current tests.
- JAX is a strong candidate for compiled autodiff workflows, especially on
  CUDA or after further maturation of JAX Metal.
- PyTorch is numerically compatible, but PyTorch MPS was not performant for the
  indexing-heavy workload tested here.

## Caveats

- This folder does not replace the TensorFlow engine.
- The JAX validator requires the main repository's `tensor_engine.py`.
- The PyTorch backend implementation is included, but this folder does not
  currently include the full PyTorch benchmark runner.
- JAX compilation is shape-dependent. Report first-JIT and steady-state timing
  separately.
- Apple JAX Metal is experimental and should be validated on each machine.
