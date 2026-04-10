# Bogotá Hardware Benchmark Scripts

This folder contains two independent scripts for running the Bogotá benchmark on other machines:

- `download_zenodo_inputs.py`
- `run_trisk_hardware_benchmark.py`
- `run_openquake_hardware_benchmark.py`
- `plot_hardware_benchmark_results.py`

Each script is intended to be run inside its own environment. The T-Risk script requires TensorFlow. The OpenQuake script requires OpenQuake and uses direct in-memory calls to OpenQuake risk-library functions, not `oq engine --run` or OpenQuake datastore exports.

## Required Input Files

Place these two files in the same directory as the scripts:

- `bogota_trisk_inputs.npz`
- `bogota_hazard_chia.npz`

These are the only data inputs required by both scripts.

If `bogota_hazard_chia.npz` is not stored directly in the repository, download it from Zenodo:

DOI:

- `https://doi.org/10.5281/zenodo.19501460`

Downloader command:

```bash
python download_zenodo_inputs.py
```

By default, the downloader stores:

- `bogota_hazard_chia.npz`

in the same directory as the scripts.

If you already have the file locally, you can simply place it beside the scripts instead of using the downloader.

## T-Risk Script

Run inside the TensorFlow/T-Risk environment:

```bash
python run_trisk_hardware_benchmark.py
```

By default, this runs all four IM groups and benchmarks:

- forward AAL computation
- vulnerability gradient by TensorFlow automatic differentiation
- exposure gradient by TensorFlow automatic differentiation
- hazard gradient by TensorFlow automatic differentiation

The output is:

```text
trisk_hardware_benchmark_summary.json
```

Recommended full command:

```bash
MPLCONFIGDIR=/tmp/trisk_mpl_config XDG_CACHE_HOME=/tmp/codex_cache \
python run_trisk_hardware_benchmark.py \
  --out trisk_hardware_benchmark_summary.json \
  --repeat 3 \
  --warmup 1
```

Small smoke test:

```bash
MPLCONFIGDIR=/tmp/trisk_mpl_config XDG_CACHE_HOME=/tmp/codex_cache \
python run_trisk_hardware_benchmark.py \
  --groups SA_0p1 \
  --max-assets 200 \
  --max-events 20 \
  --repeat 1 \
  --warmup 0 \
  --out trisk_smoke_summary.json
```

To run only selected modes:

```bash
python run_trisk_hardware_benchmark.py --modes forward vulnerability exposure
```

## OpenQuake Script

Run inside the OpenQuake environment:

```bash
python run_openquake_hardware_benchmark.py
```

By default, this runs all four IM groups and benchmarks:

- forward AAL computation
- vulnerability-gradient finite differences for one representative typology per IM group
- exposure-gradient finite differences for the full asset set

The default does not run the full hazard-gradient finite-difference benchmark because it can take several minutes. To include it, use `--modes all` or explicitly include `hazard`.

The output is:

```text
openquake_hardware_benchmark_summary.json
```

Recommended full command without hazard gradient:

```bash
NUMBA_CACHE_DIR=/tmp/oq_numba_cache \
MPLCONFIGDIR=/tmp/oq_mpl_config \
XDG_CACHE_HOME=/tmp/codex_cache \
python run_openquake_hardware_benchmark.py \
  --out openquake_hardware_benchmark_summary.json \
  --repeat 1 \
  --warmup 0
```

Full command including hazard-gradient finite differences:

```bash
NUMBA_CACHE_DIR=/tmp/oq_numba_cache \
MPLCONFIGDIR=/tmp/oq_mpl_config \
XDG_CACHE_HOME=/tmp/codex_cache \
python run_openquake_hardware_benchmark.py \
  --modes all \
  --out openquake_hardware_benchmark_with_hazard_summary.json \
  --repeat 1 \
  --warmup 0
```

Small smoke test:

```bash
NUMBA_CACHE_DIR=/tmp/oq_numba_cache \
MPLCONFIGDIR=/tmp/oq_mpl_config \
XDG_CACHE_HOME=/tmp/codex_cache \
python run_openquake_hardware_benchmark.py \
  --groups SA_0p1 \
  --max-assets 200 \
  --max-events 20 \
  --modes forward vulnerability exposure \
  --repeat 1 \
  --warmup 0 \
  --out openquake_smoke_summary.json
```

## Timing Definition

The scripts report both input load time and compute timings. The main benchmark timing fields are compute-only timings on already loaded in-memory arrays.

For each timed operation, the JSON records:

- `times_s`
- `min_s`
- `mean_s`
- `std_s`
- `repeat`
- `warmup`

For manuscript-style reporting, use the compute timings, not the load time.

## Processing and Plotting Results

After running both benchmark scripts, place their JSON outputs in the same folder. For example:

```text
hardware_results/
  trisk_hardware_benchmark_summary.json
  openquake_hardware_benchmark_summary.json
```

Then run:

```bash
MPLCONFIGDIR=/tmp/trisk_mpl_config XDG_CACHE_HOME=/tmp/codex_cache \
python plot_hardware_benchmark_results.py \
  --results-dir hardware_results
```

The script automatically discovers the T-Risk and OpenQuake JSON files and writes figures to:

```text
hardware_results/figures/
```

Expected outputs include:

- `hardware_benchmark_timing_summary.csv`
- `hardware_benchmark_plot_summary.json`
- `forward_runtime_by_im.png`
- `vulnerability_runtime_by_im.png`
- `exposure_runtime_by_im.png`
- `speedup_summary_by_im.png`

If both benchmark JSON files contain hazard-gradient timings, it also writes:

- `hazard_runtime_by_im.png`
- `hazard_gradient_runtime_scaling.png`
- `hazard_gradient_runtime_share.png`

If the JSON files have different names, pass them explicitly:

```bash
python plot_hardware_benchmark_results.py \
  --results-dir hardware_results \
  --trisk-json hardware_results/my_trisk_results.json \
  --openquake-json hardware_results/my_openquake_results.json
```

## OpenQuake and Numba Timing Assumptions

The OpenQuake benchmark should be run with `NUMBA_CACHE_DIR` pointing to a writable directory. This allows numba-compiled functions inside the OpenQuake dependency stack to cache successfully and avoids known cache-location failures in some environments.

This does not mean that every operation in the OpenQuake benchmark is numba-accelerated. The forward calculation uses OpenQuake risk-library functionality and the installed NumPy/OpenQuake/numba stack where applicable. The finite-difference gradients, however, are implemented as explicit perturbation loops around OpenQuake's vulnerability evaluation. Those loops are the practical finite-difference baseline used for comparison; they are not a native OpenQuake analytic-gradient implementation and are not presented as a fully hand-optimized custom numba implementation of the gradient.

The intended interpretation is:

- OpenQuake is not penalized with file I/O, process startup, or datastore export overhead.
- OpenQuake is run in a configured environment with numba caching enabled.
- The gradient comparison measures the practical cost of obtaining sensitivities from OpenQuake by finite differences under harmonized deterministic inputs.
- The comparison should not be described as proving superiority over every possible custom-optimized finite-difference implementation.

## Notes

- `download_zenodo_inputs.py` constructs the direct Zenodo download URL from the DOI and downloads `bogota_hazard_chia.npz`.
- The OpenQuake script uses `openquake.risklib.scientific.VulnerabilityFunction` directly.
- The OpenQuake script suppresses vulnerability uncertainty by using zero coefficients of variation and deterministic mean loss ratios.
- The T-Risk script clips interpolation at the vulnerability-grid bounds to match the OpenQuake helper behavior.
- If the OpenQuake environment raises a numba cache error, set `NUMBA_CACHE_DIR` to a writable directory, as shown in the commands above.
