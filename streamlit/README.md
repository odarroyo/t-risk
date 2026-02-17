# Tensor Risk Engine - Streamlit Web Application

Interactive web interface for the Tensor Risk Engine, providing a user-friendly platform for catastrophe risk assessment with full gradient analysis.

## Features

- 📤 **Multiple Data Input Options**: Upload CSV/XLSX files, generate synthetic portfolios, or load saved analyses
- ⚡ **GPU-Accelerated Computing**: Leverages TensorFlow for fast computation on large portfolios
- 📊 **Interactive Visualizations**: 15+ Plotly-based interactive charts for comprehensive risk analysis
- 🎯 **Gradient Analysis**: Full sensitivity analysis with automatic differentiation
- 💾 **Save/Load Sessions**: Save complete analyses as ZIP archives for later review
- 🔧 **Retrofit Optimizer**: Use gradients to prioritize building retrofits within budget constraints

## Quick Start

### Installation

1. Ensure you have Python 3.12+ installed

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the App

From the `streamlit/` directory, run:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## User Guide

Video walkthrough is available at: https://youtu.be/sELH2Gjvyns

### Tab 1: Setup

Choose your data source:
- **Upload Files**: Provide CSV/XLSX files with exposure, vulnerability, hazard, and optionally lambda rates
- **Generate Synthetic**: Create test portfolios with configurable parameters (N, Q, K, M)
- **Load Saved Analysis**: Resume a previously saved session

Download template files to see expected formats.

### Tab 2: Inputs

- Upload and validate your data files
- Preview data with expandable sections
- See real-time validation messages

**Required files:**
- Assets: exposure and typology for each asset
- Vulnerability: K curves × M intensity points
- Hazard: N assets × Q events intensity matrix

**Optional files:**
- Lambda rates: custom occurrence rates for events

### Tab 3: Run Analysis

- Review configuration summary
- View memory usage estimate
- Choose whether to compute gradients
- Run the analysis and view computation time

### Tab 4: Results Dashboard

View comprehensive risk metrics and visualizations:
- Portfolio summary metrics (AAL, total rate, mean/max event loss)
- Vulnerability curves with hazard overlays
- AAL vs Exposure scatter plots
- Distribution comparisons by typology
- Top 20 assets table by AAL

**Export options:**
- Results CSV with asset-level metrics
- Summary TXT report
- Complete analysis ZIP (with all inputs, results, and gradients)

### Tab 5: Gradients & Sensitivity

Explore sensitivity analysis:

**Exposure Gradients (∂AAL/∂v)**
- Top 100 assets by sensitivity
- Retrofit optimizer with ROI calculations

**Vulnerability Gradients (∂AAL/∂C)**
- Heatmap showing critical curve regions

**Hazard Gradients (∂AAL/∂H)**
- Sensitivity vs return period scatter
- Sampled heatmap visualization

**Scenario Importance (∂AAL/∂λ)**
- Event contribution to AAL
- Loss vs occurrence rate relationships

## File Structure

```
streamlit/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # CSV/XLSX parsing and template generation
│   ├── validators.py          # Input validation functions
│   ├── visualizations.py      # Plotly chart creation (15+ functions)
│   └── persistence.py         # Save/load functionality
└── templates/                  # (Auto-generated template downloads)
```

## Data File Formats

### Assets File (CSV/XLSX)
```csv
asset_id,exposure,typology,latitude,longitude,description
1,150000,0,37.7749,-122.4194,Single-family home
2,500000,2,37.7849,-122.4094,Commercial building
```

**Required columns:** `exposure`, `typology`

### Vulnerability Curves (CSV/XLSX)
```csv
typology_name,intensity_0.0g,intensity_0.05g,...,intensity_1.5g
Old Masonry,0.00,0.05,...,0.98
Wood Frame,0.00,0.02,...,0.85
```

K rows (typologies) × M columns (intensity points)

### Hazard Matrix (CSV/XLSX)
```csv
asset_id,event_1,event_2,event_3,...
1,0.25,0.45,0.12,...
2,0.30,0.50,0.15,...
```

N rows (assets) × Q columns (events)

### Scenario Rates (CSV/XLSX) - Optional
```csv
event_id,lambda_per_year,return_period_years
1,0.03125,32
2,0.01,100
```

## Save/Load Format

Saved analyses are ZIP archives containing:
- `metadata.json`: Human-readable configuration and summary
- `inputs.npz`: Compressed NumPy arrays (v, u, C, x_grid, H, lambdas)
- `results.npz`: Compressed metrics and loss matrix
- `gradients.npz`: Gradient arrays (if computed)

## Performance Notes

**Typical computation times (Apple M4 Pro Metal):**
- N=1,000, Q=5,000: ~2 seconds (with gradients)
- N=10,000, Q=10,000: ~15 seconds (with gradients)
- N=10,000, Q=100,000: ~2 minutes (with gradients)

**Memory usage:**
- Loss matrix: N × Q × 4 bytes
- For N=10,000, Q=100,000: ~4 GB

Consider reducing Q or using event chunking for very large portfolios.

## Troubleshooting

### "No module named 'tensor_engine'"
- Ensure you're running from the `streamlit/` directory
- The app automatically adds the parent directory to Python path

### ValidationError: "Intensity grid not strictly increasing"
- Check that your vulnerability curve intensity values are in ascending order
- No duplicate values allowed

### Memory errors with large portfolios
- Reduce number of events (Q)
- Disable gradient computation
- Use a machine with more RAM or GPU memory

### TensorFlow not using GPU
- For macOS: Install `tensorflow-metal`
- For CUDA GPUs: Install `tensorflow-gpu`

## Advanced Usage

### Custom Lambda Distributions

Create custom occurrence rate distributions for:
- **Importance sampling**: Focus on critical events
- **Climate scenarios**: Time-varying hazard rates
- **Catalog weighting**: Synthetic vs historic events

### Gradient-Based Optimization

Use exposure gradients to:
- Optimize portfolio composition
- Allocate risk-based capital
- Prioritize retrofits with budget constraints
- Identify risk concentration

### Batch Processing

For multiple portfolio analyses:
1. Run analysis with each configuration
2. Save results as ZIP
3. Compare across scenarios using loaded analyses

## Contributing

For issues, feature requests, or contributions, please refer to the main Tensor Risk Engine repository.

## License

See main repository LICENSE file.

## Citation

If you use this software in academic work, please cite:
[Manuscript reference to be added]

---

**Tensor Risk Engine Team**  
February 2026
