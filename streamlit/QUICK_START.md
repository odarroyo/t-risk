# Tensor Risk Engine - Web App Quick Start

**Get started in 5 minutes!**

## Installation

```bash
cd streamlit
pip install -r requirements.txt
```

## Launch

```bash
streamlit run app.py
```

Browser opens automatically at `http://localhost:8501`

## First Analysis (Synthetic Data)

1. **Setup Tab** → Click "🎲 Generate Synthetic Data"
   - N = 1000 assets
   - Q = 5000 events  
   - K = 5 typologies
   - Click "Generate Synthetic Portfolio"

2. **Run Analysis Tab** → Check "Compute Gradients" → Click "Run Risk Analysis"
   - Wait ~2-5 seconds (GPU) or ~15 seconds (CPU)

3. **Results Dashboard** → Explore:
   - Portfolio AAL (top metric cards)
   - Vulnerability curves
   - AAL vs Exposure scatter
   - Event loss distribution
   - Top risky assets table

4. **Gradients Tab** → Try:
   - Exposure gradient chart (retrofit priorities)
   - Retrofit optimizer (set budget, see recommendations)
   - Vulnerability gradient heatmap
   - Event contribution analysis

5. **Export** → Download complete analysis ZIP for later

## First Analysis (Your Data)

1. **Setup Tab** → Click template downloads:
   - 📄 Assets Template
   - 📄 Vulnerability Template  
   - 📄 Hazard Template

2. **Fill templates** with your data:
   - **Assets**: `exposure`, `typology` columns (required)
   - **Vulnerability**: K rows × M columns, values in [0, 1]
   - **Hazard**: N rows × Q columns, intensity values

3. **Inputs Tab** → Upload files → Click "Load and Validate Files"
   - Fix any validation errors (red messages)
   - Green checkmarks = ready to run

4. **Run Analysis Tab** → Run → Explore results

## Key Features

- **Flexible Input**: CSV/XLSX, synthetic data, or load saved analyses
- **Real-time Validation**: Immediate feedback on data quality
- **Interactive Charts**: 15+ Plotly visualizations (zoom, hover, export)
- **Gradient Analysis**: Full sensitivity for all parameters
- **Save/Load**: Resume work anytime (ZIP archives)
- **Retrofit Optimizer**: Budget-constrained mitigation planning

## File Formats

### Assets (CSV)
```csv
asset_id,exposure,typology
1,150000,0
2,500000,2
3,300000,1
```

### Vulnerability (CSV)
```csv
typology_name,0.00g,0.05g,0.10g,0.20g,...,1.50g
Old Masonry,0.00,0.05,0.15,0.35,...,0.98
Wood Frame,0.00,0.02,0.08,0.20,...,0.85
RC Frame,0.00,0.01,0.05,0.15,...,0.60
```

### Hazard (CSV - wide format preferred)
```csv
asset_id,event_1,event_2,event_3,...,event_Q
1,0.25,0.45,0.12,...,0.88
2,0.30,0.50,0.15,...,0.90
```

### Scenario Rates (Optional CSV)
```csv
event_id,lambda_per_year
1,0.03125
2,0.01000
3,0.00200
```

## Tabs Overview

| Tab | Purpose |
|-----|---------|
| 🏠 **Setup** | Choose how to provide data |
| 📥 **Inputs** | Upload files and validate |
| ⚡ **Run Analysis** | Execute computation |
| 📊 **Dashboard** | Explore metrics and charts |
| 🎯 **Gradients** | Sensitivity analysis & optimization |

## Common Issues

**"Typology index exceeds max allowed"**
→ Add more rows to vulnerability file or reduce typology indices

**"Intensity grid not strictly increasing"**  
→ Remove duplicate intensity values in vulnerability file

**App slow / freezes**
→ Start with smaller synthetic data (N=100, Q=1000) to test

**Can't upload file**
→ Ensure CSV UTF-8 encoding, XLSX not password-protected

## Next Steps

- **Detailed Guide**: See [APP_DOCUMENTATION.md](APP_DOCUMENTATION.md) (75+ pages)
- **Examples**: Check `examples/` folder for Python scripts
- **Core Engine**: Import directly in Python for custom workflows

## Support

- GitHub Issues: [Report bugs](https://github.com/yourusername/Tensor_Risk_Engine/issues)
- Documentation: APP_DOCUMENTATION.md, API_DOCUMENTATION.md
- Examples: minimum_example_*.py scripts

---

**Built with:** Streamlit • TensorFlow • Plotly • NumPy
