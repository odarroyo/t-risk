# Tensor Risk Engine - Web Application Documentation

**Interactive Web Interface for Catastrophe Risk Assessment with Automatic Gradient Computation**

Version 1.0 | February 2026

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Getting Started](#getting-started)
4. [User Interface Guide](#user-interface-guide)
5. [Data Input Formats](#data-input-formats)
6. [Analysis Workflow](#analysis-workflow)
7. [Visualizations Reference](#visualizations-reference)
8. [Save/Load Functionality](#saveload-functionality)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)

---

## Introduction

The Tensor Risk Engine Web Application provides an intuitive, browser-based interface for performing catastrophe risk assessments with full gradient analysis. Built with Streamlit and Plotly, it offers:

- **📤 Flexible Data Input**: Upload CSV/XLSX files, generate synthetic portfolios, or load saved analyses
- **⚡ GPU-Accelerated Computing**: Leverages TensorFlow for fast computation on portfolios with millions of combinations
- **📊 Interactive Visualizations**: 15+ Plotly charts with zoom, pan, hover, and export capabilities
- **🎯 Gradient Analysis**: Complete sensitivity analysis using automatic differentiation
- **💾 Session Persistence**: Save and resume analyses as compressed ZIP archives
- **🔧 Optimization Tools**: Budget-constrained retrofit optimizer using exposure gradients

### Key Innovations

Unlike traditional catastrophe modeling software:
- **Fully Differentiable**: Entire pipeline supports automatic differentiation
- **No Discretization Errors**: Continuous interpolation instead of binned lookups
- **Real-time Validation**: Immediate feedback on data quality and consistency
- **Transparent Computation**: See exactly what's being calculated and why

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- 4GB RAM minimum (16GB+ recommended for large portfolios)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Optional: GPU for acceleration (Metal on macOS, CUDA on Linux/Windows)

### Step 1: Install Dependencies

```bash
cd streamlit
pip install -r requirements.txt
```

**Dependencies installed:**
- `streamlit >= 1.31.0` - Web application framework
- `numpy >= 1.24.0` - Numerical computing
- `pandas >= 2.0.0` - Data manipulation
- `tensorflow >= 2.15.0` - Deep learning framework (automatic differentiation)
- `plotly >= 5.18.0` - Interactive visualizations
- `matplotlib >= 3.7.0` - Additional plotting
- `openpyxl >= 3.1.0` - Excel file support
- `scipy >= 1.11.0` - Scientific computing

### Step 2: Verify Installation

```bash
streamlit --version
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

### Step 3: Launch Application

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### GPU Setup (Optional but Recommended)

**macOS (M1/M2/M3/M4):**
```bash
pip install tensorflow-metal
```

**Linux/Windows with NVIDIA GPU:**
```bash
pip install tensorflow[and-cuda]
```

**Verify GPU availability:**
```python
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
```

---

## Getting Started

### Quick Start: 5-Minute Demo

1. **Launch the app**: `streamlit run app.py`
2. **Go to Setup tab** → Select "🎲 Generate Synthetic Data"
3. **Configure portfolio**:
   - N = 1000 assets
   - Q = 5000 events
   - K = 5 typologies
   - Lambda = exponential
4. **Click "Generate"** → Go to "Run Analysis" tab
5. **Check "Compute Gradients"** → Click "Run Risk Analysis"
6. **Explore results** in Dashboard and Gradients tabs

**Expected time:** ~2-5 seconds (with GPU), ~10-20 seconds (CPU only)

### Your First Analysis with Real Data

#### Step 1: Download Templates
- Go to **Setup** tab
- Click "📄 Assets Template", "📄 Vulnerability Template", "📄 Hazard Template"
- Open templates in Excel or any CSV editor

#### Step 2: Fill Templates with Your Data

**Assets Template:**
```csv
asset_id,exposure,typology,latitude,longitude,description
1,150000,0,37.7749,-122.4194,Single-family home
2,500000,2,37.7849,-122.4094,Commercial building
...
```

**Vulnerability Template:**
- Each row = one building typology
- Each column = one intensity point (e.g., 0.0g, 0.05g, 0.1g, ...)
- Values = Mean Damage Ratio (0 to 1)

**Hazard Template:**
- Each row = one asset
- Each column = one stochastic event
- Values = ground motion intensity in g

#### Step 3: Upload & Validate
- Go to **Inputs** tab
- Upload your three files
- Check validation messages (green = pass, red = fix issues)

#### Step 4: Run Analysis
- Go to **Run Analysis** tab
- Review configuration
- Enable/disable gradients
- Click "Run Risk Analysis"

#### Step 5: Explore Results
- **Dashboard tab**: See portfolio metrics and visualizations
- **Gradients tab**: Analyze sensitivities (if gradients computed)
- **Export**: Download CSV, TXT report, or complete ZIP

---

## User Interface Guide

### Layout Overview

```
┌─────────────────────────────────────────────────────┐
│  Sidebar                  Main Content Area          │
│  ┌────────┐              ┌──────────────────────┐   │
│  │Quick   │              │  Tab Navigation      │   │
│  │Load    │              │ 🏠 │ 📥 │ ⚡ │ 📊 │ 🎯 │   │
│  ├────────┤              └──────────────────────┘   │
│  │Reset   │              ┌──────────────────────┐   │
│  ├────────┤              │                      │   │
│  │Docs    │              │   Active Tab         │   │
│  ├────────┤              │   Content            │   │
│  │Loaded  │              │                      │   │
│  │Info    │              │                      │   │
│  └────────┘              └──────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Sidebar Features

**🚀 Quick Load**
- Upload saved analysis ZIP
- Instantly resumes previous session
- Bypasses all data input steps

**🔄 Reset All**
- Clears entire session state
- Returns to fresh start
- Use when switching projects

**📚 Documentation**
- Links to API docs
- Links to engine source code

**📂 Loaded Analysis Info** (when applicable)
- Shows metadata from loaded analysis
- Timestamp, dimensions, AAL
- Quick reference without navigating tabs

### Tab Navigation

#### Tab 1: 🏠 Setup

**Purpose:** Choose how to provide data

**Options:**

1. **📤 Upload CSV/XLSX Files**
   - Download templates (5 types)
   - Prepare your data
   - Proceed to Inputs tab

2. **🎲 Generate Synthetic Data**
   - **Two modes available:**
     - **🎯 Simple Mode**: Random portfolios with uniform/exponential lambda distributions
     - **🏙️ Advanced Mode**: City-representative portfolios with category-based structure
   - Sliders for N, Q, K, M
   - Memory estimate shown
   - Instant generation
   - New: Total Rate (Λ) display with interpretation

3. **📂 Load Saved Analysis**
   - Upload previous ZIP
   - Shows metadata summary
   - Note about gradient availability
   - Resume from where you left off

**Memory Estimates:**
- Small (< 100 MB): Very fast, no concerns
- Medium (100-1000 MB): Fast on GPU
- Large (1-4 GB): May need chunking
- Very Large (> 4 GB): Reduce Q or use batching

#### Tab 2: 📥 Inputs

**Purpose:** Upload and validate data files

**File Uploaders:**

1. **Assets File** (Required)
   - Columns: `exposure`, `typology`
   - Optional: `asset_id`, `latitude`, `longitude`, `description`
   - Formats: CSV or XLSX

2. **Vulnerability Curves** (Required)
   - K rows × M columns
   - First row can have intensity headers
   - Values in [0, 1]
   - Formats: CSV or XLSX

3. **Hazard Matrix** (Required)
   - N rows × Q columns
   - Can have `asset_id` column
   - Wide format preferred
   - Formats: CSV or XLSX

4. **Scenario Rates** (Optional)
   - Q values
   - Column: `lambda` or `lambda_per_year`
   - If omitted: uniform rates (1/Q)

**Validation Display:**

After clicking "Load and Validate Files", you'll see:

```
Shapes:
✓ Shapes valid: N=1000 assets, Q=5000 events, K=5 typologies, M=20 points

Monotonicity:
✓ Intensity grid is strictly monotonic (0.000g to 1.500g)

Value Ranges:
✓ Vulnerability in valid range [0, 1]
✓ Hazard intensities reasonable: [0.025g, 1.164g]
✓ Occurrence rates valid: Λ=1.000000 events/year
```

**Error Examples:**

```
❌ Typology index 5 exceeds max allowed (4). Need at least 6 vulnerability curves.
```
→ Fix: Add more rows to vulnerability file or reduce max typology in assets

```
❌ Intensity grid not strictly increasing at position 7: x[7]=0.35, x[8]=0.35
```
→ Fix: Remove duplicate intensity value in vulnerability file

```
❌ Vulnerability values outside [0,1]: range=[−0.05, 1.03]
```
→ Fix: Clip or correct vulnerability values to valid range

**Preview Sections:**

- **Loaded Analysis**: Read-only view with expandable dataframes
- **Synthetic Data**: Summary metrics and preview
- **Uploaded Data**: Automatically shown after successful validation

#### Tab 3: ⚡ Run Analysis

**Purpose:** Execute risk computation with optional gradients

**Configuration Display:**
- N assets, Q events, K typologies, M curve points
- Total occurrence rate (Λ)
- Memory usage estimate

**Analysis Options:**

- ☐ **Compute Gradients** (checkbox)
  - Enabled: Full sensitivity analysis (~50% more time)
  - Disabled: Faster (metrics only, no gradient tab)

**Run Button:**
- Initializes TensorFlow engine
- Shows progress bar (33% → 66% → 100%)
- Displays computation time
- Shows quick summary metrics

**Computation Time Examples:**

| Portfolio | Time (GPU) | Time (CPU) |
|-----------|------------|------------|
| N=100, Q=1K | 0.05s | 0.2s |
| N=1K, Q=5K | 2s | 15s |
| N=10K, Q=10K | 15s | 180s |
| N=10K, Q=100K | 120s | ~30min |

**Status Messages:**
```
Initializing Tensor Risk Engine...
✓ Engine initialized
Computing loss matrix...
Metrics computed. Computing gradients...
✓ Analysis complete!
```

**Results Summary:**
```
✓ Analysis completed in 2.34 seconds

Portfolio AAL: $1,250,345.67
Mean Loss/Event: $250.07
Max Event Loss: $15,432.19
```

#### Tab 4: 📊 Results Dashboard

**Purpose:** Explore risk metrics and visualizations

**Portfolio Summary (Top Cards):**
```
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│Portfolio AAL │ Total Rate Λ │Mean Loss/Evt │Max Event Loss│Total Exposure│
│ $1,250,346   │  1.000000    │   $250       │  $15,432    │  $500,000,000│
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

**Interactive Visualizations:**

1. **🔵 Vulnerability Curves**
   - Line chart with markers
   - K curves colored by typology
   - Shaded regions: frequent vs rare events
   - Hover: intensity, MDR
   - Zoom/pan enabled

2. **🔴 AAL vs Exposure Scatter**
   - Points colored by typology
   - Diagonal reference line (AAL = Exposure)
   - Hover: asset details, AAL ratio
   - Identify risk concentration

3. **📊 Exposure Distribution**
   - Histogram overlay by typology
   - 30 bins per typology
   - Shows portfolio composition

4. **📊 AAL Distribution**
   - Box plots by typology
   - Shows median, quartiles, outliers
   - Compare risk profiles

5. **📊 Event Loss Distribution**
   - Histogram of portfolio loss per event
   - Vertical lines: mean (red), median (green)
   - 50 bins for resolution

**Top Assets Table:**
- Sortable by any column
- Shows rank, ID, AAL, Exposure, AAL/Exposure ratio, Typology
- Interactive filtering
- Top 20 (or all if fewer assets)

**Export Options:**

1. **📄 Download Results CSV**
   - Asset-level metrics
   - AAL, Exposure, Typology, Ratios, Std Dev
   - Filename: `risk_results_YYYYMMDD_HHMMSS.csv`

2. **📄 Download Summary TXT**
   - Portfolio configuration
   - All summary metrics
   - Computation details
   - Filename: `analysis_summary_YYYYMMDD_HHMMSS.txt`

3. **💾 Save Complete Analysis**
   - ZIP with all inputs, results, gradients
   - Includes metadata (timestamp, source, etc.)
   - Can be loaded later to resume
   - Filename: `tensor_analysis_YYYYMMDD_HHMMSS.zip`

#### Tab 5: 🎯 Gradients & Sensitivity

**Purpose:** Explore sensitivity analysis and optimize decisions

**Available only if gradients were computed**

**Section 1: Exposure Sensitivity (∂AAL/∂v)**

Interpretation: How much does portfolio AAL increase per $1 of exposure at each asset?

**Top Assets Chart:**
- Horizontal bar chart
- Color by typology
- Shows top N assets (up to 100)
- Hover shows exposure value
- Y-axis: asset IDs (hidden for clarity)

**Retrofit Optimizer:**

Interactive tool using exposure gradients:

```
Retrofit Budget: [$0 ──────●────── $250,000,000]
Retrofit Effectiveness: [10% ──────────●────── 100%] (50%)
```

**Calculations:**
- Retrofit cost = 30% of exposure (configurable assumption)
- AAL reduction = ∂AAL/∂v × Exposure × Effectiveness
- ROI = AAL reduction / Retrofit cost
- Selects assets sorted by ROI within budget

**Output Table:**
```
┌──────────┬──────────────┬──────────────┬──────┬─────────────┐
│Asset ID  │Retrofit Cost │AAL Reduction │ ROI  │Current AAL  │
├──────────┼──────────────┼──────────────┼──────┼─────────────┤
│   142    │   $45,000    │   $2,250     │ 0.050│   $4,500    │
│   873    │   $150,000   │   $6,750     │ 0.045│  $13,500    │
│   ...    │   ...        │   ...        │ ...  │   ...       │
└──────────┴──────────────┴──────────────┴──────┴─────────────┘

Assets to Retrofit: 15
Total Cost: $2,850,000
AAL Reduction: $127,500
```

**Section 2: Vulnerability Sensitivity (∂AAL/∂C)**

Interpretation: How does AAL change when vulnerability curve values change?

**Vulnerability Gradient Heatmap:**
- K rows (typologies) × M columns (intensities)
- Color scale: Red = high positive gradient, Green = negative
- Hover: typology, intensity, gradient value
- Identifies critical damage thresholds

**Use Cases:**
- Prioritize which curves need better calibration
- Identify most impactful intensity ranges
- Understand vulnerability model sensitivity

**Section 3: Hazard Sensitivity (∂AAL/∂H)**

Interpretation: Portfolio AAL change per unit intensity change at each asset-event combination

**Hazard Sensitivity vs Return Period:**
- Scatter plot (sampled for performance)
- X-axis: Return period (log scale)
- Y-axis: |∂AAL/∂H| (log scale)
- Color: occurrence rate λ
- Shows relationship between event frequency and sensitivity

**Hazard Gradient Heatmap:**
- Sampled N×Q matrix (max 50 assets × 100 events)
- Color scale: Blue-Red diverging (centered at 0)
- Shows spatial-temporal sensitivity pattern
- Asset index vs Event index

**Use Cases:**
- Identify critical asset-event combinations
- Quantify hazard model uncertainty impact
- Guide hazard model refinement

**Section 4: Scenario Importance (∂AAL/∂λ)**

Interpretation: ∂AAL/∂λ_q = Total portfolio loss in event q

**Event Contribution to AAL:**
- Scatter plot: Return Period vs (λ × Loss)
- Color: event loss magnitude
- Hover: return period, contribution, λ
- Identifies which events drive AAL

**Scenario Loss vs Occurrence Rate:**
- Log-log scatter plot
- X-axis: λ (events/year)
- Y-axis: Portfolio loss ($)
- Color: λ values on viridis scale
- Shows importance sampling distribution

**Gradient Statistics Panel:**

```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│ Exposure Gradients   │Vulnerability Gradnts │  Hazard Gradients    │
├──────────────────────┼──────────────────────┼──────────────────────┤
│ Mean: 0.1234         │ Mean: 1.23e+03       │ Mean: 4.56e+02       │
│ Std:  0.0456         │ Std:  5.67e+02       │ Std:  2.34e+02       │
│ Max:  0.8901         │ Max:  8.90e+03       │ Max:  9.87e+03       │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

---

## Data Input Formats

### Assets File

**Required Columns:**
- `exposure` (or `exposure_usd`, `value`, `replacement_cost`): Numeric, > 0
- `typology`: Integer, 0 to K-1

**Optional Columns:**
- `asset_id`: Identifier (numeric or string)
- `latitude`, `longitude`: Geographic coordinates
- `description`: Text description

**Example (CSV):**
```csv
asset_id,exposure,typology,latitude,longitude,description
1,150000,0,37.7749,-122.4194,"Single-family home, wood frame"
2,500000,2,37.7849,-122.4094,"Commercial building, RC frame"
3,300000,1,37.7949,-122.3994,"Multi-family, masonry"
4,200000,0,37.8049,-122.4294,"Single-family home, wood frame"
5,750000,3,37.8149,-122.4394,"Office building, steel frame"
```

**Constraints:**
- All exposure values must be > 0
- Typology indices must be 0 to K-1 (no gaps allowed in range)
- N rows = number of assets

### Vulnerability Curves File

**Format 1: CSV with Intensity Headers**
```csv
typology_name,intensity_0.00g,intensity_0.05g,intensity_0.10g,...,intensity_1.50g
Old Masonry (Type 0),0.00,0.05,0.15,...,0.98
Wood Frame (Type 1),0.00,0.02,0.08,...,0.85
RC Frame (Type 2),0.00,0.01,0.05,...,0.60
Steel Frame (Type 3),0.00,0.005,0.03,...,0.45
Prefab Metal (Type 4),0.00,0.002,0.01,...,0.30
```

**Format 2: XLSX with Multiple Sheets**
- Sheet "intensity_grid": Single column with M intensity values
- Sheet "curves": K rows × M columns (vulnerability values)

**Constraints:**
- K rows (one per typology, must match max(u)+1 from assets)
- M columns (intensity points)
- All values in [0, 1]
- Intensity grid must be strictly monotonically increasing
- First column can be typology names (will be detected and excluded)

**Typical Intensity Ranges:**
- **Earthquake**: 0.0g to 2.0g (peak ground acceleration)
- **Hurricane**: 0 to 200 mph (wind speed) - scale to [0, 2]
- **Flood**: 0 to 10m (inundation depth) - scale appropriately

### Hazard Matrix File

**Format 1: Wide Format (Preferred)**
```csv
asset_id,event_1,event_2,event_3,...,event_Q
1,0.25,0.45,0.12,...,0.88
2,0.30,0.50,0.15,...,0.90
3,0.22,0.42,0.10,...,0.85
...
N,0.28,0.48,0.14,...,0.92
```

**Format 2: Long Format**
```csv
asset_id,event_id,intensity
1,1,0.25
1,2,0.45
1,3,0.12
...
N,Q,0.92
```
Will be automatically pivoted to wide format.

**Constraints:**
- N rows (one per asset)
- Q columns (one per event)
- Values typically in [0, 2.0] for earthquake (g units)
- No negative values
- Can have `asset_id` column (will be excluded from matrix)

**Sources:**
- OpenQuake calculations
- USGS ShakeMap scenarios
- RiskScape hazard outputs
- Custom hazard modeling

### Scenario Rates File (Optional)

**Format:**
```csv
event_id,lambda_per_year,return_period_years,magnitude,description
1,0.03125,32.0,5.5,"Frequent local event"
2,0.01,100.0,6.5,"Moderate regional event"
3,0.002,500.0,7.5,"Rare major event"
4,0.0002,5000.0,8.5,"Very rare catastrophic event"
```

**Required Column:**
- `lambda` (or `lambda_per_year`, `rate`, `occurrence_rate`): Numeric, ≥ 0

**Optional Columns:**
- `event_id`: Event identifier
- `return_period_years`: 1/λ calculation reference
- Any other metadata

**Constraints:**
- Q rows (one per event)
- All λ_q ≥ 0
- Typically sum(λ) ≈ 1.0 for normalized probabilities
- If omitted: uniform rates λ_q = 1/Q

**Lambda Distributions:**

1. **Uniform** (default if file not provided)
   ```
   λ_q = 1/Q for all q
   ```
   Use when: events equally likely, Monte Carlo simulation

2. **Exponential** (realistic seismic recurrence)
   ```
   λ_q ∝ exp(-βq)  where β controls decay rate
   ```
   Use when: mimicking realistic earthquake catalogs

3. **Custom** (importance sampling)
   ```
   λ_q specified per event based on magnitude, distance, etc.
   ```
   Use when: have actual event catalog with recurrence rates

---

## Synthetic Portfolio Generation

The application offers **two modes** for generating synthetic portfolios, each serving different purposes:

### Mode 1: 🎯 Simple (Random Portfolios)

**Purpose:** Quick exploration and testing with fully randomized data

**Configuration:**
- **Portfolio Size:** N assets, Q events, K typologies, M curve points
- **Lambda Distribution:** 
  - Uniform: Equal probability for all events
  - Exponential: Realistic decay (frequent → rare events)

**Generation Process:**
1. Random exposure values: Uniform distribution [100k, 500k]
2. Random typology assignments: 0 to K-1
3. Sigmoid vulnerability curves: Steepness and midpoint vary by typology
4. Random hazard intensities: Uniform [0, 1.5g]
5. Lambda rates: Based on selected distribution

**Use Cases:**
- Quick testing and prototyping
- Algorithm verification
- Performance benchmarking
- Teaching basic concepts

**Limitations:**
- No realistic portfolio structure
- Random exposure/vulnerability relationships
- May not represent real-world scenarios

---

### Mode 2: 🏙️ Advanced (City-Representative Portfolios)

**Purpose:** Realistic portfolios with category-based structure and RP-dependent intensities

**Key Features:**

1. **Asset Categories**
   - Define 1-5 categories (e.g., "Low-Value Homes", "Old Vulnerable", "Modern Protected")
   - Control percentage distribution across portfolio
   - Set cost ranges (min/max) per category
   - Assign typologies (specific, random, or list)

2. **Return Period (RP) Spacing Modes**
   
   **Exponential Spacing** (Realistic):
   - Mimics actual seismic catalogs
   - Many frequent events (low RP), few rare events (high RP)
   - Creates higher total rates (Λ) due to clustering at low RP
   - Formula: `RP = exp(linspace(log(rp_min), log(rp_max), Q))`
   - **Caution:** Can produce Λ > 1.0 with narrow RP ranges or many events
   
   **Linear Spacing** (Educational):
   - Events uniformly distributed across RP range
   - Equal spacing between consecutive return periods
   - Creates lower total rates (Λ)
   - Formula: `RP = linspace(rp_min, rp_max, Q)`
   - **Best for:** Teaching, parameter exploration, avoiding unrealistic total rates

3. **RP-Dependent Intensity Coupling**
   - **Frequent events** (low RP): Low intensity range (e.g., 0.02-0.06g)
   - **Rare events** (high RP): High intensity range (e.g., 0.85-0.95g)
   - **Inverse relationship:** Higher occurrence rate → lower intensity
   - Realistic seismic behavior

**Total Rate (Λ) Interpretation:**

The total rate Λ = Σλ represents **expected number of damaging events per year**:

- **Λ < 0.3** 🟢: Realistic (one event every 3+ years)
- **0.3 ≤ Λ < 1.0** 🟡: High seismicity (one event every 1-3 years)
- **Λ ≥ 1.0** 🔴: Unrealistic (multiple events yearly, AAL may exceed portfolio value!)

**Guidelines for Realistic Λ:**
- Use **linear spacing** for most configurations, OR
- Use **exponential spacing** with wider RP ranges (e.g., 50-10000 years), OR
- Reduce number of events Q, OR
- Increase minimum return period

**Preset Templates:**

1. **Residential City**
   - 60% Low-Value Homes ($50k-$300k, typology 1-2)
   - 30% Mid-Value Homes ($300k-$800k, typology 2-3)
   - 10% High-Value Homes ($800k-$2M, typology 4)
   - RP: 32-5000 years, exponential spacing
   - Intensities: Frequent 0.02-0.06g, Rare 0.85-0.95g

2. **High-Risk Zone**
   - 70% Standard Homes ($100k-$200k, random typology)
   - 10% Old Vulnerable ($500k-$600k, typology 0 - weakest)
   - 20% Modern Protected ($400k-$500k, typology 4 - strongest)
   - RP: 32-5000 years, exponential spacing
   - Intensities: Frequent 0.02-0.06g, Rare 0.85-0.95g
   - **Demonstrates:** Risk concentration in vulnerable subportfolio

3. **Commercial District**
   - 40% Retail ($200k-$500k, typology 2)
   - 30% Office Buildings ($500k-$1.5M, typology 3)
   - 30% Mixed-Use ($300k-$800k, typology 2-3)
   - RP: 32-5000 years, exponential spacing
   - Intensities: Frequent 0.02-0.06g, Rare 0.75-0.90g

**Custom Categories:**

Create your own portfolio structure:
1. Click "Add Category" (up to 5 total)
2. Set name, percentage, cost range, typology
3. Ensure percentages sum to 100%
4. Validation checks automatically

**Example Custom Configuration:**

| Category | % | Cost Range | Typology | Purpose |
|----------|---|------------|----------|---------|
| Unreinforced Masonry | 20% | $200k-$400k | 0 | High vulnerability |
| Wood Frame | 50% | $150k-$350k | 2-3 | Moderate |
| Steel Frame | 30% | $500k-$1M | 4 | Low vulnerability |

**Comparison: Exponential vs Linear Spacing**

Example with N=100, Q=500, RP=[32, 5000] years:

| Spacing | Total Rate Λ | Inter-Event Time | RP < 100yr | RP > 1000yr | Use Case |
|---------|--------------|------------------|------------|-------------|----------|
| Exponential | 3.08 | 0.32 years | 368 events | 37 events | Realistic catalogs* |
| Linear | 0.52 | 1.91 years | 13 events | 245 events | Teaching/exploration |

*Requires wider RP range or fewer events for realistic Λ

**Educational Use:**

The dual-mode system is designed for teaching catastrophe modeling:

1. **Demonstrate RP spacing impact:**
   - Generate same portfolio with both spacing modes
   - Compare total rates and AAL values
   - Show how event clustering affects risk

2. **Explore unrealistic scenarios:**
   - Intentionally create Λ > 1.0 configurations
   - Show why AAL exceeds portfolio value
   - Teach importance of realistic recurrence assumptions

3. **Category-based risk concentration:**
   - Use "High-Risk Zone" preset
   - Show how 10% of assets can drive 40%+ of AAL
   - Demonstrate gradient-based asset prioritization

---

## Analysis Workflow

### Workflow 1: Exploration (Synthetic Data - Simple Mode)

**Goal:** Understand engine capabilities without data preparation

1. **Setup** → Generate Synthetic Data → **Simple Mode**
   - N = 1000, Q = 5000, K = 5, M = 20
   - Lambda = exponential
   
2. **Run Analysis** → Enable gradients → Run

3. **Dashboard** → Explore all visualizations
   - Note portfolio AAL
   - Examine top assets
   - Check event loss distribution

4. **Gradients** → Try retrofit optimizer
   - Budget = 10% of total exposure
   - Effectiveness = 50%
   - See recommended assets

---

### Workflow 1b: Realistic City Portfolio (Advanced Mode)

**Goal:** Generate realistic city-representative portfolio with category-based structure

1. **Setup** → Generate Synthetic Data → **Advanced Mode**
   - Choose preset: "Residential City", "High-Risk Zone", or "Commercial District"
   - Or customize: Define asset categories with percentages, cost ranges, typologies
   - Configure return period range (e.g., 32-5000 years)
   - Choose RP spacing: exponential (realistic) or linear (educational)
   - Set intensity ranges for frequent vs rare events
   
2. **Review Total Rate (Λ)**
   - Green (Λ < 0.3): Realistic seismicity
   - Yellow (0.3 ≤ Λ < 1.0): High seismicity
   - Red (Λ ≥ 1.0): Unrealistic - consider linear spacing or wider RP range
   
3. **Run Analysis** → Enable gradients → Run

3. **Dashboard** → Explore all visualizations
   - Note portfolio AAL
   - Examine top assets
   - Check event loss distribution

4. **Gradients** → Try retrofit optimizer
   - Budget = 10% of total exposure
   - Effectiveness = 50%
   - See recommended assets

5. **Export** → Save complete analysis
   - Download ZIP for later reference

**Time:** ~5 minutes

### Workflow 2: Your Data (First Time)

**Goal:** Analyze your actual portfolio

1. **Setup** → Download all templates

2. **Prepare Data** (outside app)
   - Fill assets.csv with your buildings
   - Create vulnerability curves (literature or calibrated)
   - Obtain hazard from OpenQuake/other source
   - Optional: prepare lambda rates

3. **Inputs** → Upload files
   - Fix any validation errors
   - Review previews

4. **Run Analysis** → Enable gradients → Run

5. **Dashboard** → Interpret results
   - Portfolio AAL vs your budget/limits
   - Which typologies drive risk?
   - Top risky assets?

6. **Gradients** → Sensitivity analysis
   - Which assets to retrofit first?
   - Are vulnerability curves well-calibrated?
   - Which events matter most?

7. **Export** → Save everything
   - ZIP for archival
   - CSV for reporting
   - TXT for documentation

**Time:** 1-2 hours (mostly data prep)

### Workflow 3: Scenario Comparison

**Goal:** Compare different assumptions/models

1. **Run baseline** (e.g., current vulnerability curves)
   - Save as `baseline_analysis.zip`

2. **Modify inputs** (e.g., improved building codes)
   - Update vulnerability curves
   - Re-upload

3. **Run comparison**
   - Save as `improved_analysis.zip`

4. **Manual comparison** (load each)
   - Compare Portfolio AAL
   - Compare top assets
   - Quantify AAL reduction

**Enhancement idea for v1.1:** Built-in comparison mode with side-by-side views

### Workflow 4: Uncertainty Quantification

**Goal:** Understand parameter uncertainty impact

1. **Run with mean vulnerability** → Note AAL

2. **Run with mean + 1 std** → Note AAL increase

3. **Run with mean - 1 std** → Note AAL decrease

4. **Use gradients** for faster estimate:
   ```
   ΔAAL ≈ Σ (∂AAL/∂C[k,m] × ΔC[k,m])
   ```

5. **Monte Carlo** (advanced):
   - Run N times with sampled parameters
   - Build AAL distribution
   - Report confidence intervals

---

## Visualizations Reference

### 1. Vulnerability Curves

**Type:** Line chart with markers

**Purpose:** Show Mean Damage Ratio vs Ground Motion Intensity for each typology

**Features:**
- Interactive legend (click to hide/show curves)
- Shaded regions for frequent/rare events
- Hover: exact intensity and MDR
- Zoom: rectangle select
- Pan: click and drag

**Interpretation:**
- Steeper curves = more fragile buildings
- Curve at 0.5g ≈ threshold for significant damage
- Compare your curves to literature (HAZUS, Spence, etc.)

**Use Cases:**
- Validate vulnerability inputs
- Understand damage progression
- Compare building types

### 2. AAL vs Exposure Scatter

**Type:** Scatter plot with typology coloring

**Purpose:** Identify assets with high risk per dollar

**Features:**
- Color by typology
- Diagonal reference line (AAL = Exposure)
- Hover: AAL/Exposure ratio
- Zoom to cluster

**Interpretation:**
- Points above diagonal: AAL > Exposure (impossible, check data)
- Points near diagonal: very high risk
- Clusters: similar exposure + typology + hazard
- Outliers: investigate individually

**Use Cases:**
- Risk concentration analysis
- Identify underinsured assets
- Portfolio composition review

### 3. Exposure Distribution

**Type:** Overlaid histograms

**Purpose:** Show portfolio composition by building type

**Features:**
- 30 bins per typology
- Transparency for overlap visibility
- Legend toggleable

**Interpretation:**
- Tall bars = many assets at this value
- Wide spread = diverse portfolio
- Concentration = potential correlated risk

**Use Cases:**
- Portfolio balance check
- Identify dominant exposure ranges
- Prepare for aggregation

### 4. AAL Distribution

**Type:** Box plots

**Purpose:** Compare risk profiles across typologies

**Features:**
- Box: IQR (25th-75th percentile)
- Whiskers: 1.5×IQR
- Outliers: individual points
- Hover: exact values

**Interpretation:**
- Median line: typical asset AAL
- Box width: variability within typology
- Outliers: exceptional risk
- Compare medians: which typology is riskiest?

**Use Cases:**
- Typology risk ranking
- Identify outlier assets
- Understand within-type variability

### 5. Event Loss Distribution

**Type:** Histogram with statistics

**Purpose:** Show portfolio loss across all stochastic events

**Features:**
- 50 bins
- Vertical lines: mean (red), median (green)
- Hover: bin range, count

**Interpretation:**
- Shape: often right-skewed (rare high losses)
- Mean vs median: degree of skewness
- Tail: catastrophic scenarios
- Mode: most common outcome

**Use Cases:**
- Understand event variability
- Exceedance probability estimation
- Reinsurance layer placement

### 6. Vulnerability Gradient Heatmap

**Type:** 2D heatmap (K × M)

**Purpose:** Show sensitivity of AAL to vulnerability curve changes

**Features:**
- Color: gradient magnitude
- Hover: typology, intensity, gradient
- Zoom to region

**Interpretation:**
- Red (high positive): increasing MDR here increases AAL most
- Green (negative): rare, check data
- Bright spots: critical thresholds
- Uniform column: all typologies sensitive at this intensity

**Use Cases:**
- Prioritize vulnerability research
- Identify calibration targets
- Understand model sensitivity

### 7. Top Assets by Exposure Gradient

**Type:** Horizontal bar chart

**Purpose:** Rank assets by retrofit effectiveness

**Features:**
- Color by typology
- Hover: exposure value
- Sorted by gradient (highest first)

**Interpretation:**
- Longer bar = higher ∂AAL/∂v
- Means: $1 more exposure → more AAL
- Product gradient × exposure = potential AAL reduction

**Use Cases:**
- Retrofit prioritization
- Asset-level risk contribution
- Portfolio optimization

### 8. Hazard Sensitivity vs Return Period

**Type:** Scatter plot (log-log)

**Purpose:** Relate event frequency to hazard sensitivity

**Features:**
- Sample: 1000 points (if N×Q > 1000)
- Color: λ value
- Both axes logarithmic

**Interpretation:**
- X-axis: how often event occurs
- Y-axis: how sensitive AAL is to hazard change
- Pattern: typically high sensitivity for moderate return periods

**Use Cases:**
- Understand critical return periods
- Hazard model uncertainty impact
- Guide hazard refinement

### 9. Hazard Gradient Heatmap

**Type:** 2D heatmap (sampled N × Q)

**Purpose:** Show asset-event sensitivity pattern

**Features:**
- Sampled for performance (max 50×100)
- Diverging colorscale (blue-white-red)
- Hover: asset, event, gradient

**Interpretation:**
- Red: intensity increase → AAL increase
- Blue: intensity decrease → AAL increase (unusual)
- White: low sensitivity
- Row patterns: asset-specific
- Column patterns: event-specific

**Use Cases:**
- Identify critical asset-event pairs
- Understand spatial patterns
- Guide detailed analysis

### 10. Event Contribution to AAL

**Type:** Scatter plot

**Purpose:** Show which events drive AAL (λ × Loss)

**Features:**
- X: return period (log scale)
- Y: contribution to AAL
- Color: event loss

**Interpretation:**
- Y-value = event's contribution to portfolio AAL
- High points = important events
- Rare events: high loss but low contribution (if λ small)
- Frequent events: moderate loss but high contribution

**Use Cases:**
- Identify critical scenarios
- Validate event catalog
- Explain AAL to stakeholders

### 11. Scenario Loss vs Occurrence Rate

**Type:** Log-log scatter

**Purpose:** Show relationship between event frequency and severity

**Features:**
- Both axes logarithmic
- Color: λ value
- Hover: exact values

**Interpretation:**
- Typically: inverse relationship (rare = severe)
- Scatter: variability in event set
- Outliers: unusual events

**Use Cases:**
- Validate event catalog
- Understand importance sampling
- Communication with technical stakeholders

---

## Save/Load Functionality

### Save Format

**File Type:** ZIP archive

**Contents:**
1. `metadata.json` - Human-readable configuration
2. `inputs.npz` - Compressed NumPy arrays (v, u, C, x_grid, H, lambdas)
3. `results.npz` - Compressed metrics and loss matrix
4. `gradients.npz` - Gradient arrays (only if computed)

**Metadata Structure:**
```json
{
  "version": "1.0",
  "engine": "TensorialRiskEngine",
  "timestamp": "2026-02-09T14:30:00",
  "dimensions": {
    "N": 1000,
    "Q": 5000,
    "K": 5,
    "M": 20
  },
  "data_source": "uploaded",
  "uploaded_filenames": {
    "assets": "my_portfolio.csv",
    "vulnerability": "my_curves.xlsx",
    "hazard": "my_hazard.csv",
    "lambdas": "my_lambdas.csv"
  },
  "lambda_mode": "custom",
  "gradients_computed": true,
  "computation_time_seconds": 2.34,
  "total_rate": 1.000000,
  "aal_portfolio": 1250345.67
}
```

### Saving an Analysis

**From Results Dashboard:**
1. Scroll to "Export Results" section
2. Click "💾 Save Complete Analysis"
3. File downloads: `tensor_analysis_YYYYMMDD_HHMMSS.zip`

**What's Saved:**
- All input arrays (v, u, C, x_grid, H, lambdas)
- All computed metrics (AAL per asset, variance, etc.)
- Complete loss matrix (N×Q)
- All gradients (if computed)
- Metadata (when, how, from what)

**File Size:**
- Compressed efficiently
- Typical: 5-50 MB
- Large portfolios (N=10K, Q=100K): up to 500 MB

### Loading an Analysis

**Method 1: Quick Load (Sidebar)**
1. Click "Load Saved Analysis" in sidebar
2. Select ZIP file
3. Automatically populates all data
4. Switches to Results tab

**Method 2: Full Load (Setup Tab)**
1. Go to Setup tab
2. Select "Load Saved Analysis" option
3. Upload ZIP file
4. View metadata summary
5. Navigate to Results or Gradients tabs

**What Happens:**
- All inputs restored to session state
- Results and gradients available immediately
- No re-computation needed
- Can view/export as normal

**Validation:**
- Automatic shape checking
- Array integrity verification
- Version compatibility check (if different versions)
- Warning if gradients missing

### Use Cases for Save/Load

**1. Archival**
- Save analyses for regulatory compliance
- Document portfolio evolution over time
- Maintain analysis trail

**2. Collaboration**
- Share analyses with colleagues
- Send to clients/stakeholders
- Peer review

**3. Comparison**
- Save baseline
- Save scenario variations
- Manual comparison (load each)

**4. Resume Work**
- Exit app mid-analysis
- Load next day to continue
- No data re-entry

**5. Presentation**
- Prepare analysis
- Load during meeting
- Show live visualizations

---

## Advanced Features

### 1. Retrofit Optimizer

**Purpose:** Use exposure gradients to optimize budget-constrained mitigation

**Algorithm:**
1. Compute retrofit cost per asset (default: 30% of exposure)
2. Compute AAL reduction = ∂AAL/∂v × Exposure × Effectiveness
3. Compute ROI = AAL reduction / Retrofit cost
4. Sort assets by ROI (descending)
5. Select assets until budget exhausted

**Controls:**
- **Budget Slider**: Total available funds ($0 to 50% of portfolio exposure)
- **Effectiveness Slider**: Expected vulnerability reduction (10% to 100%)

**Assumptions (Customizable in Code):**
- Retrofit cost = 0.3 × Exposure (can modify in app.py line ~795)
- Effectiveness applies uniformly (could make typology-specific)
- Linear relationship (could add diminishing returns)

**Output:**
- List of recommended assets
- Total cost (≤ budget)
- Total AAL reduction
- Individual ROI for each asset

**Extensions:**
- Multiple retrofit types (different cost/effectiveness)
- Non-linear cost models
- Constraints (e.g., max assets, geographic spread)

### 2. Gradient Statistics

**Purpose:** Summarize gradient distributions for reporting

**Metrics Shown:**
- **Mean**: Average gradient across all elements
- **Std**: Standard deviation (variability)
- **Max**: Maximum absolute gradient (most sensitive point)

**Interpretation:**

**Exposure Gradients:**
- Mean ~ 0.10: Average \$0.10 AAL per \$1 exposure
- High std: Large variability in asset risk
- Max ~ 0.90: Some assets very risky

**Vulnerability Gradients:**
- Typically large values (many assets affected)
- Max indicates critical curve point
- Compare across typologies

**Hazard Gradients:**
- Typically largest in magnitude
- Max indicates critical asset-event pair
- Average shows overall hazard sensitivity

### 3. Memory Management

**Automatic:**
- Arrays stored as float32 (not float64)
- Compression in NPZ save (typically 5-10× reduction)
- Sampling in large heatmaps

**Manual:**
- Disable gradients if not needed
- Reduce Q (event chunking if needed)
- Use synthetic data for testing before full analysis

**Monitoring:**
```python
import tensorflow as tf
print(tf.config.experimental.get_memory_info('GPU:0'))
```

### 4. Batch Processing (Advanced Users)

**Scenario:** Run multiple portfolios automatically

**Approach:**
1. Create script that calls Streamlit session state programmatically
2. Loop over data files
3. Save each result

**Example Framework:**
```python
# batch_run.py
import streamlit as st
from app import run_analysis

for datafile in data_files:
    inputs = load_data(datafile)
    results, gradients = run_analysis(inputs)
    save_analysis(results, f"results_{datafile}.zip")
```

Note: Streamlit is designed for interactive use; batch processing may require custom wrapper.

### 5. Best Practices for Synthetic Portfolio Generation

**Choosing Between Simple and Advanced Modes:**

| Scenario | Recommended Mode | Settings |
|----------|------------------|----------|
| Quick testing | Simple | N=100, Q=1000, Exponential |
| Algorithm validation | Simple | Uniform lambda for reproducibility |
| Teaching basics | Simple | Small N/Q for fast iteration |
| Realistic city portfolio | Advanced | Use presets, check Λ |
| Research/publication | Advanced | Custom categories, document Λ |
| Parameter sensitivity | Advanced | Linear spacing for exploration |

**Managing Total Rate (Λ):**

**Target:** Λ < 1.0 for realistic portfolios

**If Λ > 1.0:**

1. **Switch to linear spacing** (easiest fix)
   - Reduces Λ by 80-90% typically
   - Maintains RP range coverage
   - Good for most applications

2. **Widen RP range** (keep exponential)
   - Increase `rp_min` (e.g., 32 → 50 years)
   - Increase `rp_max` (e.g., 5000 → 10000 years)
   - Spreads events across wider timespan

3. **Reduce number of events**
   - Lower Q (e.g., 1000 → 500)
   - Fewer events = lower total rate
   - May reduce resolution

4. **Accept for educational purposes**
   - Demonstrate unrealistic scenarios
   - Show AAL > portfolio value problem
   - Teach importance of recurrence modeling

**When to Use Each RP Spacing Mode:**

**Exponential Spacing:**
- Realistic seismic catalog representation
- Research requiring realism
- Demonstrating tail risk importance
- When you need event clustering behavior
- **Caution:** Monitor Λ closely

**Linear Spacing:**
- Teaching and exploration
- Parameter sensitivity studies
- When you need Λ < 1.0 guaranteed
- Uniform coverage of RP range
- Easier interpretation for non-experts

**Category Design Tips:**

1. **Percentages:**
   - Must sum to exactly 100%
   - Consider actual city demographics
   - Example: 60% residential, 30% commercial, 10% industrial

2. **Cost Ranges:**
   - Non-overlapping is clearer but not required
   - Use realistic values for region
   - Consider inflation if comparing across years

3. **Typology Assignment:**
   - `'random'`: Diverse portfolio within category
   - Specific value (0-4): Uniform vulnerability
   - List [1,2,3]: Mix of specific types
   - Lower index = more vulnerable

4. **Intensity Ranges:**
   - Frequent events: 0.01-0.10g typical
   - Rare events: 0.5-1.5g typical
   - Ensure rare > frequent
   - Consider local seismic hazard (e.g., Chile vs. UK)

**Educational Workflow - Teaching Seismic Risk:**

**Lesson 1: Event Frequency Impact**
1. Generate portfolio with exponential spacing, RP=[32, 5000]
2. Note Λ and AAL
3. Regenerate with linear spacing, same RP range
4. Compare Λ (should be ~5× lower)
5. Compare AAL (should be similar despite different Λ)
6. **Takeaway:** Event distribution affects total rate but not necessarily AAL

**Lesson 2: Risk Concentration**
1. Use "High-Risk Zone" preset
2. Run analysis with gradients
3. Dashboard → Examine AAL by category
4. Note: 10% of assets (old vulnerable) drive 40%+ of AAL
5. Gradients → See high ∂AAL/∂v for vulnerable assets
6. **Takeaway:** Portfolio composition matters enormously

**Lesson 3: Return Period vs Intensity**
1. Generate advanced portfolio
2. Dashboard → Vulnerability curves
3. Note intensity ranges overlaid
4. Understand: frequent events hit low-intensity portions of curves
5. Rare events hit high-intensity (catastrophic) portions
6. **Takeaway:** RP-intensity coupling is critical for realism

**Common Misconceptions to Address:**

- **"More events = higher AAL"**: False. AAL = Σ(λ × L). More low-λ events may not increase AAL.
- **"Λ should equal 1.0"**: Only if events are normalized probabilities. Physical interpretation: events/year.
- **"Exponential spacing is always better"**: Not for teaching or when causing Λ > 1.0.
- **"All portfolios need 1000+ events"**: Depends on application. 500 often sufficient for teaching.

---

## Troubleshooting

### Installation Issues

**Problem:** `ImportError: No module named 'streamlit'`

**Solution:**
```bash
pip install streamlit
# or if using conda:
conda install -c conda-forge streamlit
```

---

**Problem:** TensorFlow installation fails on macOS

**Solution:**
```bash
# Use conda for cleaner install
conda create -n tensor_risk python=3.11
conda activate tensor_risk
conda install -c apple tensorflow-deps
pip install tensorflow-macos tensorflow-metal
```

---

**Problem:** `ModuleNotFoundError: No module named 'tensor_engine'`

**Solution:**
- Ensure you're running from `streamlit/` directory
- App adds parent directory to path automatically
- If fails, manually:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Tensor_Risk_Engine"
streamlit run app.py
```

---

### Data Input Issues

**Problem:** "Intensity grid not strictly increasing"

**Cause:** Duplicate or decreasing intensity values in vulnerability file

**Solution:**
```python
# Check your x_grid
import pandas as pd
df = pd.read_csv('vulnerability.csv')
intensities = [float(col.split('_')[1].replace('g', '')) 
               for col in df.columns if 'intensity' in col]
print(sorted(intensities))  # Should be strictly ascending with no gaps
```

---

**Problem:** "Typology index exceeds max allowed"

**Cause:** Asset references typology not defined in vulnerability file

**Solution:**
- If max(u) = 5, you need at least 6 rows in vulnerability file (types 0-5)
- Either add more curves or reduce typology indices in assets

---

**Problem:** "All arrays must be of the same length" (Pandas error)

**Cause:** Fixed in latest version; occurs when N < requested top_n

**Solution:** Update app.py to latest version (includes min() checks)

---

### Computation Issues

**Problem:** App freezes during "Run Analysis"

**Possible Causes:**
1. **Very large portfolio**: N×Q > 100M elements
2. **Memory exhaustion**: Check system RAM/GPU memory
3. **TensorFlow not initialized**: Restart app

**Solutions:**
1. Reduce Q (e.g., sample events)
2. Disable gradients first run
3. Use synthetic data to test (N=100, Q=1000)
4. Check memory:
```python
import psutil
print(f"RAM: {psutil.virtual_memory().percent}%")
```

---

**Problem:** "ResourceExhaustedError" from TensorFlow

**Cause:** GPU memory insufficient

**Solutions:**
1. Reduce portfolio size
2. Force CPU:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```
3. Chunk events (process Q/10 at a time)

---

**Problem:** Very slow computation (>5 minutes for N=1000, Q=5000)

**Possible Causes:**
1. Running on CPU (no GPU detected)
2. TensorFlow compiling (@tf.function first call)
3. Swap memory usage (RAM full)

**Diagnostics:**
```python
import tensorflow as tf
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
```

**Solutions:**
- Install tensorflow-metal (macOS) or CUDA toolkit
- First run always slower (compilation), second run faster
- Close other applications to free RAM

---

### Visualization Issues

**Problem:** Plots not displaying / blank charts

**Cause:** Plotly rendering issue in browser

**Solutions:**
1. Refresh page (Ctrl+R or Cmd+R)
2. Clear browser cache
3. Try different browser (Chrome recommended)
4. Check JavaScript enabled

---

**Problem:** "NaN detected in gradients"

**Cause:** Division by zero or invalid interpolation

**Solutions:**
1. Check for zero exposure values (all v > 0)
2. Check for flat vulnerability curves (all C constant)
3. Check for extreme hazard values (H > 10.0g unusual)

---

### File I/O Issues

**Problem:** Cannot upload CSV file

**Cause:** File encoding or format issue

**Solutions:**
1. Save as UTF-8 encoded CSV
2. Check for special characters in headers
3. Ensure consistent delimiter (comma)
4. Remove BOM if present:
```python
df = pd.read_csv('file.csv', encoding='utf-8-sig')
df.to_csv('file_fixed.csv', index=False, encoding='utf-8')
```

---

**Problem:** "Invalid ZIP file" when loading analysis

**Cause:** Corrupted or incomplete download

**Solutions:**
1. Re-download save file
2. Check file size (should be > 1 KB)
3. Try saving again
4. Verify using:
```bash
unzip -t tensor_analysis_*.zip
```

---

### Performance Tips

1. **Start Small**: Test with N=100, Q=1000 before full portfolio
2. **Synthetic First**: Validate workflow before uploading real data
3. **Disable Gradients**: If only need AAL, skip gradients (2× faster)
4. **Save Often**: Save milestones to avoid re-computation
5. **Use GPU**: 10-50× speedup for large portfolios
6. **Sample Events**: If Q > 100K, consider representative subset
7. **Monitor Memory**: Use system monitor during large runs

---

## API Reference

### Session State Variables

The app uses Streamlit session state to persist data across tab switches:

```python
st.session_state.inputs           # Dict: v, u, C, x_grid, H, lambdas
st.session_state.results          # Dict: all metrics + loss_matrix
st.session_state.gradients        # Dict: grad_v, grad_C, grad_H, grad_lambdas
st.session_state.validation_status # Bool or None
st.session_state.data_source      # Str: 'uploaded', 'synthetic', 'loaded'
st.session_state.uploaded_filenames # Dict: original filenames
st.session_state.lambda_mode      # Str: 'uniform', 'exponential', 'custom'
st.session_state.computation_time # Float: seconds
st.session_state.analysis_loaded  # Bool: is this a loaded session?
st.session_state.load_metadata    # Dict: metadata from loaded ZIP
```

### Utility Functions

Located in `streamlit/utils/`:

#### `data_loader.py`

```python
load_assets_file(uploaded_file) -> (v, u, metadata_df)
load_vulnerability_file(uploaded_file) -> (C, x_grid, typology_names)
load_hazard_file(uploaded_file) -> H
load_lambdas_file(uploaded_file) -> lambdas

generate_synthetic_data(N, Q, K, M, lambda_mode) -> dict
generate_synthetic_data_advanced(N, Q, K, M, asset_categories, rp_min, rp_max, 
                                 rp_spacing, min_intensity_frequent, 
                                 max_intensity_frequent, min_intensity_rare, 
                                 max_intensity_rare) -> dict
get_portfolio_preset(preset_name) -> dict
generate_assets_template() -> BytesIO
generate_vulnerability_template() -> BytesIO
generate_hazard_template() -> BytesIO
generate_lambdas_template() -> BytesIO
```

#### `validators.py`

```python
validate_shapes(v, u, C, x_grid, H, lambdas) -> (valid, message)
validate_monotonic(x_grid) -> (valid, message)
validate_ranges(C, H, lambdas) -> (valid, message)
validate_dtypes(v, u, C, x_grid, H, lambdas) -> (converted_dict, message)
validate_all(v, u, C, x_grid, H, lambdas) -> (valid, converted, message)
estimate_memory_usage(N, Q, K, M) -> str
```

#### `visualizations.py`

All return `go.Figure` (Plotly figure objects):

```python
create_vulnerability_curves_plot(C, x_grid, H, typology_names)
create_aal_vs_exposure_scatter(aal_per_asset, v, u, typology_names)
create_exposure_distribution(v, u, typology_names)
create_aal_distribution(aal_per_asset, u, typology_names)
create_event_loss_distribution(loss_per_event)
create_vulnerability_gradient_heatmap(grad_C, x_grid, typology_names)
create_exposure_gradient_chart(grad_v, v, u, top_n, typology_names)
create_hazard_sensitivity_vs_return_period(grad_H, lambdas, sample_size)
create_hazard_gradient_heatmap(grad_H, max_assets, max_events)
create_event_contribution_plot(loss_per_event, lambdas)
create_scenario_loss_vs_rate_plot(loss_per_event, lambdas)
create_top_assets_table(aal_per_asset, v, u, top_n, typology_names)  # Returns DataFrame
create_portfolio_summary_metrics(metrics) -> dict
```

#### `persistence.py`

```python
save_analysis(inputs, results, gradients, metadata) -> BytesIO
load_analysis(uploaded_file) -> dict
validate_loaded_data(data_dict) -> (valid, message)
create_metadata(inputs, results, gradients, ...) -> dict
```

---

## Frequently Asked Questions

### General

**Q: Can I use this for commercial projects?**

A: Yes, subject to the MIT license terms. Attribution appreciated.

---

**Q: What peril types are supported?**

A: Any peril where:
- Intensity can be quantified (g, mph, m depth, etc.)
- Vulnerability curves exist (MDR vs intensity)
- Hazard can be modeled spatially

Currently demonstrated for earthquake; easily adaptable for hurricane, flood, etc.

---

**Q: How does this compare to HAZUS/RiskScape/OpenQuake?**

A: 
- **HAZUS**: Similar methodology, but we add full differentiability
- **RiskScape**: Similar modularity, we add gradient analysis
- **OpenQuake**: Can use OpenQuake hazard as input to our engine
- **Unique**: Gradient-based optimization not available in traditional tools

---

### Technical

**Q: Why TensorFlow instead of NumPy?**

A: Automatic differentiation. TensorFlow's GradientTape computes exact gradients without finite differences, enabling sensitivity analysis at scale.

---

**Q: Can I run without GPU?**

A: Yes, CPU works fine. GPU provides 10-50× speedup for large portfolios but isn't required.

---

**Q: Maximum portfolio size?**

A: 
- **GPU (16GB)**: N=10K, Q=100K (~4GB loss matrix)
- **CPU (64GB RAM)**: Similar, but slower
- **Workaround**: Event chunking (process Q/10 at a time)

---

**Q: Why are some visualizations sampled?**

A: Performance. Rendering 10K×100K heatmap (1B points) crashes browsers. Sampling preserves patterns while maintaining responsiveness.

---

### Data

**Q: Where to get vulnerability curves?**

A:
- **HAZUS**: Free FEMA methodology
- **Literature**: Spence et al., Porter et al.
- **Calibration**: Fit to historical losses (use our gradients!)
- **Engineering judgment**: Consult structural engineers

---

**Q: Where to get hazard?**

A:
- **OpenQuake**: Open-source seismic hazard
- **USGS**: ShakeMaps for scenarios
- **Commercial**: AIR, RMS models
- **Custom**: Your own modeling

---

**Q: Can I use this with OpenQuake outputs?**

A: Yes! Export OpenQuake ground motion fields as CSV, format as hazard matrix, upload.

---

**Q: What if I have correlation between events?**

A: Current version treats events independently. Correlation (future version) would modify the loss matrix aggregation.

---

### Methodological

**Q: Why interpolation instead of bins?**

A: Differentiability. Interpolation is continuous and smooth, enabling gradients. Bins create discontinuities that break autodiff.

---

**Q: Are gradients exact or approximate?**

A: Exact (within numerical precision). TensorFlow's reverse-mode autodiff computes true derivatives, not finite differences.

---

**Q: Can I optimize multiple parameters simultaneously?**

A: Yes! Gradients for v, C, H, λ computed in one pass. Use gradient descent:
```
v_new = v - learning_rate * ∂AAL/∂v
```

---

**Q: What about epistemic uncertainty?**

A: Use gradients to propagate:
```
Var(AAL) ≈ Σ (∂AAL/∂θᵢ)² Var(θᵢ)
```
where θᵢ are uncertain parameters.

---

## Glossary

- **AAL**: Average Annual Loss - Rate-weighted expected loss per year
- **MDR**: Mean Damage Ratio - Expected fraction of replacement cost lost given intensity
- **Intensity**: Hazard severity (e.g., g for earthquake, mph for wind)
- **Typology**: Building class with common vulnerability (e.g., "wood frame")
- **Gradient**: Derivative of AAL w.r.t. parameter (sensitivity)
- **Occurrence Rate (λ)**: Events per year for a scenario
- **Total Rate (Λ)**: Sum of all occurrence rates (Σλ), expected damaging events per year
- **Return Period (RP)**: 1/λ, average time between events of a given magnitude
- **RP Spacing**: Distribution of return periods across event catalog
  - **Exponential Spacing**: Logarithmic distribution mimicking seismic recurrence (many frequent, few rare)
  - **Linear Spacing**: Uniform distribution across RP range (equal spacing)
- **Asset Category**: Portfolio subdivision by characteristics (cost range, typology, location)
- **City-Representative Portfolio**: Synthetic portfolio with realistic category-based structure
- **RP-Intensity Coupling**: Inverse relationship where frequent events have low intensity, rare events have high intensity
- **Stochastic Event**: Simulated scenario from probabilistic catalog
- **Loss Matrix**: N×Q matrix where J[i,q] = loss of asset i in event q
- **Exposure**: Replacement cost or insured value of asset
- **Portfolio**: Collection of all assets being analyzed
- **GPU**: Graphics Processing Unit - accelerates tensor operations
- **Autodiff**: Automatic differentiation - computing gradients programmatically

---

## Changelog

### Version 1.1 (February 2026)

**Major Update: Dual-Mode Synthetic Portfolio Generation**

New Features:
- **Advanced Mode** for city-representative portfolios
  - Category-based portfolio structure (1-5 asset categories)
  - Control percentage, cost range, and typology per category
  - Three preset templates (Residential City, High-Risk Zone, Commercial District)
- **RP Spacing Modes**: Exponential (realistic) vs Linear (educational)
- **Total Rate (Λ) Display** with color-coded interpretation
  - Green (Λ < 0.3): Realistic seismicity
  - Yellow (0.3 ≤ Λ < 1.0): High seismicity
  - Red (Λ ≥ 1.0): Unrealistic - AAL may exceed portfolio value
- **Educational Tooltips** explaining spacing modes and total rate implications
- **RP-Dependent Intensity Coupling** for realistic seismic behavior

Enhancements:
- Comprehensive documentation for teaching catastrophe modeling
- Best practices guide for synthetic portfolio generation
- Example workflows for educational use
- Expanded glossary with new concepts

Bug Fixes:
- Fixed hazard gradient visualization indexing error
- Improved session state management for advanced mode

### Version 1.0 (February 2026)

**Initial Release**

Features:
- Complete web interface with 5 tabs
- Support for CSV/XLSX uploads
- Simple synthetic data generator
- 15+ interactive Plotly visualizations
- Full gradient analysis (∂AAL/∂v, ∂AAL/∂C, ∂AAL/∂H, ∂AAL/∂λ)
- Save/load as ZIP archives
- Retrofit optimizer
- Real-time validation
- Template downloads
- Comprehensive error handling

Known Limitations:
- No built-in scenario comparison mode
- Manual batch processing
- Single-peril only
- No spatial correlation

Planned for v1.2:
- Event chunking for Q > 1M
- Side-by-side comparison mode
- Multi-peril support
- Enhanced retrofit models

---

## License

MIT License - See main repository LICENSE file

---

## Support & Contact

- **GitHub Issues**: [Report bugs](https://github.com/yourusername/Tensor_Risk_Engine/issues)
- **Documentation**: This file + [API_DOCUMENTATION.md](../Documentation/API_DOCUMENTATION.md)
- **Examples**: See [minimum_example_*.py](../examples/)

---

**Last Updated:** February 11, 2026

**Maintained by:** [Your Name/Team]

**Built with:** Streamlit • TensorFlow • Plotly • NumPy

