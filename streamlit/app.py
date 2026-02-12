"""
Tensor Risk Engine - Streamlit Web Application
===============================================
Interactive web interface for catastrophe risk assessment with full gradient analysis.

Author: Tensor Risk Engine Team
Date: February 2026
"""

import sys
import os
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

from tensor_engine import TensorialRiskEngine
from utils.data_loader import (
    load_assets_file, load_vulnerability_file, load_hazard_file, 
    load_lambdas_file, generate_synthetic_data, generate_synthetic_data_advanced,
    get_portfolio_preset, generate_assets_template, generate_vulnerability_template,
    generate_hazard_template, generate_lambdas_template
)
from utils.validators import (
    validate_all, estimate_memory_usage, validate_asset_categories, 
    validate_intensity_ranges
)
from utils.visualizations import *
from utils.persistence import save_analysis, load_analysis, validate_loaded_data, create_metadata

# Page configuration
st.set_page_config(
    page_title="Tensor Risk Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    if 'inputs' not in st.session_state:
        st.session_state.inputs = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'gradients' not in st.session_state:
        st.session_state.gradients = None
    if 'validation_status' not in st.session_state:
        st.session_state.validation_status = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None
    if 'uploaded_filenames' not in st.session_state:
        st.session_state.uploaded_filenames = {}
    if 'lambda_mode' not in st.session_state:
        st.session_state.lambda_mode = 'uniform'
    if 'computation_time' not in st.session_state:
        st.session_state.computation_time = 0.0
    if 'analysis_loaded' not in st.session_state:
        st.session_state.analysis_loaded = False
    if 'load_metadata' not in st.session_state:
        st.session_state.load_metadata = None

init_session_state()

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Tensor Risk Engine")
    st.markdown("v1.0 - February 2026")
    st.markdown("---")
    
    # Quick Load Section
    st.markdown("#### 🚀 Quick Load")
    quick_load_file = st.file_uploader(
        "Load Saved Analysis",
        type=['zip'],
        key='quick_load',
        help="Upload a previously saved analysis ZIP file"
    )
    
    if quick_load_file is not None:
        try:
            with st.spinner("Loading analysis..."):
                loaded_data = load_analysis(quick_load_file)
                valid, message = validate_loaded_data(loaded_data)
                
                if valid:
                    st.session_state.inputs = loaded_data['inputs']
                    st.session_state.results = loaded_data['results']
                    st.session_state.gradients = loaded_data['gradients']
                    st.session_state.load_metadata = loaded_data['metadata']
                    st.session_state.analysis_loaded = True
                    st.session_state.data_source = 'loaded'
                    st.success("✓ Analysis loaded successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Invalid analysis file: {message}")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 Reset All", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session_state()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📚 Documentation")
    st.markdown("[API Documentation](../Documentation/API_DOCUMENTATION.md)")
    st.markdown("[Tensor Engine Code](../tensor_engine.py)")
    
    # Show loaded analysis info
    if st.session_state.analysis_loaded and st.session_state.load_metadata:
        st.markdown("---")
        st.markdown("### 📂 Loaded Analysis")
        meta = st.session_state.load_metadata
        st.text(f"Date: {meta['timestamp'][:10]}")
        st.text(f"Assets: {meta['dimensions']['N']}")
        st.text(f"Events: {meta['dimensions']['Q']}")
        st.text(f"Typologies: {meta['dimensions']['K']}")
        st.text(f"AAL: ${meta['aal_portfolio']:,.0f}")

# Main content
st.markdown('<div class="main-header">Tensor Risk Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Differentiable Catastrophe Risk Assessment with Automatic Gradient Computation</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Setup", 
    "📥 Inputs", 
    "⚡ Run Analysis", 
    "📊 Results Dashboard", 
    "🎯 Gradients & Sensitivity"
])

# ===========================
# TAB 1: SETUP
# ===========================
with tab1:
    st.header("Welcome to Tensor Risk Engine")
    
    st.markdown("""
    This application provides a **fully differentiable catastrophe risk assessment framework** with:
    
    - 🔢 **Vectorized Operations**: GPU-accelerated TensorFlow computations
    - 📈 **Automatic Gradients**: Sensitivity w.r.t. all input parameters
    - 🎲 **Stochastic Events**: Support for large event catalogs (Q > 100,000)
    - 🏗️ **Multi-Typology**: Multiple building vulnerability classes
    - 📊 **Rate-Weighted Metrics**: Non-uniform scenario occurrence rates
    
    ---
    """)
    
    # Data source selection
    st.subheader("Select Data Source")
    
    data_option = st.radio(
        "How would you like to provide data?",
        ["📤 Upload CSV/XLSX Files", "🎲 Generate Synthetic Data", "📂 Load Saved Analysis"],
        index=0 if not st.session_state.analysis_loaded else 2
    )
    
    if data_option == "🎲 Generate Synthetic Data":
        st.markdown("### Synthetic Portfolio Generator")
        
        # Mode selector
        gen_mode = st.radio(
            "Generation Mode",
            ["🎯 Simple (Random)", "🏙️ Advanced (City-Representative)"],
            help="Simple: fully random portfolio. Advanced: control asset categories and hazard ranges."
        )
        
        if gen_mode == "🎯 Simple (Random)":
            # Simple mode - original behavior
            col1, col2 = st.columns(2)
            
            with col1:
                N = st.slider("Number of Assets (N)", 100, 10000, 1000, 100)
                Q = st.slider("Number of Events (Q)", 100, 100000, 5000, 100)
            
            with col2:
                K = st.slider("Number of Typologies (K)", 3, 10, 5, 1)
                M = st.slider("Curve Points (M)", 10, 50, 20, 5)
            
            lambda_dist = st.selectbox(
                "Lambda Distribution",
                ["uniform", "exponential"],
                help="Uniform: equal rates for all events. Exponential: realistic seismic recurrence."
            )
            
            # Memory estimate
            mem_est = estimate_memory_usage(N, Q, K, M)
            st.info(f"💾 Estimated memory usage: {mem_est}")
            
            if st.button("🎲 Generate Simple Portfolio", type="primary", use_container_width=True):
                with st.spinner("Generating synthetic data..."):
                    synthetic_data = generate_synthetic_data(N, Q, K, M, lambda_dist)
                    st.session_state.inputs = synthetic_data
                    st.session_state.data_source = 'synthetic'
                    st.session_state.lambda_mode = lambda_dist
                    st.session_state.uploaded_filenames = {}
                    st.success(f"✓ Generated portfolio: {N} assets, {Q} events, {K} typologies")
        
        else:
            # Advanced mode - category-based generation
            col1, col2 = st.columns(2)
            
            with col1:
                N = st.slider("Number of Assets (N)", 100, 10000, 1000, 100, key='adv_n')
                Q = st.slider("Number of Events (Q)", 100, 100000, 5000, 100, key='adv_q')
            
            with col2:
                K = st.slider("Number of Typologies (K)", 3, 10, 5, 1, key='adv_k')
                M = st.slider("Curve Points (M)", 10, 50, 20, 5, key='adv_m')
            
            # Preset selector
            st.markdown("#### 📋 Portfolio Template")
            preset_name = st.selectbox(
                "Choose Preset or Customize",
                ["Residential City", "High-Risk Zone", "Commercial District", "Custom"],
                help="Select a preset or choose Custom to define your own categories"
            )
            
            # Initialize categories from preset or session state
            if preset_name != "Custom":
                preset_config = get_portfolio_preset(preset_name)
                categories = preset_config['categories']
                intensity_ranges = preset_config['intensity_ranges']
                rp_min, rp_max = preset_config['rp_min'], preset_config['rp_max']
                rp_spacing = preset_config.get('rp_spacing', 'exponential')
            else:
                # Custom mode - use session state or defaults
                if 'custom_categories' not in st.session_state:
                    st.session_state.custom_categories = [
                        {'name': 'Category 1', 'percentage': 100, 'cost_min': 100, 'cost_max': 500, 'typology': 'random'}
                    ]
                categories = st.session_state.custom_categories
                rp_min, rp_max = 32.0, 5000.0
                rp_spacing = 'exponential'
                intensity_ranges = {
                    'min_intensity_frequent': 0.02,
                    'max_intensity_frequent': 0.06,
                    'min_intensity_rare': 0.85,
                    'max_intensity_rare': 0.95
                }
            
            # Asset categories section
            with st.expander("📊 Asset Categories", expanded=(preset_name == "Custom")):
                if preset_name == "Custom":
                    # Add/remove category buttons
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("➕ Add Category") and len(categories) < 5:
                            categories.append({
                                'name': f'Category {len(categories)+1}',
                                'percentage': 0,
                                'cost_min': 100,
                                'cost_max': 500,
                                'typology': 'random'
                            })
                            st.session_state.custom_categories = categories
                            st.rerun()
                    
                    with col_btn2:
                        if st.button("➖ Remove Last") and len(categories) > 1:
                            categories.pop()
                            st.session_state.custom_categories = categories
                            st.rerun()
                    
                    st.markdown("---")
                
                # Display/edit categories
                edited_categories = []
                for i, cat in enumerate(categories):
                    st.markdown(f"**{cat['name']}**")
                    cat_col1, cat_col2, cat_col3 = st.columns(3)
                    
                    with cat_col1:
                        pct = st.slider(
                            "Percentage (%)",
                            0, 100, cat['percentage'],
                            key=f"cat_{i}_pct",
                            disabled=(preset_name != "Custom")
                        )
                    
                    with cat_col2:
                        cost_min = st.number_input(
                            "Min Cost ($k)",
                            min_value=1, max_value=10000, value=cat['cost_min'],
                            key=f"cat_{i}_min",
                            disabled=(preset_name != "Custom")
                        )
                    
                    with cat_col3:
                        cost_max = st.number_input(
                            "Max Cost ($k)",
                            min_value=1, max_value=10000, value=cat['cost_max'],
                            key=f"cat_{i}_max",
                            disabled=(preset_name != "Custom")
                        )
                    
                    # Typology selector
                    typology_options = ['random'] + list(range(K))
                    typology_display = ['Random'] + [f'Type {k}' for k in range(K)]
                    
                    if isinstance(cat['typology'], list):
                        default_typ_idx = 0  # Random
                    elif cat['typology'] == 'random':
                        default_typ_idx = 0
                    else:
                        default_typ_idx = cat['typology'] + 1
                    
                    typology = st.selectbox(
                        "Typology",
                        typology_display,
                        index=default_typ_idx,
                        key=f"cat_{i}_typ",
                        disabled=(preset_name != "Custom")
                    )
                    
                    # Convert typology back to proper format
                    if typology == 'Random':
                        typ_value = 'random'
                    else:
                        typ_value = int(typology.split()[1])
                    
                    edited_categories.append({
                        'name': cat['name'],
                        'percentage': pct,
                        'cost_min': cost_min,
                        'cost_max': cost_max,
                        'typology': typ_value
                    })
                    
                    if i < len(categories) - 1:
                        st.markdown("---")
                
                if preset_name == "Custom":
                    st.session_state.custom_categories = edited_categories
                
                categories = edited_categories
                
                # Show total percentage
                total_pct = sum(cat['percentage'] for cat in categories)
                if abs(total_pct - 100) < 1:
                    st.success(f"✓ Total: {total_pct}%")
                else:
                    st.warning(f"⚠️ Total: {total_pct}% (must be 100%)")
            
            # Hazard intensity ranges section
            with st.expander("🌍 Hazard Intensity Ranges", expanded=False):
                st.markdown("**Return Period Range**")
                col_rp1, col_rp2 = st.columns(2)
                with col_rp1:
                    rp_min = st.number_input("Min RP (years)", min_value=1.0, max_value=10000.0, 
                                             value=rp_min, key='rp_min')
                with col_rp2:
                    rp_max = st.number_input("Max RP (years)", min_value=1.0, max_value=10000.0, 
                                             value=rp_max, key='rp_max')
                
                st.markdown("**Return Period Spacing**")
                rp_spacing = st.radio(
                    "Spacing Mode",
                    ["exponential", "linear"],
                    index=0 if rp_spacing == 'exponential' else 1,
                    horizontal=True,
                    help="**Exponential** (realistic): Many frequent events, few rare events. " +
                         "Creates higher total rates (Λ). Use for realistic seismic catalogs.\n\n" +
                         "**Linear** (teaching): Events uniformly distributed across RP range. " +
                         "Creates lower total rates. Use for educational exploration.",
                    key='rp_spacing'
                )
                
                st.markdown("**Frequent Events (Low RP)**")
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    min_int_freq = st.number_input("Min Intensity (g)", min_value=0.0, max_value=2.0, 
                                                   value=intensity_ranges['min_intensity_frequent'], 
                                                   step=0.01, format="%.3f", key='min_int_freq')
                with col_f2:
                    max_int_freq = st.number_input("Max Intensity (g)", min_value=0.0, max_value=2.0, 
                                                   value=intensity_ranges['max_intensity_frequent'],
                                                   step=0.01, format="%.3f", key='max_int_freq')
                
                st.markdown("**Rare Events (High RP)**")
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    min_int_rare = st.number_input("Min Intensity (g)", min_value=0.0, max_value=2.0, 
                                                   value=intensity_ranges['min_intensity_rare'],
                                                   step=0.01, format="%.3f", key='min_int_rare')
                with col_r2:
                    max_int_rare = st.number_input("Max Intensity (g)", min_value=0.0, max_value=2.0, 
                                                   value=intensity_ranges['max_intensity_rare'],
                                                   step=0.01, format="%.3f", key='max_int_rare')
                
                st.info("💡 Realistic: Frequent events have low intensities, rare events have high intensities")
            
            # Memory estimate
            mem_est = estimate_memory_usage(N, Q, K, M)
            st.info(f"💾 Estimated memory usage: {mem_est}")
            
            # Validation and generation
            if st.button("🏙️ Generate City-Representative Portfolio", type="primary", use_container_width=True):
                # Validate categories
                valid_cat, msg_cat = validate_asset_categories(categories, K)
                if not valid_cat:
                    st.error(msg_cat)
                else:
                    # Validate intensity ranges
                    valid_int, msg_int = validate_intensity_ranges(
                        min_int_freq, max_int_freq, min_int_rare, max_int_rare
                    )
                    if not valid_int:
                        st.error(msg_int)
                    else:
                        with st.spinner("Generating city-representative portfolio..."):
                            synthetic_data = generate_synthetic_data_advanced(
                                N=N, Q=Q, K=K, M=M,
                                asset_categories=categories,
                                rp_min=rp_min,
                                rp_max=rp_max,
                                rp_spacing=rp_spacing,
                                min_intensity_frequent=min_int_freq,
                                max_intensity_frequent=max_int_freq,
                                min_intensity_rare=min_int_rare,
                                max_intensity_rare=max_int_rare
                            )
                            st.session_state.inputs = synthetic_data
                            st.session_state.data_source = 'synthetic_advanced'
                            st.session_state.lambda_mode = rp_spacing
                            st.session_state.uploaded_filenames = {}
                            st.session_state.portfolio_config = {
                                'categories': categories,
                                'rp_min': rp_min,
                                'rp_max': rp_max,
                                'rp_spacing': rp_spacing,
                                'intensity_ranges': {
                                    'min_intensity_frequent': min_int_freq,
                                    'max_intensity_frequent': max_int_freq,
                                    'min_intensity_rare': min_int_rare,
                                    'max_intensity_rare': max_int_rare
                                }
                            }
                            st.success(f"✓ Generated city-representative portfolio: {N} assets, {Q} events")
                            st.info(f"✓ {len(categories)} asset categories with RP-dependent intensities")
                            
                            # Display total rate with educational context
                            total_rate = synthetic_data['total_rate']
                            avg_inter_event = 1.0 / total_rate if total_rate > 0 else float('inf')
                            
                            if total_rate < 0.3:
                                st.success(f"📊 Total Rate (Λ): {total_rate:.4f} - One event every {avg_inter_event:.1f} years (realistic seismicity)")
                            elif total_rate < 1.0:
                                st.warning(f"📊 Total Rate (Λ): {total_rate:.4f} - One event every {avg_inter_event:.1f} years (high seismicity)")
                            else:
                                st.error(f"📊 Total Rate (Λ): {total_rate:.4f} - {total_rate:.1f} events per year (unrealistic - AAL may exceed portfolio value!)")
                            
                            # Educational comparison
                            with st.expander("ℹ️ Understanding Total Rate (Λ)"):
                                st.markdown(f"""
                                **Current Configuration:**
                                - RP Range: [{rp_min:.0f}, {rp_max:.0f}] years
                                - Spacing: {rp_spacing.upper()}
                                - Total Rate Λ = Σλ = {total_rate:.4f}
                                - Average Inter-Event Time: {avg_inter_event:.1f} years
                                
                                **Interpretation:**
                                - Λ < 0.3: Realistic for most seismic regions (event every 3+ years)
                                - Λ ≈ 0.5: High seismicity zone (event every 2 years)
                                - Λ ≈ 1.0: Extreme seismicity (one event per year)
                                - Λ > 1.0: Unrealistic (multiple events yearly, AAL > exposure)
                                
                                **Teaching Note:**
                                Exponential spacing creates denser event clusters at low RP values 
                                (realistic), while linear spacing distributes events uniformly 
                                (better for exploring parameter sensitivity).
                                """)
    
    elif data_option == "📤 Upload CSV/XLSX Files":
        st.markdown("### Template Downloads")
        st.markdown("Download these templates to see the expected file formats:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📄 Assets Template",
                data=generate_assets_template(),
                file_name="assets_template.csv",
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                "📄 Vulnerability Template",
                data=generate_vulnerability_template(),
                file_name="vulnerability_template.csv",
                mime="text/csv"
            )
        
        with col3:
            st.download_button(
                "📄 Hazard Template",
                data=generate_hazard_template(),
                file_name="hazard_template.csv",
                mime="text/csv"
            )
        
        st.download_button(
            "📄 Lambda Rates Template",
            data=generate_lambdas_template(),
            file_name="lambdas_template.csv",
            mime="text/csv"
        )
        
        st.info("ℹ️ Proceed to the **Inputs** tab to upload your files.")
    
    elif data_option == "📂 Load Saved Analysis":
        st.markdown("### Load Previously Saved Analysis")
        
        load_file = st.file_uploader(
            "Upload Analysis ZIP File",
            type=['zip'],
            key='main_load',
            help="Upload a complete analysis saved from a previous session"
        )
        
        if load_file is not None:
            try:
                with st.spinner("Loading and validating analysis..."):
                    loaded_data = load_analysis(load_file)
                    valid, message = validate_loaded_data(loaded_data)
                    
                    if valid:
                        st.session_state.inputs = loaded_data['inputs']
                        st.session_state.results = loaded_data['results']
                        st.session_state.gradients = loaded_data['gradients']
                        st.session_state.load_metadata = loaded_data['metadata']
                        st.session_state.analysis_loaded = True
                        st.session_state.data_source = 'loaded'
                        
                        st.success(message)
                        
                        # Show metadata
                        st.markdown("### Loaded Analysis Summary")
                        meta = loaded_data['metadata']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Assets (N)", meta['dimensions']['N'])
                        col2.metric("Events (Q)", meta['dimensions']['Q'])
                        col3.metric("Typologies (K)", meta['dimensions']['K'])
                        col4.metric("Curve Points (M)", meta['dimensions']['M'])
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Portfolio AAL", f"${meta['aal_portfolio']:,.2f}")
                        col2.metric("Total Rate (Λ)", f"{meta['total_rate']:.6f}")
                        
                        st.text(f"Original timestamp: {meta['timestamp']}")
                        st.text(f"Data source: {meta['data_source']}")
                        st.text(f"Lambda mode: {meta['lambda_mode']}")
                        st.text(f"Gradients computed: {meta['gradients_computed']}")
                        
                        if not meta['gradients_computed']:
                            st.warning("⚠️ This analysis does not include gradients. You can recompute them in the 'Run Analysis' tab.")
                    else:
                        st.error(f"❌ {message}")
            
            except Exception as e:
                st.error(f"❌ Error loading analysis: {str(e)}")

# ===========================
# TAB 2: INPUTS
# ===========================
with tab2:
    st.header("Data Inputs")
    
    if st.session_state.analysis_loaded:
        st.info("📂 Analysis loaded from file. Data shown below is read-only.")
        
        # Show loaded inputs
        inputs = st.session_state.inputs
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Assets (N)", len(inputs['v']))
        col2.metric("Events (Q)", inputs['H'].shape[1])
        col3.metric("Typologies (K)", inputs['C'].shape[0])
        col4.metric("Curve Points (M)", len(inputs['x_grid']))
        
        with st.expander("View Exposure & Typology Data"):
            df = pd.DataFrame({
                'Asset ID': range(len(inputs['v'])),
                'Exposure ($)': inputs['v'],
                'Typology': inputs['u']
            })
            st.dataframe(df.head(100), use_container_width=True)
        
        with st.expander("View Vulnerability Curves"):
            st.text(f"Shape: {inputs['C'].shape} (K typologies × M points)")
            st.dataframe(pd.DataFrame(inputs['C']), use_container_width=True)
        
        with st.expander("View Hazard Matrix"):
            st.text(f"Shape: {inputs['H'].shape} (N assets × Q events)")
            st.dataframe(pd.DataFrame(inputs['H'][:50, :10]), use_container_width=True)
            st.caption("Showing first 50 assets × 10 events")
    
    elif st.session_state.data_source == 'synthetic':
        st.success("✓ Synthetic data generated. View summary below:")
        
        inputs = st.session_state.inputs
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Assets (N)", len(inputs['v']))
        col2.metric("Events (Q)", inputs['H'].shape[1])
        col3.metric("Typologies (K)", inputs['C'].shape[0])
        col4.metric("Curve Points (M)", len(inputs['x_grid']))
        
        with st.expander("Preview Data"):
            st.subheader("Exposure Vector (first 10)")
            st.write(inputs['v'][:10])
            
            st.subheader("Vulnerability Curves")
            st.dataframe(pd.DataFrame(inputs['C']), use_container_width=True)
    
    else:
        st.markdown("### Upload Portfolio Data Files")
        
        # File uploaders
        col1, col2 = st.columns(2)
        
        with col1:
            assets_file = st.file_uploader(
                "1️⃣ Assets File (exposure & typology)",
                type=['csv', 'xlsx'],
                help="Required: exposure and typology columns"
            )
            
            vuln_file = st.file_uploader(
                "2️⃣ Vulnerability Curves File",
                type=['csv', 'xlsx'],
                help="K rows (typologies) × M columns (intensity points)"
            )
        
        with col2:
            hazard_file = st.file_uploader(
                "3️⃣ Hazard Matrix File",
                type=['csv', 'xlsx'],
                help="N rows (assets) × Q columns (events)"
            )
            
            use_custom_lambdas = st.checkbox("Use custom scenario rates (λ)", value=False)
            
            if use_custom_lambdas:
                lambdas_file = st.file_uploader(
                    "4️⃣ Scenario Rates File (optional)",
                    type=['csv', 'xlsx'],
                    help="Q values: occurrence rate for each event"
                )
            else:
                lambdas_file = None
                st.info("Using uniform rates: λ_q = 1/Q for all events")
        
        # Process uploads
        if st.button("📤 Load and Validate Files", type="primary", use_container_width=True):
            try:
                if assets_file is None or vuln_file is None or hazard_file is None:
                    st.error("❌ Please upload at least Assets, Vulnerability, and Hazard files")
                else:
                    with st.spinner("Loading files..."):
                        # Load data
                        v, u, assets_df = load_assets_file(assets_file)
                        C, x_grid, typology_names = load_vulnerability_file(vuln_file)
                        H = load_hazard_file(hazard_file)
                        
                        if lambdas_file is not None:
                            lambdas = load_lambdas_file(lambdas_file)
                            lambda_mode = 'custom'
                        else:
                            lambdas = None
                            lambda_mode = 'uniform'
                        
                        # Validate
                        valid, converted, message = validate_all(v, u, C, x_grid, H, lambdas)
                        
                        st.text(message)
                        
                        if valid:
                            # Create uniform lambdas if not provided
                            if converted['lambdas'] is None:
                                Q = converted['H'].shape[1]
                                converted['lambdas'] = np.ones(Q, dtype=np.float32) / Q
                            
                            st.session_state.inputs = converted
                            st.session_state.data_source = 'uploaded'
                            st.session_state.lambda_mode = lambda_mode
                            st.session_state.uploaded_filenames = {
                                'assets': assets_file.name,
                                'vulnerability': vuln_file.name,
                                'hazard': hazard_file.name,
                                'lambdas': lambdas_file.name if lambdas_file else None
                            }
                            st.success("✓ All files loaded and validated successfully!")
                        else:
                            st.error("❌ Validation failed. Please check your data and try again.")
            
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")

# ===========================
# TAB 3: RUN ANALYSIS
# ===========================
with tab3:
    st.header("Run Risk Analysis")
    
    if st.session_state.inputs is None:
        st.warning("⚠️ Please provide input data first (Setup or Inputs tab)")
    else:
        inputs = st.session_state.inputs
        
        # Show configuration
        st.markdown("### Current Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        N = inputs['v'].shape[0]
        Q = inputs['H'].shape[1]
        K = inputs['C'].shape[0]
        M = inputs['x_grid'].shape[0]
        
        col1.metric("Assets (N)", N)
        col2.metric("Events (Q)", Q)
        col3.metric("Typologies (K)", K)
        col4.metric("Curve Points (M)", M)
        
        # Lambda info
        if inputs['lambdas'] is not None:
            Lambda_total = float(np.sum(inputs['lambdas']))
            st.text(f"Total occurrence rate (Λ): {Lambda_total:.6f} events/year")
        else:
            st.text("Using uniform occurrence rates: λ_q = 1/Q")
        
        # Memory estimate
        mem_est = estimate_memory_usage(N, Q, K, M)
        st.info(f"💾 Estimated memory usage: {mem_est}")
        
        # Analysis options
        st.markdown("### Analysis Options")
        
        compute_gradients = st.checkbox(
            "Compute Gradients",
            value=True,
            help="Enable sensitivity analysis (adds ~50% computation time)"
        )
        
        # Run button
        if st.button("⚡ Run Risk Analysis", type="primary", use_container_width=True):
            with st.spinner("Initializing Tensor Risk Engine..."):
                # Initialize engine
                engine = TensorialRiskEngine(
                    v=inputs['v'],
                    u=inputs['u'],
                    C=inputs['C'],
                    x_grid=inputs['x_grid'],
                    H=inputs['H'],
                    lambdas=inputs['lambdas']
                )
                
                st.success("✓ Engine initialized")
            
            # Compute loss and metrics
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Computing loss matrix...")
                progress_bar.progress(33)
                
                start_time = time.time()
                
                J_matrix, metrics = engine.compute_loss_and_metrics()
                
                # Convert to NumPy
                results = {
                    'loss_matrix': J_matrix.numpy(),
                    'aal_per_asset': metrics['aal_per_asset'].numpy(),
                    'aal_portfolio': float(metrics['aal_portfolio'].numpy()),
                    'mean_per_event_per_asset': metrics['mean_per_event_per_asset'].numpy(),
                    'variance_per_asset': metrics['variance_per_asset'].numpy(),
                    'std_per_asset': metrics['std_per_asset'].numpy(),
                    'loss_per_event': metrics['loss_per_event'].numpy(),
                    'total_rate': float(metrics['total_rate'].numpy())
                }
                
                progress_bar.progress(66)
                status_text.text("Metrics computed. Computing gradients...")
                
                # Compute gradients if requested
                gradients = None
                if compute_gradients:
                    analysis = engine.full_gradient_analysis()
                    gradients = {
                        'grad_exposure': analysis['grad_exposure'].numpy(),
                        'grad_vulnerability': analysis['grad_vulnerability'].numpy(),
                        'grad_hazard': analysis['grad_hazard'].numpy(),
                        'grad_lambdas': analysis['grad_lambdas'].numpy()
                    }
                
                end_time = time.time()
                computation_time = end_time - start_time
                
                progress_bar.progress(100)
                status_text.text("✓ Analysis complete!")
                
                # Store results
                st.session_state.results = results
                st.session_state.gradients = gradients
                st.session_state.computation_time = computation_time
                
                # Display summary
                st.success(f"✓ Analysis completed in {computation_time:.2f} seconds")
                
                st.markdown("### Results Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Portfolio AAL", f"${results['aal_portfolio']:,.2f}")
                col2.metric("Mean Loss/Event", f"${np.mean(results['loss_per_event']):,.2f}")
                col3.metric("Max Event Loss", f"${np.max(results['loss_per_event']):,.2f}")
                
                st.info("📊 View detailed results in the 'Results Dashboard' tab")
            
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                progress_bar.empty()
                status_text.empty()

# ===========================
# TAB 4: RESULTS DASHBOARD
# ===========================
with tab4:
    st.header("Results Dashboard")
    
    if st.session_state.results is None:
        st.warning("⚠️ No results available. Please run analysis first.")
    else:
        results = st.session_state.results
        inputs = st.session_state.inputs
        
        # Loaded analysis indicator
        if st.session_state.analysis_loaded:
            st.info(f"📂 Loaded from: {st.session_state.load_metadata['timestamp']}")
        
        # Portfolio Summary
        st.markdown("### 📈 Portfolio Summary Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Portfolio AAL", f"${results['aal_portfolio']:,.0f}")
        col2.metric("Total Rate (Λ)", f"{results['total_rate']:.6f}")
        col3.metric("Mean Loss/Event", f"${np.mean(results['loss_per_event']):,.0f}")
        col4.metric("Max Event Loss", f"${np.max(results['loss_per_event']):,.0f}")
        col5.metric("Total Exposure", f"${np.sum(inputs['v']):,.0f}")
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### 📊 Risk Visualizations")
        
        # Vulnerability curves
        with st.expander("🔵 Vulnerability Curves", expanded=True):
            fig1 = create_vulnerability_curves_plot(
                inputs['C'], inputs['x_grid'], inputs['H']
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # AAL vs Exposure
        with st.expander("🔴 AAL vs. Exposure Scatter"):
            fig2 = create_aal_vs_exposure_scatter(
                results['aal_per_asset'], inputs['v'], inputs['u']
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📊 Exposure Distribution"):
                fig3 = create_exposure_distribution(inputs['v'], inputs['u'])
                st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            with st.expander("📊 AAL Distribution"):
                fig4 = create_aal_distribution(results['aal_per_asset'], inputs['u'])
                st.plotly_chart(fig4, use_container_width=True)
        
        # Event loss distribution
        with st.expander("📊 Event Loss Distribution"):
            fig5 = create_event_loss_distribution(results['loss_per_event'])
            st.plotly_chart(fig5, use_container_width=True)
        
        # Top assets table
        st.markdown("### 🏆 Top Assets by AAL")
        n_assets = len(results['aal_per_asset'])
        display_n = min(20, n_assets)  # Show top 20 or all if fewer
        top_assets_df = create_top_assets_table(
            results['aal_per_asset'], inputs['v'], inputs['u'], top_n=display_n
        )
        st.dataframe(top_assets_df, use_container_width=True, hide_index=True)
        
        # Export section
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export results CSV
            results_df = pd.DataFrame({
                'Asset_ID': range(len(results['aal_per_asset'])),
                'AAL': results['aal_per_asset'],
                'Exposure': inputs['v'],
                'Typology': inputs['u'],
                'AAL_Ratio': results['aal_per_asset'] / inputs['v'],
                'Std_Dev': results['std_per_asset']
            })
            
            csv_buffer = results_df.to_csv(index=False).encode()
            st.download_button(
                "📄 Download Results CSV",
                data=csv_buffer,
                file_name=f"risk_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export summary report
            summary_text = f"""Tensor Risk Engine - Analysis Summary
=====================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Source: {st.session_state.data_source}

Portfolio Configuration:
- Assets (N): {len(inputs['v'])}
- Events (Q): {inputs['H'].shape[1]}
- Typologies (K): {inputs['C'].shape[0]}
- Curve Points (M): {len(inputs['x_grid'])}

Risk Metrics:
- Portfolio AAL: ${results['aal_portfolio']:,.2f}
- Total Exposure: ${np.sum(inputs['v']):,.2f}
- AAL/Exposure Ratio: {results['aal_portfolio']/np.sum(inputs['v']):.4f}
- Total Rate (Λ): {results['total_rate']:.6f} events/year
- Mean Loss per Event: ${np.mean(results['loss_per_event']):,.2f}
- Median Loss per Event: ${np.median(results['loss_per_event']):,.2f}
- Max Event Loss: ${np.max(results['loss_per_event']):,.2f}

Computation:
- Time: {st.session_state.computation_time:.2f} seconds
- Gradients Computed: {st.session_state.gradients is not None}
"""
            
            st.download_button(
                "📄 Download Summary TXT",
                data=summary_text,
                file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col3:
            # Save complete analysis
            try:
                metadata = create_metadata(
                    inputs=inputs,
                    results=results,
                    gradients=st.session_state.gradients,
                    data_source=st.session_state.data_source,
                    uploaded_filenames=st.session_state.uploaded_filenames,
                    lambda_mode=st.session_state.lambda_mode,
                    computation_time=st.session_state.computation_time
                )
                
                zip_buffer = save_analysis(
                    inputs=inputs,
                    results=results,
                    gradients=st.session_state.gradients,
                    metadata=metadata
                )
                
                st.download_button(
                    "💾 Save Complete Analysis",
                    data=zip_buffer,
                    file_name=f"tensor_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    help="Save all inputs, results, and gradients for later use"
                )
            except Exception as e:
                st.error(f"Error creating save file: {str(e)}")

# ===========================
# TAB 5: GRADIENTS & SENSITIVITY
# ===========================
with tab5:
    st.header("Gradients & Sensitivity Analysis")
    
    if st.session_state.gradients is None:
        st.warning("⚠️ Gradients not computed. Please run analysis with 'Compute Gradients' enabled.")
    else:
        gradients = st.session_state.gradients
        results = st.session_state.results
        inputs = st.session_state.inputs
        
        st.markdown("""
        Gradient analysis reveals how portfolio AAL changes with respect to each input parameter.
        This enables optimization, sensitivity analysis, and risk-based decision making.
        """)
        
        # Exposure Gradients
        st.markdown("### 🎯 Exposure Sensitivity (∂AAL/∂v)")
        st.markdown("Identifies which assets contribute most to portfolio risk per dollar of exposure.")
        
        with st.expander("Top Assets by Exposure Gradient", expanded=True):
            n_assets = len(gradients['grad_exposure'])
            display_n = min(100, n_assets)  # Show top 100 or all if fewer
            fig_grad_v = create_exposure_gradient_chart(
                gradients['grad_exposure'], inputs['v'], inputs['u'], top_n=display_n
            )
            st.plotly_chart(fig_grad_v, use_container_width=True)
        
        # Retrofit Optimizer
        with st.expander("🔧 Retrofit Optimizer"):
            st.markdown("**Use gradients to identify optimal retrofit prioritization**")
            
            budget = st.slider(
                "Retrofit Budget ($)",
                min_value=0,
                max_value=int(np.sum(inputs['v']) * 0.5),
                value=int(np.sum(inputs['v']) * 0.1),
                step=100000
            )
            
            effectiveness = st.slider(
                "Retrofit Effectiveness (%)",
                min_value=10,
                max_value=100,
                value=50,
                help="% reduction in vulnerability after retrofit"
            ) / 100.0
            
            # Calculate retrofit recommendations
            retrofit_cost_per_asset = inputs['v'] * 0.3  # Assume 30% of exposure
            grad_v = gradients['grad_exposure']
            
            # ROI = (AAL reduction) / (retrofit cost)
            aal_reduction = grad_v * inputs['v'] * effectiveness
            roi = aal_reduction / (retrofit_cost_per_asset + 1e-6)
            
            # Sort by ROI
            sorted_indices = np.argsort(roi)[::-1]
            
            # Select assets within budget
            cumulative_cost = 0
            selected_assets = []
            max_display = min(50, len(grad_v))  # Limit display to 50 or fewer
            for idx in sorted_indices:
                if cumulative_cost + retrofit_cost_per_asset[idx] <= budget:
                    selected_assets.append(idx)
                    cumulative_cost += retrofit_cost_per_asset[idx]
                if len(selected_assets) >= max_display:
                    break
            
            if len(selected_assets) > 0:
                total_aal_reduction = np.sum(aal_reduction[selected_assets])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Assets to Retrofit", len(selected_assets))
                col2.metric("Total Cost", f"${cumulative_cost:,.0f}")
                col3.metric("AAL Reduction", f"${total_aal_reduction:,.0f}")
                
                retrofit_df = pd.DataFrame({
                    'Asset ID': selected_assets,
                    'Retrofit Cost': retrofit_cost_per_asset[selected_assets],
                    'AAL Reduction': aal_reduction[selected_assets],
                    'ROI': roi[selected_assets],
                    'Current AAL': results['aal_per_asset'][selected_assets]
                })
                
                st.dataframe(retrofit_df, use_container_width=True, hide_index=True)
        
        # Vulnerability Gradients
        st.markdown("### 📉 Vulnerability Sensitivity (∂AAL/∂C)")
        st.markdown("Shows which parts of vulnerability curves have greatest impact on risk.")
        
        with st.expander("Vulnerability Gradient Heatmap"):
            fig_grad_C = create_vulnerability_gradient_heatmap(
                gradients['grad_vulnerability'], inputs['x_grid']
            )
            st.plotly_chart(fig_grad_C, use_container_width=True)
        
        # Hazard Gradients
        st.markdown("### 🌍 Hazard Sensitivity (∂AAL/∂H)")
        st.markdown("Quantifies sensitivity to hazard intensity changes.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("Hazard Sensitivity vs Return Period"):
                fig_hazard_rp = create_hazard_sensitivity_vs_return_period(
                    gradients['grad_hazard'], inputs['lambdas']
                )
                st.plotly_chart(fig_hazard_rp, use_container_width=True)
        
        with col2:
            with st.expander("Hazard Gradient Heatmap (Sampled)"):
                fig_grad_H = create_hazard_gradient_heatmap(
                    gradients['grad_hazard'], max_assets=50, max_events=100
                )
                st.plotly_chart(fig_grad_H, use_container_width=True)
        
        # Scenario Importance
        st.markdown("### 🎲 Scenario Importance (∂AAL/∂λ)")
        st.markdown("Identifies which events contribute most to portfolio AAL.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("Event Contribution to AAL"):
                fig_contrib = create_event_contribution_plot(
                    results['loss_per_event'], inputs['lambdas']
                )
                st.plotly_chart(fig_contrib, use_container_width=True)
        
        with col2:
            with st.expander("Scenario Loss vs Occurrence Rate"):
                fig_loss_rate = create_scenario_loss_vs_rate_plot(
                    results['loss_per_event'], inputs['lambdas']
                )
                st.plotly_chart(fig_loss_rate, use_container_width=True)
        
        # Gradient statistics
        with st.expander("📊 Gradient Statistics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Exposure Gradients**")
                st.text(f"Mean: {np.mean(gradients['grad_exposure']):.4f}")
                st.text(f"Std: {np.std(gradients['grad_exposure']):.4f}")
                st.text(f"Max: {np.max(gradients['grad_exposure']):.4f}")
            
            with col2:
                st.markdown("**Vulnerability Gradients**")
                st.text(f"Mean: {np.mean(gradients['grad_vulnerability']):.2e}")
                st.text(f"Std: {np.std(gradients['grad_vulnerability']):.2e}")
                st.text(f"Max: {np.max(np.abs(gradients['grad_vulnerability'])):.2e}")
            
            with col3:
                st.markdown("**Hazard Gradients**")
                st.text(f"Mean: {np.mean(gradients['grad_hazard']):.2e}")
                st.text(f"Std: {np.std(gradients['grad_hazard']):.2e}")
                st.text(f"Max: {np.max(np.abs(gradients['grad_hazard'])):.2e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Tensor Risk Engine v1.0</strong></p>
    <p>Fully Differentiable Catastrophe Risk Assessment Framework</p>
    <p>Powered by TensorFlow | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
