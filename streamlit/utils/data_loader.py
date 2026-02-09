"""
Data Loading Module for Tensor Risk Engine
===========================================
Handles CSV/XLSX file parsing, template generation, and synthetic data creation.
"""

import numpy as np
import pandas as pd
from io import BytesIO, StringIO
from typing import Tuple, Dict, Optional
import sys
import os

# Add parent directory to path to import tensor_engine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tensor_engine import generate_synthetic_portfolio


def load_assets_file(uploaded_file) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load assets file containing exposure and typology information.
    
    Expected columns:
    - exposure (or exposure_usd): Required, replacement cost per asset
    - typology: Required, integer typology index (0, 1, 2, ...)
    - asset_id: Optional, asset identifier
    - latitude, longitude: Optional, geographic coordinates
    - description: Optional, text description
    
    Parameters
    ----------
    uploaded_file : UploadedFile
        CSV or XLSX file
    
    Returns
    -------
    v : np.ndarray, shape (N,)
        Exposure vector
    u : np.ndarray, shape (N,)
        Typology indices
    metadata_df : pd.DataFrame
        Full dataframe with all columns
    """
    # Read file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("File must be CSV or XLSX format")
    
    # Find exposure column
    exposure_col = None
    for col in ['exposure', 'exposure_usd', 'value', 'replacement_cost']:
        if col in df.columns:
            exposure_col = col
            break
    
    if exposure_col is None:
        raise ValueError("No exposure column found. Expected: 'exposure', 'exposure_usd', 'value', or 'replacement_cost'")
    
    # Find typology column
    if 'typology' not in df.columns:
        raise ValueError("No 'typology' column found")
    
    # Extract arrays
    v = df[exposure_col].values.astype(np.float64)
    u = df['typology'].values.astype(np.int32)
    
    return v, u, df


def load_vulnerability_file(uploaded_file) -> Tuple[np.ndarray, np.ndarray, Optional[list]]:
    """
    Load vulnerability curves file.
    
    Format Option 1 (CSV with header row):
    intensity_0.0g, intensity_0.05g, intensity_0.1g, ...
    0.00, 0.01, 0.05, ...  # Typology 0
    0.00, 0.005, 0.02, ... # Typology 1
    
    Format Option 2 (XLSX with two sheets):
    Sheet 'intensity_grid': Single column with M values
    Sheet 'curves': K rows × M columns
    
    Parameters
    ----------
    uploaded_file : UploadedFile
        CSV or XLSX file
    
    Returns
    -------
    C : np.ndarray, shape (K, M)
        Vulnerability matrix
    x_grid : np.ndarray, shape (M,)
        Intensity grid
    typology_names : list or None
        Names of typologies if provided
    """
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        
        # Try to extract intensity grid from column names
        x_grid = []
        for col in df.columns:
            if 'intensity' in col.lower():
                # Extract numeric value from column name like "intensity_0.05g"
                try:
                    val = float(col.split('_')[1].replace('g', ''))
                    x_grid.append(val)
                except:
                    pass
        
        if len(x_grid) == 0:
            # Assume all columns are vulnerability values, create default grid
            M = len(df.columns)
            x_grid = np.linspace(0.0, 1.5, M)
        else:
            x_grid = np.array(x_grid)
        
        # Extract vulnerability curves (all rows, numeric columns only)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        C = df[numeric_cols].values.astype(np.float64)
        
        # Check for typology names
        typology_names = None
        if 'typology_name' in df.columns:
            typology_names = df['typology_name'].tolist()
        
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        # Read all sheets
        excel_file = pd.ExcelFile(uploaded_file)
        
        if 'intensity_grid' in excel_file.sheet_names and 'curves' in excel_file.sheet_names:
            # Format 2: Separate sheets
            x_grid = pd.read_excel(excel_file, sheet_name='intensity_grid').iloc[:, 0].values
            curves_df = pd.read_excel(excel_file, sheet_name='curves')
            C = curves_df.select_dtypes(include=[np.number]).values
            typology_names = None
        else:
            # Format 1: Single sheet
            df = pd.read_excel(excel_file, sheet_name=0)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            C = df[numeric_cols].values
            M = C.shape[1]
            x_grid = np.linspace(0.0, 1.5, M)
            typology_names = None
    else:
        raise ValueError("File must be CSV or XLSX format")
    
    return C.astype(np.float32), x_grid.astype(np.float32), typology_names


def load_intensity_grid_file(uploaded_file) -> np.ndarray:
    """
    Load intensity grid file.
    
    Expected format: Single column of intensity values
    
    Parameters
    ----------
    uploaded_file : UploadedFile
        CSV or XLSX file
    
    Returns
    -------
    x_grid : np.ndarray, shape (M,)
        Intensity grid
    """
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Take first numeric column
    x_grid = df.iloc[:, 0].values.astype(np.float32)
    
    return x_grid


def load_hazard_file(uploaded_file) -> np.ndarray:
    """
    Load hazard intensity matrix.
    
    Format Option 1 (Wide format):
    asset_id, event_1, event_2, event_3, ...
    1, 0.25, 0.45, 0.12, ...
    2, 0.30, 0.50, 0.15, ...
    
    Format Option 2 (Long format):
    asset_id, event_id, intensity
    1, 1, 0.25
    1, 2, 0.45
    2, 1, 0.30
    ...
    
    Parameters
    ----------
    uploaded_file : UploadedFile
        CSV or XLSX file
    
    Returns
    -------
    H : np.ndarray, shape (N, Q)
        Hazard intensity matrix
    """
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Detect format
    if 'intensity' in df.columns and ('asset_id' in df.columns or 'event_id' in df.columns):
        # Long format - pivot to wide
        if 'asset_id' not in df.columns or 'event_id' not in df.columns:
            raise ValueError("Long format requires 'asset_id', 'event_id', and 'intensity' columns")
        
        df_pivot = df.pivot(index='asset_id', columns='event_id', values='intensity')
        H = df_pivot.values
    else:
        # Wide format - take all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude asset_id if present
        numeric_cols = [col for col in numeric_cols if 'id' not in col.lower()]
        H = df[numeric_cols].values
    
    return H.astype(np.float32)


def load_lambdas_file(uploaded_file) -> np.ndarray:
    """
    Load scenario occurrence rates.
    
    Expected columns:
    - lambda (or lambda_per_year): Occurrence rate
    - event_id: Optional
    - return_period_years: Optional (will be ignored, lambda takes precedence)
    
    Parameters
    ----------
    uploaded_file : UploadedFile
        CSV or XLSX file
    
    Returns
    -------
    lambdas : np.ndarray, shape (Q,)
        Occurrence rates
    """
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Find lambda column
    lambda_col = None
    for col in ['lambda', 'lambda_per_year', 'rate', 'occurrence_rate']:
        if col in df.columns:
            lambda_col = col
            break
    
    if lambda_col is None:
        raise ValueError("No lambda column found. Expected: 'lambda', 'lambda_per_year', 'rate', or 'occurrence_rate'")
    
    lambdas = df[lambda_col].values.astype(np.float32)
    
    return lambdas


def generate_assets_template() -> BytesIO:
    """Generate example assets CSV template."""
    csv_content = """asset_id,exposure,typology,latitude,longitude,description
1,150000,0,37.7749,-122.4194,Single-family home - Old Masonry
2,500000,2,37.7849,-122.4094,Commercial building - RC Frame
3,300000,1,37.7949,-122.3994,Multi-family - Wood Frame
4,200000,0,37.8049,-122.4294,Single-family home - Old Masonry
5,750000,3,37.8149,-122.4394,Office building - Steel Frame
6,180000,1,37.7649,-122.4494,Single-family home - Wood Frame
7,420000,2,37.7549,-122.4594,Mid-rise apartment - RC Frame
8,250000,4,37.8249,-122.4694,Warehouse - Prefab Metal
9,350000,1,37.8349,-122.4794,Duplex - Wood Frame
10,600000,2,37.8449,-122.4894,School building - RC Frame"""
    
    return BytesIO(csv_content.encode())


def generate_vulnerability_template() -> BytesIO:
    """Generate example vulnerability curves CSV template."""
    # Create intensity grid
    x_grid = np.linspace(0.0, 1.5, 20)
    
    # Create example curves for 5 typologies
    curves = []
    typology_names = [
        "Old Masonry (Type 0)",
        "Wood Frame (Type 1)",
        "RC Frame (Type 2)",
        "Steel Frame (Type 3)",
        "Prefab Metal (Type 4)"
    ]
    
    for k, name in enumerate(typology_names):
        steepness = 8.0 + k * 2.0
        midpoint = 0.4 + k * 0.1
        curve = 1.0 / (1.0 + np.exp(-steepness * (x_grid - midpoint)))
        curves.append(curve)
    
    # Create DataFrame
    df = pd.DataFrame(curves)
    df.insert(0, 'typology_name', typology_names)
    
    # Rename columns to show intensity values
    for i, intensity in enumerate(x_grid):
        df.rename(columns={i: f'intensity_{intensity:.2f}g'}, inplace=True)
    
    # Convert to CSV
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return csv_buffer


def generate_intensity_grid_template() -> BytesIO:
    """Generate example intensity grid CSV template."""
    x_grid = np.linspace(0.0, 1.5, 20)
    df = pd.DataFrame({'intensity': x_grid})
    
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return csv_buffer


def generate_hazard_template() -> BytesIO:
    """Generate example hazard matrix CSV template (small example)."""
    np.random.seed(42)
    N_sample = 10
    Q_sample = 5
    H_sample = np.random.uniform(0.0, 1.2, (N_sample, Q_sample))
    
    df = pd.DataFrame(H_sample, columns=[f'event_{i+1}' for i in range(Q_sample)])
    df.insert(0, 'asset_id', range(1, N_sample + 1))
    
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return csv_buffer


def generate_lambdas_template() -> BytesIO:
    """Generate example scenario rates CSV template."""
    # Example: exponential distribution
    Q_sample = 10
    lambdas_sample = np.exp(-np.linspace(0, 3, Q_sample))
    lambdas_sample = lambdas_sample / lambdas_sample.sum()
    
    return_periods = 1.0 / lambdas_sample
    
    df = pd.DataFrame({
        'event_id': range(1, Q_sample + 1),
        'lambda_per_year': lambdas_sample,
        'return_period_years': return_periods,
        'description': [f'Event {i+1}' for i in range(Q_sample)]
    })
    
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return csv_buffer


def generate_synthetic_data(N: int, Q: int, K: int = 5, M: int = 20,
                           lambda_mode: str = 'exponential') -> Dict[str, np.ndarray]:
    """
    Generate synthetic portfolio data using tensor_engine function.
    
    Parameters
    ----------
    N : int
        Number of assets
    Q : int
        Number of events
    K : int
        Number of typologies
    M : int
        Number of curve points
    lambda_mode : str
        'uniform' or 'exponential'
    
    Returns
    -------
    dict
        Dictionary with keys: v, u, C, x_grid, H, lambdas
    """
    v, u, C, x_grid, H, lambdas = generate_synthetic_portfolio(
        n_assets=N,
        n_events=Q,
        n_typologies=K,
        n_curve_points=M,
        lambda_distribution=lambda_mode
    )
    
    return {
        'v': v,
        'u': u,
        'C': C,
        'x_grid': x_grid,
        'H': H,
        'lambdas': lambdas
    }
