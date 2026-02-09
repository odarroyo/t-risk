"""
Data Validation Module for Tensor Risk Engine
==============================================
Validates input data for consistency, shapes, types, and value constraints.
"""

import numpy as np
from typing import Tuple, Dict, Optional
import streamlit as st


def validate_shapes(v: np.ndarray, u: np.ndarray, C: np.ndarray, 
                   x_grid: np.ndarray, H: np.ndarray, 
                   lambdas: Optional[np.ndarray] = None) -> Tuple[bool, str]:
    """
    Validate that all array shapes are consistent.
    
    Parameters
    ----------
    v : np.ndarray, shape (N,)
        Exposure vector
    u : np.ndarray, shape (N,)
        Typology indices
    C : np.ndarray, shape (K, M)
        Vulnerability matrix
    x_grid : np.ndarray, shape (M,)
        Intensity grid
    H : np.ndarray, shape (N, Q)
        Hazard intensity matrix
    lambdas : np.ndarray, shape (Q,), optional
        Scenario occurrence rates
    
    Returns
    -------
    valid : bool
        True if all shapes are consistent
    message : str
        Error message if invalid, success message if valid
    """
    try:
        N = v.shape[0]
        Q = H.shape[1]
        K = C.shape[0]
        M = C.shape[1]
        
        # Check N consistency
        if u.shape[0] != N:
            return False, f"❌ Exposure (N={N}) and Typology (N={u.shape[0]}) size mismatch"
        
        if H.shape[0] != N:
            return False, f"❌ Exposure (N={N}) and Hazard rows (N={H.shape[0]}) mismatch"
        
        # Check M consistency
        if x_grid.shape[0] != M:
            return False, f"❌ Vulnerability columns (M={M}) and Intensity grid (M={x_grid.shape[0]}) mismatch"
        
        # Check Q consistency
        if lambdas is not None and lambdas.shape[0] != Q:
            return False, f"❌ Hazard events (Q={Q}) and Lambda rates (Q={lambdas.shape[0]}) mismatch"
        
        # Check typology indices are valid
        max_typology = np.max(u)
        if max_typology >= K:
            return False, f"❌ Typology index {max_typology} exceeds max allowed ({K-1}). Need at least {max_typology+1} vulnerability curves."
        
        # Check minimum dimensions
        if N == 0:
            return False, "❌ No assets (N=0)"
        if Q == 0:
            return False, "❌ No events (Q=0)"
        if K == 0:
            return False, "❌ No typologies (K=0)"
        if M < 2:
            return False, f"❌ Need at least 2 intensity points, got M={M}"
        
        return True, f"✓ Shapes valid: N={N} assets, Q={Q} events, K={K} typologies, M={M} intensity points"
    
    except Exception as e:
        return False, f"❌ Shape validation error: {str(e)}"


def validate_monotonic(x_grid: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that intensity grid is strictly monotonically increasing.
    
    Parameters
    ----------
    x_grid : np.ndarray, shape (M,)
        Intensity grid
    
    Returns
    -------
    valid : bool
        True if strictly increasing
    message : str
        Error message or success message
    """
    try:
        if len(x_grid) < 2:
            return False, "❌ Intensity grid must have at least 2 points"
        
        diffs = np.diff(x_grid)
        
        if np.any(diffs <= 0):
            # Find first non-increasing point
            idx = np.where(diffs <= 0)[0][0]
            return False, f"❌ Intensity grid not strictly increasing at position {idx}: x[{idx}]={x_grid[idx]:.3f}, x[{idx+1}]={x_grid[idx+1]:.3f}"
        
        return True, f"✓ Intensity grid is strictly monotonic ({x_grid[0]:.3f}g to {x_grid[-1]:.3f}g)"
    
    except Exception as e:
        return False, f"❌ Monotonicity check error: {str(e)}"


def validate_ranges(C: np.ndarray, H: np.ndarray, 
                   lambdas: Optional[np.ndarray] = None) -> Tuple[bool, str]:
    """
    Validate value ranges for vulnerability, hazard, and lambdas.
    
    Parameters
    ----------
    C : np.ndarray, shape (K, M)
        Vulnerability matrix (should be in [0, 1])
    H : np.ndarray, shape (N, Q)
        Hazard intensities (typically [0, 2.0]g)
    lambdas : np.ndarray, shape (Q,), optional
        Occurrence rates (must be non-negative)
    
    Returns
    -------
    valid : bool
        True if all ranges are valid
    message : str
        Error message or success message
    """
    messages = []
    
    # Check vulnerability in [0, 1]
    if np.any(C < 0) or np.any(C > 1):
        c_min, c_max = np.min(C), np.max(C)
        messages.append(f"❌ Vulnerability values outside [0,1]: range=[{c_min:.3f}, {c_max:.3f}]")
    else:
        messages.append(f"✓ Vulnerability in valid range [0, 1]")
    
    # Check hazard for reasonableness (warn if > 2.0g)
    h_min, h_max = np.min(H), np.max(H)
    if h_min < 0:
        messages.append(f"❌ Negative hazard intensities: min={h_min:.3f}g")
    elif h_max > 3.0:
        messages.append(f"⚠️ Very high hazard intensities: max={h_max:.3f}g (unusual but allowed)")
    else:
        messages.append(f"✓ Hazard intensities reasonable: [{h_min:.3f}g, {h_max:.3f}g]")
    
    # Check lambdas non-negative
    if lambdas is not None:
        if np.any(lambdas < 0):
            messages.append(f"❌ Negative occurrence rates: min={np.min(lambdas):.6f}")
        else:
            total_rate = np.sum(lambdas)
            messages.append(f"✓ Occurrence rates valid: Λ={total_rate:.6f} events/year")
    
    # Determine overall validity
    has_error = any(msg.startswith("❌") for msg in messages)
    
    return not has_error, "\n".join(messages)


def validate_dtypes(v: np.ndarray, u: np.ndarray, C: np.ndarray,
                   x_grid: np.ndarray, H: np.ndarray,
                   lambdas: Optional[np.ndarray] = None) -> Tuple[Dict, str]:
    """
    Validate and convert data types to required formats.
    
    Parameters
    ----------
    v, u, C, x_grid, H, lambdas : np.ndarray
        Input arrays
    
    Returns
    -------
    converted : dict
        Dictionary with converted arrays
    message : str
        Conversion summary message
    """
    messages = []
    converted = {}
    
    # Convert exposure to float32
    if v.dtype != np.float32:
        messages.append(f"Converted exposure: {v.dtype} → float32")
        converted['v'] = v.astype(np.float32)
    else:
        converted['v'] = v
    
    # Convert typology to int32
    if u.dtype != np.int32:
        messages.append(f"Converted typology: {u.dtype} → int32")
        converted['u'] = u.astype(np.int32)
    else:
        converted['u'] = u
    
    # Convert vulnerability to float32
    if C.dtype != np.float32:
        messages.append(f"Converted vulnerability: {C.dtype} → float32")
        converted['C'] = C.astype(np.float32)
    else:
        converted['C'] = C
    
    # Convert grid to float32
    if x_grid.dtype != np.float32:
        messages.append(f"Converted grid: {x_grid.dtype} → float32")
        converted['x_grid'] = x_grid.astype(np.float32)
    else:
        converted['x_grid'] = x_grid
    
    # Convert hazard to float32
    if H.dtype != np.float32:
        messages.append(f"Converted hazard: {H.dtype} → float32")
        converted['H'] = H.astype(np.float32)
    else:
        converted['H'] = H
    
    # Convert lambdas to float32
    if lambdas is not None:
        if lambdas.dtype != np.float32:
            messages.append(f"Converted lambdas: {lambdas.dtype} → float32")
            converted['lambdas'] = lambdas.astype(np.float32)
        else:
            converted['lambdas'] = lambdas
    else:
        converted['lambdas'] = None
    
    # Check for NaN/Inf
    if np.any(~np.isfinite(converted['v'])):
        messages.append("❌ NaN/Inf detected in exposure")
    if np.any(~np.isfinite(converted['C'])):
        messages.append("❌ NaN/Inf detected in vulnerability")
    if np.any(~np.isfinite(converted['x_grid'])):
        messages.append("❌ NaN/Inf detected in intensity grid")
    if np.any(~np.isfinite(converted['H'])):
        messages.append("❌ NaN/Inf detected in hazard")
    if lambdas is not None and np.any(~np.isfinite(converted['lambdas'])):
        messages.append("❌ NaN/Inf detected in lambdas")
    
    message = "\n".join(messages) if messages else "✓ All dtypes correct"
    return converted, message


def validate_all(v: np.ndarray, u: np.ndarray, C: np.ndarray,
                x_grid: np.ndarray, H: np.ndarray,
                lambdas: Optional[np.ndarray] = None) -> Tuple[bool, Dict, str]:
    """
    Run all validations and return converted data if valid.
    
    Parameters
    ----------
    v, u, C, x_grid, H, lambdas : np.ndarray
        Input arrays
    
    Returns
    -------
    valid : bool
        True if all validations pass
    converted : dict
        Converted arrays with correct dtypes
    message : str
        Combined validation message
    """
    all_messages = []
    
    # 1. Validate and convert dtypes
    converted, dtype_msg = validate_dtypes(v, u, C, x_grid, H, lambdas)
    if dtype_msg:
        all_messages.append("Data Types:\n" + dtype_msg)
    
    # Use converted arrays for remaining checks
    v_c = converted['v']
    u_c = converted['u']
    C_c = converted['C']
    x_grid_c = converted['x_grid']
    H_c = converted['H']
    lambdas_c = converted['lambdas']
    
    # 2. Validate shapes
    shapes_valid, shapes_msg = validate_shapes(v_c, u_c, C_c, x_grid_c, H_c, lambdas_c)
    all_messages.append("\nShapes:\n" + shapes_msg)
    
    # 3. Validate monotonicity
    mono_valid, mono_msg = validate_monotonic(x_grid_c)
    all_messages.append("\nMonotonicity:\n" + mono_msg)
    
    # 4. Validate ranges
    ranges_valid, ranges_msg = validate_ranges(C_c, H_c, lambdas_c)
    all_messages.append("\nValue Ranges:\n" + ranges_msg)
    
    # Overall validity
    overall_valid = shapes_valid and mono_valid and ranges_valid
    
    combined_message = "\n".join(all_messages)
    
    return overall_valid, converted, combined_message


def estimate_memory_usage(N: int, Q: int, K: int, M: int) -> str:
    """
    Estimate memory usage for given dimensions.
    
    Parameters
    ----------
    N, Q, K, M : int
        Portfolio dimensions
    
    Returns
    -------
    str
        Human-readable memory estimate
    """
    # Loss matrix: N × Q × 4 bytes (float32)
    loss_matrix_mb = (N * Q * 4) / (1024**2)
    
    # Hazard matrix: N × Q × 4 bytes
    hazard_matrix_mb = (N * Q * 4) / (1024**2)
    
    # Gradients (if computed): ~ 2 × N × Q × 4 bytes
    gradient_mb = (2 * N * Q * 4) / (1024**2)
    
    total_mb = loss_matrix_mb + hazard_matrix_mb + gradient_mb
    
    if total_mb < 100:
        return f"~{total_mb:.1f} MB (small, very fast)"
    elif total_mb < 1000:
        return f"~{total_mb:.1f} MB (medium, fast on GPU)"
    elif total_mb < 4000:
        return f"~{total_mb/1024:.2f} GB (large, may need chunking)"
    else:
        return f"~{total_mb/1024:.2f} GB (very large, recommend reducing Q or using chunking)"
