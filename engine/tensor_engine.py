"""
Tensorial Risk Engine - Full Implementation
============================================
Based on the manuscript formulation for differentiable catastrophe risk assessment.

Implements:
- Section 2: Deterministic Hazard Formulation
- Section 3: Probabilistic Hazard Formulation  
- Section 4: Gradient of Vulnerability
- Section 5: Gradient of Exposure and Hazard

Mathematical Foundation:
- Multi-typology support (vector u)
- Per-asset AAL and variance
- Full gradient computation for all inputs
"""

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

# ==========================================
# 1. DATA GENERATION (Manuscript-Compliant)
# ==========================================

def generate_synthetic_portfolio(n_assets: int, n_events: int, n_typologies: int = 5,
                                 n_curve_points: int = 20,
                                 lambdas: Optional[np.ndarray] = None,
                                 lambda_distribution: str = 'exponential',
                                 sigma_fraction: Optional[float] = None) -> Tuple:
    """
    Generate synthetic portfolio data matching manuscript notation.
    
    Creates realistic synthetic data for catastrophe risk modeling including
    exposure values, building typologies, vulnerability curves, intensity grids,
    hazard intensity matrices, and scenario occurrence rates.
    
    Parameters
    ----------
    n_assets : int
        Number of assets in the portfolio (N)
    n_events : int
        Number of stochastic event realizations (Q)
    n_typologies : int, optional
        Number of building typologies/vulnerability curves (K), default is 5
    n_curve_points : int, optional
        Number of discretization points for vulnerability curves (M), default is 20
    lambdas : np.ndarray, optional
        Pre-specified scenario occurrence rates ∈ ℝ^Q. If None, will be generated
    lambda_distribution : str, optional
        Distribution type for generated lambdas: 'uniform' or 'exponential', default is 'exponential'
    sigma_fraction : float, optional
        Fraction of maximum Beta std dev to use for vulnerability uncertainty.
        When provided, generates Sigma = sigma_fraction * sqrt(C * (1 - C)).
        Typical values: 0.2–0.5. If None, no Sigma is generated.

    Returns
    -------
    v_exposure : np.ndarray, shape (N,)
        Exposure vector containing replacement costs for each asset ∈ ℝ^N
        Values range from $100,000 to $1,000,000
    u_typology : np.ndarray, shape (N,)
        Typology index vector mapping each asset to a vulnerability curve ∈ ℤ^N
        Values in {0, 1, ..., K-1}
    C_matrix : np.ndarray, shape (K, M)
        Vulnerability matrix containing K curves with M points each ∈ ℝ^(K×M)
        Each curve represents Mean Damage Ratio as function of intensity
    x_grid : np.ndarray, shape (M,)
        Common intensity grid vector for all curves ∈ ℝ^M
        Values range from 0.0g to 1.5g
    H_intensities : np.ndarray, shape (N, Q)
        Hazard intensity matrix ∈ ℝ^(N×Q)
        H[i,q] = ground motion intensity at asset i during event q
    lambdas_out : np.ndarray, shape (Q,)
        Scenario occurrence rate vector ∈ ℝ^Q
        λ_q = annual occurrence rate of event q (events/year)
    Sigma_matrix : np.ndarray, shape (K, M) or None
        Vulnerability std dev matrix ∈ ℝ^(K×M), or None if sigma_fraction not set
    
    Notes
    -----
    - Uses sigmoid functions to generate realistic vulnerability curves
    - Different typologies have varying fragility (steepness) and thresholds
    - Hazard intensities are uniformly distributed for demonstration purposes
    - Exponential distribution for lambdas mimics importance sampling typical in CAT modeling
    - Fixed random seed (42) ensures reproducible results
    
    Examples
    --------
    >>> v, u, C, x, H, lambdas, Sigma = generate_synthetic_portfolio(1000, 5000, 5, 20)
    >>> print(v.shape, u.shape, C.shape, x.shape, H.shape, lambdas.shape)
    (1000,) (1000,) (5, 20) (20,) (1000, 5000) (5000,)
    >>> print(Sigma)  # None when sigma_fraction not set
    None
    """
    np.random.seed(42)
    
    # Exposure Vector v ∈ ℝ^N
    v_exposure = np.random.uniform(100_000, 1_000_000, n_assets).astype(np.float32)
    
    # Typology Index Vector u ∈ ℤ^N
    # Each asset is assigned to one of K building typologies
    u_typology = np.random.randint(0, n_typologies, n_assets).astype(np.int32)
    
    # Intensity Grid x ∈ ℝ^M (common for all curves)
    x_grid = np.linspace(0.0, 1.5, n_curve_points).astype(np.float32)
    
    # Vulnerability Matrix C ∈ ℝ^(K×M)
    # Generate K different vulnerability curves (different building types)
    C_matrix = np.zeros((n_typologies, n_curve_points), dtype=np.float32)
    
    for k in range(n_typologies):
        # Different curves for different building types
        # Varying fragility: masonry (steep) vs. reinforced concrete (gradual)
        steepness = 8.0 + k * 2.0  # More fragile buildings have steeper curves
        midpoint = 0.4 + k * 0.1   # Different damage thresholds
        C_matrix[k, :] = 1.0 / (1.0 + np.exp(-steepness * (x_grid - midpoint)))
    
    # Hazard Intensity Matrix H ∈ ℝ^(N×Q)
    # Different intensity at each building for each earthquake scenario
    H_intensities = np.random.uniform(0.0, 1.2, (n_assets, n_events)).astype(np.float32)
    
    # Scenario Occurrence Rate Vector λ ∈ ℝ^Q (Manuscript Section 3c)
    if lambdas is None:
        if lambda_distribution == 'uniform':
            # Uniform rates: λ_q = 1/Q for all q
            lambdas_out = np.ones(n_events, dtype=np.float32) / n_events
        elif lambda_distribution == 'exponential':
            # Exponential decay: typical for importance sampling in CAT modeling
            # Higher rates for more frequent (lower intensity) events
            lambdas_out = np.exp(-np.linspace(0, 3, n_events)).astype(np.float32)
            # Normalize so that sum equals 1.0 (interpretable as probabilities)
            lambdas_out = lambdas_out / lambdas_out.sum()
        else:
            raise ValueError(f"Unknown lambda_distribution: {lambda_distribution}")
    else:
        lambdas_out = lambdas.astype(np.float32)

    # Vulnerability uncertainty Sigma ∈ ℝ^(K×M)
    if sigma_fraction is not None:
        # Sigma = fraction × sqrt(C × (1 - C)), the maximum std dev of a Beta
        # distribution with mean C. Clamp C away from 0 and 1 for stability.
        C_clamped = np.clip(C_matrix, 1e-6, 1.0 - 1e-6)
        Sigma_matrix = (sigma_fraction * np.sqrt(C_clamped * (1.0 - C_clamped))).astype(np.float32)
    else:
        Sigma_matrix = None

    return v_exposure, u_typology, C_matrix, x_grid, H_intensities, lambdas_out, Sigma_matrix

# ==========================================
# 2. DETERMINISTIC FORMULATION (Section 2)
# ==========================================

@tf.function
def deterministic_loss(v: tf.Tensor, u: tf.Tensor, C: tf.Tensor, 
                       x_grid: tf.Tensor, h: tf.Tensor) -> tf.Tensor:
    """
    Compute deterministic loss for a single hazard scenario (Manuscript Section 2).
    
    Implements the differentiable interpolation formulation from manuscript equations 1-3:
    - Eq. 1: α = (h - x_j) / (x_{j+1} - x_j)
    - Eq. 2: MDR_i = (1-α)·C[u_i,j] + α·C[u_i,j+1]
    - Eq. 3: J = Σ v_i · MDR_i
    
    This function uses linear interpolation to calculate Mean Damage Ratios (MDR)
    from vulnerability curves based on intensity values, then aggregates with
    exposure values to compute total loss.
    
    Parameters
    ----------
    v : tf.Tensor, shape (N,), dtype float32
        Exposure vector containing replacement costs for N assets ∈ ℝ^N
    u : tf.Tensor, shape (N,), dtype int32
        Typology index vector mapping assets to vulnerability curves ∈ ℤ^N
        Values must be in range [0, K-1]
    C : tf.Tensor, shape (K, M), dtype float32
        Vulnerability matrix with K curves and M intensity points ∈ ℝ^(K×M)
        C[k,m] represents Mean Damage Ratio for typology k at intensity x[m]
    x_grid : tf.Tensor, shape (M,), dtype float32
        Intensity grid vector defining curve discretization ∈ ℝ^M
        Must be monotonically increasing
    h : tf.Tensor, shape (N,), dtype float32
        Intensity vector for single scenario ∈ ℝ^N
        h[i] is the ground motion intensity at asset i
    
    Returns
    -------
    J : tf.Tensor, scalar, dtype float32
        Total portfolio loss for this scenario
    
    Notes
    -----
    - Uses tf.searchsorted for efficient grid lookup
    - Adds epsilon (1e-8) to denominators for numerical stability
    - Employs flat indexing for GPU compatibility (Metal backend)
    - Fully differentiable w.r.t. all inputs except u (integer indices)
    
    Mathematical Details
    --------------------
    For each asset i:
    1. Find grid index j such that x[j] ≤ h[i] < x[j+1]
    2. Compute interpolation weight α = (h[i] - x[j]) / (x[j+1] - x[j])
    3. Interpolate damage: MDR[i] = (1-α)·C[u[i],j] + α·C[u[i],j+1]
    4. Compute asset loss: L[i] = v[i] · MDR[i]
    5. Aggregate: J = Σ L[i]
    
    Examples
    --------
    >>> v = tf.constant([100000., 200000., 300000.], dtype=tf.float32)
    >>> u = tf.constant([0, 1, 0], dtype=tf.int32)
    >>> C = tf.constant([[0.0, 0.5, 1.0], [0.0, 0.3, 0.8]], dtype=tf.float32)
    >>> x_grid = tf.constant([0.0, 0.5, 1.0], dtype=tf.float32)
    >>> h = tf.constant([0.25, 0.75, 0.5], dtype=tf.float32)
    >>> loss = deterministic_loss(v, u, C, x_grid, h)
    >>> print(loss.numpy())
    """
    N = tf.shape(v)[0]
    
    # Find grid indices: j such that x[j] ≤ h < x[j+1]
    idx = tf.searchsorted(x_grid, h, side='right') - 1
    idx = tf.clip_by_value(idx, 0, tf.shape(x_grid)[0] - 2)
    
    # Grid boundaries
    x_lower = tf.gather(x_grid, idx)
    x_upper = tf.gather(x_grid, idx + 1)
    
    # Interpolation weight α (Eq. 1)
    alpha = (h - x_lower) / (x_upper - x_lower + 1e-8)
    
    # Gather vulnerability values based on typology u
    # C[u[i], idx[i]] for each asset i
    # Use loops for Metal GPU compatibility
    K = tf.shape(C)[0]
    M = tf.shape(C)[1]
    
    # Flatten C and compute flat indices
    c_flat = tf.reshape(C, [-1])
    flat_idx_lower = u * M + idx
    flat_idx_upper = u * M + (idx + 1)
    
    c_lower = tf.gather(c_flat, flat_idx_lower)
    c_upper = tf.gather(c_flat, flat_idx_upper)
    
    # Mean Damage Ratio (Eq. 2)
    mdr = (1.0 - alpha) * c_lower + alpha * c_upper
    
    # Total Loss (Eq. 3)
    J = tf.reduce_sum(v * mdr)
    
    return J


# ==========================================
# 3. PROBABILISTIC FORMULATION (Section 3)
# ==========================================

@tf.function
def _interpolate_matrix(u: tf.Tensor, Matrix: tf.Tensor,
                        x_grid: tf.Tensor, H: tf.Tensor) -> tf.Tensor:
    """
    Interpolate a (K, M) matrix at hazard intensities for each asset.

    Shared helper that implements the flat-indexing linear interpolation
    pattern used throughout the engine. For each asset i and event q,
    looks up Matrix[u[i], :] at intensity H[i,q] via linear interpolation
    on x_grid.

    Parameters
    ----------
    u : tf.Tensor, shape (N,), dtype int32
        Typology index per asset, values in {0, ..., K-1}.
    Matrix : tf.Tensor, shape (K, M), dtype float32
        Any per-typology matrix to interpolate (e.g., C or Sigma).
    x_grid : tf.Tensor, shape (M,), dtype float32
        Intensity grid vector.
    H : tf.Tensor, shape (N, Q), dtype float32
        Hazard intensity matrix.

    Returns
    -------
    result : tf.Tensor, shape (N, Q), dtype float32
        Interpolated values for each asset-event pair.

    Notes
    -----
    Fully differentiable w.r.t. Matrix and H.
    Uses flat indexing for Metal GPU compatibility.
    """
    N = tf.shape(H)[0]
    Q = tf.shape(H)[1]
    M = tf.shape(x_grid)[0]

    H_flat = tf.reshape(H, [-1])

    idx = tf.searchsorted(x_grid, H_flat, side='right') - 1
    idx = tf.clip_by_value(idx, 0, M - 2)

    x_lower = tf.gather(x_grid, idx)
    x_upper = tf.gather(x_grid, idx + 1)
    alpha = (H_flat - x_lower) / (x_upper - x_lower + 1e-8)

    u_repeated = tf.tile(tf.expand_dims(u, 1), [1, Q])
    u_flat = tf.reshape(u_repeated, [-1])

    m_flat = tf.reshape(Matrix, [-1])
    flat_idx_lower = u_flat * M + idx
    flat_idx_upper = u_flat * M + (idx + 1)

    m_lower = tf.gather(m_flat, flat_idx_lower)
    m_upper = tf.gather(m_flat, flat_idx_upper)

    result_flat = (1.0 - alpha) * m_lower + alpha * m_upper
    return tf.reshape(result_flat, [N, Q])


@tf.function
def probabilistic_loss_matrix(v: tf.Tensor, u: tf.Tensor, C: tf.Tensor,
                               x_grid: tf.Tensor, H: tf.Tensor) -> tf.Tensor:
    """
    Compute probabilistic loss matrix for all assets and events (Manuscript Section 3).
    
    Implements the stochastic loss calculation from manuscript equation 4:
    J[i,q] = v[i] × [(1-α[i,q])·C[u[i],j[i,q]] + α[i,q]·C[u[i],j[i,q]+1]]
    
    This function extends deterministic loss to handle Q stochastic event
    realizations simultaneously, computing the complete N×Q loss matrix in a
    single vectorized operation.
    
    Parameters
    ----------
    v : tf.Tensor, shape (N,), dtype float32
        Exposure vector for N assets ∈ ℝ^N
    u : tf.Tensor, shape (N,), dtype int32
        Typology index vector ∈ ℤ^N
    C : tf.Tensor, shape (K, M), dtype float32
        Vulnerability matrix ∈ ℝ^(K×M)
    x_grid : tf.Tensor, shape (M,), dtype float32
        Intensity grid vector ∈ ℝ^M
    H : tf.Tensor, shape (N, Q), dtype float32
        Hazard intensity matrix ∈ ℝ^(N×Q)
        H[i,q] = intensity at asset i during event q
    
    Returns
    -------
    J_matrix : tf.Tensor, shape (N, Q), dtype float32
        Loss matrix where J[i,q] is the loss of asset i in event q ∈ ℝ^(N×Q)
    
    Notes
    -----
    - Flattens H matrix to (N*Q,) for vectorized interpolation
    - Tiles typology indices Q times to match flattened shape
    - Uses flat indexing for GPU compatibility
    - Reshapes result back to (N, Q) matrix form
    - Fully differentiable w.r.t. v, C, and H
    
    Memory Considerations
    ---------------------
    For large portfolios, the loss matrix can be substantial:
    - N=10,000 assets, Q=100,000 events → 1B elements (4GB float32)
    - Consider chunking over events for very large problems
    
    Mathematical Details
    --------------------
    The function processes all N×Q combinations simultaneously:
    1. Flatten H to vector of length N*Q
    2. Find grid indices for all intensities
    3. Compute interpolation weights α for all combinations
    4. Tile typology vector Q times
    5. Gather vulnerability values using (typology, intensity) pairs
    6. Interpolate MDR for all N*Q combinations
    7. Multiply by exposure (broadcast v across Q events)
    8. Reshape result to (N, Q) matrix
    
    Examples
    --------
    >>> v = tf.constant([100000., 200000.], dtype=tf.float32)
    >>> u = tf.constant([0, 1], dtype=tf.int32)
    >>> C = tf.constant([[0.0, 0.5, 1.0], [0.0, 0.3, 0.8]], dtype=tf.float32)
    >>> x_grid = tf.constant([0.0, 0.5, 1.0], dtype=tf.float32)
    >>> H = tf.constant([[0.25, 0.75], [0.5, 0.9]], dtype=tf.float32)
    >>> J = probabilistic_loss_matrix(v, u, C, x_grid, H)
    >>> print(J.shape)
    (2, 2)
    """
    # Interpolate vulnerability matrix C at hazard intensities
    mdr_matrix = _interpolate_matrix(u, C, x_grid, H)

    # Loss Matrix (Manuscript Eq. 4): J[i,q] = v[i] × MDR[i,q]
    J_matrix = tf.expand_dims(v, 1) * mdr_matrix

    return J_matrix


@tf.function
def compute_risk_metrics(J_matrix: tf.Tensor, lambdas: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
    """
    Compute comprehensive risk metrics from loss matrix (Manuscript Section 3c).
    
    Derives catastrophe risk metrics by aggregating the loss matrix with
    scenario occurrence rates. Implements rate-weighted manuscript equations:
    - AAL_i = Σ_q λ_q × J[i,q]  (rate-weighted annual loss)
    - μ_i = AAL_i / Λ = Σ_q w_q × J[i,q]  (mean per event)
    - σ²_i = Σ_q w_q × (J[i,q] - μ_i)²  (rate-weighted variance)
    
    Parameters
    ----------
    J_matrix : tf.Tensor, shape (N, Q), dtype float32
        Loss matrix where J[i,q] is loss of asset i in event q ∈ ℝ^(N×Q)
    lambdas : tf.Tensor, shape (Q,), dtype float32, optional
        Scenario occurrence rate vector ∈ ℝ^Q where λ_q ≥ 0
        If None, uniform rates (1/Q) are used for backward compatibility
    
    Returns
    -------
    metrics : dict
        Dictionary containing the following risk metrics:
        
        aal_per_asset : tf.Tensor, shape (N,), dtype float32
            Rate-weighted Average Annual Loss per asset ∈ ℝ^N
            AAL_i = Σ_q λ_q × J[i,q]
            
        aal_portfolio : tf.Tensor, scalar, dtype float32
            Total portfolio Average Annual Loss
            AAL_portfolio = Σ_i AAL_i
            
        mean_per_event_per_asset : tf.Tensor, shape (N,), dtype float32
            Mean loss per event occurrence per asset ∈ ℝ^N
            μ_i = AAL_i / Λ = Σ_q w_q × J[i,q]
            
        variance_per_asset : tf.Tensor, shape (N,), dtype float32
            Rate-weighted variance of loss per asset ∈ ℝ^N
            σ²_i = Σ_q w_q × (J[i,q] - μ_i)²
            
        std_per_asset : tf.Tensor, shape (N,), dtype float32
            Standard deviation of loss per asset ∈ ℝ^N
            σ_i = √(σ²_i)
            
        loss_per_event : tf.Tensor, shape (Q,), dtype float32
            Total portfolio loss for each event ∈ ℝ^Q
            L_q = Σ_i J[i,q]
            
        total_rate : tf.Tensor, scalar, dtype float32
            Total occurrence rate Λ = Σ_q λ_q
    
    Notes
    -----
    - All computations are differentiable w.r.t. J_matrix and lambdas
    - AAL represents rate-weighted expected annual loss
    - Variance is weighted by normalized rates w_q = λ_q / Λ
    - When lambdas is uniform (λ_q = 1/Q), reduces to simple averaging
    - Loss per event is independent of rates (intrinsic to scenarios)
    
    Applications
    ------------
    - Portfolio risk quantification with importance sampling
    - Non-uniform event catalogs (varying return periods)
    - Asset-level risk ranking
    - Uncertainty analysis
    - Risk-based capital requirements
    
    Examples
    --------
    >>> J = tf.constant([[1000., 2000., 3000.],
    ...                   [500., 1500., 2500.]], dtype=tf.float32)
    >>> lambdas = tf.constant([0.5, 0.3, 0.2], dtype=tf.float32)
    >>> metrics = compute_risk_metrics(J, lambdas)
    >>> print(f"Portfolio AAL: ${metrics['aal_portfolio'].numpy():.2f}")
    >>> print(f"Asset 0 AAL: ${metrics['aal_per_asset'][0].numpy():.2f}")
    """
    Q = tf.cast(tf.shape(J_matrix)[1], tf.float32)
    
    # Handle backward compatibility: if no lambdas, use uniform rates
    if lambdas is None:
        lambdas = tf.ones(tf.shape(J_matrix)[1], dtype=tf.float32) / Q
    
    # Total occurrence rate Λ = Σ_q λ_q
    Lambda = tf.reduce_sum(lambdas)
    
    # Normalized weights w_q = λ_q / Λ (sum to 1)
    w = lambdas / (Lambda + 1e-10)  # Add epsilon for numerical stability
    
    # Per-asset AAL (Manuscript Section 3c): AAL_i = Σ_q λ_q × J[i,q]
    # Shape: (N,) = (N, Q) @ (Q,)
    aal_per_asset = tf.reduce_sum(J_matrix * tf.expand_dims(lambdas, 0), axis=1)
    
    # Portfolio AAL: sum across all assets
    aal_portfolio = tf.reduce_sum(aal_per_asset)
    
    # Mean loss per event: μ_i = AAL_i / Λ = Σ_q w_q × J[i,q]
    mean_per_event_per_asset = aal_per_asset / (Lambda + 1e-10)
    
    # Rate-weighted variance per asset: σ²_i = Σ_q w_q × (J[i,q] - μ_i)²
    deviations_sq = tf.square(J_matrix - tf.expand_dims(mean_per_event_per_asset, 1))
    variance_per_asset = tf.reduce_sum(deviations_sq * tf.expand_dims(w, 0), axis=1)
    std_per_asset = tf.sqrt(variance_per_asset)
    
    # Loss per event (independent of rates)
    loss_per_event = tf.reduce_sum(J_matrix, axis=0)
    
    return {
        'aal_per_asset': aal_per_asset,
        'aal_portfolio': aal_portfolio,
        'mean_per_event_per_asset': mean_per_event_per_asset,
        'variance_per_asset': variance_per_asset,
        'std_per_asset': std_per_asset,
        'loss_per_event': loss_per_event,
        'total_rate': Lambda
    }


@tf.function
def compute_risk_metrics_with_uncertainty(
        J_matrix: tf.Tensor, lambdas: tf.Tensor,
        v: tf.Tensor, sigma_interpolated: tf.Tensor) -> Dict[str, tf.Tensor]:
    """
    Compute risk metrics including vulnerability uncertainty via the law of
    total variance.

    Extends compute_risk_metrics by decomposing total loss variance into:
    - Aleatory (event) variance: Var_events[E[Loss | event]]
    - Vulnerability variance: E_events[Var_vuln[Loss | event]]

    Using the law of total variance:
        Var_total = Var_aleatory + Var_vulnerability
        Var_vuln_i = Σ_q w_q × v_i² × σ²[u_i, m(i,q)]

    Parameters
    ----------
    J_matrix : tf.Tensor, shape (N, Q), dtype float32
        Loss matrix ∈ ℝ^(N×Q)
    lambdas : tf.Tensor, shape (Q,), dtype float32
        Scenario occurrence rates ∈ ℝ^Q
    v : tf.Tensor, shape (N,), dtype float32
        Exposure vector ∈ ℝ^N
    sigma_interpolated : tf.Tensor, shape (N, Q), dtype float32
        Interpolated vulnerability std dev at each asset-event intensity.

    Returns
    -------
    metrics : dict
        All keys from compute_risk_metrics, plus:

        variance_vulnerability_per_asset : tf.Tensor, shape (N,)
            Vulnerability uncertainty variance component per asset.
        variance_vulnerability_portfolio : tf.Tensor, scalar
            Portfolio-level vulnerability variance.
        variance_total_per_asset : tf.Tensor, shape (N,)
            Total variance = aleatory + vulnerability per asset.
        std_total_per_asset : tf.Tensor, shape (N,)
            Total standard deviation per asset.
        variance_total_portfolio : tf.Tensor, scalar
            Total portfolio variance.
        std_total_portfolio : tf.Tensor, scalar
            Total portfolio standard deviation.

    Notes
    -----
    Fully differentiable w.r.t. J_matrix, lambdas, v, and sigma_interpolated.
    """
    # Base metrics (aleatory)
    metrics = compute_risk_metrics(J_matrix, lambdas)

    Lambda = metrics['total_rate']
    w = lambdas / (Lambda + 1e-10)

    # Vulnerability variance component per asset:
    # Var_vuln_i = Σ_q w_q × v_i² × σ²_interpolated[i,q]
    v_sq = tf.square(v)  # (N,)
    sigma_sq = tf.square(sigma_interpolated)  # (N, Q)
    var_vuln_per_asset = tf.reduce_sum(
        sigma_sq * tf.expand_dims(w, 0), axis=1
    ) * v_sq  # (N,)

    var_vuln_portfolio = tf.reduce_sum(var_vuln_per_asset)

    # Total variance (law of total variance)
    var_total_per_asset = metrics['variance_per_asset'] + var_vuln_per_asset
    var_total_portfolio = tf.reduce_sum(var_total_per_asset)
    std_total_per_asset = tf.sqrt(var_total_per_asset)
    std_total_portfolio = tf.sqrt(var_total_portfolio)

    metrics['variance_vulnerability_per_asset'] = var_vuln_per_asset
    metrics['variance_vulnerability_portfolio'] = var_vuln_portfolio
    metrics['variance_total_per_asset'] = var_total_per_asset
    metrics['std_total_per_asset'] = std_total_per_asset
    metrics['variance_total_portfolio'] = var_total_portfolio
    metrics['std_total_portfolio'] = std_total_portfolio

    return metrics


# ==========================================
# 4. GRADIENT COMPUTATION ENGINE
# ==========================================

class TensorialRiskEngine:
    """
    Complete differentiable catastrophe risk engine with automatic gradient computation.
    
    This class implements the full tensorial formulation from the manuscript, providing:
    - Deterministic and probabilistic loss calculations
    - Per-asset and portfolio-level risk metrics
    - Automatic differentiation for all input parameters
    - GPU acceleration via TensorFlow
    
    The engine maintains portfolio data as TensorFlow variables/constants and provides
    methods to compute gradients w.r.t. vulnerability curves, exposure values, and
    hazard intensities. This enables sensitivity analysis and optimization that would
    be impractical with traditional task-based risk engines.
    
    Attributes
    ----------
    v : tf.Variable, shape (N,), dtype float32
        Exposure vector (replacement costs) ∈ ℝ^N
    u : tf.Constant, shape (N,), dtype int32
        Typology index vector ∈ ℤ^N
    C : tf.Variable, shape (K, M), dtype float32
        Vulnerability matrix ∈ ℝ^(K×M)
    x_grid : tf.Constant, shape (M,), dtype float32
        Intensity grid vector ∈ ℝ^M
    H : tf.Variable, shape (N, Q), dtype float32
        Hazard intensity matrix ∈ ℝ^(N×Q)
    n_assets : int
        Number of assets (N)
    n_events : int
        Number of stochastic events (Q)
    n_typologies : int
        Number of building typologies (K)
    
    Methods
    -------
    compute_loss_and_metrics()
        Compute loss matrix and risk metrics
    gradient_wrt_vulnerability()
        Compute ∂(AAL)/∂C (Manuscript Section 4)
    gradient_wrt_exposure()
        Compute ∂(AAL)/∂v (Manuscript Section 5)
    gradient_wrt_hazard()
        Compute ∂(AAL)/∂H (Manuscript Section 5)
    full_gradient_analysis()
        Compute complete gradient ∇J = [∂J/∂H, ∂J/∂C, ∂J/∂v]
    
    Notes
    -----
    - Exposure (v), vulnerability (C), and hazard (H) are stored as Variables
      to enable gradient computation
    - Typology indices (u) and grid (x_grid) are Constants (not differentiable)
    - All computations leverage TensorFlow's automatic differentiation
    - Compatible with GPU acceleration (TensorFlow Metal on macOS)
    
    Examples
    --------
    >>> # Generate portfolio data
    >>> v, u, C, x_grid, H = generate_synthetic_portfolio(1000, 5000)
    >>> 
    >>> # Initialize engine
    >>> engine = TensorialRiskEngine(v, u, C, x_grid, H)
    >>> 
    >>> # Compute risk metrics
    >>> J_matrix, metrics = engine.compute_loss_and_metrics()
    >>> print(f"Portfolio AAL: ${metrics['aal_portfolio'].numpy():,.2f}")
    >>> 
    >>> # Compute gradients
    >>> grad_C, _ = engine.gradient_wrt_vulnerability()
    >>> grad_v, _ = engine.gradient_wrt_exposure()
    >>> 
    >>> # Full analysis
    >>> analysis = engine.full_gradient_analysis()
    >>> print("Most sensitive assets:", analysis['grad_exposure'].numpy().argsort()[-10:])
    """
    
    def __init__(self, v: np.ndarray, u: np.ndarray, C: np.ndarray,
                 x_grid: np.ndarray, H: np.ndarray, lambdas: Optional[np.ndarray] = None,
                 Sigma: Optional[np.ndarray] = None, CoV: Optional[np.ndarray] = None):
        """
        Initialize the tensorial risk engine with portfolio data.

        Converts NumPy arrays to TensorFlow tensors and stores them as either
        Variables (for parameters we want to differentiate) or Constants.

        Parameters
        ----------
        v : np.ndarray, shape (N,)
            Exposure vector (replacement costs) ∈ ℝ^N
        u : np.ndarray, shape (N,)
            Typology index vector ∈ ℤ^N, values in {0, ..., K-1}
        C : np.ndarray, shape (K, M)
            Vulnerability matrix ∈ ℝ^(K×M)
        x_grid : np.ndarray, shape (M,)
            Intensity grid vector ∈ ℝ^M
        H : np.ndarray, shape (N, Q)
            Hazard intensity matrix ∈ ℝ^(N×Q)
        lambdas : np.ndarray, shape (Q,), optional
            Scenario occurrence rate vector ∈ ℝ^Q
            If None, uniform rates (1/Q) will be used
        Sigma : np.ndarray, shape (K, M), optional
            Vulnerability standard deviation matrix ∈ ℝ^(K×M)
            Sigma[k,m] = std dev of MDR at intensity x_grid[m] for typology k.
            Must satisfy Sigma² < C × (1 - C) for Beta distribution validity.
            Mutually exclusive with CoV.
        CoV : np.ndarray, shape (K,) or (K, M), optional
            Coefficient of variation for vulnerability curves.
            If shape (K,): scalar CoV per typology, Sigma = CoV[k] * C.
            If shape (K, M): per-point CoV, Sigma = CoV * C.
            Mutually exclusive with Sigma.

        Notes
        -----
        The initialization creates:
        - tf.Variable for v, C, H, lambdas, Sigma (differentiable parameters)
        - tf.Constant for u, x_grid (non-differentiable indices/grid)
        - Backward compatible: lambdas defaults to uniform if not provided
        - When Sigma is provided, vulnerability variance is propagated analytically
          through the law of total variance
        """
        self.v = tf.Variable(v, dtype=tf.float32, name='exposure')
        self.u = tf.constant(u, dtype=tf.int32, name='typology')
        self.C = tf.Variable(C, dtype=tf.float32, name='vulnerability')
        self.x_grid = tf.constant(x_grid, dtype=tf.float32, name='grid')
        self.H = tf.Variable(H, dtype=tf.float32, name='hazard')

        self.n_assets = v.shape[0]
        self.n_events = H.shape[1]
        self.n_typologies = C.shape[0]

        # Scenario occurrence rates (Manuscript Section 3c)
        if lambdas is None:
            # Backward compatibility: uniform rates
            lambdas = np.ones(self.n_events, dtype=np.float32) / self.n_events
        self.lambdas = tf.Variable(lambdas, dtype=tf.float32, name='scenario_rates')

        # Vulnerability uncertainty (Sigma matrix)
        if Sigma is not None and CoV is not None:
            raise ValueError("Provide either Sigma or CoV, not both.")

        if CoV is not None:
            CoV = np.asarray(CoV, dtype=np.float32)
            if CoV.ndim == 1:
                # Scalar CoV per typology: Sigma[k,m] = CoV[k] * C[k,m]
                Sigma = CoV[:, np.newaxis] * C
            else:
                # Per-point CoV: Sigma[k,m] = CoV[k,m] * C[k,m]
                Sigma = CoV * C

        if Sigma is not None:
            Sigma = np.asarray(Sigma, dtype=np.float32)
            self.Sigma = tf.Variable(Sigma, dtype=tf.float32, name='sigma_vulnerability')
        else:
            self.Sigma = None
    
    def compute_loss_and_metrics(self) -> Tuple[tf.Tensor, Dict]:
        """
        Compute complete loss matrix and rate-weighted risk metrics.
        
        Executes the probabilistic hazard formulation to generate the full
        N×Q loss matrix, then computes all derived risk metrics using
        scenario occurrence rates (Manuscript Section 3c).
        
        Returns
        -------
        J_matrix : tf.Tensor, shape (N, Q), dtype float32
            Complete loss matrix ∈ ℝ^(N×Q)
            J[i,q] = loss of asset i in event q
        metrics : dict
            Dictionary of rate-weighted risk metrics containing:
            - aal_per_asset : Rate-weighted AAL per asset ∈ ℝ^N
            - aal_portfolio : Total portfolio AAL (scalar)
            - mean_per_event_per_asset : Mean per event ∈ ℝ^N
            - variance_per_asset : Rate-weighted variance ∈ ℝ^N
            - std_per_asset : Standard deviation ∈ ℝ^N
            - loss_per_event : Total loss per event ∈ ℝ^Q
            - total_rate : Total occurrence rate Λ
        
        Notes
        -----
        This method implements Manuscript Section 3c with non-uniform
        scenario occurrence rates via the lambda vector.
        
        Examples
        --------
        >>> J_matrix, metrics = engine.compute_loss_and_metrics()
        >>> print(f"Shape: {J_matrix.shape}")
        >>> print(f"AAL: ${metrics['aal_portfolio'].numpy():,.2f}")
        >>> print(f"Total rate Λ: {metrics['total_rate'].numpy():.4f}")

        When Sigma is provided, metrics also contain:
        - variance_vulnerability_per_asset : Vulnerability variance component ∈ ℝ^N
        - variance_vulnerability_portfolio : Portfolio vulnerability variance (scalar)
        - variance_total_per_asset : Total variance (aleatory + vulnerability) ∈ ℝ^N
        - std_total_per_asset : Total std dev ∈ ℝ^N
        - variance_total_portfolio : Total portfolio variance (scalar)
        - std_total_portfolio : Total portfolio std dev (scalar)
        """
        J_matrix = probabilistic_loss_matrix(
            self.v, self.u, self.C, self.x_grid, self.H
        )

        if self.Sigma is not None:
            # Interpolate Sigma at hazard intensities
            sigma_matrix = _interpolate_matrix(
                self.u, self.Sigma, self.x_grid, self.H
            )
            metrics = compute_risk_metrics_with_uncertainty(
                J_matrix, self.lambdas, self.v, sigma_matrix
            )
        else:
            metrics = compute_risk_metrics(J_matrix, self.lambdas)

        return J_matrix, metrics
    
    def gradient_wrt_vulnerability(self) -> Tuple[tf.Tensor, Dict]:
        """
        Compute gradient of rate-weighted AAL w.r.t. vulnerability curves (Manuscript Section 4c).
        
        Calculates ∂(AAL)/∂C using automatic differentiation, answering the question:
        "How sensitive is portfolio risk to changes in each vulnerability curve point?"
        
        Uses rate-weighted formulation: AAL = Σ_q λ_q × Σ_i J[i,q]
        
        This gradient identifies which parts of which vulnerability curves have the
        greatest impact on portfolio loss, enabling:
        - Vulnerability model calibration
        - Identification of critical damage thresholds
        - Prioritization of curve refinement efforts
        
        Returns
        -------
        grad_C : tf.Tensor, shape (K, M), dtype float32
            Gradient of portfolio AAL w.r.t. vulnerability matrix ∈ ℝ^(K×M)
            grad_C[k,m] = ∂(AAL)/∂C[k,m]
            Positive values indicate: increasing C[k,m] increases AAL
        metrics : dict
            Current risk metrics (same as compute_loss_and_metrics)
        
        Notes
        -----
        - Uses TensorFlow's GradientTape for automatic differentiation
        - Gradient w.r.t. C is always computable (C is a Variable)
        - Gradient magnitude indicates sensitivity
        - Sign indicates direction of influence
        
        Applications
        ------------
        - Calibrate vulnerability curves to match historical losses
        - Identify which typologies drive portfolio risk
        - Quantify impact of uncertainty in damage functions
        
        Examples
        --------
        >>> grad_C, metrics = engine.gradient_wrt_vulnerability()
        >>> print(f"Max gradient: {grad_C.numpy().max():.2e}")
        >>> most_sensitive = np.unravel_index(grad_C.numpy().argmax(), grad_C.shape)
        >>> print(f"Most sensitive point: Typology {most_sensitive[0]}, Point {most_sensitive[1]}")
        """
        with tf.GradientTape() as tape:
            J_matrix, metrics = self.compute_loss_and_metrics()
            aal = metrics['aal_portfolio']
        
        grad_C = tape.gradient(aal, self.C)
        return grad_C, metrics
    
    def gradient_wrt_exposure(self) -> Tuple[tf.Tensor, Dict]:
        """
        Compute gradient of rate-weighted AAL w.r.t. exposure values (Manuscript Section 5c).
        
        Calculates ∂(AAL)/∂v using automatic differentiation, answering the question:
        "How much does portfolio AAL increase per additional dollar of exposure at each asset?"
        
        Uses rate-weighted formulation: ∂AAL/∂v_i = Σ_q λ_q × MDR[i,q]
        
        This gradient provides asset-level risk importance, enabling:
        - Identification of risk concentration
        - Retrofit/mitigation prioritization
        - Portfolio optimization
        - Risk-based capital allocation
        
        Returns
        -------
        grad_v : tf.Tensor, shape (N,), dtype float32
            Gradient of portfolio AAL w.r.t. exposure vector ∈ ℝ^N
            grad_v[i] = ∂(AAL)/∂v[i]
            Units: dimensionless ($/$ of exposure)
        metrics : dict
            Current risk metrics
        
        Interpretation
        --------------
        - grad_v[i] = 0.5 means: $1 more exposure at asset i → $0.50 more AAL
        - Higher gradient = asset contributes more to portfolio risk per dollar
        - Accounts for hazard, vulnerability, and spatial correlation
        
        Applications
        ------------
        - Identify which buildings to retrofit first
        - Portfolio risk concentration analysis
        - Risk-based pricing for insurance
        - Capital allocation across locations
        
        Examples
        --------
        >>> grad_v, metrics = engine.gradient_wrt_exposure()
        >>> top_10_risky = grad_v.numpy().argsort()[-10:][::-1]
        >>> print("Top 10 risk contributors by exposure sensitivity:")
        >>> for rank, idx in enumerate(top_10_risky, 1):
        ...     print(f"{rank}. Asset {idx}: ∂AAL/∂v = {grad_v[idx].numpy():.4f}")
        """
        with tf.GradientTape() as tape:
            J_matrix, metrics = self.compute_loss_and_metrics()
            aal = metrics['aal_portfolio']
        
        grad_v = tape.gradient(aal, self.v)
        return grad_v, metrics
    
    def gradient_wrt_hazard(self) -> Tuple[tf.Tensor, Dict]:
        """
        Compute gradient of rate-weighted AAL w.r.t. hazard intensities (Manuscript Section 5c).
        
        Calculates ∂(AAL)/∂H using automatic differentiation, answering the question:
        "How sensitive is portfolio risk to changes in hazard intensity estimates?"
        
        Uses rate-weighted formulation: ∂AAL/∂H[i,q] includes λ_q factor
        
        This gradient quantifies hazard uncertainty impact, enabling:
        - Hazard model sensitivity analysis
        - Identification of critical scenarios
        - Quantification of epistemic uncertainty
        - Guided improvement of hazard models
        
        Returns
        -------
        grad_H : tf.Tensor, shape (N, Q), dtype float32
            Gradient of portfolio AAL w.r.t. hazard matrix ∈ ℝ^(N×Q)
            grad_H[i,q] = ∂(AAL)/∂H[i,q]
            Units: $/g (dollars per unit intensity change)
        metrics : dict
            Current risk metrics
        
        Interpretation
        --------------
        - grad_H[i,q] > 0: increasing intensity at asset i in event q increases AAL
        - Magnitude indicates sensitivity to hazard changes
        - Large gradients identify critical asset-event combinations
        
        Applications
        ------------
        - Assess impact of hazard model uncertainty
        - Identify which events drive portfolio risk
        - Guide hazard model refinement priorities
        - Sensitivity to ground motion models
        
        Notes
        -----
        - For N=1000, Q=5000, gradient is 5M values (20MB)
        - Consider aggregating (e.g., sum over assets) for interpretation
        - Gradients near curve endpoints may be larger due to extrapolation
        
        Examples
        --------
        >>> grad_H, metrics = engine.gradient_wrt_hazard()
        >>> print(f"Average hazard sensitivity: {tf.reduce_mean(tf.abs(grad_H)).numpy():.2e}")
        >>> # Find most critical event
        >>> event_sensitivity = tf.reduce_sum(tf.abs(grad_H), axis=0)
        >>> critical_event = tf.argmax(event_sensitivity).numpy()
        >>> print(f"Most critical event: {critical_event}")
        """
        with tf.GradientTape() as tape:
            J_matrix, metrics = self.compute_loss_and_metrics()
            aal = metrics['aal_portfolio']
        
        grad_H = tape.gradient(aal, self.H)
        return grad_H, metrics
    
    def gradient_wrt_lambdas(self) -> Tuple[tf.Tensor, Dict]:
        """
        Compute gradient of AAL w.r.t. scenario occurrence rates (Manuscript Section 3c).
        
        Calculates ∂(AAL)/∂λ using automatic differentiation, answering the question:
        "How sensitive is portfolio AAL to changes in scenario occurrence rates?"
        
        This gradient quantifies scenario importance and enables:
        - Understanding which scenarios drive portfolio risk
        - Sensitivity to event catalog composition
        - Importance sampling analysis
        - Event set optimization
        
        Returns
        -------
        grad_lambdas : tf.Tensor, shape (Q,), dtype float32
            Gradient of portfolio AAL w.r.t. scenario rates ∈ ℝ^Q
            grad_lambdas[q] = ∂(AAL)/∂λ_q = Σ_i J[i,q]
            Units: dollars (portfolio loss for event q)
        metrics : dict
            Current risk metrics
        
        Interpretation
        --------------
        - grad_lambdas[q] = total portfolio loss in event q
        - Higher values indicate scenarios that contribute more to AAL
        - Positive values (always): more frequent events increase AAL
        - Magnitude indicates criticality of each scenario
        
        Applications
        ------------
        - Identify most critical scenarios in event catalog
        - Optimize importance sampling weights
        - Sensitivity to catalog completeness
        - Event set pruning decisions
        
        Notes
        -----
        - Gradient is simply the loss per event: ∂(Σ_q λ_q × L_q)/∂λ_q = L_q
        - Useful for understanding scenario contributions to risk
        - Can guide event catalog refinement
        
        Examples
        --------
        >>> grad_lambdas, metrics = engine.gradient_wrt_lambdas()
        >>> critical_events = tf.argsort(grad_lambdas, direction='DESCENDING')[:10]
        >>> print(f"Top 10 critical events: {critical_events.numpy()}")
        >>> print(f"Most critical event loss: ${grad_lambdas[critical_events[0]].numpy():,.2f}")
        """
        with tf.GradientTape() as tape:
            J_matrix, metrics = self.compute_loss_and_metrics()
            aal = metrics['aal_portfolio']
        
        grad_lambdas = tape.gradient(aal, self.lambdas)
        return grad_lambdas, metrics

    def gradient_wrt_sigma(self) -> Tuple[tf.Tensor, Dict]:
        """
        Compute gradient of total portfolio variance w.r.t. vulnerability
        uncertainty (Sigma matrix).

        Calculates ∂(Var_total)/∂Σ using automatic differentiation.
        The target is total variance (not AAL), because ∂AAL/∂Σ = 0 by
        design — the mean loss depends only on C (mean MDR), not on Σ.

        Requires that self.Sigma is not None.

        Returns
        -------
        grad_Sigma : tf.Tensor, shape (K, M), dtype float32
            Gradient of total portfolio variance w.r.t. Sigma matrix.
            grad_Sigma[k,m] = ∂(Var_total_portfolio)/∂Σ[k,m]
        metrics : dict
            Current risk metrics including vulnerability variance decomposition.

        Raises
        ------
        ValueError
            If Sigma was not provided at engine initialization.

        Notes
        -----
        - ∂AAL/∂Σ = 0 is mathematically correct (mean is independent of variance)
        - This gradient targets variance: identifies which Sigma entries most
          affect total portfolio risk dispersion
        - Positive gradient means increasing Σ[k,m] increases total variance

        Examples
        --------
        >>> engine = TensorialRiskEngine(v, u, C, x_grid, H, lambdas, Sigma=Sigma)
        >>> grad_Sigma, metrics = engine.gradient_wrt_sigma()
        >>> print(f"Max variance sensitivity: {grad_Sigma.numpy().max():.2e}")
        """
        if self.Sigma is None:
            raise ValueError("Sigma not provided. Initialize engine with Sigma or CoV.")

        with tf.GradientTape() as tape:
            J_matrix, metrics = self.compute_loss_and_metrics()
            target = metrics['variance_total_portfolio']

        grad_Sigma = tape.gradient(target, self.Sigma)
        return grad_Sigma, metrics

    def full_gradient_analysis(self) -> Dict:
        """
        Compute complete gradient analysis for all parameters (Manuscript Sections 3c-5c).
        
        Calculates the full gradient vector ∇J = [∂J/∂H, ∂J/∂C, ∂J/∂v, ∂J/∂λ] in a single
        pass using persistent gradient tape. This provides complete sensitivity
        information for portfolio optimization and uncertainty quantification.
        
        Returns
        -------
        analysis : dict
            Complete analysis results containing:
            
            grad_hazard : tf.Tensor, shape (N, Q), dtype float32
                ∂(AAL)/∂H - Hazard intensity sensitivity (rate-weighted)
                
            grad_vulnerability : tf.Tensor, shape (K, M), dtype float32
                ∂(AAL)/∂C - Vulnerability curve sensitivity (rate-weighted)
                
            grad_exposure : tf.Tensor, shape (N,), dtype float32
                ∂(AAL)/∂v - Exposure sensitivity (rate-weighted)
                
            grad_lambdas : tf.Tensor, shape (Q,), dtype float32
                ∂(AAL)/∂λ - Scenario occurrence rate sensitivity

            grad_sigma : tf.Tensor, shape (K, M), dtype float32, optional
                ∂(Var_total)/∂Σ - Vulnerability uncertainty sensitivity.
                Only present when Sigma was provided at initialization.

            metrics : dict
                All rate-weighted risk metrics (AAL, variance, etc.)

            loss_matrix : tf.Tensor, shape (N, Q), dtype float32
                Complete loss matrix J ∈ ℝ^(N×Q)
        
        Applications
        ------------
        This comprehensive analysis enables:
        
        1. **Portfolio Optimization**
           - Use grad_exposure to identify retrofit priorities
           - Minimize AAL subject to budget constraints
           
        2. **Model Calibration**
           - Use grad_vulnerability to fit curves to loss data
           - Gradient descent on vulnerability parameters
           
        3. **Uncertainty Quantification**
           - Use grad_hazard to propagate hazard uncertainty
           - Sensitivity to ground motion models
           
        4. **Risk Management**
           - Identify key drivers of portfolio risk
           - Allocate resources to maximum-impact interventions
        
        Notes
        -----
        - Uses persistent=True on GradientTape (must be manually deleted)
        - More efficient than calling three gradient methods separately
        - All gradients computed from same forward pass
        
        Examples
        --------
        >>> analysis = engine.full_gradient_analysis()
        >>> 
        >>> # Find most impactful typology
        >>> vuln_impact = tf.reduce_sum(tf.abs(analysis['grad_vulnerability']), axis=1)
        >>> print(f"Most impactful typology: {tf.argmax(vuln_impact).numpy()}")
        >>> 
        >>> # Find assets to retrofit
        >>> retrofit_priority = tf.argsort(analysis['grad_exposure'], direction='DESCENDING')[:10]
        >>> print(f"Top 10 assets for retrofit: {retrofit_priority.numpy()}")
        >>> 
        >>> # Check hazard sensitivity
        >>> hazard_sens = tf.reduce_mean(tf.abs(analysis['grad_hazard']))
        >>> print(f"Average hazard sensitivity: ${hazard_sens.numpy():.2f}/g")
        """
        with tf.GradientTape() as tape:
            J_matrix, metrics = self.compute_loss_and_metrics()
            aal = metrics['aal_portfolio']

        # Single backward pass for ALL gradients — the cheap gradient principle.
        # With a scalar target and list of sources, TF traverses the graph once.
        grad_H, grad_C, grad_v, grad_lambdas = tape.gradient(
            aal, [self.H, self.C, self.v, self.lambdas]
        )

        result = {
            'grad_hazard': grad_H,
            'grad_vulnerability': grad_C,
            'grad_exposure': grad_v,
            'grad_lambdas': grad_lambdas,
            'metrics': metrics,
            'loss_matrix': J_matrix
        }

        # Sigma gradient targets variance, not AAL (∂AAL/∂Σ = 0 by design)
        if self.Sigma is not None:
            with tf.GradientTape() as tape_var:
                J_matrix_v, metrics_v = self.compute_loss_and_metrics()
                var_total = metrics_v['variance_total_portfolio']

            result['grad_sigma'] = tape_var.gradient(var_total, self.Sigma)

        return result


# ==========================================
# 6. CLASSICAL RISK (Hazard-Curve Convolution)
# ==========================================

@tf.function
def classical_loss(v: tf.Tensor, u: tf.Tensor, C: tf.Tensor,
                   x_grid: tf.Tensor, hazard_poes: tf.Tensor,
                   hazard_imls: tf.Tensor) -> tf.Tensor:
    """
    Compute average annual loss per asset from hazard curves via classical risk
    convolution (differentiable).

    For each asset *i*, integrates the vulnerability mean-damage-ratio curve
    against the hazard curve using the trapezoidal rule:

        AAL_i = v_i × ∫ MDR(iml) × |dPoE/diml| diml
              ≈ v_i × Σ_l  MDR(iml_l) × (PoE_{l-1} − PoE_l)

    where PoE values decrease with increasing IML (exceedance probabilities),
    so (PoE_{l-1} − PoE_l) represents the probability mass in each IML bin.

    Parameters
    ----------
    v : tf.Tensor, shape (N,), dtype float32
        Exposure values per asset.
    u : tf.Tensor, shape (N,), dtype int32
        Typology index per asset, values in {0, ..., K-1}.
    C : tf.Tensor, shape (K, M), dtype float32
        Vulnerability matrix (K curves × M curve points).
    x_grid : tf.Tensor, shape (M,), dtype float32
        Intensity grid for vulnerability curves.
    hazard_poes : tf.Tensor, shape (N, L), dtype float32
        Hazard PoE matrix — hazard_poes[i,l] = P(IML > imls[l]) at asset i.
        Values should be monotonically decreasing along axis 1.
    hazard_imls : tf.Tensor, shape (L,), dtype float32
        IML levels corresponding to the hazard curve columns.

    Returns
    -------
    avg_loss : tf.Tensor, shape (N,), dtype float32
        Average annual loss per asset.

    Notes
    -----
    Fully differentiable w.r.t. v, C, hazard_poes.
    Uses the same linear interpolation scheme as probabilistic_loss_matrix
    to evaluate vulnerability at hazard IML levels.
    """
    N = tf.shape(v)[0]
    L = tf.shape(hazard_imls)[0]
    M = tf.shape(x_grid)[0]

    # Evaluate vulnerability at each hazard IML level for each asset
    # Tile hazard_imls across assets: shape (N, L)
    imls_tiled = tf.tile(tf.expand_dims(hazard_imls, 0), [N, 1])
    imls_flat = tf.reshape(imls_tiled, [-1])  # (N*L,)

    # Find grid indices for interpolation
    idx = tf.searchsorted(x_grid, imls_flat, side='right') - 1
    idx = tf.clip_by_value(idx, 0, M - 2)

    x_lower = tf.gather(x_grid, idx)
    x_upper = tf.gather(x_grid, idx + 1)
    alpha = (imls_flat - x_lower) / (x_upper - x_lower + 1e-8)

    # Repeat typology indices L times
    u_repeated = tf.tile(tf.expand_dims(u, 1), [1, L])
    u_flat = tf.reshape(u_repeated, [-1])  # (N*L,)

    c_flat = tf.reshape(C, [-1])
    flat_idx_lower = u_flat * M + idx
    flat_idx_upper = u_flat * M + (idx + 1)

    c_lower = tf.gather(c_flat, flat_idx_lower)
    c_upper = tf.gather(c_flat, flat_idx_upper)

    mdr_flat = (1.0 - alpha) * c_lower + alpha * c_upper
    mdr_matrix = tf.reshape(mdr_flat, [N, L])  # (N, L)

    # Compute ΔPoE = PoE_{l} - PoE_{l+1}  (probability mass in each bin)
    # Append zero column at the end (PoE beyond last IML = 0)
    poe_shifted = tf.concat([hazard_poes[:, 1:],
                             tf.zeros([N, 1], dtype=tf.float32)], axis=1)
    delta_poe = hazard_poes - poe_shifted  # (N, L), all >= 0

    # AAL_i = v_i × Σ_l MDR(iml_l) × ΔPoE_l
    avg_loss = v * tf.reduce_sum(mdr_matrix * delta_poe, axis=1)

    return avg_loss


# ==========================================
# 7. FRAGILITY & DAMAGE STATE FUNCTIONS
# ==========================================

@tf.function
def fragility_damage_distribution(u: tf.Tensor, F: tf.Tensor,
                                  x_grid: tf.Tensor,
                                  H: tf.Tensor) -> tf.Tensor:
    """
    Compute damage-state probability distributions from fragility curves
    (differentiable).

    For each asset *i* and event *q*, interpolates D fragility exceedance
    curves at the intensity H[i,q], then converts exceedance probabilities
    to damage-state probabilities via differencing:

        P(ds_0) = 1 − P(exceed ds_1)
        P(ds_d) = P(exceed ds_d) − P(exceed ds_{d+1})   for d = 1..D-1
        P(ds_D) = P(exceed ds_D)

    Parameters
    ----------
    u : tf.Tensor, shape (N,), dtype int32
        Typology index per asset, values in {0, ..., K-1}.
    F : tf.Tensor, shape (K, D, M), dtype float32
        Fragility tensor — F[k,d,m] = P(exceeding limit-state d | IML = x_grid[m])
        for typology k. D limit states, M IML points.
    x_grid : tf.Tensor, shape (M,), dtype float32
        Intensity grid for fragility curves (common across all typologies).
    H : tf.Tensor, shape (N, Q), dtype float32
        Hazard intensity matrix.

    Returns
    -------
    damage_probs : tf.Tensor, shape (N, Q, D+1), dtype float32
        Probability of each damage state per asset per event.
        damage_probs[i,q,0] = P(no damage), ..., damage_probs[i,q,D] = P(complete).

    Notes
    -----
    Fully differentiable w.r.t. F and H.
    Uses the same linear interpolation as probabilistic_loss_matrix.
    """
    N = tf.shape(H)[0]
    Q = tf.shape(H)[1]
    K = tf.shape(F)[0]
    D = tf.shape(F)[1]
    M = tf.shape(F)[2]

    H_flat = tf.reshape(H, [-1])  # (N*Q,)
    NQ = tf.shape(H_flat)[0]

    # Grid lookup
    idx = tf.searchsorted(x_grid, H_flat, side='right') - 1
    idx = tf.clip_by_value(idx, 0, M - 2)

    x_lower = tf.gather(x_grid, idx)
    x_upper = tf.gather(x_grid, idx + 1)
    alpha = (H_flat - x_lower) / (x_upper - x_lower + 1e-8)  # (NQ,)

    # Repeat typology indices Q times
    u_repeated = tf.tile(tf.expand_dims(u, 1), [1, Q])
    u_flat = tf.reshape(u_repeated, [-1])  # (NQ,)

    # For each limit state d, interpolate F[u, d, :] at the intensity
    # F is (K, D, M) — flatten to (K*D*M,)
    f_flat = tf.reshape(F, [-1])

    # Vectorised across all D limit states at once (no Python/tf.range loop)
    # Build d indices: (D,) → tile to (D, NQ) → flatten to (D*NQ,)
    d_range = tf.range(D)  # (D,)
    d_tiled = tf.tile(tf.expand_dims(d_range, 1), [1, NQ])  # (D, NQ)
    d_flat = tf.reshape(d_tiled, [-1])  # (D*NQ,)

    # Repeat u_flat and idx D times
    u_rep_d = tf.tile(tf.expand_dims(u_flat, 0), [D, 1])  # (D, NQ)
    u_rep_d = tf.reshape(u_rep_d, [-1])  # (D*NQ,)
    idx_rep_d = tf.tile(tf.expand_dims(idx, 0), [D, 1])   # (D, NQ)
    idx_rep_d = tf.reshape(idx_rep_d, [-1])                # (D*NQ,)
    alpha_rep_d = tf.tile(tf.expand_dims(alpha, 0), [D, 1])  # (D, NQ)
    alpha_rep_d = tf.reshape(alpha_rep_d, [-1])               # (D*NQ,)

    flat_lower_all = u_rep_d * (D * M) + d_flat * M + idx_rep_d
    flat_upper_all = u_rep_d * (D * M) + d_flat * M + (idx_rep_d + 1)
    f_lo_all = tf.gather(f_flat, flat_lower_all)
    f_hi_all = tf.gather(f_flat, flat_upper_all)
    exceed_flat = (1.0 - alpha_rep_d) * f_lo_all + alpha_rep_d * f_hi_all
    exceed_flat = tf.clip_by_value(exceed_flat, 0.0, 1.0)

    # Reshape to (D, NQ) then transpose to (NQ, D)
    exceed_all = tf.reshape(exceed_flat, [D, NQ])
    exceed_all = tf.transpose(exceed_all)  # (NQ, D)

    # Convert exceedance to damage-state probabilities
    # P(ds_0) = 1 - exceed[:,0]
    # P(ds_d) = exceed[:,d] - exceed[:,d+1]  for d=0..D-2
    # P(ds_D) = exceed[:, D-1]
    p_no_damage = 1.0 - exceed_all[:, 0:1]  # (NQ, 1)
    # Differences between consecutive limit states
    p_intermediate = exceed_all[:, :-1] - exceed_all[:, 1:]  # (NQ, D-1)
    p_complete = exceed_all[:, -1:]  # (NQ, 1)

    damage_probs_flat = tf.concat([p_no_damage, p_intermediate, p_complete],
                                  axis=1)  # (NQ, D+1)
    damage_probs_flat = tf.maximum(damage_probs_flat, 0.0)  # numerical safety

    damage_probs = tf.reshape(damage_probs_flat, [N, Q, D + 1])
    return damage_probs


@tf.function
def consequence_loss(damage_probs: tf.Tensor, consequence_ratios: tf.Tensor,
                     v: tf.Tensor, u: tf.Tensor) -> tf.Tensor:
    """
    Compute losses from damage-state probabilities and consequence ratios
    (differentiable).

    For each asset *i* and event *q*:
        loss[i,q] = v[i] × Σ_d damage_probs[i,q,d] × consequence_ratios[u[i],d]

    Parameters
    ----------
    damage_probs : tf.Tensor, shape (N, Q, D+1), dtype float32
        Damage-state probabilities (from fragility_damage_distribution).
    consequence_ratios : tf.Tensor, shape (K, D+1), dtype float32
        Loss ratio per damage state per typology.
        consequence_ratios[k,0] = loss ratio for no-damage (typically 0),
        consequence_ratios[k,D] = loss ratio for complete damage (typically 1).
    v : tf.Tensor, shape (N,), dtype float32
        Exposure values per asset.
    u : tf.Tensor, shape (N,), dtype int32
        Typology index per asset, values in {0, ..., K-1}.

    Returns
    -------
    loss_matrix : tf.Tensor, shape (N, Q), dtype float32
        Loss per asset per event.

    Notes
    -----
    Fully differentiable w.r.t. damage_probs, consequence_ratios, v.
    """
    # Gather consequence ratios for each asset's typology: (N, D+1)
    cr_per_asset = tf.gather(consequence_ratios, u)  # (N, D+1)

    # Weighted sum over damage states: (N, Q)
    # damage_probs is (N, Q, D+1), cr_per_asset is (N, D+1)
    # Expand cr to (N, 1, D+1) for broadcasting
    cr_expanded = tf.expand_dims(cr_per_asset, 1)  # (N, 1, D+1)
    mdr = tf.reduce_sum(damage_probs * cr_expanded, axis=2)  # (N, Q)

    loss_matrix = tf.expand_dims(v, 1) * mdr  # (N, Q)
    return loss_matrix


# ==========================================
# 8. CLASSICAL DAMAGE (Hazard-Curve × Fragility)
# ==========================================

@tf.function
def classical_damage(u: tf.Tensor, F: tf.Tensor, x_grid: tf.Tensor,
                     hazard_poes: tf.Tensor,
                     hazard_imls: tf.Tensor) -> tf.Tensor:
    """
    Compute expected damage-state fractions per asset from hazard curves and
    fragility functions (differentiable).

    Integrates fragility exceedance probabilities against the hazard curve:

        E[P(ds_d)] = ∫ P(ds_d | iml) × |dPoE/diml| diml

    using the same ΔPoE discretization as classical_loss.

    Parameters
    ----------
    u : tf.Tensor, shape (N,), dtype int32
        Typology index per asset.
    F : tf.Tensor, shape (K, D, M), dtype float32
        Fragility tensor.
    x_grid : tf.Tensor, shape (M,), dtype float32
        Intensity grid for fragility curves.
    hazard_poes : tf.Tensor, shape (N, L), dtype float32
        Hazard PoE matrix.
    hazard_imls : tf.Tensor, shape (L,), dtype float32
        IML levels for hazard curves.

    Returns
    -------
    expected_damage : tf.Tensor, shape (N, D+1), dtype float32
        Expected fraction in each damage state per asset.

    Notes
    -----
    Fully differentiable w.r.t. F, hazard_poes.
    """
    N = tf.shape(hazard_poes)[0]
    L = tf.shape(hazard_imls)[0]
    K = tf.shape(F)[0]
    D = tf.shape(F)[1]
    M = tf.shape(F)[2]

    # Evaluate fragility at hazard IML levels
    imls_tiled = tf.tile(tf.expand_dims(hazard_imls, 0), [N, 1])  # (N, L)
    imls_flat = tf.reshape(imls_tiled, [-1])  # (N*L,)
    NL = tf.shape(imls_flat)[0]

    idx = tf.searchsorted(x_grid, imls_flat, side='right') - 1
    idx = tf.clip_by_value(idx, 0, M - 2)

    x_lower = tf.gather(x_grid, idx)
    x_upper = tf.gather(x_grid, idx + 1)
    alpha = (imls_flat - x_lower) / (x_upper - x_lower + 1e-8)

    u_repeated = tf.tile(tf.expand_dims(u, 1), [1, L])
    u_flat = tf.reshape(u_repeated, [-1])  # (NL,)

    f_flat = tf.reshape(F, [-1])

    # Vectorised across all D limit states at once (no tf.range loop)
    d_range = tf.range(D)
    d_tiled = tf.tile(tf.expand_dims(d_range, 1), [1, NL])  # (D, NL)
    d_flat = tf.reshape(d_tiled, [-1])  # (D*NL,)

    u_rep_d = tf.tile(tf.expand_dims(u_flat, 0), [D, 1])
    u_rep_d = tf.reshape(u_rep_d, [-1])  # (D*NL,)
    idx_rep_d = tf.tile(tf.expand_dims(idx, 0), [D, 1])
    idx_rep_d = tf.reshape(idx_rep_d, [-1])
    alpha_rep_d = tf.tile(tf.expand_dims(alpha, 0), [D, 1])
    alpha_rep_d = tf.reshape(alpha_rep_d, [-1])

    flat_lower_all = u_rep_d * (D * M) + d_flat * M + idx_rep_d
    flat_upper_all = u_rep_d * (D * M) + d_flat * M + (idx_rep_d + 1)
    f_lo_all = tf.gather(f_flat, flat_lower_all)
    f_hi_all = tf.gather(f_flat, flat_upper_all)
    exceed_flat = (1.0 - alpha_rep_d) * f_lo_all + alpha_rep_d * f_hi_all
    exceed_flat = tf.clip_by_value(exceed_flat, 0.0, 1.0)

    exceed_all = tf.reshape(exceed_flat, [D, NL])
    exceed_all = tf.transpose(exceed_all)  # (NL, D)

    # Convert to damage-state probs
    p_no_damage = 1.0 - exceed_all[:, 0:1]
    p_intermediate = exceed_all[:, :-1] - exceed_all[:, 1:]
    p_complete = exceed_all[:, -1:]
    damage_probs_flat = tf.concat([p_no_damage, p_intermediate, p_complete],
                                  axis=1)  # (NL, D+1)
    damage_probs_flat = tf.maximum(damage_probs_flat, 0.0)

    damage_probs = tf.reshape(damage_probs_flat, [N, L, D + 1])  # (N, L, D+1)

    # Compute ΔPoE
    poe_shifted = tf.concat([hazard_poes[:, 1:],
                             tf.zeros([N, 1], dtype=tf.float32)], axis=1)
    delta_poe = hazard_poes - poe_shifted  # (N, L)

    # Weight damage probs by ΔPoE and sum over IML levels
    # (N, L, D+1) × (N, L, 1) → sum over L → (N, D+1)
    delta_poe_expanded = tf.expand_dims(delta_poe, 2)  # (N, L, 1)
    expected_damage = tf.reduce_sum(damage_probs * delta_poe_expanded, axis=1)

    return expected_damage


# ==========================================
# 9. BENEFIT-COST RATIO
# ==========================================

@tf.function
def benefit_cost_ratio(aal_original: tf.Tensor, aal_retrofitted: tf.Tensor,
                       retrofit_cost: tf.Tensor,
                       interest_rate: float = 0.05,
                       asset_life_expectancy: float = 50.0) -> tf.Tensor:
    """
    Compute benefit-cost ratio for retrofitting (differentiable).

    BCR_i = (AAL_original_i − AAL_retrofitted_i) × BPVF / retrofit_cost_i

    where BPVF = (1 − (1+r)^(−T)) / r is the Benefit Present Value Factor.

    Parameters
    ----------
    aal_original : tf.Tensor, shape (N,), dtype float32
        Average annual loss per asset with original vulnerability.
    aal_retrofitted : tf.Tensor, shape (N,), dtype float32
        Average annual loss per asset with retrofitted vulnerability.
    retrofit_cost : tf.Tensor, shape (N,), dtype float32
        Cost of retrofitting each asset.
    interest_rate : float
        Annual discount rate (e.g. 0.05 for 5%).
    asset_life_expectancy : float
        Expected remaining life of the asset in years.

    Returns
    -------
    bcr : tf.Tensor, shape (N,), dtype float32
        Benefit-cost ratio per asset.

    Notes
    -----
    Fully differentiable w.r.t. aal_original, aal_retrofitted, retrofit_cost.
    """
    r = tf.constant(interest_rate, dtype=tf.float32)
    T = tf.constant(asset_life_expectancy, dtype=tf.float32)
    bpvf = (1.0 - tf.pow(1.0 + r, -T)) / (r + 1e-10)

    benefit = (aal_original - aal_retrofitted) * bpvf
    bcr = benefit / (retrofit_cost + 1e-10)
    return bcr
