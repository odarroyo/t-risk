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
                                 lambda_distribution: str = 'exponential') -> Tuple:
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
    
    Notes
    -----
    - Uses sigmoid functions to generate realistic vulnerability curves
    - Different typologies have varying fragility (steepness) and thresholds
    - Hazard intensities are uniformly distributed for demonstration purposes
    - Exponential distribution for lambdas mimics importance sampling typical in CAT modeling
    - Fixed random seed (42) ensures reproducible results
    
    Examples
    --------
    >>> v, u, C, x, H, lambdas = generate_synthetic_portfolio(1000, 5000, 5, 20)
    >>> print(v.shape, u.shape, C.shape, x.shape, H.shape, lambdas.shape)
    (1000,) (1000,) (5, 20) (20,) (1000, 5000) (5000,)
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
    
    return v_exposure, u_typology, C_matrix, x_grid, H_intensities, lambdas_out

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
    N = tf.shape(H)[0]
    Q = tf.shape(H)[1]
    M = tf.shape(x_grid)[0]
    
    # Flatten H for vectorized operations
    H_flat = tf.reshape(H, [-1])
    
    # Find grid indices
    idx = tf.searchsorted(x_grid, H_flat, side='right') - 1
    idx = tf.clip_by_value(idx, 0, M - 2)
    
    # Grid boundaries
    x_lower = tf.gather(x_grid, idx)
    x_upper = tf.gather(x_grid, idx + 1)
    
    # Interpolation weight α (broadcast for all N×Q combinations)
    alpha = (H_flat - x_lower) / (x_upper - x_lower + 1e-8)
    
    # Repeat typology indices Q times (for all events)
    u_repeated = tf.tile(tf.expand_dims(u, 1), [1, Q])
    u_flat = tf.reshape(u_repeated, [-1])
    
    # Use flat indexing for Metal GPU compatibility
    c_flat = tf.reshape(C, [-1])
    flat_idx_lower = u_flat * M + idx
    flat_idx_upper = u_flat * M + (idx + 1)
    
    c_lower = tf.gather(c_flat, flat_idx_lower)
    c_upper = tf.gather(c_flat, flat_idx_upper)
    
    # Mean Damage Ratio for all (N, Q) combinations
    mdr_flat = (1.0 - alpha) * c_lower + alpha * c_upper
    mdr_matrix = tf.reshape(mdr_flat, [N, Q])
    
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
                 x_grid: np.ndarray, H: np.ndarray, lambdas: Optional[np.ndarray] = None):
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
        
        Notes
        -----
        The initialization creates:
        - tf.Variable for v, C, H, lambdas (differentiable parameters)
        - tf.Constant for u, x_grid (non-differentiable indices/grid)
        - Backward compatible: lambdas defaults to uniform if not provided
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
        """
        J_matrix = probabilistic_loss_matrix(
            self.v, self.u, self.C, self.x_grid, self.H
        )
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
        with tf.GradientTape(persistent=True) as tape:
            J_matrix, metrics = self.compute_loss_and_metrics()
            aal = metrics['aal_portfolio']
        
        grad_H = tape.gradient(aal, self.H)
        grad_C = tape.gradient(aal, self.C)
        grad_v = tape.gradient(aal, self.v)
        grad_lambdas = tape.gradient(aal, self.lambdas)
        
        del tape  # Clean up persistent tape
        
        return {
            'grad_hazard': grad_H,
            'grad_vulnerability': grad_C,
            'grad_exposure': grad_v,
            'grad_lambdas': grad_lambdas,
            'metrics': metrics,
            'loss_matrix': J_matrix
        }
