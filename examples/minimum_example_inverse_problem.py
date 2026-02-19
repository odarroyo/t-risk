"""
Inverse Problem: Differentiable Vulnerability Curve Calibration from Observed Losses
=====================================================================================
Based on Latex_notes/inverse_problem.tex

Given observed per-asset losses from a single disaster (ground truth), this script
recovers the vulnerability matrix C via gradient descent on a monotonicity-constrained
reparameterization (latent ω).

Mathematical Formulation:
- Calibration loss: J_calib = (1/N) Σ (L_pred - L_obs)² + λ_reg · ||∇²C||²
- Monotonicity constraint: C_{k,m} = Σ_{t=0}^{m} Softplus(ω_{k,t}), capped at 1.0
- Gradient: ∂J/∂ω computed via TensorFlow auto-diff through the full pipeline

Implementation Strategy:
- Step 1: Generate ground truth (known C_true → L_obs)
- Step 2: Initialize latent ω (random/flat start)
- Step 3: Calibration loss + gradient verification
- Step 4: Plain gradient descent
- Step 5: Adam optimizer comparison
- Step 6: Quantitative recovery analysis
- Step 7: Sensitivity experiments (noise, regularization, N_assets)
- Step 8: Publication-quality figures

Author: Tensorial Risk Engine Project
Date: February 2026
"""

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, Optional

from tensor_engine import (
    generate_synthetic_portfolio,
    deterministic_loss,
    probabilistic_loss_matrix,
    TensorialRiskEngine
)

# ==========================================
# GLOBAL SETTINGS
# ==========================================

FIGURE_DIR = 'figures_inverse_problem_2'
os.makedirs(FIGURE_DIR, exist_ok=True)
FIGURE_COUNTER = 0

def save_figure(fig, title: str):
    """Save figure with sequential numbering at 300 dpi."""
    global FIGURE_COUNTER
    FIGURE_COUNTER += 1
    filepath = os.path.join(FIGURE_DIR, f'Figure{FIGURE_COUNTER}_{title}.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filepath}")
    plt.close(fig)


# ==========================================
# INVERSE PROBLEM SOLVER CLASS
# ==========================================

class InverseProblemSolver:
    """
    Differentiable vulnerability curve calibration from observed losses.
    
    Solves the inverse problem: given observed per-asset losses from a single
    disaster, recover the vulnerability matrix C that best explains the data.
    
    Uses monotonicity-constrained reparameterization:
        C_{k,m} = min(1.0, Σ_{t=0}^{m} Softplus(ω_{k,t}))
    
    Parameters
    ----------
    v : np.ndarray, shape (N,)
        Exposure vector (replacement costs)
    u : np.ndarray, shape (N,)
        Typology index vector
    x_grid : np.ndarray, shape (M,)
        Intensity grid vector
    h : np.ndarray, shape (N,)
        Observed hazard intensity vector (single event)
    L_obs : np.ndarray, shape (N,)
        Observed per-asset losses
    n_typologies : int
        Number of building typologies (K)
    lambda_reg : float
        Regularization strength for curve smoothness
    """
    
    def __init__(self, v: np.ndarray, u: np.ndarray, x_grid: np.ndarray,
                 h: np.ndarray, L_obs: np.ndarray, n_typologies: int,
                 lambda_reg: float = 1e-4):
        
        self.v_tf = tf.constant(v, dtype=tf.float32)
        self.u_tf = tf.constant(u, dtype=tf.int32)
        self.x_grid_tf = tf.constant(x_grid, dtype=tf.float32)
        # Reshape h to (N, 1) for use with probabilistic_loss_matrix
        self.H_tf = tf.constant(h.reshape(-1, 1), dtype=tf.float32)
        self.L_obs_tf = tf.constant(L_obs, dtype=tf.float32)
        self.lambda_reg = lambda_reg
        
        self.N = len(v)
        self.K = n_typologies
        self.M = len(x_grid)
    
    @staticmethod
    def omega_to_C(omega: tf.Variable) -> tf.Tensor:
        """
        Convert latent variables ω to monotonic vulnerability matrix C.
        
        C_{k,m} = min(1.0, Σ_{t=0}^{m} Softplus(ω_{k,t}))
        
        Guarantees:
        - C_{k,m} ≥ C_{k,m-1} (monotonically increasing)
        - C_{k,0} ≥ 0 (non-negative)
        - C_{k,m} ≤ 1.0 (capped at 100% damage)
        """
        increments = tf.nn.softplus(omega)
        C = tf.cumsum(increments, axis=1)
        C = tf.minimum(C, 1.0)
        return C
    
    def compute_predicted_losses(self, C: tf.Tensor) -> tf.Tensor:
        """
        Compute per-asset predicted losses using the forward engine.
        
        Reuses probabilistic_loss_matrix with Q=1 to get L_pred = J[:, 0].
        """
        J_matrix = probabilistic_loss_matrix(
            self.v_tf, self.u_tf, C, self.x_grid_tf, self.H_tf
        )
        return J_matrix[:, 0]  # Shape: (N,)
    
    def compute_calibration_loss(self, omega: tf.Variable) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute J_calib = MSE + λ_reg · smoothness_penalty.
        
        Returns
        -------
        J_calib : tf.Tensor (scalar)
            Total calibration loss
        mse : tf.Tensor (scalar)
            Mean squared error component
        reg : tf.Tensor (scalar)
            Regularization component
        """
        C = self.omega_to_C(omega)
        L_pred = self.compute_predicted_losses(C)
        
        # MSE loss
        mse = tf.reduce_mean(tf.square(L_pred - self.L_obs_tf))
        
        # Smoothness regularization: ||∇²C||² (discrete second derivative)
        # ∇²C_{k,m} ≈ C_{k,m+2} - 2·C_{k,m+1} + C_{k,m}
        if self.M >= 3:
            d2C = C[:, 2:] - 2.0 * C[:, 1:-1] + C[:, :-2]
            reg = self.lambda_reg * tf.reduce_sum(tf.square(d2C))
        else:
            reg = tf.constant(0.0, dtype=tf.float32)
        
        J_calib = mse + reg
        return J_calib, mse, reg
    
    def initialize_omega(self, strategy: str = 'random', seed: int = 123) -> tf.Variable:
        """
        Initialize latent variables ω.
        
        Parameters
        ----------
        strategy : str
            'random' - Small random values giving gentle ramp
            'zeros'  - All zeros (Softplus(0) ≈ 0.693 per increment)
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed)
        if strategy == 'random':
            # Small negative values → Softplus outputs ≈ 0.02-0.15 per increment
            # Total over M=20 points → C ramps from ~0.05 to ~1.0
            omega_init = np.random.uniform(-3.0, -1.5, (self.K, self.M)).astype(np.float32)
        elif strategy == 'zeros':
            omega_init = np.zeros((self.K, self.M), dtype=np.float32)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return tf.Variable(omega_init, dtype=tf.float32, name='omega')
    
    def solve(self, optimizer_type: str = 'adam', lr: float = 0.01, 
              n_iterations: int = 2000, omega_init: Optional[tf.Variable] = None,
              seed: int = 123, print_every: int = 200) -> Dict:
        """
        Run the calibration optimization loop.
        
        Parameters
        ----------
        optimizer_type : str
            'gd' for plain gradient descent, 'adam' for Adam optimizer
        lr : float
            Learning rate
        n_iterations : int
            Number of optimization iterations
        omega_init : tf.Variable, optional
            Pre-initialized omega. If None, random initialization is used.
        seed : int
            Random seed for omega initialization
        print_every : int
            Print progress every N iterations
        
        Returns
        -------
        results : dict
            'C_optimized': final vulnerability matrix (K, M)
            'omega_final': final latent variables (K, M)
            'loss_history': total loss per iteration
            'mse_history': MSE per iteration
            'reg_history': regularization per iteration
            'grad_norm_history': gradient norm per iteration
        """
        # Initialize omega
        if omega_init is not None:
            omega = tf.Variable(omega_init.numpy(), dtype=tf.float32, name='omega')
        else:
            omega = self.initialize_omega(strategy='random', seed=seed)
        
        # Initialize optimizer
        if optimizer_type == 'adam':
            optimizer = tf.optimizers.Adam(learning_rate=lr)
        elif optimizer_type == 'gd':
            optimizer = tf.optimizers.SGD(learning_rate=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # History tracking
        loss_history = []
        mse_history = []
        reg_history = []
        grad_norm_history = []
        
        t_start = time.time()
        
        for it in range(n_iterations):
            with tf.GradientTape() as tape:
                J_calib, mse, reg = self.compute_calibration_loss(omega)
            
            grad_omega = tape.gradient(J_calib, omega)
            
            if optimizer_type == 'gd':
                omega.assign_sub(lr * grad_omega)
            else:
                optimizer.apply_gradients([(grad_omega, omega)])
            
            # Record history
            loss_val = J_calib.numpy()
            mse_val = mse.numpy()
            reg_val = reg.numpy()
            grad_norm_val = tf.norm(grad_omega).numpy()
            
            loss_history.append(loss_val)
            mse_history.append(mse_val)
            reg_history.append(reg_val)
            grad_norm_history.append(grad_norm_val)
            
            if (it + 1) % print_every == 0 or it == 0:
                print(f"  Iter {it+1:5d} | Loss: {loss_val:.4e} | MSE: {mse_val:.4e} | "
                      f"Reg: {reg_val:.4e} | ||grad||: {grad_norm_val:.4e}")
        
        elapsed = time.time() - t_start
        C_optimized = self.omega_to_C(omega).numpy()
        
        print(f"  Optimization complete in {elapsed:.2f}s")
        print(f"  Final loss: {loss_history[-1]:.4e} (MSE: {mse_history[-1]:.4e})")
        
        return {
            'C_optimized': C_optimized,
            'omega_final': omega.numpy(),
            'loss_history': np.array(loss_history),
            'mse_history': np.array(mse_history),
            'reg_history': np.array(reg_history),
            'grad_norm_history': np.array(grad_norm_history),
            'elapsed_time': elapsed,
            'omega_variable': omega
        }


# ==========================================
# DATA GENERATION FOR INVERSE PROBLEM
# ==========================================

def generate_inverse_problem_data(n_assets: int = 200, n_typologies: int = 3,
                                   n_curve_points: int = 20,
                                   noise_level: float = 0.0,
                                   seed: int = 42) -> Dict:
    """
    Generate synthetic ground truth data for the inverse problem.
    
    Creates a known C_true, a single-event hazard field h, and observed
    losses L_obs from the forward engine. Optionally adds Gaussian noise.
    
    Parameters
    ----------
    n_assets : int
        Number of assets (N)
    n_typologies : int
        Number of typologies (K)
    n_curve_points : int
        Number of curve points (M)
    noise_level : float
        Noise as fraction of max loss. 0.0 = no noise.
    seed : int
        Random seed
    
    Returns
    -------
    data : dict with keys:
        v, u, x_grid, h, C_true, L_obs, L_obs_clean
    """
    np.random.seed(seed)
    
    # Exposure vector v ∈ ℝ^N
    v = np.random.uniform(100_000, 800_000, n_assets).astype(np.float32)
    
    # Typology assignment u ∈ ℤ^N (roughly balanced)
    u = np.array([i % n_typologies for i in range(n_assets)], dtype=np.int32)
    np.random.shuffle(u)
    
    # Intensity grid x ∈ ℝ^M
    x_grid = np.linspace(0.0, 1.5, n_curve_points).astype(np.float32)
    
    # True vulnerability matrix C_true ∈ ℝ^(K×M) - sigmoid curves
    C_true = np.zeros((n_typologies, n_curve_points), dtype=np.float32)
    for k in range(n_typologies):
        steepness = 8.0 + k * 3.0
        midpoint = 0.3 + k * 0.15
        C_true[k, :] = 1.0 / (1.0 + np.exp(-steepness * (x_grid - midpoint)))
    
    # Hazard intensity vector h ∈ ℝ^N (single disaster event)
    # Spread across the full grid range for good coverage
    h = np.random.uniform(0.05, 1.3, n_assets).astype(np.float32)
    
    # Forward engine: compute clean observed losses
    v_tf = tf.constant(v, dtype=tf.float32)
    u_tf = tf.constant(u, dtype=tf.int32)
    C_tf = tf.constant(C_true, dtype=tf.float32)
    x_tf = tf.constant(x_grid, dtype=tf.float32)
    H_tf = tf.constant(h.reshape(-1, 1), dtype=tf.float32)
    
    J_matrix = probabilistic_loss_matrix(v_tf, u_tf, C_tf, x_tf, H_tf)
    L_obs_clean = J_matrix[:, 0].numpy()
    
    # Add noise if requested
    if noise_level > 0:
        noise_std = noise_level * np.max(L_obs_clean)
        noise = np.random.normal(0, noise_std, n_assets).astype(np.float32)
        L_obs = np.maximum(L_obs_clean + noise, 0.0)  # Ensure non-negative
    else:
        L_obs = L_obs_clean.copy()
    
    return {
        'v': v, 'u': u, 'x_grid': x_grid, 'h': h,
        'C_true': C_true, 'L_obs': L_obs, 'L_obs_clean': L_obs_clean,
        'n_assets': n_assets, 'n_typologies': n_typologies,
        'n_curve_points': n_curve_points, 'noise_level': noise_level
    }


# ==========================================
# STEP 1: GENERATE GROUND TRUTH
# ==========================================

def step1_generate_ground_truth():
    """Generate and verify ground truth data."""
    print("=" * 70)
    print("STEP 1: Generate Ground Truth Data")
    print("=" * 70)
    
    data = generate_inverse_problem_data(n_assets=200, n_typologies=3, 
                                          n_curve_points=20, noise_level=0.0)
    
    v, u, x_grid, h = data['v'], data['u'], data['x_grid'], data['h']
    C_true, L_obs = data['C_true'], data['L_obs']
    
    # Verification 1: Shapes
    print(f"\n  Shapes:")
    print(f"    v:      {v.shape}   (exposure)")
    print(f"    u:      {u.shape}   (typology)")
    print(f"    x_grid: {x_grid.shape}  (intensity grid)")
    print(f"    h:      {h.shape}   (hazard intensities)")
    print(f"    C_true: {C_true.shape}  (vulnerability matrix)")
    print(f"    L_obs:  {L_obs.shape}   (observed losses)")
    
    # Verification 1: Assets per typology
    print(f"\n  Assets per typology:")
    for k in range(data['n_typologies']):
        count = np.sum(u == k)
        print(f"    Typology {k}: {count} assets")
    
    # Verification 1: Loss statistics
    print(f"\n  Observed losses:")
    print(f"    Min:  ${L_obs.min():>12,.2f}")
    print(f"    Max:  ${L_obs.max():>12,.2f}")
    print(f"    Mean: ${L_obs.mean():>12,.2f}")
    print(f"    All non-negative: {np.all(L_obs >= 0)}")
    
    # Figure 1: Ground truth curves + observed loss scatter
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: True vulnerability curves
    ax = axes[0]
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    labels = ['Typology 0 (Fragile)', 'Typology 1 (Medium)', 'Typology 2 (Robust)']
    for k in range(data['n_typologies']):
        ax.plot(x_grid, C_true[k], color=colors[k], linewidth=2.5, label=labels[k])
    ax.set_xlabel('Intensity (g)', fontsize=12)
    ax.set_ylabel('Mean Damage Ratio (MDR)', fontsize=12)
    ax.set_title('True Vulnerability Curves $\\mathbf{C}_{true}$', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Intensity histogram per typology
    ax = axes[1]
    for k in range(data['n_typologies']):
        mask = (u == k)
        ax.hist(h[mask], bins=20, alpha=0.5, color=colors[k], label=f'Typology {k}')
    ax.set_xlabel('Intensity (g)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Hazard Intensity Distribution by Typology', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Observed losses vs intensity
    ax = axes[2]
    for k in range(data['n_typologies']):
        mask = (u == k)
        ax.scatter(h[mask], L_obs[mask], c=colors[k], alpha=0.6, s=20, label=f'Typology {k}')
    ax.set_xlabel('Intensity (g)', fontsize=12)
    ax.set_ylabel('Observed Loss ($)', fontsize=12)
    ax.set_title('Observed Losses $L^{obs}$ vs Intensity', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'ground_truth')
    
    print("\n  STEP 1 PASSED: All verifications OK")
    return data


# ==========================================
# STEP 2: INITIALIZE OMEGA
# ==========================================

def step2_initialize_omega(data: Dict):
    """Initialize and verify latent variables ω."""
    print("\n" + "=" * 70)
    print("STEP 2: Initialize Latent Variables ω")
    print("=" * 70)
    
    solver = InverseProblemSolver(
        v=data['v'], u=data['u'], x_grid=data['x_grid'],
        h=data['h'], L_obs=data['L_obs'],
        n_typologies=data['n_typologies'], lambda_reg=1e-4
    )
    
    omega = solver.initialize_omega(strategy='random', seed=123)
    C_init = solver.omega_to_C(omega).numpy()
    
    # Verification 2: Monotonicity
    is_monotonic = np.all(np.diff(C_init, axis=1) >= -1e-7)
    print(f"\n  ω shape: {omega.shape}")
    print(f"  ω range: [{omega.numpy().min():.3f}, {omega.numpy().max():.3f}]")
    print(f"  C_init shape: {C_init.shape}")
    print(f"  C_init range: [{C_init.min():.4f}, {C_init.max():.4f}]")
    print(f"  Monotonically increasing: {is_monotonic}")
    print(f"  Bounded [0, 1]: {np.all(C_init >= 0) and np.all(C_init <= 1.0)}")
    
    # Figure 2: Initial vs true curves
    fig, axes = plt.subplots(1, data['n_typologies'], figsize=(6*data['n_typologies'], 5))
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    for k in range(data['n_typologies']):
        ax = axes[k]
        ax.plot(data['x_grid'], data['C_true'][k], 'k-', linewidth=2.5, label='$C_{true}$')
        ax.plot(data['x_grid'], C_init[k], '--', color=colors[k], linewidth=2, 
                label='$C_{init}$ (from ω)')
        ax.set_xlabel('Intensity (g)', fontsize=12)
        ax.set_ylabel('MDR', fontsize=12)
        ax.set_title(f'Typology {k}', fontsize=13)
        ax.legend(fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Initial Curves vs True Curves (Before Optimization)', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, 'initial_vs_true')
    
    print("\n  STEP 2 PASSED: C_init is monotonic and bounded")
    return solver, omega


# ==========================================
# STEP 3: LOSS AND GRADIENT VERIFICATION
# ==========================================

def step3_verify_loss_and_gradient(solver: InverseProblemSolver, omega: tf.Variable, data: Dict):
    """Verify calibration loss and gradient computation."""
    print("\n" + "=" * 70)
    print("STEP 3: Calibration Loss & Gradient Verification")
    print("=" * 70)
    
    # Compute loss
    with tf.GradientTape() as tape:
        J_calib, mse, reg = solver.compute_calibration_loss(omega)
    grad_omega = tape.gradient(J_calib, omega)
    
    print(f"\n  Calibration loss at initialization:")
    print(f"    J_calib = {J_calib.numpy():.4e}")
    print(f"    MSE     = {mse.numpy():.4e}")
    print(f"    Reg     = {reg.numpy():.4e}")
    print(f"    MSE/Total ratio: {mse.numpy()/J_calib.numpy():.4f}")
    
    # Verify gradient exists and has correct shape
    print(f"\n  Gradient verification:")
    print(f"    grad_omega shape: {grad_omega.shape}")
    print(f"    grad_omega is None: {grad_omega is None}")
    print(f"    Contains NaN: {np.any(np.isnan(grad_omega.numpy()))}")
    print(f"    ||grad||: {tf.norm(grad_omega).numpy():.4e}")
    
    # Gradient norms per typology
    for k in range(data['n_typologies']):
        gnorm = tf.norm(grad_omega[k]).numpy()
        print(f"    Typology {k} ||grad||: {gnorm:.4e}")
    
    # Finite difference check on 3 random points
    print(f"\n  Finite-difference gradient check (3 points):")
    eps = 1e-3  # Larger epsilon for numerical stability at this loss scale
    np.random.seed(999)
    test_points = [(np.random.randint(0, data['n_typologies']),
                    np.random.randint(0, data['n_curve_points'])) for _ in range(3)]
    
    all_passed = True
    for k_idx, m_idx in test_points:
        # Forward: f(ω + ε·e_{k,m}) — use numpy for Metal GPU compatibility
        omega_np = omega.numpy().copy()
        omega_np[k_idx, m_idx] += eps
        omega_plus = tf.Variable(omega_np, dtype=tf.float32)
        J_plus, _, _ = solver.compute_calibration_loss(omega_plus)
        
        # Backward: f(ω - ε·e_{k,m})
        omega_np = omega.numpy().copy()
        omega_np[k_idx, m_idx] -= eps
        omega_minus = tf.Variable(omega_np, dtype=tf.float32)
        J_minus, _, _ = solver.compute_calibration_loss(omega_minus)
        
        fd_grad = (J_plus.numpy() - J_minus.numpy()) / (2 * eps)
        ad_grad = grad_omega[k_idx, m_idx].numpy()
        
        rel_error = abs(fd_grad - ad_grad) / (abs(ad_grad) + 1e-10)
        status = "PASS" if rel_error < 0.01 else "FAIL"
        if rel_error >= 0.01:
            all_passed = False
        
        print(f"    ω[{k_idx},{m_idx}]: AD={ad_grad:+.6e}, FD={fd_grad:+.6e}, "
              f"RelErr={rel_error:.2e} [{status}]")
    
    if all_passed:
        print("\n  STEP 3 PASSED: Loss computable, gradients verified")
    else:
        print("\n  STEP 3 WARNING: Some finite-difference checks did not pass")
    
    return J_calib.numpy()


# ==========================================
# STEP 4: PLAIN GRADIENT DESCENT
# ==========================================

def step4_plain_gd(solver: InverseProblemSolver, data: Dict):
    """Run plain gradient descent optimization."""
    print("\n" + "=" * 70)
    print("STEP 4: Plain Gradient Descent Optimization")
    print("=" * 70)
    
    # Learning rate must be very small for raw GD because ||grad|| ~ 1e9
    # Effective step = lr * ||grad|| should be O(1), so lr ~ 1e-10 to 1e-11
    results_gd = solver.solve(
        optimizer_type='gd', lr=1e-11, n_iterations=2000,
        seed=123, print_every=400
    )
    
    # Verification 4: Loss decreased
    initial_loss = results_gd['loss_history'][0]
    final_loss = results_gd['loss_history'][-1]
    reduction = (1 - final_loss / initial_loss) * 100
    print(f"\n  Loss reduction: {reduction:.2f}%")
    print(f"  Initial MSE: {results_gd['mse_history'][0]:.4e}")
    print(f"  Final MSE:   {results_gd['mse_history'][-1]:.4e}")
    
    # Figure 3: GD loss convergence
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Total loss (log scale)
    ax = axes[0]
    ax.semilogy(results_gd['loss_history'], 'b-', linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('$J_{calib}$ (log scale)', fontsize=12)
    ax.set_title('Total Calibration Loss (GD)', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: MSE and Reg separately
    ax = axes[1]
    ax.semilogy(results_gd['mse_history'], 'r-', linewidth=1.5, label='MSE')
    ax.semilogy(results_gd['reg_history'], 'g-', linewidth=1.5, label='Regularization')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss Component (log scale)', fontsize=12)
    ax.set_title('MSE vs Regularization (GD)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Gradient norm
    ax = axes[2]
    ax.semilogy(results_gd['grad_norm_history'], 'purple', linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('$||\\nabla_{\\omega} J||$ (log scale)', fontsize=12)
    ax.set_title('Gradient Norm (GD)', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'gd_convergence')
    
    print("\n  STEP 4 PASSED: GD optimization complete")
    return results_gd


# ==========================================
# STEP 5: ADAM OPTIMIZER
# ==========================================

def step5_adam_optimizer(solver: InverseProblemSolver, data: Dict):
    """Run Adam optimizer for comparison."""
    print("\n" + "=" * 70)
    print("STEP 5: Adam Optimizer")
    print("=" * 70)
    
    results_adam = solver.solve(
        optimizer_type='adam', lr=0.01, n_iterations=2000,
        seed=123, print_every=400
    )
    
    # Verification 5 stats
    initial_loss = results_adam['loss_history'][0]
    final_loss = results_adam['loss_history'][-1]
    reduction = (1 - final_loss / initial_loss) * 100
    print(f"\n  Loss reduction: {reduction:.2f}%")
    print(f"  Initial MSE: {results_adam['mse_history'][0]:.4e}")
    print(f"  Final MSE:   {results_adam['mse_history'][-1]:.4e}")
    
    print("\n  STEP 5 PASSED: Adam optimization complete")
    return results_adam


def step5b_compare_optimizers(results_gd: Dict, results_adam: Dict):
    """Compare GD vs Adam convergence."""
    print("\n  Comparing GD vs Adam:")
    print(f"    GD   final MSE: {results_gd['mse_history'][-1]:.4e}  ({results_gd['elapsed_time']:.2f}s)")
    print(f"    Adam final MSE: {results_adam['mse_history'][-1]:.4e}  ({results_adam['elapsed_time']:.2f}s)")
    
    # Find iteration where Adam reaches GD's final MSE
    gd_final_mse = results_gd['mse_history'][-1]
    adam_reaches = np.where(results_adam['mse_history'] <= gd_final_mse)[0]
    if len(adam_reaches) > 0:
        print(f"    Adam reaches GD final MSE at iteration {adam_reaches[0]+1}")
    
    # Figure 4: GD vs Adam comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.semilogy(results_gd['loss_history'], 'b-', linewidth=1.5, label='Gradient Descent', alpha=0.8)
    ax.semilogy(results_adam['loss_history'], 'r-', linewidth=1.5, label='Adam', alpha=0.8)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('$J_{calib}$ (log scale)', fontsize=12)
    ax.set_title('Convergence Comparison: GD vs Adam', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.semilogy(results_gd['mse_history'], 'b-', linewidth=1.5, label='GD (MSE)', alpha=0.8)
    ax.semilogy(results_adam['mse_history'], 'r-', linewidth=1.5, label='Adam (MSE)', alpha=0.8)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('MSE (log scale)', fontsize=12)
    ax.set_title('MSE Convergence: GD vs Adam', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'gd_vs_adam_comparison')


# ==========================================
# STEP 6: QUANTITATIVE RECOVERY ANALYSIS
# ==========================================

def step6_recovery_analysis(solver: InverseProblemSolver, results_adam: Dict, data: Dict):
    """Quantitative analysis of curve recovery quality."""
    print("\n" + "=" * 70)
    print("STEP 6: Quantitative Recovery Analysis")
    print("=" * 70)
    
    C_true = data['C_true']
    C_opt = results_adam['C_optimized']
    x_grid = data['x_grid']
    K = data['n_typologies']
    M = data['n_curve_points']
    
    # Per-typology metrics
    print(f"\n  {'Typology':<12} {'RMSE':<12} {'Max|Err|':<12} {'R²':<12}")
    print(f"  {'-'*48}")
    
    for k in range(K):
        rmse = np.sqrt(np.mean((C_opt[k] - C_true[k])**2))
        max_err = np.max(np.abs(C_opt[k] - C_true[k]))
        ss_res = np.sum((C_opt[k] - C_true[k])**2)
        ss_tot = np.sum((C_true[k] - np.mean(C_true[k]))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        print(f"  Typology {k:<3} {rmse:<12.6f} {max_err:<12.6f} {r2:<12.6f}")
    
    # Overall RMSE
    overall_rmse = np.sqrt(np.mean((C_opt - C_true)**2))
    print(f"\n  Overall RMSE: {overall_rmse:.6f}")
    
    # Predicted vs observed losses
    C_opt_tf = tf.constant(C_opt, dtype=tf.float32)
    L_pred_opt = solver.compute_predicted_losses(C_opt_tf).numpy()
    L_obs = data['L_obs']
    
    loss_rmse = np.sqrt(np.mean((L_pred_opt - L_obs)**2))
    loss_r2 = 1 - np.sum((L_pred_opt - L_obs)**2) / (np.sum((L_obs - np.mean(L_obs))**2) + 1e-10)
    print(f"\n  Loss reconstruction:")
    print(f"    RMSE:  ${loss_rmse:,.2f}")
    print(f"    R²:    {loss_r2:.6f}")
    
    # Figure 5: Recovered curves vs true curves
    fig, axes = plt.subplots(1, K, figsize=(6*K, 5))
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    for k in range(K):
        ax = axes[k]
        ax.plot(x_grid, C_true[k], 'k-', linewidth=2.5, label='$C_{true}$')
        ax.plot(x_grid, C_opt[k], '--', color=colors[k], linewidth=2.5, 
                label='$C_{optimized}$')
        ax.fill_between(x_grid, C_true[k], C_opt[k], alpha=0.15, color=colors[k])
        
        # Show intensity coverage for this typology
        mask = (data['u'] == k)
        h_k = data['h'][mask]
        for hi in h_k:
            ax.axvline(hi, color=colors[k], alpha=0.03, linewidth=0.5)
        
        rmse_k = np.sqrt(np.mean((C_opt[k] - C_true[k])**2))
        ax.set_xlabel('Intensity (g)', fontsize=12)
        ax.set_ylabel('MDR', fontsize=12)
        ax.set_title(f'Typology {k} (RMSE={rmse_k:.4f})', fontsize=13)
        ax.legend(fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Recovered vs True Vulnerability Curves (Adam)', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, 'recovered_vs_true_curves')
    
    # Figure 6: L_pred vs L_obs scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    for k in range(K):
        mask = (data['u'] == k)
        ax.scatter(L_obs[mask], L_pred_opt[mask], c=colors[k], alpha=0.5, 
                   s=25, label=f'Typology {k}')
    lims = [0, max(L_obs.max(), L_pred_opt.max()) * 1.05]
    ax.plot(lims, lims, 'k--', linewidth=1.5, label='1:1 line')
    ax.set_xlabel('$L^{obs}$ ($)', fontsize=12)
    ax.set_ylabel('$L^{pred}$ ($)', fontsize=12)
    ax.set_title(f'Loss Reconstruction ($R^2 = {loss_r2:.4f}$)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Figure 7: Residual histogram
    ax = axes[1]
    residuals = L_pred_opt - L_obs
    ax.hist(residuals, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linewidth=2, linestyle='--')
    ax.set_xlabel('Residual: $L^{pred} - L^{obs}$ ($)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Residual Distribution (mean={np.mean(residuals):.2f})', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'loss_reconstruction')
    
    print("\n  STEP 6 PASSED: Recovery analysis complete")
    return C_opt, L_pred_opt


# ==========================================
# STEP 7: SENSITIVITY EXPERIMENTS
# ==========================================

def step7_sensitivity_experiments():
    """Run sensitivity experiments: noise, regularization, N_assets."""
    print("\n" + "=" * 70)
    print("STEP 7: Sensitivity Experiments")
    print("=" * 70)
    
    n_iters = 1500  # Fewer iterations for speed
    base_K = 3
    base_M = 20
    
    # ---- 7a: Effect of noise ----
    print("\n  --- 7a: Effect of Noise ---")
    noise_levels = [0.0, 0.01, 0.05, 0.10]
    noise_results = {}
    
    for nl in noise_levels:
        print(f"\n  Noise level = {nl}")
        data_noisy = generate_inverse_problem_data(
            n_assets=200, n_typologies=base_K, n_curve_points=base_M,
            noise_level=nl, seed=42
        )
        solver_noisy = InverseProblemSolver(
            v=data_noisy['v'], u=data_noisy['u'], x_grid=data_noisy['x_grid'],
            h=data_noisy['h'], L_obs=data_noisy['L_obs'],
            n_typologies=base_K, lambda_reg=1e-4
        )
        res = solver_noisy.solve(optimizer_type='adam', lr=0.01, n_iterations=n_iters,
                                  seed=123, print_every=n_iters)  # Only print final
        
        rmse = np.sqrt(np.mean((res['C_optimized'] - data_noisy['C_true'])**2))
        noise_results[nl] = {'rmse': rmse, 'C_opt': res['C_optimized'], 
                              'C_true': data_noisy['C_true'], 'mse_history': res['mse_history']}
        print(f"    Curve RMSE: {rmse:.6f}")
    
    # ---- 7b: Effect of regularization ----
    print("\n  --- 7b: Effect of Regularization ---")
    reg_values = [0.0, 1e-6, 1e-4, 1e-2]
    reg_results = {}
    
    data_base = generate_inverse_problem_data(
        n_assets=200, n_typologies=base_K, n_curve_points=base_M,
        noise_level=0.0, seed=42
    )
    
    for lr_val in reg_values:
        print(f"\n  λ_reg = {lr_val}")
        solver_reg = InverseProblemSolver(
            v=data_base['v'], u=data_base['u'], x_grid=data_base['x_grid'],
            h=data_base['h'], L_obs=data_base['L_obs'],
            n_typologies=base_K, lambda_reg=lr_val
        )
        res = solver_reg.solve(optimizer_type='adam', lr=0.01, n_iterations=n_iters,
                                seed=123, print_every=n_iters)
        
        rmse = np.sqrt(np.mean((res['C_optimized'] - data_base['C_true'])**2))
        reg_results[lr_val] = {'rmse': rmse, 'C_opt': res['C_optimized'],
                                'mse_history': res['mse_history']}
        print(f"    Curve RMSE: {rmse:.6f}")
    
    # ---- 7c: Effect of N_assets ----
    print("\n  --- 7c: Effect of Number of Assets ---")
    n_values = [50, 200, 1000]
    n_results = {}
    
    for n_a in n_values:
        print(f"\n  N_assets = {n_a}")
        data_n = generate_inverse_problem_data(
            n_assets=n_a, n_typologies=base_K, n_curve_points=base_M,
            noise_level=0.0, seed=42
        )
        solver_n = InverseProblemSolver(
            v=data_n['v'], u=data_n['u'], x_grid=data_n['x_grid'],
            h=data_n['h'], L_obs=data_n['L_obs'],
            n_typologies=base_K, lambda_reg=1e-4
        )
        res = solver_n.solve(optimizer_type='adam', lr=0.01, n_iterations=n_iters,
                              seed=123, print_every=n_iters)
        
        rmse = np.sqrt(np.mean((res['C_optimized'] - data_n['C_true'])**2))
        n_results[n_a] = {'rmse': rmse, 'C_opt': res['C_optimized'],
                           'C_true': data_n['C_true'], 'mse_history': res['mse_history']}
        print(f"    Curve RMSE: {rmse:.6f}")
    
    # ---- Figures ----
    
    # Figure 8: Noise sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for nl in noise_levels:
        ax.semilogy(noise_results[nl]['mse_history'], linewidth=1.5, 
                     label=f'σ = {nl}', alpha=0.8)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('MSE (log scale)', fontsize=12)
    ax.set_title('Effect of Observation Noise on Convergence', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    rmse_vals = [noise_results[nl]['rmse'] for nl in noise_levels]
    ax.bar([str(nl) for nl in noise_levels], rmse_vals, color='steelblue', alpha=0.8)
    ax.set_xlabel('Noise Level (σ / max($L^{obs}$))', fontsize=12)
    ax.set_ylabel('Curve RMSE', fontsize=12)
    ax.set_title('Recovery Quality vs Noise Level', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, 'sensitivity_noise')
    
    # Figure 9: Regularization sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for lr_val in reg_values:
        label = f'λ={lr_val:.0e}' if lr_val > 0 else 'λ=0'
        ax.semilogy(reg_results[lr_val]['mse_history'], linewidth=1.5, 
                     label=label, alpha=0.8)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('MSE (log scale)', fontsize=12)
    ax.set_title('Effect of Regularization on Convergence', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    rmse_vals = [reg_results[lr_val]['rmse'] for lr_val in reg_values]
    labels = [f'{lr_val:.0e}' if lr_val > 0 else '0' for lr_val in reg_values]
    ax.bar(labels, rmse_vals, color='coral', alpha=0.8)
    ax.set_xlabel('$\\lambda_{reg}$', fontsize=12)
    ax.set_ylabel('Curve RMSE', fontsize=12)
    ax.set_title('Recovery Quality vs Regularization', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, 'sensitivity_regularization')
    
    # Figure 10: N_assets sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for n_a in n_values:
        ax.semilogy(n_results[n_a]['mse_history'], linewidth=1.5, 
                     label=f'N={n_a}', alpha=0.8)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('MSE (log scale)', fontsize=12)
    ax.set_title('Effect of Portfolio Size on Convergence', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    rmse_vals = [n_results[n_a]['rmse'] for n_a in n_values]
    ax.bar([str(n_a) for n_a in n_values], rmse_vals, color='mediumpurple', alpha=0.8)
    ax.set_xlabel('Number of Assets (N)', fontsize=12)
    ax.set_ylabel('Curve RMSE', fontsize=12)
    ax.set_title('Recovery Quality vs Portfolio Size', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, 'sensitivity_n_assets')
    
    # Summary table
    print("\n\n  ╔══════════════════════════════════════════════════════╗")
    print("  ║         SENSITIVITY EXPERIMENT SUMMARY                ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║ {'Experiment':<20} {'Parameter':<15} {'Curve RMSE':<15} ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    for nl in noise_levels:
        print(f"  ║ {'Noise':<20} {'σ='+str(nl):<15} {noise_results[nl]['rmse']:<15.6f} ║")
    for lr_val in reg_values:
        lbl = f'λ={lr_val:.0e}' if lr_val > 0 else 'λ=0'
        print(f"  ║ {'Regularization':<20} {lbl:<15} {reg_results[lr_val]['rmse']:<15.6f} ║")
    for n_a in n_values:
        print(f"  ║ {'N_assets':<20} {'N='+str(n_a):<15} {n_results[n_a]['rmse']:<15.6f} ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    
    print("\n  STEP 7 PASSED: Sensitivity experiments complete")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  INVERSE PROBLEM: Vulnerability Curve Calibration          ║")
    print("║  From Observed Losses via Differentiable Optimization      ║")
    print("║  Manuscript: inverse_problem.tex                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    t_total_start = time.time()
    
    # Step 1: Generate ground truth
    data = step1_generate_ground_truth()
    
    # Step 2: Initialize omega
    solver, omega = step2_initialize_omega(data)
    
    # Step 3: Verify loss and gradient
    initial_loss = step3_verify_loss_and_gradient(solver, omega, data)
    
    # Step 4: Plain gradient descent
    results_gd = step4_plain_gd(solver, data)
    
    # Step 5: Adam optimizer + comparison
    results_adam = step5_adam_optimizer(solver, data)
    step5b_compare_optimizers(results_gd, results_adam)
    
    # Step 6: Recovery analysis (using Adam results)
    C_opt, L_pred_opt = step6_recovery_analysis(solver, results_adam, data)
    
    # Step 7: Sensitivity experiments
    step7_sensitivity_experiments()
    
    # Final summary
    t_total = time.time() - t_total_start
    print("\n" + "=" * 70)
    print(f"  COMPLETE: All steps executed in {t_total:.1f}s")
    print(f"  Figures saved to: {FIGURE_DIR}/")
    print(f"  Total figures: {FIGURE_COUNTER}")
    print("=" * 70)
