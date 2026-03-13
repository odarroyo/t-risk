"""
Inverse Problem — Large-Scale Realistic Portfolio (K=12, N=12,000)
===================================================================
Based on minimum_example_inverse_problem_limited_range.py

This script tests the inverse problem solver under a realistic scale:
- 12 vulnerability typologies with diverse steepness and midpoint values
- 12,000 assets with non-uniform typology distribution
- h_max = 0.65 giving mean TRC ≈ 49% — a challenging scenario with
  2 typologies at TRC=0%, 2 in the adversarial zone, and 8 reliably covered

Goals:
  1. Verify the solver works at realistic scale (K=12, N=12,000)
  2. Confirm the TRC framework holds with 12 diverse typologies
  3. Profile compute performance on Apple M4 Pro (Metal GPU)

Steps:
  1: Generate ground truth with realistic portfolio
  2: Initialize ω and verify gradients
  3: Adam optimization with per-iteration timing
  4: Per-typology recovery analysis
  5: Performance profiling (K=3/N=200 reference vs K=12/N=12K)
  6: Mini TRC sweep (5 h_max values)
  7: Summary

Author: Tensorial Risk Engine Project
Date: February 2026
"""

import numpy as np
import tensorflow as tf
import time
import matplotlib
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, Optional, List

from tensor_engine import (
    generate_synthetic_portfolio,
    deterministic_loss,
    probabilistic_loss_matrix,
    TensorialRiskEngine
)

# ==========================================
# GLOBAL SETTINGS
# ==========================================

FIGURE_DIR = 'figures_inverse_problem_big'
os.makedirs(FIGURE_DIR, exist_ok=True)
FIGURE_COUNTER = 0

# Configuration
H_MAX = 0.65
H_MIN = 0.05
N_ASSETS = 12_000
N_TYPOLOGIES = 12
N_CURVE_POINTS = 20

# 12-typology realistic portfolio
# Ordered roughly from fragile (low midpoint) to robust (high midpoint)
TYPOLOGY_NAMES = [
    'Adobe/Earthen',        # k=0
    'URM Light',            # k=1
    'URM Heavy',            # k=2
    'RC Low-Rise Inf.',     # k=3
    'RC Mid-Rise Inf.',     # k=4
    'RC Low-Rise Eng.',     # k=5
    'RC Mid-Rise Eng.',     # k=6
    'Steel Low-Rise',       # k=7
    'Steel Mid-Rise',       # k=8
    'Steel High-Rise',      # k=9
    'Timber',               # k=10
    'Base-Isolated',        # k=11
]

MIDPOINTS = np.array([0.15, 0.25, 0.30, 0.40, 0.50, 0.60,
                       0.65, 0.75, 0.85, 0.95, 1.10, 1.25], dtype=np.float32)
STEEPNESSES = np.array([12.0, 15.0, 8.0, 10.0, 14.0, 9.0,
                         18.0, 7.0, 11.0, 16.0, 6.0, 13.0], dtype=np.float32)

# Non-uniform typology weights (must sum to 1.0)
# Masonry and basic RC dominate; steel/base-isolated are rare
TYPOLOGY_WEIGHTS = np.array([
    0.03,   # k=0 Adobe/Earthen
    0.08,   # k=1 URM Light
    0.22,   # k=2 URM Heavy        ← dominant
    0.18,   # k=3 RC Low-Rise Inf. ← dominant
    0.07,   # k=4 RC Mid-Rise Inf.
    0.15,   # k=5 RC Low-Rise Eng. ← dominant
    0.06,   # k=6 RC Mid-Rise Eng.
    0.05,   # k=7 Steel Low-Rise
    0.04,   # k=8 Steel Mid-Rise
    0.02,   # k=9 Steel High-Rise
    0.08,   # k=10 Timber
    0.02,   # k=11 Base-Isolated
], dtype=np.float64)

assert abs(TYPOLOGY_WEIGHTS.sum() - 1.0) < 1e-10, "Weights must sum to 1.0"

# 12-color palette (distinguishable)
COLORS_12 = [
    '#e74c3c',  # red
    '#e67e22',  # orange
    '#f1c40f',  # yellow
    '#2ecc71',  # green
    '#1abc9c',  # teal
    '#3498db',  # blue
    '#2980b9',  # dark blue
    '#9b59b6',  # purple
    '#8e44ad',  # dark purple
    '#34495e',  # dark grey
    '#16a085',  # dark teal
    '#d35400',  # dark orange
]


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
    
    Uses monotonicity-constrained reparameterization:
        C_{k,m} = min(1.0, Σ_{t=0}^{m} Softplus(ω_{k,t}))
    """
    
    def __init__(self, v: np.ndarray, u: np.ndarray, x_grid: np.ndarray,
                 h: np.ndarray, L_obs: np.ndarray, n_typologies: int,
                 lambda_reg: float = 1e-4):
        
        self.v_tf = tf.constant(v, dtype=tf.float32)
        self.u_tf = tf.constant(u, dtype=tf.int32)
        self.x_grid_tf = tf.constant(x_grid, dtype=tf.float32)
        self.H_tf = tf.constant(h.reshape(-1, 1), dtype=tf.float32)
        self.L_obs_tf = tf.constant(L_obs, dtype=tf.float32)
        self.lambda_reg = lambda_reg
        
        self.N = len(v)
        self.K = n_typologies
        self.M = len(x_grid)
    
    @staticmethod
    def omega_to_C(omega: tf.Variable) -> tf.Tensor:
        """Convert latent variables ω to monotonic vulnerability matrix C."""
        increments = tf.nn.softplus(omega)
        C = tf.cumsum(increments, axis=1)
        C = tf.minimum(C, 1.0)
        return C
    
    def compute_predicted_losses(self, C: tf.Tensor) -> tf.Tensor:
        """Compute per-asset predicted losses using the forward engine."""
        J_matrix = probabilistic_loss_matrix(
            self.v_tf, self.u_tf, C, self.x_grid_tf, self.H_tf
        )
        return J_matrix[:, 0]
    
    def compute_calibration_loss(self, omega: tf.Variable) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute J_calib = MSE + λ_reg · smoothness_penalty."""
        C = self.omega_to_C(omega)
        L_pred = self.compute_predicted_losses(C)
        
        mse = tf.reduce_mean(tf.square(L_pred - self.L_obs_tf))
        
        if self.M >= 3:
            d2C = C[:, 2:] - 2.0 * C[:, 1:-1] + C[:, :-2]
            reg = self.lambda_reg * tf.reduce_sum(tf.square(d2C))
        else:
            reg = tf.constant(0.0, dtype=tf.float32)
        
        J_calib = mse + reg
        return J_calib, mse, reg
    
    def initialize_omega(self, strategy: str = 'random', seed: int = 123) -> tf.Variable:
        """Initialize latent variables ω."""
        np.random.seed(seed)
        if strategy == 'random':
            omega_init = np.random.uniform(-3.0, -1.5, (self.K, self.M)).astype(np.float32)
        elif strategy == 'zeros':
            omega_init = np.zeros((self.K, self.M), dtype=np.float32)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return tf.Variable(omega_init, dtype=tf.float32, name='omega')
    
    def solve(self, optimizer_type: str = 'adam', lr: float = 0.01,
              n_iterations: int = 2000, omega_init: Optional[tf.Variable] = None,
              seed: int = 123, print_every: int = 200,
              record_timing: bool = False) -> Dict:
        """
        Run the calibration optimization loop.
        
        Parameters
        ----------
        record_timing : bool
            If True, record per-iteration wall-clock time.
        
        Returns
        -------
        results : dict with C_optimized, omega_final, loss/mse/reg/grad histories,
                  elapsed_time, omega_variable, and optionally iteration_times.
        """
        if omega_init is not None:
            omega = tf.Variable(omega_init.numpy(), dtype=tf.float32, name='omega')
        else:
            omega = self.initialize_omega(strategy='random', seed=seed)
        
        if optimizer_type == 'adam':
            optimizer = tf.optimizers.Adam(learning_rate=lr)
        elif optimizer_type == 'gd':
            optimizer = tf.optimizers.SGD(learning_rate=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        loss_history = []
        mse_history = []
        reg_history = []
        grad_norm_history = []
        iteration_times = [] if record_timing else None
        
        t_start = time.time()
        
        for it in range(n_iterations):
            if record_timing:
                t_iter_start = time.perf_counter()
            
            with tf.GradientTape() as tape:
                J_calib, mse, reg = self.compute_calibration_loss(omega)
            
            grad_omega = tape.gradient(J_calib, omega)
            
            if optimizer_type == 'gd':
                omega.assign_sub(lr * grad_omega)
            else:
                optimizer.apply_gradients([(grad_omega, omega)])
            
            loss_val = J_calib.numpy()
            mse_val = mse.numpy()
            reg_val = reg.numpy()
            grad_norm_val = tf.norm(grad_omega).numpy()
            
            loss_history.append(loss_val)
            mse_history.append(mse_val)
            reg_history.append(reg_val)
            grad_norm_history.append(grad_norm_val)
            
            if record_timing:
                iteration_times.append(time.perf_counter() - t_iter_start)
            
            if (it + 1) % print_every == 0 or it == 0:
                print(f"  Iter {it+1:5d} | Loss: {loss_val:.4e} | MSE: {mse_val:.4e} | "
                      f"Reg: {reg_val:.4e} | ||grad||: {grad_norm_val:.4e}")
        
        elapsed = time.time() - t_start
        C_optimized = self.omega_to_C(omega).numpy()
        
        print(f"  Optimization complete in {elapsed:.2f}s")
        print(f"  Final loss: {loss_history[-1]:.4e} (MSE: {mse_history[-1]:.4e})")
        
        results = {
            'C_optimized': C_optimized,
            'omega_final': omega.numpy(),
            'loss_history': np.array(loss_history),
            'mse_history': np.array(mse_history),
            'reg_history': np.array(reg_history),
            'grad_norm_history': np.array(grad_norm_history),
            'elapsed_time': elapsed,
            'omega_variable': omega,
        }
        if record_timing:
            results['iteration_times'] = np.array(iteration_times)
        return results


# ==========================================
# DATA GENERATION
# ==========================================

def generate_inverse_problem_data(n_assets: int = 12_000, n_typologies: int = 12,
                                   n_curve_points: int = 20,
                                   noise_level: float = 0.0,
                                   h_range: Tuple[float, float] = (0.05, 0.65),
                                   midpoints: Optional[np.ndarray] = None,
                                   steepnesses: Optional[np.ndarray] = None,
                                   typology_weights: Optional[np.ndarray] = None,
                                   seed: int = 42) -> Dict:
    """
    Generate synthetic ground truth data for the inverse problem.
    
    Parameters
    ----------
    midpoints : ndarray, optional
        Sigmoid midpoints per typology. If None, uses linear spacing.
    steepnesses : ndarray, optional
        Sigmoid steepness per typology. If None, uses linear spacing.
    typology_weights : ndarray, optional
        Multinomial weights for typology assignment. If None, uniform.
    """
    np.random.seed(seed)
    
    # Exposure
    v = np.random.uniform(100_000, 800_000, n_assets).astype(np.float32)
    
    # Typology assignment: non-uniform multinomial
    if typology_weights is not None:
        w = np.array(typology_weights, dtype=np.float64)
        w /= w.sum()  # normalize
        u = np.random.choice(n_typologies, size=n_assets, p=w).astype(np.int32)
    else:
        u = np.array([i % n_typologies for i in range(n_assets)], dtype=np.int32)
        np.random.shuffle(u)
    
    # Intensity grid
    x_grid = np.linspace(0.0, 1.5, n_curve_points).astype(np.float32)
    
    # True vulnerability curves
    if midpoints is None:
        midpoints = np.array([0.3 + k * 0.15 for k in range(n_typologies)], dtype=np.float32)
    if steepnesses is None:
        steepnesses = np.array([8.0 + k * 3.0 for k in range(n_typologies)], dtype=np.float32)
    
    C_true = np.zeros((n_typologies, n_curve_points), dtype=np.float32)
    for k in range(n_typologies):
        C_true[k, :] = 1.0 / (1.0 + np.exp(-steepnesses[k] * (x_grid - midpoints[k])))
    
    # Hazard
    h = np.random.uniform(h_range[0], h_range[1], n_assets).astype(np.float32)
    
    # Forward engine
    v_tf = tf.constant(v, dtype=tf.float32)
    u_tf = tf.constant(u, dtype=tf.int32)
    C_tf = tf.constant(C_true, dtype=tf.float32)
    x_tf = tf.constant(x_grid, dtype=tf.float32)
    H_tf = tf.constant(h.reshape(-1, 1), dtype=tf.float32)
    
    J_matrix = probabilistic_loss_matrix(v_tf, u_tf, C_tf, x_tf, H_tf)
    L_obs_clean = J_matrix[:, 0].numpy()
    
    if noise_level > 0:
        noise_std = noise_level * np.max(L_obs_clean)
        noise = np.random.normal(0, noise_std, n_assets).astype(np.float32)
        L_obs = np.maximum(L_obs_clean + noise, 0.0)
    else:
        L_obs = L_obs_clean.copy()
    
    return {
        'v': v, 'u': u, 'x_grid': x_grid, 'h': h,
        'C_true': C_true, 'L_obs': L_obs, 'L_obs_clean': L_obs_clean,
        'n_assets': n_assets, 'n_typologies': n_typologies,
        'n_curve_points': n_curve_points, 'noise_level': noise_level,
        'h_range': h_range, 'midpoints': midpoints, 'steepnesses': steepnesses,
    }


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def shade_unobserved_region(ax, h_max, x_max=1.5):
    """Add shading to the unobserved intensity region."""
    ax.axvspan(h_max, x_max, alpha=0.12, color='red', zorder=0)
    ax.axvline(h_max, color='red', linewidth=1.0, linestyle='--', alpha=0.7)


def compute_region_rmse(C_opt, C_true, x_grid, h_max):
    """Compute RMSE in observed and unobserved regions separately."""
    obs_mask = x_grid <= h_max
    unobs_mask = x_grid > h_max
    
    rmse_total = np.sqrt(np.mean((C_opt - C_true)**2))
    rmse_obs = np.sqrt(np.mean((C_opt[:, obs_mask] - C_true[:, obs_mask])**2)) if obs_mask.sum() > 0 else np.nan
    rmse_unobs = np.sqrt(np.mean((C_opt[:, unobs_mask] - C_true[:, unobs_mask])**2)) if unobs_mask.sum() > 0 else np.nan
    
    return rmse_obs, rmse_unobs, rmse_total


def compute_transition_coverage(h_max, midpoints, steepnesses, h_min=0.05):
    """Compute per-typology TRC."""
    ln19 = np.log(19.0)
    K = len(midpoints)
    trc = np.zeros(K)
    regions = []
    for k in range(K):
        delta_k = ln19 / steepnesses[k]
        lo = midpoints[k] - delta_k
        hi = midpoints[k] + delta_k
        regions.append((lo, hi))
        width = hi - lo
        covered = max(0.0, min(h_max, hi) - max(h_min, lo))
        trc[k] = np.clip(covered / width, 0.0, 1.0)
    return trc, np.mean(trc), regions


def compute_per_typology_rmse(C_opt, C_true, x_grid, h_max):
    """Compute RMSE per typology: [rmse_obs, rmse_unobs, rmse_all]."""
    obs_mask = x_grid <= h_max
    unobs_mask = x_grid > h_max
    K = C_opt.shape[0]
    rmse_per_typ = np.zeros((K, 3))
    for k in range(K):
        rmse_per_typ[k, 2] = np.sqrt(np.mean((C_opt[k] - C_true[k])**2))
        if obs_mask.sum() > 0:
            rmse_per_typ[k, 0] = np.sqrt(np.mean((C_opt[k, obs_mask] - C_true[k, obs_mask])**2))
        else:
            rmse_per_typ[k, 0] = np.nan
        if unobs_mask.sum() > 0:
            rmse_per_typ[k, 1] = np.sqrt(np.mean((C_opt[k, unobs_mask] - C_true[k, unobs_mask])**2))
        else:
            rmse_per_typ[k, 1] = np.nan
    return rmse_per_typ


def trc_regime_color(trc_val):
    """Return color based on TRC regime: red=adversarial, orange=moderate, green=reliable."""
    if trc_val <= 0.001:
        return '#95a5a6'  # grey for zero
    elif trc_val < 0.30:
        return '#e74c3c'  # red for adversarial
    elif trc_val < 0.50:
        return '#f39c12'  # orange for borderline
    else:
        return '#27ae60'  # green for reliable


# ==========================================
# STEP 1: GENERATE GROUND TRUTH
# ==========================================

def step1_generate_ground_truth():
    """Generate realistic portfolio with K=12 typologies, non-uniform distribution."""
    print("=" * 70)
    print("STEP 1: Generate Ground Truth (K=12, N=12,000, Realistic Portfolio)")
    print("=" * 70)
    
    data = generate_inverse_problem_data(
        n_assets=N_ASSETS, n_typologies=N_TYPOLOGIES, n_curve_points=N_CURVE_POINTS,
        noise_level=0.0, h_range=(H_MIN, H_MAX),
        midpoints=MIDPOINTS, steepnesses=STEEPNESSES,
        typology_weights=TYPOLOGY_WEIGHTS
    )
    
    v, u, x_grid, h = data['v'], data['u'], data['x_grid'], data['h']
    C_true, L_obs = data['C_true'], data['L_obs']
    K = data['n_typologies']
    
    # Print shapes
    print(f"\n  Shapes: v={v.shape}, u={u.shape}, x_grid={x_grid.shape}, "
          f"h={h.shape}, C_true={C_true.shape}, L_obs={L_obs.shape}")
    print(f"  Grid range: [{x_grid[0]:.2f}, {x_grid[-1]:.2f}], "
          f"Hazard range: [{h.min():.3f}, {h.max():.3f}]")
    
    # Asset distribution
    print(f"\n  Asset distribution (non-uniform):")
    unique, counts = np.unique(u, return_counts=True)
    for k in range(K):
        cnt = counts[unique == k][0] if k in unique else 0
        pct = 100.0 * cnt / len(u)
        print(f"    k={k:2d} ({TYPOLOGY_NAMES[k]:22s}): {cnt:5d} assets ({pct:5.1f}%), "
              f"x0={MIDPOINTS[k]:.2f}, s={STEEPNESSES[k]:.0f}")
    
    # TRC analysis
    trc, trc_mean, regions = compute_transition_coverage(H_MAX, MIDPOINTS, STEEPNESSES, H_MIN)
    print(f"\n  Transition Region Coverage at h_max={H_MAX}:")
    print(f"  {'k':>3s}  {'Name':22s}  {'T_lo':>6s}  {'T_hi':>6s}  {'Width':>6s}  {'TRC':>6s}  {'Regime'}")
    print(f"  {'-'*3}  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*20}")
    for k in range(K):
        lo, hi = regions[k]
        width = hi - lo
        regime = "ZERO" if trc[k] < 0.01 else "ADVERSARIAL" if trc[k] < 0.30 else "BORDERLINE" if trc[k] < 0.50 else "RELIABLE"
        print(f"  {k:3d}  {TYPOLOGY_NAMES[k]:22s}  {lo:6.2f}  {hi:6.2f}  {width:6.3f}  {trc[k]:5.0%}  {regime}")
    print(f"\n  Mean TRC: {trc_mean:.1%}")
    
    # ----- Figure 1: Ground truth vulnerability curves (3x4 grid) -----
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(f'Ground Truth Vulnerability Curves (K={K})\n'
                 f'$h_{{max}}$={H_MAX}, Grid coverage={100*H_MAX/1.5:.0f}%, '
                 f'$\\overline{{TRC}}$={trc_mean:.0%}',
                 fontsize=14, fontweight='bold')
    
    for k in range(K):
        row, col = k // 4, k % 4
        ax = axes[row, col]
        ax.plot(x_grid, C_true[k], color=COLORS_12[k], linewidth=2)
        shade_unobserved_region(ax, H_MAX)
        
        # Shade transition region
        lo, hi = regions[k]
        ax.axvspan(max(0, lo), min(1.5, hi), alpha=0.08, color=COLORS_12[k])
        ax.axvline(MIDPOINTS[k], color=COLORS_12[k], alpha=0.4, linestyle=':')
        
        ax.set_title(f'k={k}: {TYPOLOGY_NAMES[k]}\n'
                     f'$x_0$={MIDPOINTS[k]:.2f}, s={STEEPNESSES[k]:.0f}, '
                     f'TRC={trc[k]:.0%}',
                     fontsize=9)
        ax.set_xlim(0, 1.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Intensity $x$', fontsize=8)
        ax.set_ylabel('$C_k(x)$', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, 'ground_truth_12_typologies')
    
    # ----- Figure 2: Asset distribution bar chart -----
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_colors = [trc_regime_color(trc[k]) for k in range(K)]
    bars = ax.bar(range(K), [counts[unique == k][0] for k in range(K)],
                  color=bar_colors, edgecolor='black', linewidth=0.5)
    
    # Annotate each bar with TRC and N_k
    for k in range(K):
        cnt = counts[unique == k][0] if k in unique else 0
        ax.text(k, cnt + 50, f'TRC={trc[k]:.0%}\nN={cnt}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    
    ax.set_xticks(range(K))
    ax.set_xticklabels([f'k={k}\n{TYPOLOGY_NAMES[k]}' for k in range(K)],
                       fontsize=7.5, rotation=30, ha='right')
    ax.set_ylabel('Number of Assets $N_k$', fontsize=11)
    ax.set_title(f'Asset Distribution by Typology (N={N_ASSETS:,}, K={K})\n'
                 f'Colors: grey=TRC 0%, red=adversarial (<30%), '
                 f'orange=borderline (30-50%), green=reliable (>50%)',
                 fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    save_figure(fig, 'asset_distribution')
    
    print(f"\n  STEP 1: PASS ✓")
    return data


# ==========================================
# STEP 2: INITIALIZE OMEGA AND VERIFY GRADIENTS
# ==========================================

def step2_init_and_gradient_check(data):
    """Initialize omega, create solver, verify gradients via finite differences."""
    print("\n" + "=" * 70)
    print("STEP 2: Initialize ω and Verify Gradients")
    print("=" * 70)
    
    v, u, x_grid, h, L_obs = data['v'], data['u'], data['x_grid'], data['h'], data['L_obs']
    C_true = data['C_true']
    K = data['n_typologies']
    
    solver = InverseProblemSolver(v, u, x_grid, h, L_obs, K, lambda_reg=1e-4)
    omega = solver.initialize_omega(strategy='random', seed=123)
    C_init = InverseProblemSolver.omega_to_C(omega).numpy()
    
    print(f"  ω shape: {omega.shape} = {omega.shape[0] * omega.shape[1]} parameters")
    print(f"  C_init: min={C_init.min():.4f}, max={C_init.max():.4f}")
    
    # Gradient verification (3 random components)
    print(f"\n  Gradient verification (finite differences, ε=1e-3):")
    
    epsilon = 1e-3
    fd_checks = []
    
    np.random.seed(999)
    test_indices = [(np.random.randint(0, K), np.random.randint(0, solver.M)) for _ in range(3)]
    
    with tf.GradientTape() as tape:
        J, mse, reg = solver.compute_calibration_loss(omega)
    grad_auto = tape.gradient(J, omega).numpy()
    
    for (ki, mi) in test_indices:
        omega_plus = omega.numpy().copy()
        omega_plus[ki, mi] += epsilon
        omega_minus = omega.numpy().copy()
        omega_minus[ki, mi] -= epsilon
        
        omega_p = tf.Variable(omega_plus, dtype=tf.float32)
        omega_m = tf.Variable(omega_minus, dtype=tf.float32)
        
        J_p, _, _ = solver.compute_calibration_loss(omega_p)
        J_m, _, _ = solver.compute_calibration_loss(omega_m)
        
        fd_grad = (J_p.numpy() - J_m.numpy()) / (2 * epsilon)
        auto_grad = grad_auto[ki, mi]
        
        if abs(auto_grad) > 1e-10:
            rel_err = abs(fd_grad - auto_grad) / abs(auto_grad)
        else:
            rel_err = abs(fd_grad - auto_grad)
        
        status = "OK" if rel_err < 0.05 else "WARNING"
        print(f"    ω[{ki},{mi}]: AD={auto_grad:.6e}, FD={fd_grad:.6e}, "
              f"rel_err={rel_err:.2e} [{status}]")
        fd_checks.append(rel_err < 0.05)
    
    # ----- Figure 3: Initial curves vs ground truth (3x4) -----
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(f'Initial vs True Vulnerability Curves (K={K})\nω ~ Uniform(-3, -1.5)',
                 fontsize=14, fontweight='bold')
    
    for k in range(K):
        row, col = k // 4, k % 4
        ax = axes[row, col]
        ax.plot(x_grid, C_true[k], color=COLORS_12[k], linewidth=2, label='True')
        ax.plot(x_grid, C_init[k], color='grey', linewidth=1.5, linestyle='--', label='Init')
        shade_unobserved_region(ax, H_MAX)
        ax.set_title(f'k={k}: {TYPOLOGY_NAMES[k]}', fontsize=9)
        ax.set_xlim(0, 1.5)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, 'initial_vs_true')
    
    all_ok = all(fd_checks)
    print(f"\n  STEP 2: {'PASS' if all_ok else 'WARNING (gradient check)'} "
          f"{'✓' if all_ok else '⚠'}")
    return solver, omega


# ==========================================
# STEP 3: ADAM OPTIMIZATION WITH TIMING
# ==========================================

def step3_adam_optimization(solver, data):
    """Run Adam optimizer with per-iteration timing."""
    print("\n" + "=" * 70)
    print("STEP 3: Adam Optimization (K=12, N=12,000, 2000 iterations)")
    print("=" * 70)
    
    t0 = time.time()
    results = solver.solve(
        optimizer_type='adam', lr=0.01, n_iterations=2000,
        seed=123, print_every=200, record_timing=True
    )
    total_time = time.time() - t0
    
    loss_h = results['loss_history']
    mse_h = results['mse_history']
    reg_h = results['reg_history']
    grad_h = results['grad_norm_history']
    iter_times = results['iteration_times']
    
    print(f"\n  Performance summary:")
    print(f"    Total wall-clock: {total_time:.1f}s")
    print(f"    Mean iteration:   {np.mean(iter_times)*1000:.1f}ms")
    print(f"    Median iteration: {np.median(iter_times)*1000:.1f}ms")
    print(f"    Std iteration:    {np.std(iter_times)*1000:.1f}ms")
    print(f"    First 10 iters:   {np.mean(iter_times[:10])*1000:.1f}ms (includes tracing)")
    print(f"    Last 100 iters:   {np.mean(iter_times[-100:])*1000:.1f}ms (steady state)")
    print(f"    Loss reduction:   {loss_h[0]:.4e} → {loss_h[-1]:.4e} "
          f"({100*(1 - loss_h[-1]/loss_h[0]):.2f}%)")
    
    # ----- Figure 4: Convergence + timing (4-panel) -----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Adam Optimization Convergence (K={N_TYPOLOGIES}, N={N_ASSETS:,})\n'
                 f'Total time: {total_time:.1f}s, {np.mean(iter_times)*1000:.1f}ms/iter',
                 fontsize=14, fontweight='bold')
    
    iters = np.arange(1, len(loss_h) + 1)
    cum_time = np.cumsum(iter_times)
    
    # Panel 1: Total loss
    axes[0, 0].semilogy(iters, loss_h, 'b-', linewidth=0.8)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Total Loss $J_{calib}$')
    axes[0, 0].set_title('Total Calibration Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel 2: MSE vs Regularization
    ax2 = axes[0, 1]
    ax2.semilogy(iters, mse_h, 'b-', linewidth=0.8, label='MSE')
    ax2.semilogy(iters, reg_h, 'r-', linewidth=0.8, label='Reg')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Component Loss')
    ax2.set_title('MSE vs Regularization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Gradient norm
    axes[1, 0].semilogy(iters, grad_h, 'g-', linewidth=0.8)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('$\\||\\nabla_\\omega J\\||$')
    axes[1, 0].set_title('Gradient Norm')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel 4: Cumulative time + per-iter time
    ax4 = axes[1, 1]
    ax4.plot(iters, cum_time, 'b-', linewidth=1.0, label='Cumulative')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Cumulative Time (s)', color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4r = ax4.twinx()
    # Plot smoothed per-iteration time
    window = 50
    if len(iter_times) > window:
        smoothed = np.convolve(iter_times * 1000, np.ones(window)/window, mode='valid')
        ax4r.plot(iters[window-1:], smoothed, 'r-', alpha=0.6, linewidth=0.8, label='Per-iter (smoothed)')
    ax4r.set_ylabel('Per-Iteration Time (ms)', color='r')
    ax4r.tick_params(axis='y', labelcolor='r')
    ax4.set_title('Wall-Clock Timing')
    ax4.grid(True, alpha=0.3)
    
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, 'convergence_timing')
    
    print(f"\n  STEP 3: PASS ✓")
    return results


# ==========================================
# STEP 4: PER-TYPOLOGY RECOVERY ANALYSIS
# ==========================================

def step4_recovery_analysis(solver, results, data):
    """Analyze per-typology recovery quality."""
    print("\n" + "=" * 70)
    print("STEP 4: Per-Typology Recovery Analysis")
    print("=" * 70)
    
    C_opt = results['C_optimized']
    C_true = data['C_true']
    x_grid = data['x_grid']
    K = data['n_typologies']
    u = data['u']
    
    # TRC
    trc, trc_mean, regions = compute_transition_coverage(H_MAX, MIDPOINTS, STEEPNESSES, H_MIN)
    
    # Per-typology RMSE
    rmse_per = compute_per_typology_rmse(C_opt, C_true, x_grid, H_MAX)
    
    # Pooled RMSE
    rmse_obs, rmse_unobs, rmse_total = compute_region_rmse(C_opt, C_true, x_grid, H_MAX)
    
    # Asset counts
    unique, counts = np.unique(u, return_counts=True)
    n_per_k = np.zeros(K, dtype=int)
    for k in range(K):
        if k in unique:
            n_per_k[k] = int(counts[unique == k][0])
    
    # Print table
    print(f"\n  {'k':>3s}  {'Name':22s}  {'N_k':>5s}  {'TRC':>5s}  "
          f"{'RMSE_obs':>9s}  {'RMSE_unobs':>11s}  {'RMSE_all':>9s}  {'Regime'}")
    print(f"  {'-'*3}  {'-'*22}  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*11}  {'-'*9}  {'-'*15}")
    for k in range(K):
        regime = "ZERO" if trc[k] < 0.01 else "ADVERSARIAL" if trc[k] < 0.30 else "BORDERLINE" if trc[k] < 0.50 else "RELIABLE"
        print(f"  {k:3d}  {TYPOLOGY_NAMES[k]:22s}  {n_per_k[k]:5d}  {trc[k]:4.0%}  "
              f"{rmse_per[k,0]:9.4f}  {rmse_per[k,1]:11.4f}  {rmse_per[k,2]:9.4f}  {regime}")
    
    print(f"\n  Pooled: RMSE_obs={rmse_obs:.4f}, RMSE_unobs={rmse_unobs:.4f}, "
          f"RMSE_total={rmse_total:.4f}")
    
    # ----- Figure 5: Recovered vs True (3x4 grid) -----
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle(f'Recovered vs True Vulnerability Curves (K={K})\n'
                 f'$h_{{max}}$={H_MAX}, $\\overline{{TRC}}$={trc_mean:.0%}, '
                 f'Pooled RMSE={rmse_total:.4f}',
                 fontsize=14, fontweight='bold')
    
    for k in range(K):
        row, col = k // 4, k % 4
        ax = axes[row, col]
        ax.plot(x_grid, C_true[k], color=COLORS_12[k], linewidth=2, label='True')
        ax.plot(x_grid, C_opt[k], color='black', linewidth=1.5, linestyle='--', label='Recovered')
        
        # Error band
        error = np.abs(C_opt[k] - C_true[k])
        ax.fill_between(x_grid, C_true[k] - error, C_true[k] + error,
                        alpha=0.15, color=COLORS_12[k])
        
        shade_unobserved_region(ax, H_MAX)
        
        ax.set_title(f'k={k}: {TYPOLOGY_NAMES[k]}\n'
                     f'TRC={trc[k]:.0%}, RMSE_unobs={rmse_per[k,1]:.4f}',
                     fontsize=9)
        ax.set_xlim(0, 1.5)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, 'recovered_vs_true')
    
    # ----- Figure 6: Per-typology RMSE bar chart -----
    fig, ax = plt.subplots(figsize=(16, 7))
    
    x_pos = np.arange(K)
    width = 0.35
    bars_obs = ax.bar(x_pos - width/2, rmse_per[:, 0], width,
                      color=[trc_regime_color(trc[k]) for k in range(K)],
                      edgecolor='black', linewidth=0.5, alpha=0.7, label='RMSE$_{obs}$')
    bars_unobs = ax.bar(x_pos + width/2, rmse_per[:, 1], width,
                        color=[trc_regime_color(trc[k]) for k in range(K)],
                        edgecolor='black', linewidth=0.5, label='RMSE$_{unobs}$')
    
    # Annotate TRC on each bar
    for k in range(K):
        y_val = max(rmse_per[k, 0], rmse_per[k, 1])
        ax.text(k, y_val + 0.003, f'TRC={trc[k]:.0%}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'k={k}\n{TYPOLOGY_NAMES[k]}' for k in range(K)],
                       fontsize=7.5, rotation=30, ha='right')
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'Per-Typology RMSE (Observed vs Unobserved)\n'
                 f'$h_{{max}}$={H_MAX}, $\\overline{{TRC}}$={trc_mean:.0%}',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    save_figure(fig, 'per_typology_rmse_bars')
    
    # ----- Figure 7: Loss reconstruction scatter -----
    C_opt_tf = tf.constant(C_opt, dtype=tf.float32)
    L_pred = solver.compute_predicted_losses(C_opt_tf).numpy()
    L_obs = data['L_obs']
    
    # R² computation
    ss_res = np.sum((L_obs - L_pred)**2)
    ss_tot = np.sum((L_obs - np.mean(L_obs))**2)
    r2 = 1.0 - ss_res / ss_tot
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    for k in range(K):
        mask_k = data['u'] == k
        ax1.scatter(L_obs[mask_k], L_pred[mask_k], c=COLORS_12[k],
                    s=3, alpha=0.3, label=TYPOLOGY_NAMES[k])
    
    lim = max(L_obs.max(), L_pred.max()) * 1.05
    ax1.plot([0, lim], [0, lim], 'r-', linewidth=1, label='Perfect fit')
    ax1.set_xlabel('$L^{obs}$', fontsize=12)
    ax1.set_ylabel('$L^{pred}$', fontsize=12)
    ax1.set_title(f'Loss Reconstruction ($R^2$ = {r2:.6f})', fontsize=12)
    ax1.legend(fontsize=6, ncol=3, loc='lower right', markerscale=3)
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.grid(True, alpha=0.3)
    
    # Residuals
    residuals = L_obs - L_pred
    ax2.hist(residuals, bins=80, color='steelblue', edgecolor='black', linewidth=0.3)
    ax2.axvline(0, color='red', linewidth=1)
    ax2.set_xlabel('Residual $L^{obs} - L^{pred}$', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Residual Distribution (N={N_ASSETS:,})', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_figure(fig, 'loss_reconstruction')
    
    print(f"  R² = {r2:.6f}")
    print(f"\n  STEP 4: PASS ✓")
    return rmse_per, trc


# ==========================================
# STEP 5: PERFORMANCE PROFILING
# ==========================================

def step5_performance_profiling(results_big, data):
    """Compare K=12/N=12K performance against K=3/N=200 reference."""
    print("\n" + "=" * 70)
    print("STEP 5: Performance Profiling (K=12/N=12K vs K=3/N=200)")
    print("=" * 70)
    
    # ---- Run K=3, N=200 reference solve ----
    print("\n  Running K=3, N=200 reference solve (1500 iterations)...")
    
    ref_midpoints = np.array([0.30, 0.45, 0.60], dtype=np.float32)
    ref_steepnesses = np.array([8.0, 11.0, 14.0], dtype=np.float32)
    
    ref_data = generate_inverse_problem_data(
        n_assets=200, n_typologies=3, n_curve_points=20,
        noise_level=0.0, h_range=(H_MIN, H_MAX),
        midpoints=ref_midpoints, steepnesses=ref_steepnesses,
        seed=42
    )
    
    ref_solver = InverseProblemSolver(
        ref_data['v'], ref_data['u'], ref_data['x_grid'],
        ref_data['h'], ref_data['L_obs'], 3, lambda_reg=1e-4
    )
    
    ref_results = ref_solver.solve(
        optimizer_type='adam', lr=0.01, n_iterations=1500,
        seed=123, print_every=500, record_timing=True
    )
    
    # ---- Compare ----
    big_times = results_big['iteration_times']
    ref_times = ref_results['iteration_times']
    
    # Skip first 10 iterations (tracing overhead)
    big_steady = big_times[10:]
    ref_steady = ref_times[10:]
    
    print(f"\n  Performance comparison (steady-state, skipping first 10 iters):")
    print(f"    {'Metric':<30s}  {'K=3,N=200':>12s}  {'K=12,N=12K':>12s}  {'Ratio':>8s}")
    print(f"    {'-'*30}  {'-'*12}  {'-'*12}  {'-'*8}")
    
    metrics = [
        ('Mean iter time (ms)', np.mean(ref_steady)*1000, np.mean(big_steady)*1000),
        ('Median iter time (ms)', np.median(ref_steady)*1000, np.median(big_steady)*1000),
        ('Std iter time (ms)', np.std(ref_steady)*1000, np.std(big_steady)*1000),
        ('Total time (s)', ref_results['elapsed_time'], results_big['elapsed_time']),
        ('Parameters (K×M)', 3*20, 12*20),
        ('Data points (N)', 200, 12000),
    ]
    
    for name, ref_val, big_val in metrics:
        ratio = big_val / ref_val if ref_val > 0 else float('inf')
        print(f"    {name:<30s}  {ref_val:12.2f}  {big_val:12.2f}  {ratio:7.1f}x")
    
    # ----- Figure 8: Performance profile (3-panel) -----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('Compute Performance: K=3/N=200 vs K=12/N=12,000',
                 fontsize=14, fontweight='bold')
    
    # Panel 1: Per-iteration time histogram
    ax = axes[0]
    ax.hist(ref_steady * 1000, bins=40, alpha=0.6, color='steelblue',
            label=f'K=3, N=200\nμ={np.mean(ref_steady)*1000:.1f}ms', density=True)
    ax.hist(big_steady * 1000, bins=40, alpha=0.6, color='coral',
            label=f'K=12, N=12K\nμ={np.mean(big_steady)*1000:.1f}ms', density=True)
    ax.set_xlabel('Time per Iteration (ms)')
    ax.set_ylabel('Density')
    ax.set_title('Iteration Time Distribution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Cumulative time vs iteration
    ax = axes[1]
    ax.plot(np.arange(1, len(ref_times)+1), np.cumsum(ref_times),
            color='steelblue', linewidth=1.5, label='K=3, N=200')
    ax.plot(np.arange(1, len(big_times)+1), np.cumsum(big_times),
            color='coral', linewidth=1.5, label='K=12, N=12K')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cumulative Time (s)')
    ax.set_title('Cumulative Wall-Clock Time')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Scaling summary bar chart
    ax = axes[2]
    categories = ['Mean Iter\n(ms)', 'Total\n(s)', 'Parameters\n(K×M)', 'Data\n(N)']
    ref_vals = [np.mean(ref_steady)*1000, ref_results['elapsed_time'], 60, 200]
    big_vals = [np.mean(big_steady)*1000, results_big['elapsed_time'], 240, 12000]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    ax.bar(x_pos - width/2, ref_vals, width, color='steelblue', label='K=3, N=200')
    ax.bar(x_pos + width/2, big_vals, width, color='coral', label='K=12, N=12K')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Value')
    ax.set_title('Scaling Comparison')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add ratio annotations
    for i, (rv, bv) in enumerate(zip(ref_vals, big_vals)):
        ratio = bv / rv if rv > 0 else 0
        ax.text(i, max(rv, bv) * 1.5, f'{ratio:.1f}x', ha='center',
                va='bottom', fontsize=9, fontweight='bold')
    
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, 'performance_profile')
    
    print(f"\n  STEP 5: PASS ✓")
    return ref_results


# ==========================================
# STEP 6: MINI TRC SWEEP
# ==========================================

def step6_trc_sweep(data):
    """Run 5-point TRC sweep to verify three-regime framework at K=12."""
    print("\n" + "=" * 70)
    print("STEP 6: Mini TRC Sweep (5 h_max values, K=12)")
    print("=" * 70)
    
    h_max_values = [0.45, 0.55, 0.65, 0.85, 1.10]
    
    C_true = data['C_true']
    K = data['n_typologies']
    
    # Storage
    all_trc = []
    all_rmse_per = []
    all_rmse_pooled = []
    sweep_times = []
    
    for idx, h_val in enumerate(h_max_values):
        print(f"\n  --- Sweep point {idx+1}/5: h_max={h_val:.2f} ---")
        
        trc, trc_mean, regions = compute_transition_coverage(h_val, MIDPOINTS, STEEPNESSES, H_MIN)
        all_trc.append(trc.copy())
        
        print(f"  Mean TRC: {trc_mean:.0%}")
        
        # Generate data for this h_max
        sweep_data = generate_inverse_problem_data(
            n_assets=N_ASSETS, n_typologies=N_TYPOLOGIES,
            n_curve_points=N_CURVE_POINTS,
            noise_level=0.0, h_range=(H_MIN, h_val),
            midpoints=MIDPOINTS, steepnesses=STEEPNESSES,
            typology_weights=TYPOLOGY_WEIGHTS, seed=42
        )
        
        # Create solver and run
        sweep_solver = InverseProblemSolver(
            sweep_data['v'], sweep_data['u'], sweep_data['x_grid'],
            sweep_data['h'], sweep_data['L_obs'], K, lambda_reg=1e-4
        )
        
        t_start = time.time()
        sweep_results = sweep_solver.solve(
            optimizer_type='adam', lr=0.01, n_iterations=1500,
            seed=123, print_every=500
        )
        sweep_time = time.time() - t_start
        sweep_times.append(sweep_time)
        
        C_opt = sweep_results['C_optimized']
        
        # Per-typology RMSE
        rmse_per = compute_per_typology_rmse(C_opt, C_true, data['x_grid'], h_val)
        all_rmse_per.append(rmse_per.copy())
        
        rmse_obs, rmse_unobs, rmse_total = compute_region_rmse(C_opt, C_true, data['x_grid'], h_val)
        all_rmse_pooled.append((rmse_obs, rmse_unobs, rmse_total))
        
        print(f"  Pooled RMSE: obs={rmse_obs:.4f}, unobs={rmse_unobs:.4f}, "
              f"total={rmse_total:.4f}, time={sweep_time:.1f}s")
    
    all_trc = np.array(all_trc)          # (5, K)
    all_rmse_per = np.array(all_rmse_per)  # (5, K, 3)
    all_rmse_pooled = np.array(all_rmse_pooled)  # (5, 3)
    
    # Summary table
    print(f"\n  TRC Sweep Summary:")
    print(f"  {'h_max':>5s}  {'TRC_mean':>8s}  {'RMSE_unobs':>10s}  {'RMSE_total':>10s}  {'Time(s)':>8s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")
    for idx, h_val in enumerate(h_max_values):
        trc_mean = all_trc[idx].mean()
        print(f"  {h_val:5.2f}  {trc_mean:7.0%}  {all_rmse_pooled[idx,1]:10.4f}  "
              f"{all_rmse_pooled[idx,2]:10.4f}  {sweep_times[idx]:8.1f}")
    
    # ----- Figure 9: Analytical TRC curves (12 typologies) -----
    fig, ax = plt.subplots(figsize=(14, 7))
    
    h_range_plot = np.linspace(0.05, 1.5, 200)
    for k in range(K):
        trc_curve = []
        for hv in h_range_plot:
            t, _, _ = compute_transition_coverage(hv, [MIDPOINTS[k]], [STEEPNESSES[k]], H_MIN)
            trc_curve.append(t[0])
        ax.plot(h_range_plot, trc_curve, color=COLORS_12[k], linewidth=1.5,
                label=f'k={k} {TYPOLOGY_NAMES[k]}')
    
    # Mark sweep points
    for h_val in h_max_values:
        ax.axvline(h_val, color='black', alpha=0.3, linestyle=':')
        trc_at_h, trc_mean_at_h, _ = compute_transition_coverage(h_val, MIDPOINTS, STEEPNESSES, H_MIN)
        ax.plot(h_val, trc_mean_at_h, 'ks', markersize=8, zorder=10)
    
    ax.axhline(0.50, color='grey', linestyle='--', alpha=0.5, label='TRC=50% threshold')
    ax.set_xlabel('$h_{max}$', fontsize=13)
    ax.set_ylabel('$TRC_k$', fontsize=13)
    ax.set_title(f'Transition Region Coverage vs $h_{{max}}$ (K={K})\n'
                 f'Black squares: sweep points', fontsize=13)
    ax.legend(fontsize=7, ncol=3, loc='lower right')
    ax.set_xlim(0.05, 1.5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_figure(fig, 'trc_analytical_curves')
    
    # ----- Figure 10: Per-typology RMSE_unobs vs h_max -----
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for k in range(K):
        rmse_unobs_k = all_rmse_per[:, k, 1]
        ax.plot(h_max_values, rmse_unobs_k, 'o-', color=COLORS_12[k],
                linewidth=1.5, markersize=6, label=f'k={k} {TYPOLOGY_NAMES[k]}')
    
    # Pooled
    ax.plot(h_max_values, all_rmse_pooled[:, 1], 'ks-', linewidth=2,
            markersize=8, label='Pooled', zorder=10)
    
    ax.set_xlabel('$h_{max}$', fontsize=13)
    ax.set_ylabel('$RMSE_{unobs}^{(k)}$', fontsize=13)
    ax.set_title(f'Per-Typology Unobserved RMSE vs $h_{{max}}$ (K={K})', fontsize=13)
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_figure(fig, 'per_typology_rmse_vs_hmax')
    
    # ----- Figure 11: Universality plot (RMSE vs own TRC) -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: colored by typology
    for k in range(K):
        trc_k = all_trc[:, k]
        rmse_k = all_rmse_per[:, k, 1]
        ax1.scatter(trc_k * 100, rmse_k, c=COLORS_12[k], s=60,
                    edgecolors='black', linewidth=0.5, zorder=5,
                    label=f'k={k}')
    
    ax1.set_xlabel('$TRC_k$ (%)', fontsize=13)
    ax1.set_ylabel('$RMSE_{unobs}^{(k)}$', fontsize=13)
    ax1.set_title('RMSE vs Own TRC (colored by typology)', fontsize=12)
    ax1.legend(fontsize=7, ncol=3, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Shade regimes
    for a in [ax1, ax2]:
        a.axvspan(0, 1, alpha=0.06, color='grey')    # zero
        a.axvspan(1, 30, alpha=0.08, color='red')     # adversarial
        a.axvspan(30, 50, alpha=0.06, color='orange')  # borderline
        a.axvspan(50, 100, alpha=0.06, color='green')  # reliable
    
    # Right: all points, with binned mean
    all_trc_flat = (all_trc * 100).flatten()
    all_rmse_flat = all_rmse_per[:, :, 1].flatten()
    
    # Remove NaN
    valid = ~np.isnan(all_rmse_flat) & ~np.isnan(all_trc_flat)
    trc_valid = all_trc_flat[valid]
    rmse_valid = all_rmse_flat[valid]
    
    ax2.scatter(trc_valid, rmse_valid, c='steelblue', s=40,
                edgecolors='black', linewidth=0.5, alpha=0.7, zorder=5)
    
    # Binned mean guide
    bins = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bin_centers = []
    bin_means = []
    for i in range(len(bins)-1):
        mask_bin = (trc_valid >= bins[i]) & (trc_valid < bins[i+1])
        if mask_bin.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_means.append(np.mean(rmse_valid[mask_bin]))
    
    if bin_centers:
        ax2.plot(bin_centers, bin_means, 'r-o', linewidth=2, markersize=8,
                 label='Binned mean', zorder=10)
    
    ax2.set_xlabel('$TRC_k$ (%)', fontsize=13)
    ax2.set_ylabel('$RMSE_{unobs}^{(k)}$', fontsize=13)
    ax2.set_title(f'Universality Test (K={K}, {len(trc_valid)} points)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add regime labels
    for a in [ax1, ax2]:
        y_top = a.get_ylim()[1] * 0.95
        a.text(0.5, y_top, 'Zero', fontsize=7, ha='center', va='top', color='grey')
        a.text(15, y_top, 'Adversarial', fontsize=7, ha='center', va='top', color='red')
        a.text(40, y_top, 'Border', fontsize=7, ha='center', va='top', color='orange')
        a.text(75, y_top, 'Reliable', fontsize=7, ha='center', va='top', color='green')
    
    fig.suptitle('TRC Universality: Does the Three-Regime Structure Hold at K=12?',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, 'universality_rmse_vs_trc')
    
    print(f"\n  Total sweep time: {sum(sweep_times):.1f}s")
    print(f"\n  STEP 6: PASS ✓")
    return all_trc, all_rmse_per, all_rmse_pooled, h_max_values


# ==========================================
# STEP 7: SUMMARY
# ==========================================

def step7_summary(data, results, rmse_per, trc, sweep_data, total_start_time):
    """Print overall summary."""
    print("\n" + "=" * 70)
    print("STEP 7: Summary")
    print("=" * 70)
    
    all_trc_sweep, all_rmse_sweep, all_rmse_pooled, h_max_vals = sweep_data
    K = data['n_typologies']
    total_time = time.time() - total_start_time
    
    print(f"\n  Experiment Configuration:")
    print(f"    Typologies:    K = {K}")
    print(f"    Assets:        N = {N_ASSETS:,}")
    print(f"    Grid points:   M = {N_CURVE_POINTS}")
    print(f"    Parameters:    K×M = {K * N_CURVE_POINTS}")
    print(f"    Observation:   h ∈ [{H_MIN}, {H_MAX}]")
    print(f"    Grid coverage: {100*H_MAX/1.5:.0f}%")
    print(f"    Mean TRC:      {trc.mean():.0%}")
    
    print(f"\n  Recovery Quality at h_max={H_MAX}:")
    rmse_obs, rmse_unobs, rmse_total = compute_region_rmse(
        results['C_optimized'], data['C_true'], data['x_grid'], H_MAX)
    print(f"    Pooled RMSE_obs:    {rmse_obs:.4f}")
    print(f"    Pooled RMSE_unobs:  {rmse_unobs:.4f}")
    print(f"    Pooled RMSE_total:  {rmse_total:.4f}")
    
    # TRC regime breakdown
    n_zero = np.sum(trc < 0.01)
    n_adv = np.sum((trc >= 0.01) & (trc < 0.30))
    n_border = np.sum((trc >= 0.30) & (trc < 0.50))
    n_reliable = np.sum(trc >= 0.50)
    
    print(f"\n  TRC Regime Breakdown:")
    print(f"    Zero (TRC=0%):       {n_zero} typologies")
    print(f"    Adversarial (<30%):  {n_adv} typologies")
    print(f"    Borderline (30-50%): {n_border} typologies")
    print(f"    Reliable (≥50%):     {n_reliable} typologies")
    
    print(f"\n  Timing:")
    print(f"    Main solve (2000 iter): {results['elapsed_time']:.1f}s")
    print(f"    Total experiment:       {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Final per-typology table
    unique, counts = np.unique(data['u'], return_counts=True)
    n_per_k = np.zeros(K, dtype=int)
    for k in range(K):
        if k in unique:
            n_per_k[k] = int(counts[unique == k][0])
    
    print(f"\n  Final Per-Typology Results:")
    print(f"  {'k':>3s}  {'Name':22s}  {'N_k':>5s}  {'TRC':>5s}  "
          f"{'RMSE_obs':>9s}  {'RMSE_unobs':>11s}  {'RMSE_all':>9s}")
    print(f"  {'-'*3}  {'-'*22}  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*11}  {'-'*9}")
    for k in range(K):
        print(f"  {k:3d}  {TYPOLOGY_NAMES[k]:22s}  {n_per_k[k]:5d}  {trc[k]:4.0%}  "
              f"{rmse_per[k,0]:9.4f}  {rmse_per[k,1]:11.4f}  {rmse_per[k,2]:9.4f}")
    
    print(f"\n  STEP 7: PASS ✓")
    print(f"\n{'='*70}")
    print(f"ALL STEPS COMPLETE. Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Figures saved to: {FIGURE_DIR}/")
    print(f"{'='*70}")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == '__main__':
    total_start = time.time()
    
    print("\n" + "#" * 70)
    print("# LARGE-SCALE INVERSE PROBLEM EXPERIMENT")
    print(f"# K={N_TYPOLOGIES} typologies, N={N_ASSETS:,} assets, h_max={H_MAX}")
    print(f"# Non-uniform distribution, TRC-based analysis")
    print("#" * 70 + "\n")
    
    # Step 1: Ground truth
    data = step1_generate_ground_truth()
    
    # Step 2: Init + gradient check
    solver, omega = step2_init_and_gradient_check(data)
    
    # Step 3: Adam optimization with timing
    results = step3_adam_optimization(solver, data)
    
    # Step 4: Recovery analysis
    rmse_per, trc = step4_recovery_analysis(solver, results, data)
    
    # Step 5: Performance profiling
    ref_results = step5_performance_profiling(results, data)
    
    # Step 6: Mini TRC sweep
    sweep_data = step6_trc_sweep(data)
    
    # Step 7: Summary
    step7_summary(data, results, rmse_per, trc, sweep_data, total_start)
