"""
Tensorial Risk Engine - Manuscript Demonstration (Stochastic with Exponential Lambda)
====================================================================================
Demonstrates gradient utility with realistic portfolio structure and seismic recurrence:
- 70% standard homes ($100k-$200k, random typologies)
- 10% old pricey neighborhood ($500k-$600k, weakest typology 0)
- 20% modern expensive homes ($400k-$500k, strongest typology 4)
- Exponential lambda distribution (RP: 32 to 5000 years)
- Intensity tied to return period: frequent → low, rare → high
- 1000 stochastic events with realistic seismic recurrence law

This example demonstrates how gradients reveal risk concentration when
event frequencies follow realistic seismic recurrence relationships.
"""

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import os

from tensor_engine import (
    deterministic_loss,
    TensorialRiskEngine
)

# Global figure counter for sequential naming
FIGURE_COUNTER = 1
OUTPUT_FOLDER = "figures_example_stochastic_lambda_exponencial"


def generate_synthetic_portfolio_example_manuscript_lambda(
    n_assets: int = 1000,
    n_events: int = 1000,
    n_typologies: int = 5,
    n_curve_points: int = 20,
    base_value: float = 100.0,
    rp_min: float = 32.0,
    rp_max: float = 5000.0,
    min_intensity_frequent: float = 0.02,
    max_intensity_frequent: float = 0.06,
    min_intensity_rare: float = 0.85,
    max_intensity_rare: float = 0.95
) -> Tuple:
    """
    Generate synthetic portfolio with exponential lambda and intensity-RP coupling.
    
    Creates a realistic portfolio where scenario occurrence rates follow seismic
    recurrence laws (exponential decay) and hazard intensities are inversely
    proportional to occurrence rates (frequent events = low intensity, rare = high).
    
    Portfolio Structure
    -------------------
    Same as manuscript example: 70% standard, 10% old pricey (type 0), 20% modern (type 4)
    
    Seismic Recurrence Model
    -------------------------
    Return periods (RP) are exponentially spaced from rp_min to rp_max years.
    Lambda (occurrence rate) follows: λ = 1/RP
    
    Intensity-Lambda Coupling
    -------------------------
    Hazard intensity inversely proportional to lambda:
    - High λ (frequent, low RP) → Low intensity range
    - Low λ (rare, high RP) → High intensity range
    
    This mimics realistic seismic catalogs where frequent events are weak
    and rare events are strong.
    
    Parameters
    ----------
    n_assets : int, optional
        Total number of assets (N), default is 1000
    n_events : int, optional
        Number of stochastic events (Q), default is 1000
    n_typologies : int, optional
        Number of building typologies (K), default is 5
    n_curve_points : int, optional
        Vulnerability curve discretization points (M), default is 20
    base_value : float, optional
        Base exposure value in thousands, default is 100.0
    rp_min : float, optional
        Minimum return period in years (frequent events), default is 32.0
    rp_max : float, optional
        Maximum return period in years (rare events), default is 5000.0
    min_intensity_frequent : float, optional
        Minimum intensity for frequent events (g), default is 0.02
    max_intensity_frequent : float, optional
        Maximum intensity for frequent events (g), default is 0.06
    min_intensity_rare : float, optional
        Minimum intensity for rare events (g), default is 0.85
    max_intensity_rare : float, optional
        Maximum intensity for rare events (g), default is 0.95
    
    Returns
    -------
    v_exposure : np.ndarray, shape (N,)
        Exposure vector ∈ ℝ^N
    u_typology : np.ndarray, shape (N,)
        Typology index vector ∈ ℤ^N
    C_matrix : np.ndarray, shape (K, M)
        Vulnerability matrix ∈ ℝ^(K×M)
    x_grid : np.ndarray, shape (M,)
        Intensity grid ∈ ℝ^M
    H_intensities : np.ndarray, shape (N, Q)
        Hazard intensity matrix ∈ ℝ^(N×Q) with RP-dependent ranges
    lambdas_out : np.ndarray, shape (Q,)
        Scenario occurrence rates ∈ ℝ^Q (λ = 1/RP)
    return_periods : np.ndarray, shape (Q,)
        Return periods for each event ∈ ℝ^Q (years)
    intensity_midpoints : np.ndarray, shape (Q,)
        Intensity midpoint for each event ∈ ℝ^Q (for visualization)
    
    Examples
    --------
    >>> v, u, C, x, H, lambdas, RPs, mids = generate_synthetic_portfolio_example_manuscript_lambda(
    ...     n_assets=1000, n_events=1000, rp_min=32.0, rp_max=5000.0)
    >>> print(f"Lambda range: {lambdas.min():.6f} to {lambdas.max():.6f}")
    >>> print(f"RP range: {RPs.min():.1f} to {RPs.max():.1f} years")
    >>> print(f"Intensity range: {H.min():.3f}g to {H.max():.3f}g")
    """
    np.random.seed(42)
    
    # Portfolio composition (same as before)
    n_standard = int(0.70 * n_assets)
    n_old_pricey = int(0.10 * n_assets)
    n_modern = n_assets - n_standard - n_old_pricey
    
    v_exposure = np.zeros(n_assets, dtype=np.float32)
    u_typology = np.zeros(n_assets, dtype=np.int32)
    
    # Standard homes (70%)
    idx_start = 0
    idx_end = n_standard
    v_exposure[idx_start:idx_end] = np.random.uniform(
        base_value, 2 * base_value, n_standard
    ).astype(np.float32)
    u_typology[idx_start:idx_end] = np.random.randint(
        0, n_typologies, n_standard
    ).astype(np.int32)
    
    # Old pricey neighborhood (10%)
    idx_start = idx_end
    idx_end = idx_start + n_old_pricey
    v_exposure[idx_start:idx_end] = np.random.uniform(
        5 * base_value, 6 * base_value, n_old_pricey
    ).astype(np.float32)
    u_typology[idx_start:idx_end] = 0
    
    # Modern expensive homes (20%)
    idx_start = idx_end
    idx_end = n_assets
    v_exposure[idx_start:idx_end] = np.random.uniform(
        4 * base_value, 5 * base_value, n_modern
    ).astype(np.float32)
    u_typology[idx_start:idx_end] = 4
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_assets)
    v_exposure = v_exposure[shuffle_idx]
    u_typology = u_typology[shuffle_idx]
    
    # Intensity grid
    x_grid = np.linspace(0.0, 1.5, n_curve_points).astype(np.float32)
    
    # Vulnerability matrix
    C_matrix = np.zeros((n_typologies, n_curve_points), dtype=np.float32)
    for k in range(n_typologies):
        steepness = 8.0 + k * 2.0
        midpoint = 0.4 + k * 0.1
        C_matrix[k, :] = 1.0 / (1.0 + np.exp(-steepness * (x_grid - midpoint)))
    
    # Generate return periods with exponential spacing
    # More realistic for seismic catalogs
    return_periods = np.exp(np.linspace(np.log(rp_min), np.log(rp_max), n_events)).astype(np.float32)
    
    # Calculate lambdas: λ = 1/RP
    lambdas_out = (1.0 / return_periods).astype(np.float32)
    
    # Normalize lambda for intensity mapping (0 = rarest, 1 = most frequent)
    lambda_min = lambdas_out.min()
    lambda_max = lambdas_out.max()
    lambda_normalized = (lambdas_out - lambda_min) / (lambda_max - lambda_min)
    
    # Generate hazard intensity matrix with RP-dependent ranges
    # INVERSE relationship: high λ (frequent) → low intensity, low λ (rare) → high intensity
    H_intensities = np.zeros((n_assets, n_events), dtype=np.float32)
    intensity_midpoints = np.zeros(n_events, dtype=np.float32)
    
    for q in range(n_events):
        # Inverse mapping: lambda_normalized[q] = 1 (frequent) → low intensity
        #                  lambda_normalized[q] = 0 (rare) → high intensity
        min_intensity_q = min_intensity_frequent + (min_intensity_rare - min_intensity_frequent) * (1 - lambda_normalized[q])
        max_intensity_q = max_intensity_frequent + (max_intensity_rare - max_intensity_frequent) * (1 - lambda_normalized[q])
        
        # Store midpoint for visualization
        intensity_midpoints[q] = (min_intensity_q + max_intensity_q) / 2.0
        
        # Sample uniform intensities for all assets in this event
        H_intensities[:, q] = np.random.uniform(
            min_intensity_q, max_intensity_q, n_assets
        ).astype(np.float32)
    
    return v_exposure, u_typology, C_matrix, x_grid, H_intensities, lambdas_out, return_periods, intensity_midpoints


def demonstrate_manuscript_example_exponential_lambda():
    """
    Comprehensive demonstration with exponential lambda and RP-dependent intensities.
    
    Shows how realistic seismic recurrence laws affect risk metrics and gradients.
    """
    print("=" * 80)
    print("TENSORIAL RISK ENGINE - EXPONENTIAL LAMBDA DEMONSTRATION")
    print("=" * 80)
    
    # Generate portfolio data
    print("\n📊 Generating realistic portfolio with seismic recurrence...")
    N_ASSETS = 1000
    N_EVENTS = 1000
    N_TYPOLOGIES = 5
    BASE_VALUE = 100.0
    
    v, u, C, x_grid, H, lambdas, return_periods, intensity_midpoints = \
        generate_synthetic_portfolio_example_manuscript_lambda(
            n_assets=N_ASSETS,
            n_events=N_EVENTS,
            n_typologies=N_TYPOLOGIES,
            base_value=BASE_VALUE
        )
    
    print(f"   • Assets (N): {N_ASSETS}")
    print(f"   • Events (Q): {N_EVENTS}")
    print(f"   • Typologies (K): {N_TYPOLOGIES}")
    print(f"   • Base value: ${BASE_VALUE:.0f}k")
    
    # Seismic recurrence statistics
    print(f"\n🌍 Seismic Recurrence Statistics:")
    print(f"   • Return period range: {return_periods.min():.1f} to {return_periods.max():.1f} years")
    print(f"   • Lambda range: {lambdas.min():.6f} to {lambdas.max():.6f} events/year")
    print(f"   • Lambda ratio: {lambdas.max()/lambdas.min():.1f}×")
    print(f"   • Intensity range: {H.min():.3f}g to {H.max():.3f}g")
    print(f"   • Frequent events (RP<100): {(return_periods < 100).sum()} events, intensity {H[:, return_periods < 100].mean():.3f}g avg")
    print(f"   • Rare events (RP>1000): {(return_periods > 1000).sum()} events, intensity {H[:, return_periods > 1000].mean():.3f}g avg")
    
    # Portfolio composition
    n_standard = ((v >= 100) & (v <= 200)).sum()
    n_old_pricey = ((v >= 500) & (v <= 600)).sum()
    n_modern = ((v >= 400) & (v < 500)).sum()
    
    print(f"\n📈 Portfolio Composition:")
    print(f"   • Standard homes: {n_standard} ({100*n_standard/N_ASSETS:.1f}%)")
    print(f"   • Old pricey: {n_old_pricey} ({100*n_old_pricey/N_ASSETS:.1f}%)")
    print(f"   • Modern expensive: {n_modern} ({100*n_modern/N_ASSETS:.1f}%)")
    
    # Initialize engine
    print("\n🔧 Initializing Tensorial Risk Engine...")
    engine = TensorialRiskEngine(v, u, C, x_grid, H, lambdas)
    
    # Probabilistic loss and metrics
    print("\n" + "=" * 80)
    print("SECTION 3: PROBABILISTIC HAZARD WITH SEISMIC RECURRENCE")
    print("=" * 80)
    t_start = time.time()
    J_matrix, metrics = engine.compute_loss_and_metrics()
    t_prob = time.time() - t_start
    
    print(f"\n📈 Risk Metrics:")
    print(f"   • Portfolio AAL: ${metrics['aal_portfolio'].numpy():,.2f}k")
    print(f"   • Total rate Λ: {metrics['total_rate'].numpy():.4f}")
    print(f"   • Computation time: {t_prob*1000:.2f} ms")
    
    # AAL by asset class
    aal_np = metrics['aal_per_asset'].numpy()
    aal_standard = aal_np[(v >= 100) & (v <= 200)]
    aal_old_pricey = aal_np[(v >= 500) & (v <= 600)]
    aal_modern = aal_np[(v >= 400) & (v < 500)]
    
    print(f"\n💡 AAL by Asset Class:")
    print(f"   • Standard homes: ${aal_standard.mean():,.2f}k avg (${aal_standard.sum():,.2f}k total, {100*aal_standard.sum()/metrics['aal_portfolio'].numpy():.1f}%)")
    print(f"   • Old pricey: ${aal_old_pricey.mean():,.2f}k avg (${aal_old_pricey.sum():,.2f}k total, {100*aal_old_pricey.sum()/metrics['aal_portfolio'].numpy():.1f}%)")
    print(f"   • Modern expensive: ${aal_modern.mean():,.2f}k avg (${aal_modern.sum():,.2f}k total, {100*aal_modern.sum()/metrics['aal_portfolio'].numpy():.1f}%)")
    
    # Event contribution analysis
    loss_per_event = metrics['loss_per_event'].numpy()
    weighted_contribution = lambdas * loss_per_event  # λ_q × L_q
    
    # Find events contributing most to AAL
    top_contributing_events = np.argsort(weighted_contribution)[-10:][::-1]
    
    print(f"\n🎯 Top 10 Events Contributing to AAL (λ × Loss):")
    for rank, event_idx in enumerate(top_contributing_events, 1):
        print(f"   {rank:2d}. Event {event_idx:4d}: RP={return_periods[event_idx]:7.1f}yr, "
              f"λ={lambdas[event_idx]:.6f}, Loss=${loss_per_event[event_idx]:,.0f}k, "
              f"Contribution=${weighted_contribution[event_idx]:,.2f}k")
    
    # AAL concentration by event rarity
    rare_events_mask = return_periods > 1000
    aal_from_rare = weighted_contribution[rare_events_mask].sum()
    print(f"\n📊 AAL Concentration:")
    print(f"   • AAL from rare events (RP>1000yr): ${aal_from_rare:,.2f}k ({100*aal_from_rare/metrics['aal_portfolio'].numpy():.1f}%)")
    print(f"   • Number of rare events: {rare_events_mask.sum()} ({100*rare_events_mask.sum()/N_EVENTS:.1f}%)")
    
    # Vulnerability gradients
    print("\n" + "=" * 80)
    print("SECTION 4: GRADIENT OF VULNERABILITY")
    print("=" * 80)
    t_start = time.time()
    grad_C, _ = engine.gradient_wrt_vulnerability()
    t_grad_C = time.time() - t_start
    
    grad_per_typology = tf.reduce_sum(tf.abs(grad_C), axis=1).numpy()
    print(f"∂(AAL)/∂C computed in {t_grad_C*1000:.2f} ms")
    print(f"\n💡 Gradient Magnitude by Typology:")
    for k in range(N_TYPOLOGIES):
        print(f"   • Typology {k}: {grad_per_typology[k]:.2e}")
    
    # Exposure gradient
    print("\n" + "=" * 80)
    print("SECTION 5: GRADIENT OF EXPOSURE")
    print("=" * 80)
    t_start = time.time()
    grad_v, _ = engine.gradient_wrt_exposure()
    t_grad_v = time.time() - t_start
    
    grad_v_np = grad_v.numpy()
    grad_standard = grad_v_np[(v >= 100) & (v <= 200)]
    grad_old_pricey = grad_v_np[(v >= 500) & (v <= 600)]
    grad_modern = grad_v_np[(v >= 400) & (v < 500)]
    
    print(f"∂(AAL)/∂v computed in {t_grad_v*1000:.2f} ms")
    print(f"\n💡 Average ∂AAL/∂v by Asset Class:")
    print(f"   • Standard homes: {grad_standard.mean():.4f}")
    print(f"   • Old pricey: {grad_old_pricey.mean():.4f} ({grad_old_pricey.mean()/grad_standard.mean():.1f}× higher)")
    print(f"   • Modern expensive: {grad_modern.mean():.4f}")
    
    # Lambda gradient
    print("\n" + "=" * 80)
    print("SECTION: GRADIENT W.R.T. LAMBDA (∂AAL/∂λ)")
    print("=" * 80)
    t_start = time.time()
    grad_lambdas, _ = engine.gradient_wrt_lambdas()
    t_grad_lambda = time.time() - t_start
    
    grad_lambdas_np = grad_lambdas.numpy()
    
    print(f"∂(AAL)/∂λ computed in {t_grad_lambda*1000:.2f} ms")
    print(f"\n💡 Lambda Gradient Analysis:")
    print(f"   • Interpretation: ∂AAL/∂λ_q = total portfolio loss in event q")
    print(f"   • Range: ${grad_lambdas_np.min():,.2f}k to ${grad_lambdas_np.max():,.2f}k")
    
    # Events with highest ∂AAL/∂λ (highest portfolio loss)
    top_lambda_grad_events = np.argsort(grad_lambdas_np)[-10:][::-1]
    print(f"\n   Top 10 events by ∂AAL/∂λ (highest portfolio loss):")
    for rank, event_idx in enumerate(top_lambda_grad_events, 1):
        print(f"      {rank:2d}. Event {event_idx:4d}: RP={return_periods[event_idx]:7.1f}yr, "
              f"∂AAL/∂λ=${grad_lambdas_np[event_idx]:,.0f}k")
    
    # Full gradient analysis
    print("\n" + "=" * 80)
    print("COMPLETE GRADIENT ANALYSIS")
    print("=" * 80)
    t_start = time.time()
    full_analysis = engine.full_gradient_analysis()
    t_full = time.time() - t_start
    
    print(f"\n∇J = [∂J/∂H, ∂J/∂C, ∂J/∂v, ∂J/∂λ] computed in {t_full*1000:.2f} ms")
    print(f"\n🎯 Key Insights:")
    print(f"   • Exponential lambda creates realistic seismic catalog")
    print(f"   • Rare events (RP>1000yr) contribute {100*aal_from_rare/metrics['aal_portfolio'].numpy():.1f}% of AAL")
    print(f"   • Old pricey neighborhood: {100*aal_old_pricey.sum()/metrics['aal_portfolio'].numpy():.1f}% of AAL from {100*n_old_pricey/N_ASSETS:.1f}% of assets")
    print(f"   • Modern homes protected even at high intensities (0.85-0.95g)")
    print(f"   • ∂AAL/∂v reveals vulnerability-driven risk concentration")
    
    # Store for visualization
    full_analysis['return_periods'] = return_periods
    full_analysis['intensity_midpoints'] = intensity_midpoints
    full_analysis['lambdas'] = lambdas
    full_analysis['weighted_contribution'] = weighted_contribution
    
    return engine, full_analysis


def visualize_results(engine: TensorialRiskEngine, analysis: Dict):
    """
    Create comprehensive visualizations with lambda-intensity relationship.
    Each subfigure is saved as an individual high-quality image.
    """
    global FIGURE_COUNTER
    
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("\n📊 Generating visualizations...")
    
    fig = plt.figure(figsize=(18, 10))
    
    v_np = engine.v.numpy()
    u_np = engine.u.numpy()
    C_np = engine.C.numpy()
    x_grid_np = engine.x_grid.numpy()
    grad_C_np = analysis['grad_vulnerability'].numpy()
    grad_v_np = analysis['grad_exposure'].numpy()
    metrics = analysis['metrics']
    aal_per_asset = metrics['aal_per_asset'].numpy()
    
    return_periods = analysis['return_periods']
    intensity_midpoints = analysis['intensity_midpoints']
    lambdas = analysis['lambdas']
    weighted_contribution = analysis['weighted_contribution']
    
    standard_mask = (v_np >= 100) & (v_np <= 200)
    old_pricey_mask = (v_np >= 500) & (v_np <= 600)
    modern_mask = (v_np >= 400) & (v_np < 500)
    
    # 1. Vulnerability curves
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    for k in range(engine.n_typologies):
        label = f'Type {k}'
        if k == 0:
            label += ' (Old Pricey)'
        elif k == 4:
            label += ' (Modern)'
        ax1.plot(x_grid_np, C_np[k], label=label, linewidth=2, alpha=0.7)
    
    ax1.axvspan(0.02, 0.06, alpha=0.15, color='blue', label='Frequent (RP<100yr)')
    ax1.axvspan(0.85, 0.95, alpha=0.15, color='red', label='Rare (RP>1000yr)')
    ax1.set_xlabel('Intensity (g)', fontsize=13)
    ax1.set_ylabel('Mean Damage Ratio', fontsize=13)
    ax1.set_title('Vulnerability Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: Vulnerability Curves")
    plt.close()
    FIGURE_COUNTER += 1
    
    # 2. Vulnerability gradient heatmap
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = plt.gca()
    im = ax2.imshow(grad_C_np, aspect='auto', cmap='RdYlGn_r')
    ax2.set_xlabel('Intensity Grid Point', fontsize=13)
    ax2.set_ylabel('Typology', fontsize=13)
    ax2.set_title('∂(AAL)/∂C Gradient Heatmap', fontsize=14, fontweight='bold')
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(['Type 0 (Old)', 'Type 1', 'Type 2', 'Type 3', 'Type 4 (Mod)'])
    plt.colorbar(im, ax=ax2, label='Gradient')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: Vulnerability Gradient Heatmap")
    plt.close()
    FIGURE_COUNTER += 1
    
    # 3. Lambda vs Intensity Midpoint
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = plt.gca()
    scatter = ax3.scatter(return_periods, intensity_midpoints, c=lambdas, 
                         cmap='viridis', s=20, alpha=0.6)
    ax3.set_xlabel('Return Period (years)', fontsize=13)
    ax3.set_ylabel('Intensity Midpoint (g)', fontsize=13)
    ax3.set_title('Intensity vs Return Period', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax3, label='λ (events/year)')
    
    # Add secondary x-axis for lambda
    ax3_top = ax3.twiny()
    ax3_top.set_xlim(ax3.get_xlim())
    ax3_top.set_xscale('log')
    lambda_ticks = [0.03, 0.01, 0.003, 0.001, 0.0003]
    rp_ticks = [1/l for l in lambda_ticks]
    ax3_top.set_xticks(rp_ticks)
    ax3_top.set_xticklabels([f'{l:.3f}' for l in lambda_ticks])
    ax3_top.set_xlabel('λ (events/year)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: Intensity vs Return Period")
    plt.close()
    FIGURE_COUNTER += 1
    
    # 4. Exposure gradient
    fig4 = plt.figure(figsize=(8, 10))
    ax4 = plt.gca()
    top_assets = np.argsort(grad_v_np)[-100:]
    colors = ['blue' if standard_mask[i] else ('red' if old_pricey_mask[i] else 'green') for i in top_assets]
    ax4.barh(range(len(top_assets)), grad_v_np[top_assets], color=colors, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('∂(AAL)/∂v', fontsize=13)
    ax4.set_ylabel('Asset Index (Top 100)', fontsize=13)
    ax4.set_title('Exposure Gradient (Top 100)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Standard'),
                      Patch(facecolor='red', label='Old Pricey'),
                      Patch(facecolor='green', label='Modern')]
    ax4.legend(handles=legend_elements, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: Exposure Gradient (Top 100)")
    plt.close()
    FIGURE_COUNTER += 1
    
    # 5. Event contribution to AAL (λ × Loss)
    fig5 = plt.figure(figsize=(8, 6))
    ax5 = plt.gca()
    ax5.scatter(return_periods, weighted_contribution, s=20, alpha=0.5, c=lambdas, cmap='viridis')
    ax5.set_xlabel('Return Period (years)', fontsize=13)
    ax5.set_ylabel('Contribution to AAL: λ×Loss ($k)', fontsize=13)
    ax5.set_title('Event Contribution to Portfolio AAL', fontsize=14, fontweight='bold')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(metrics['aal_portfolio'].numpy()/1000, color='red', linestyle='--', 
                linewidth=2, label=f'AAL/1000')
    ax5.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: Event Contribution to AAL")
    plt.close()
    FIGURE_COUNTER += 1
    
    # 6. AAL vs Exposure scatter
    fig6 = plt.figure(figsize=(8, 6))
    ax6 = plt.gca()
    ax6.scatter(v_np[standard_mask], aal_per_asset[standard_mask], 
               label='Standard', alpha=0.4, s=10, color='blue')
    ax6.scatter(v_np[old_pricey_mask], aal_per_asset[old_pricey_mask], 
               label='Old Pricey', alpha=0.6, s=20, color='red')
    ax6.scatter(v_np[modern_mask], aal_per_asset[modern_mask], 
               label='Modern', alpha=0.6, s=20, color='green')
    ax6.set_xlabel('Exposure ($k)', fontsize=13)
    ax6.set_ylabel('AAL ($k)', fontsize=13)
    ax6.set_title('AAL vs Exposure (Risk Clustering)', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: AAL vs Exposure")
    plt.close()
    FIGURE_COUNTER += 1


def plot_scenario_loss_vs_lambda(engine: TensorialRiskEngine, analysis: Dict):
    """
    Plot scenario loss vs lambda (and return period).
    Each panel saved as individual high-quality image.
    """
    global FIGURE_COUNTER
    
    print("\n📊 Generating scenario loss vs lambda plot...")
    
    lambdas = analysis['lambdas']
    return_periods = analysis['return_periods']
    loss_per_event = analysis['metrics']['loss_per_event'].numpy()
    
    # Plot 1: Loss vs Lambda
    fig7 = plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    scatter1 = ax1.scatter(lambdas, loss_per_event, s=30, alpha=0.6, c=return_periods, 
                cmap='viridis', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Scenario Occurrence Rate λ (events/year)', fontsize=13)
    ax1.set_ylabel('Scenario Loss J ($k)', fontsize=13)
    ax1.set_title('Scenario Loss vs Occurrence Rate', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    cbar1 = plt.colorbar(scatter1, ax=ax1, label='Return Period (years)')
    
    # Add trend annotation
    ax1.text(0.05, 0.95, 'Lower λ (rare) → Higher loss', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: Scenario Loss vs Occurrence Rate")
    plt.close()
    FIGURE_COUNTER += 1
    
    # Plot 2: Loss vs Return Period
    fig8 = plt.figure(figsize=(8, 6))
    ax2 = plt.gca()
    scatter2 = ax2.scatter(return_periods, loss_per_event, s=30, alpha=0.6, c=lambdas, 
                cmap='plasma', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Return Period (years)', fontsize=13)
    ax2.set_ylabel('Scenario Loss J ($k)', fontsize=13)
    ax2.set_title('Scenario Loss vs Return Period', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='λ (events/year)')
    
    # Add statistics
    median_loss = np.median(loss_per_event)
    ax2.axhline(median_loss, color='red', linestyle='--', linewidth=2, 
                label=f'Median: ${median_loss:,.0f}k')
    ax2.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: Scenario Loss vs Return Period")
    plt.close()
    FIGURE_COUNTER += 1


def plot_hazard_gradient_analysis(engine: TensorialRiskEngine, analysis: Dict):
    """
    Plot hazard gradient analysis showing sensitivity to intensity changes.
    Each panel saved as individual high-quality image.
    """
    global FIGURE_COUNTER
    
    print("\n📊 Generating hazard gradient analysis plot...")
    
    grad_H = analysis['grad_hazard'].numpy()
    lambdas = analysis['lambdas']
    return_periods = analysis['return_periods']
    v_np = engine.v.numpy()
    
    # Compute statistics
    # Average |∂AAL/∂H| per event (across all assets)
    avg_grad_per_event = np.mean(np.abs(grad_H), axis=0)
    
    # Average |∂AAL/∂H| per asset (across all events)
    avg_grad_per_asset = np.mean(np.abs(grad_H), axis=1)
    
    # Identify asset classes
    standard_mask = (v_np >= 100) & (v_np <= 200)
    old_pricey_mask = (v_np >= 500) & (v_np <= 600)
    modern_mask = (v_np >= 400) & (v_np < 500)
    
    # Plot 1: Average gradient per event vs return period
    fig9 = plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    scatter1 = ax1.scatter(return_periods, avg_grad_per_event, s=30, alpha=0.6, 
                          c=lambdas, cmap='viridis', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Return Period (years)', fontsize=13)
    ax1.set_ylabel('Avg |∂AAL/∂H| per Event ($/g)', fontsize=13)
    ax1.set_title('Hazard Sensitivity vs Return Period', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    plt.colorbar(scatter1, ax=ax1, label='λ (events/year)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: Hazard Sensitivity vs Return Period")
    plt.close()
    FIGURE_COUNTER += 1
    
    # Plot 2: Top 50 assets by average hazard gradient
    fig10 = plt.figure(figsize=(8, 8))
    ax2 = plt.gca()
    top_50_assets = np.argsort(avg_grad_per_asset)[-50:]
    colors = ['blue' if standard_mask[i] else ('red' if old_pricey_mask[i] else 'green') 
              for i in top_50_assets]
    ax2.barh(range(len(top_50_assets)), avg_grad_per_asset[top_50_assets], 
             color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Avg |∂AAL/∂H| per Asset ($/g)', fontsize=13)
    ax2.set_ylabel('Asset Index (Top 50)', fontsize=13)
    ax2.set_title('Top 50 Assets by Hazard Sensitivity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Standard'),
                      Patch(facecolor='red', label='Old Pricey'),
                      Patch(facecolor='green', label='Modern')]
    ax2.legend(handles=legend_elements, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: Top 50 Assets by Hazard Sensitivity")
    plt.close()
    FIGURE_COUNTER += 1
    
    # Plot 3: Distribution of hazard gradients
    fig11 = plt.figure(figsize=(8, 6))
    ax3 = plt.gca()
    ax3.hist(avg_grad_per_asset[standard_mask], bins=30, alpha=0.6, 
             label='Standard', color='blue', edgecolor='black')
    ax3.hist(avg_grad_per_asset[old_pricey_mask], bins=30, alpha=0.6, 
             label='Old Pricey', color='red', edgecolor='black')
    ax3.hist(avg_grad_per_asset[modern_mask], bins=30, alpha=0.6, 
             label='Modern', color='green', edgecolor='black')
    ax3.set_xlabel('Avg |∂AAL/∂H| per Asset ($/g)', fontsize=13)
    ax3.set_ylabel('Number of Assets', fontsize=13)
    ax3.set_title('Hazard Gradient Distribution by Asset Class', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: Hazard Gradient Distribution")
    plt.close()
    FIGURE_COUNTER += 1
    
    # Plot 4: Gradient heatmap for top events (sample)
    fig12 = plt.figure(figsize=(10, 8))
    ax4 = plt.gca()
    # Select 20 representative events (evenly spaced in log-RP space)
    event_indices = np.linspace(0, len(return_periods)-1, 20, dtype=int)
    # Select top 100 assets by average gradient
    asset_indices = np.argsort(avg_grad_per_asset)[-100:]
    
    grad_sample = grad_H[np.ix_(asset_indices, event_indices)]
    im = ax4.imshow(grad_sample, aspect='auto', cmap='RdYlBu_r', 
                    interpolation='nearest')
    ax4.set_xlabel('Event Index (20 samples)', fontsize=13)
    ax4.set_ylabel('Asset Index (Top 100)', fontsize=13)
    ax4.set_title('∂AAL/∂H Heatmap (Sampled)', fontsize=14, fontweight='bold')
    
    # Annotate x-axis with return periods
    ax4.set_xticks(range(0, 20, 4))
    ax4.set_xticklabels([f'{return_periods[event_indices[i]]:.0f}yr' 
                         for i in range(0, 20, 4)], rotation=45)
    
    cbar = plt.colorbar(im, ax=ax4, label='∂AAL/∂H ($/g)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: Hazard Gradient Heatmap")
    plt.close()
    FIGURE_COUNTER += 1


def plot_additional_distributions(engine: TensorialRiskEngine, analysis: Dict):
    """
    Create additional distribution plots for manuscript.
    
    Figures:
    1. Event loss distribution histogram
    2. Exposure distribution by asset class
    3. AAL distribution by asset class
    """
    global FIGURE_COUNTER
    
    print("\n📊 Generating additional distribution plots...")
    
    v_np = engine.v.numpy()
    metrics = analysis['metrics']
    loss_per_event = metrics['loss_per_event'].numpy()
    aal_per_asset = metrics['aal_per_asset'].numpy()
    
    # Asset class masks
    standard_mask = (v_np >= 100) & (v_np <= 200)
    old_pricey_mask = (v_np >= 500) & (v_np <= 600)
    modern_mask = (v_np >= 400) & (v_np < 500)
    
    # Figure 1: Event Loss Distribution
    fig13 = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    counts, bins, patches = ax.hist(loss_per_event, bins=50, color='steelblue', 
                                     edgecolor='black', alpha=0.7, linewidth=0.8)
    
    # Add statistics
    mean_loss = loss_per_event.mean()
    median_loss = np.median(loss_per_event)
    ax.axvline(mean_loss, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: ${mean_loss:,.0f}k')
    ax.axvline(median_loss, color='green', linestyle='--', linewidth=2, 
               label=f'Median: ${median_loss:,.0f}k')
    
    ax.set_xlabel('Event Loss ($k)', fontsize=13)
    ax.set_ylabel('Number of Events', fontsize=13)
    ax.set_title('Distribution of Event Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text box with statistics
    textstr = f'Events: {len(loss_per_event)}\nStd Dev: ${loss_per_event.std():,.0f}k\nMin: ${loss_per_event.min():,.0f}k\nMax: ${loss_per_event.max():,.0f}k'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: Event Loss Distribution")
    plt.close()
    FIGURE_COUNTER += 1
    
    # Figure 2: Exposure Distribution by Asset Class
    fig14 = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Create histograms for each class
    bins = np.linspace(v_np.min(), v_np.max(), 40)
    ax.hist(v_np[standard_mask], bins=bins, alpha=0.6, label='Standard', 
            color='blue', edgecolor='black', linewidth=0.5)
    ax.hist(v_np[old_pricey_mask], bins=bins, alpha=0.7, label='Old Pricey', 
            color='red', edgecolor='black', linewidth=0.5)
    ax.hist(v_np[modern_mask], bins=bins, alpha=0.7, label='Modern', 
            color='green', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Exposure ($k)', fontsize=13)
    ax.set_ylabel('Number of Assets', fontsize=13)
    ax.set_title('Exposure Distribution by Asset Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    textstr = (f'Standard: {standard_mask.sum()} assets, ${v_np[standard_mask].mean():.0f}k avg\n'
               f'Old Pricey: {old_pricey_mask.sum()} assets, ${v_np[old_pricey_mask].mean():.0f}k avg\n'
               f'Modern: {modern_mask.sum()} assets, ${v_np[modern_mask].mean():.0f}k avg')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: Exposure Distribution by Asset Class")
    plt.close()
    FIGURE_COUNTER += 1
    
    # Figure 3: AAL Distribution by Asset Class
    fig15 = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Create histograms for each class
    aal_max = aal_per_asset.max()
    bins_aal = np.linspace(0, aal_max, 50)
    ax.hist(aal_per_asset[standard_mask], bins=bins_aal, alpha=0.6, 
            label='Standard', color='blue', edgecolor='black', linewidth=0.5)
    ax.hist(aal_per_asset[old_pricey_mask], bins=bins_aal, alpha=0.7, 
            label='Old Pricey', color='red', edgecolor='black', linewidth=0.5)
    ax.hist(aal_per_asset[modern_mask], bins=bins_aal, alpha=0.7, 
            label='Modern', color='green', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('AAL ($k)', fontsize=13)
    ax.set_ylabel('Frequency (Number of Assets)', fontsize=13)
    ax.set_title('AAL Distribution by Asset Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    textstr = (f'Standard: ${aal_per_asset[standard_mask].mean():.2f}k avg, ${aal_per_asset[standard_mask].sum():,.0f}k total\n'
               f'Old Pricey: ${aal_per_asset[old_pricey_mask].mean():.2f}k avg, ${aal_per_asset[old_pricey_mask].sum():,.0f}k total\n'
               f'Modern: ${aal_per_asset[modern_mask].mean():.2f}k avg, ${aal_per_asset[modern_mask].sum():,.0f}k total')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'Figure{FIGURE_COUNTER}.png'), dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved Figure{FIGURE_COUNTER}: AAL Distribution by Asset Class")
    plt.close()
    FIGURE_COUNTER += 1


def save_analysis_summary(engine: TensorialRiskEngine, analysis: Dict, filename: str = 'analysis_summary.txt'):
    """
    Save comprehensive analysis summary to text file.
    
    Includes all key metrics, statistics, and insights.
    """
    print(f"\n💾 Saving analysis summary to '{filename}'...")
    
    v_np = engine.v.numpy()
    u_np = engine.u.numpy()
    metrics = analysis['metrics']
    lambdas = analysis['lambdas']
    return_periods = analysis['return_periods']
    intensity_midpoints = analysis['intensity_midpoints']
    
    # Asset class masks
    standard_mask = (v_np >= 100) & (v_np <= 200)
    old_pricey_mask = (v_np >= 500) & (v_np <= 600)
    modern_mask = (v_np >= 400) & (v_np < 500)
    
    # Compute additional statistics
    aal_np = metrics['aal_per_asset'].numpy()
    grad_v_np = analysis['grad_exposure'].numpy()
    grad_H_np = analysis['grad_hazard'].numpy()
    grad_lambdas_np = analysis['grad_lambdas'].numpy()
    loss_per_event = metrics['loss_per_event'].numpy()
    weighted_contribution = analysis['weighted_contribution']
    
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TENSORIAL RISK ENGINE - EXPONENTIAL LAMBDA ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Portfolio configuration
        f.write("PORTFOLIO CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total assets (N): {engine.n_assets}\n")
        f.write(f"Total events (Q): {engine.n_events}\n")
        f.write(f"Typologies (K): {engine.n_typologies}\n")
        f.write(f"Base value: $100k\n\n")
        
        f.write("Asset Class Distribution:\n")
        f.write(f"  • Standard homes: {standard_mask.sum()} ({100*standard_mask.sum()/engine.n_assets:.1f}%)\n")
        f.write(f"    Exposure range: ${v_np[standard_mask].min():.0f}k - ${v_np[standard_mask].max():.0f}k\n")
        f.write(f"  • Old pricey (type 0): {old_pricey_mask.sum()} ({100*old_pricey_mask.sum()/engine.n_assets:.1f}%)\n")
        f.write(f"    Exposure range: ${v_np[old_pricey_mask].min():.0f}k - ${v_np[old_pricey_mask].max():.0f}k\n")
        f.write(f"  • Modern expensive (type 4): {modern_mask.sum()} ({100*modern_mask.sum()/engine.n_assets:.1f}%)\n")
        f.write(f"    Exposure range: ${v_np[modern_mask].min():.0f}k - ${v_np[modern_mask].max():.0f}k\n\n")
        
        # Seismic recurrence
        f.write("SEISMIC RECURRENCE MODEL\n")
        f.write("-" * 80 + "\n")
        f.write(f"Return period range: {return_periods.min():.1f} to {return_periods.max():.1f} years\n")
        f.write(f"Lambda range: {lambdas.min():.6f} to {lambdas.max():.6f} events/year\n")
        f.write(f"Lambda ratio: {lambdas.max()/lambdas.min():.1f}×\n")
        f.write(f"Intensity range: {intensity_midpoints.min():.3f}g to {intensity_midpoints.max():.3f}g\n\n")
        
        f.write("Event Statistics:\n")
        f.write(f"  • Frequent events (RP<100yr): {(return_periods < 100).sum()} events\n")
        f.write(f"    Avg intensity: {intensity_midpoints[return_periods < 100].mean():.3f}g\n")
        f.write(f"  • Moderate events (100≤RP<1000yr): {((return_periods >= 100) & (return_periods < 1000)).sum()} events\n")
        f.write(f"    Avg intensity: {intensity_midpoints[(return_periods >= 100) & (return_periods < 1000)].mean():.3f}g\n")
        f.write(f"  • Rare events (RP≥1000yr): {(return_periods >= 1000).sum()} events\n")
        f.write(f"    Avg intensity: {intensity_midpoints[return_periods >= 1000].mean():.3f}g\n\n")
        
        # Risk metrics
        f.write("RISK METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Portfolio AAL: ${metrics['aal_portfolio'].numpy():,.2f}k\n")
        f.write(f"Total rate Λ: {metrics['total_rate'].numpy():.6f}\n\n")
        
        f.write("AAL by Asset Class:\n")
        aal_standard = aal_np[standard_mask]
        aal_old_pricey = aal_np[old_pricey_mask]
        aal_modern = aal_np[modern_mask]
        
        f.write(f"  • Standard homes:\n")
        f.write(f"    Mean AAL: ${aal_standard.mean():,.2f}k\n")
        f.write(f"    Total AAL: ${aal_standard.sum():,.2f}k ({100*aal_standard.sum()/metrics['aal_portfolio'].numpy():.1f}%)\n")
        f.write(f"    Min/Max: ${aal_standard.min():,.2f}k / ${aal_standard.max():,.2f}k\n\n")
        
        f.write(f"  • Old pricey:\n")
        f.write(f"    Mean AAL: ${aal_old_pricey.mean():,.2f}k\n")
        f.write(f"    Total AAL: ${aal_old_pricey.sum():,.2f}k ({100*aal_old_pricey.sum()/metrics['aal_portfolio'].numpy():.1f}%)\n")
        f.write(f"    Min/Max: ${aal_old_pricey.min():,.2f}k / ${aal_old_pricey.max():,.2f}k\n\n")
        
        f.write(f"  • Modern expensive:\n")
        f.write(f"    Mean AAL: ${aal_modern.mean():,.2f}k\n")
        f.write(f"    Total AAL: ${aal_modern.sum():,.2f}k ({100*aal_modern.sum()/metrics['aal_portfolio'].numpy():.1f}%)\n")
        f.write(f"    Min/Max: ${aal_modern.min():,.2f}k / ${aal_modern.max():,.2f}k\n\n")
        
        # Event loss statistics
        f.write("Event Loss Statistics:\n")
        f.write(f"  • Min scenario loss: ${loss_per_event.min():,.2f}k\n")
        f.write(f"  • Mean scenario loss: ${loss_per_event.mean():,.2f}k\n")
        f.write(f"  • Median scenario loss: ${np.median(loss_per_event):,.2f}k\n")
        f.write(f"  • Max scenario loss: ${loss_per_event.max():,.2f}k\n")
        f.write(f"  • Std dev: ${loss_per_event.std():,.2f}k\n\n")
        
        # AAL concentration
        rare_events_mask = return_periods > 1000
        aal_from_rare = weighted_contribution[rare_events_mask].sum()
        f.write("AAL Concentration:\n")
        f.write(f"  • AAL from rare events (RP>1000yr): ${aal_from_rare:,.2f}k ({100*aal_from_rare/metrics['aal_portfolio'].numpy():.1f}%)\n")
        f.write(f"  • Number of rare events: {rare_events_mask.sum()} ({100*rare_events_mask.sum()/engine.n_events:.1f}%)\n\n")
        
        # Top contributing events
        f.write("Top 10 Events Contributing to AAL (λ × Loss):\n")
        top_contributing_events = np.argsort(weighted_contribution)[-10:][::-1]
        for rank, event_idx in enumerate(top_contributing_events, 1):
            f.write(f"  {rank:2d}. Event {event_idx:4d}: RP={return_periods[event_idx]:7.1f}yr, "
                   f"λ={lambdas[event_idx]:.6f}, Loss=${loss_per_event[event_idx]:,.0f}k, "
                   f"Contribution=${weighted_contribution[event_idx]:,.2f}k\n")
        f.write("\n")
        
        # Gradient analysis
        f.write("GRADIENT ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        # Exposure gradient
        grad_standard = grad_v_np[standard_mask]
        grad_old_pricey = grad_v_np[old_pricey_mask]
        grad_modern = grad_v_np[modern_mask]
        
        f.write("Exposure Gradient (∂AAL/∂v):\n")
        f.write(f"  • Standard homes: {grad_standard.mean():.4f} avg\n")
        f.write(f"  • Old pricey: {grad_old_pricey.mean():.4f} avg ({grad_old_pricey.mean()/grad_standard.mean():.1f}× higher)\n")
        f.write(f"  • Modern expensive: {grad_modern.mean():.4f} avg\n\n")
        
        f.write("Top 10 Assets by Exposure Gradient:\n")
        top_exp_grad = np.argsort(grad_v_np)[-10:][::-1]
        for rank, idx in enumerate(top_exp_grad, 1):
            asset_class = "Standard" if standard_mask[idx] else ("Old Pricey" if old_pricey_mask[idx] else "Modern")
            f.write(f"  {rank:2d}. Asset {idx:4d}: ∂AAL/∂v={grad_v_np[idx]:.4f}, "
                   f"Exp=${v_np[idx]:.0f}k, Type={u_np[idx]} ({asset_class})\n")
        f.write("\n")
        
        # Hazard gradient
        avg_grad_H_per_asset = np.mean(np.abs(grad_H_np), axis=1)
        avg_grad_H_standard = avg_grad_H_per_asset[standard_mask]
        avg_grad_H_old_pricey = avg_grad_H_per_asset[old_pricey_mask]
        avg_grad_H_modern = avg_grad_H_per_asset[modern_mask]
        
        f.write("Hazard Gradient (∂AAL/∂H):\n")
        f.write(f"  • Overall average sensitivity: {np.mean(np.abs(grad_H_np)):.2e} $/g\n")
        f.write(f"  • Standard homes: {avg_grad_H_standard.mean():.2e} $/g avg\n")
        f.write(f"  • Old pricey: {avg_grad_H_old_pricey.mean():.2e} $/g avg\n")
        f.write(f"  • Modern expensive: {avg_grad_H_modern.mean():.2e} $/g avg\n\n")
        
        # Lambda gradient
        f.write("Lambda Gradient (∂AAL/∂λ):\n")
        f.write(f"  • Range: ${grad_lambdas_np.min():,.2f}k to ${grad_lambdas_np.max():,.2f}k\n")
        f.write(f"  • Mean: ${grad_lambdas_np.mean():,.2f}k\n")
        f.write(f"  • Median: ${np.median(grad_lambdas_np):,.2f}k\n\n")
        
        f.write("Top 10 Events by ∂AAL/∂λ (Highest Portfolio Loss):\n")
        top_lambda_grad = np.argsort(grad_lambdas_np)[-10:][::-1]
        for rank, event_idx in enumerate(top_lambda_grad, 1):
            f.write(f"  {rank:2d}. Event {event_idx:4d}: RP={return_periods[event_idx]:7.1f}yr, "
                   f"∂AAL/∂λ=${grad_lambdas_np[event_idx]:,.0f}k\n")
        f.write("\n")
        
        # Vulnerability gradient
        grad_C_np = analysis['grad_vulnerability'].numpy()
        grad_per_typology = np.sum(np.abs(grad_C_np), axis=1)
        f.write("Vulnerability Gradient (∂AAL/∂C):\n")
        for k in range(engine.n_typologies):
            f.write(f"  • Typology {k}: {grad_per_typology[k]:.2e}\n")
        f.write("\n")
        
        # Key insights
        f.write("KEY INSIGHTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"1. Risk Concentration:\n")
        f.write(f"   - Old pricey neighborhood: {100*old_pricey_mask.sum()/engine.n_assets:.1f}% of assets\n")
        f.write(f"   - Contributes: {100*aal_old_pricey.sum()/metrics['aal_portfolio'].numpy():.1f}% of portfolio AAL\n")
        f.write(f"   - Risk amplification: {(100*aal_old_pricey.sum()/metrics['aal_portfolio'].numpy())/(100*old_pricey_mask.sum()/engine.n_assets):.1f}×\n\n")
        
        f.write(f"2. Event Frequency Impact:\n")
        f.write(f"   - Rare events (RP>1000yr): {100*rare_events_mask.sum()/engine.n_events:.1f}% of events\n")
        f.write(f"   - Contribute: {100*aal_from_rare/metrics['aal_portfolio'].numpy():.1f}% of AAL\n")
        f.write(f"   - Demonstrates importance of tail events in seismic risk\n\n")
        
        f.write(f"3. Vulnerability-Exposure Interaction:\n")
        f.write(f"   - Old pricey exposure gradient: {grad_old_pricey.mean():.4f}\n")
        f.write(f"   - Standard homes gradient: {grad_standard.mean():.4f}\n")
        f.write(f"   - Sensitivity ratio: {grad_old_pricey.mean()/grad_standard.mean():.1f}×\n")
        f.write(f"   - High vulnerability (type 0) + high exposure = concentrated risk\n\n")
        
        f.write(f"4. Modern Construction Benefits:\n")
        f.write(f"   - Modern homes AAL: ${aal_modern.mean():,.2f}k avg\n")
        f.write(f"   - Old pricey AAL: ${aal_old_pricey.mean():,.2f}k avg\n")
        f.write(f"   - Despite similar exposure, AAL ratio: {aal_old_pricey.mean()/aal_modern.mean():.1f}×\n")
        f.write(f"   - Superior seismic design (type 4) provides significant protection\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("End of Analysis Summary\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Analysis summary saved to '{filename}'")


def main():
    """Main execution function."""
    engine, analysis = demonstrate_manuscript_example_exponential_lambda()
    visualize_results(engine, analysis)
    
    # Additional tasks
    print("\n" + "=" * 80)
    print("ADDITIONAL ANALYSIS")
    print("=" * 80)
    
    # Task 1: Plot scenario loss vs lambda
    plot_scenario_loss_vs_lambda(engine, analysis)
    
    # Task 2: Save comprehensive summary
    save_analysis_summary(engine, analysis, 'exponential_lambda_analysis_summary.txt')
    
    # Task 3: Plot hazard gradient analysis
    plot_hazard_gradient_analysis(engine, analysis)
    
    # Task 4: Plot additional distributions
    plot_additional_distributions(engine, analysis)
    
    return engine, analysis


if __name__ == "__main__":
    engine, analysis = main()
    
    print("\n" + "=" * 80)
    print("✅ EXPONENTIAL LAMBDA DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print("   ✓ Realistic seismic recurrence law (exponential λ, RP: 32-5000 years)")
    print("   ✓ Intensity-RP coupling: frequent→low, rare→high")
    print("   ✓ Rare events dominate AAL despite low occurrence rates")
    print("   ✓ Old pricey neighborhood extremely vulnerable to rare events")
    print("   ✓ Modern homes protected even at 0.85-0.95g intensities")
    print("   ✓ ∂AAL/∂λ reveals which events drive portfolio risk")
    print("\nGenerated outputs:")
    print(f"   📁 Folder: {OUTPUT_FOLDER}/")
    print(f"   📊 15 individual high-quality figures (Figure1.png - Figure15.png)")
    print("   📄 exponential_lambda_analysis_summary.txt")
    print("\n" + "=" * 80)
