"""
JAX implementation of the Tensorial Risk Engine.

This module mirrors the public risk API of ``tensor_engine.py`` while keeping
the canonical TensorFlow implementation untouched. The core functions are pure
JAX functions so they can be used directly with ``jax.jit`` and ``jax.grad``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp


ArrayLike = Union[np.ndarray, jax.Array]
TensorDict = Dict[str, jax.Array]


def _as_float_array(x: ArrayLike, *, device: Optional[jax.Device] = None) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.float32)
    return jax.device_put(arr, device) if device is not None else arr


def _as_index_array(x: ArrayLike, *, device: Optional[jax.Device] = None) -> jax.Array:
    arr = jnp.asarray(x, dtype=jnp.int32)
    return jax.device_put(arr, device) if device is not None else arr


def generate_synthetic_portfolio(
    n_assets: int,
    n_events: int,
    n_typologies: int = 5,
    n_curve_points: int = 20,
    lambdas: Optional[np.ndarray] = None,
    lambda_distribution: str = "exponential",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic portfolio data matching the TensorFlow implementation."""
    np.random.seed(42)

    v_exposure = np.random.uniform(100_000, 1_000_000, n_assets).astype(np.float32)
    u_typology = np.random.randint(0, n_typologies, n_assets).astype(np.int32)
    x_grid = np.linspace(0.0, 1.5, n_curve_points).astype(np.float32)

    C_matrix = np.zeros((n_typologies, n_curve_points), dtype=np.float32)
    for k in range(n_typologies):
        steepness = 8.0 + k * 2.0
        midpoint = 0.4 + k * 0.1
        C_matrix[k, :] = 1.0 / (1.0 + np.exp(-steepness * (x_grid - midpoint)))

    H_intensities = np.random.uniform(0.0, 1.2, (n_assets, n_events)).astype(np.float32)

    if lambdas is None:
        if lambda_distribution == "uniform":
            lambdas_out = np.ones(n_events, dtype=np.float32) / n_events
        elif lambda_distribution == "exponential":
            lambdas_out = np.exp(-np.linspace(0, 3, n_events)).astype(np.float32)
            lambdas_out = lambdas_out / lambdas_out.sum()
        else:
            raise ValueError(f"Unknown lambda_distribution: {lambda_distribution}")
    else:
        lambdas_out = lambdas.astype(np.float32)

    return v_exposure, u_typology, C_matrix, x_grid, H_intensities, lambdas_out


def deterministic_loss(
    v: jax.Array,
    u: jax.Array,
    C: jax.Array,
    x_grid: jax.Array,
    h: jax.Array,
) -> jax.Array:
    """Compute deterministic loss for a single hazard scenario."""
    u = u.astype(jnp.int32)
    n_points = x_grid.shape[0]

    h_eval = jnp.clip(h, x_grid[0], x_grid[-1])
    valid_iml = h >= x_grid[0]

    idx = jnp.searchsorted(x_grid, h_eval, side="right") - 1
    idx = jnp.clip(idx, 0, n_points - 2).astype(jnp.int32)

    x_lower = x_grid[idx]
    x_upper = x_grid[idx + 1]
    alpha = (h_eval - x_lower) / (x_upper - x_lower + jnp.asarray(1e-8, dtype=x_grid.dtype))

    c_flat = jnp.reshape(C, (-1,))
    c_lower = c_flat[u * n_points + idx]
    c_upper = c_flat[u * n_points + idx + 1]

    mdr = (1.0 - alpha) * c_lower + alpha * c_upper
    mdr = jnp.where(valid_iml, mdr, jnp.zeros_like(mdr))
    return jnp.sum(v * mdr)


def probabilistic_loss_matrix(
    v: jax.Array,
    u: jax.Array,
    C: jax.Array,
    x_grid: jax.Array,
    H: jax.Array,
) -> jax.Array:
    """Compute the event loss matrix for all assets and events."""
    u = u.astype(jnp.int32)
    n_assets = H.shape[0]
    n_events = H.shape[1]
    n_points = x_grid.shape[0]

    H_flat = jnp.reshape(H, (-1,))
    H_eval = jnp.clip(H_flat, x_grid[0], x_grid[-1])
    valid_iml = H_flat >= x_grid[0]

    idx = jnp.searchsorted(x_grid, H_eval, side="right") - 1
    idx = jnp.clip(idx, 0, n_points - 2).astype(jnp.int32)

    x_lower = x_grid[idx]
    x_upper = x_grid[idx + 1]
    alpha = (H_eval - x_lower) / (x_upper - x_lower + jnp.asarray(1e-8, dtype=x_grid.dtype))

    u_flat = jnp.reshape(jnp.broadcast_to(u[:, None], (n_assets, n_events)), (-1,))
    c_flat = jnp.reshape(C, (-1,))
    c_lower = c_flat[u_flat * n_points + idx]
    c_upper = c_flat[u_flat * n_points + idx + 1]

    mdr_flat = (1.0 - alpha) * c_lower + alpha * c_upper
    mdr_flat = jnp.where(valid_iml, mdr_flat, jnp.zeros_like(mdr_flat))
    mdr_matrix = jnp.reshape(mdr_flat, (n_assets, n_events))
    return v[:, None] * mdr_matrix


def compute_risk_metrics(
    J_matrix: jax.Array,
    lambdas: Optional[jax.Array] = None,
) -> TensorDict:
    """Compute rate-weighted risk metrics from the loss matrix."""
    if lambdas is None:
        q = J_matrix.shape[1]
        lambdas = jnp.ones(q, dtype=J_matrix.dtype) / float(q)
    else:
        lambdas = jnp.asarray(lambdas, dtype=J_matrix.dtype)

    total_rate = jnp.sum(lambdas)
    weights = lambdas / (total_rate + jnp.asarray(1e-10, dtype=J_matrix.dtype))

    aal_per_asset = jnp.sum(J_matrix * lambdas[None, :], axis=1)
    aal_portfolio = jnp.sum(aal_per_asset)
    mean_per_event_per_asset = aal_per_asset / (total_rate + jnp.asarray(1e-10, dtype=J_matrix.dtype))
    deviations_sq = jnp.square(J_matrix - mean_per_event_per_asset[:, None])
    variance_per_asset = jnp.sum(deviations_sq * weights[None, :], axis=1)
    std_per_asset = jnp.sqrt(variance_per_asset)
    loss_per_event = jnp.sum(J_matrix, axis=0)

    return {
        "aal_per_asset": aal_per_asset,
        "aal_portfolio": aal_portfolio,
        "mean_per_event_per_asset": mean_per_event_per_asset,
        "variance_per_asset": variance_per_asset,
        "std_per_asset": std_per_asset,
        "loss_per_event": loss_per_event,
        "total_rate": total_rate,
    }


def portfolio_aal(
    v: jax.Array,
    u: jax.Array,
    C: jax.Array,
    x_grid: jax.Array,
    H: jax.Array,
    lambdas: jax.Array,
) -> jax.Array:
    """Scalar portfolio AAL used as the automatic-differentiation target."""
    J_matrix = probabilistic_loss_matrix(v, u, C, x_grid, H)
    return compute_risk_metrics(J_matrix, lambdas)["aal_portfolio"]


def compute_loss_and_metrics_pure(
    v: jax.Array,
    u: jax.Array,
    C: jax.Array,
    x_grid: jax.Array,
    H: jax.Array,
    lambdas: jax.Array,
) -> Tuple[jax.Array, TensorDict]:
    """Pure functional loss and metrics API, convenient for ``jax.jit``."""
    J_matrix = probabilistic_loss_matrix(v, u, C, x_grid, H)
    return J_matrix, compute_risk_metrics(J_matrix, lambdas)


def full_gradient_analysis_pure(
    v: jax.Array,
    u: jax.Array,
    C: jax.Array,
    x_grid: jax.Array,
    H: jax.Array,
    lambdas: jax.Array,
) -> Dict[str, Union[jax.Array, TensorDict]]:
    """Pure functional full gradient analysis."""
    J_matrix, metrics = compute_loss_and_metrics_pure(v, u, C, x_grid, H, lambdas)
    grad_v, grad_C, grad_H, grad_lambdas = jax.grad(
        portfolio_aal,
        argnums=(0, 2, 4, 5),
    )(v, u, C, x_grid, H, lambdas)
    return {
        "grad_hazard": grad_H,
        "grad_vulnerability": grad_C,
        "grad_exposure": grad_v,
        "grad_lambdas": grad_lambdas,
        "metrics": metrics,
        "loss_matrix": J_matrix,
    }


class TensorialRiskEngine:
    """JAX backend with a high-level API matching ``tensor_engine.TensorialRiskEngine``."""

    def __init__(
        self,
        v: np.ndarray,
        u: np.ndarray,
        C: np.ndarray,
        x_grid: np.ndarray,
        H: np.ndarray,
        lambdas: Optional[np.ndarray] = None,
        device: Optional[jax.Device] = None,
    ):
        self.device = device
        self.v = _as_float_array(v, device=device)
        self.u = _as_index_array(u, device=device)
        self.C = _as_float_array(C, device=device)
        self.x_grid = _as_float_array(x_grid, device=device)
        self.H = _as_float_array(H, device=device)

        self.n_assets = int(v.shape[0])
        self.n_events = int(H.shape[1])
        self.n_typologies = int(C.shape[0])

        if lambdas is None:
            lambdas = np.ones(self.n_events, dtype=np.float32) / self.n_events
        self.lambdas = _as_float_array(lambdas, device=device)

    def compute_loss_and_metrics(self) -> Tuple[jax.Array, TensorDict]:
        return compute_loss_and_metrics_pure(self.v, self.u, self.C, self.x_grid, self.H, self.lambdas)

    def gradient_wrt_vulnerability(self) -> Tuple[jax.Array, TensorDict]:
        grad_C = jax.grad(portfolio_aal, argnums=2)(
            self.v, self.u, self.C, self.x_grid, self.H, self.lambdas
        )
        _, metrics = self.compute_loss_and_metrics()
        return grad_C, metrics

    def gradient_wrt_exposure(self) -> Tuple[jax.Array, TensorDict]:
        grad_v = jax.grad(portfolio_aal, argnums=0)(
            self.v, self.u, self.C, self.x_grid, self.H, self.lambdas
        )
        _, metrics = self.compute_loss_and_metrics()
        return grad_v, metrics

    def gradient_wrt_hazard(self) -> Tuple[jax.Array, TensorDict]:
        grad_H = jax.grad(portfolio_aal, argnums=4)(
            self.v, self.u, self.C, self.x_grid, self.H, self.lambdas
        )
        _, metrics = self.compute_loss_and_metrics()
        return grad_H, metrics

    def gradient_wrt_lambdas(self) -> Tuple[jax.Array, TensorDict]:
        grad_lambdas = jax.grad(portfolio_aal, argnums=5)(
            self.v, self.u, self.C, self.x_grid, self.H, self.lambdas
        )
        _, metrics = self.compute_loss_and_metrics()
        return grad_lambdas, metrics

    def full_gradient_analysis(self) -> Dict[str, Union[jax.Array, TensorDict]]:
        return full_gradient_analysis_pure(self.v, self.u, self.C, self.x_grid, self.H, self.lambdas)


def classical_loss(
    v: jax.Array,
    u: jax.Array,
    C: jax.Array,
    x_grid: jax.Array,
    hazard_poes: jax.Array,
    hazard_imls: jax.Array,
) -> jax.Array:
    """Compute average annual loss per asset from hazard curves."""
    u = u.astype(jnp.int32)
    n_assets = v.shape[0]
    n_levels = hazard_imls.shape[0]
    n_points = x_grid.shape[0]

    imls_flat = jnp.reshape(jnp.broadcast_to(hazard_imls[None, :], (n_assets, n_levels)), (-1,))
    imls_eval = jnp.clip(imls_flat, x_grid[0], x_grid[-1])
    valid_iml = imls_flat >= x_grid[0]

    idx = jnp.searchsorted(x_grid, imls_eval, side="right") - 1
    idx = jnp.clip(idx, 0, n_points - 2).astype(jnp.int32)
    x_lower = x_grid[idx]
    x_upper = x_grid[idx + 1]
    alpha = (imls_eval - x_lower) / (x_upper - x_lower + jnp.asarray(1e-8, dtype=x_grid.dtype))

    u_flat = jnp.reshape(jnp.broadcast_to(u[:, None], (n_assets, n_levels)), (-1,))
    c_flat = jnp.reshape(C, (-1,))
    c_lower = c_flat[u_flat * n_points + idx]
    c_upper = c_flat[u_flat * n_points + idx + 1]
    mdr_flat = (1.0 - alpha) * c_lower + alpha * c_upper
    mdr_flat = jnp.where(valid_iml, mdr_flat, jnp.zeros_like(mdr_flat))
    mdr_matrix = jnp.reshape(mdr_flat, (n_assets, n_levels))

    poe_shifted = jnp.concatenate(
        [hazard_poes[:, 1:], jnp.zeros((n_assets, 1), dtype=hazard_poes.dtype)],
        axis=1,
    )
    delta_poe = hazard_poes - poe_shifted
    return v * jnp.sum(mdr_matrix * delta_poe, axis=1)


def fragility_damage_distribution(
    u: jax.Array,
    F: jax.Array,
    x_grid: jax.Array,
    H: jax.Array,
) -> jax.Array:
    """Compute damage-state probability distributions from fragility curves."""
    u = u.astype(jnp.int32)
    n_assets = H.shape[0]
    n_events = H.shape[1]
    n_states = F.shape[1]
    n_points = F.shape[2]

    H_flat = jnp.reshape(H, (-1,))
    n_asset_events = H_flat.shape[0]
    H_eval = jnp.clip(H_flat, x_grid[0], x_grid[-1])

    idx = jnp.searchsorted(x_grid, H_eval, side="right") - 1
    idx = jnp.clip(idx, 0, n_points - 2).astype(jnp.int32)
    x_lower = x_grid[idx]
    x_upper = x_grid[idx + 1]
    alpha = (H_eval - x_lower) / (x_upper - x_lower + jnp.asarray(1e-8, dtype=x_grid.dtype))

    u_flat = jnp.reshape(jnp.broadcast_to(u[:, None], (n_assets, n_events)), (-1,))
    d_flat = jnp.reshape(jnp.broadcast_to(jnp.arange(n_states)[:, None], (n_states, n_asset_events)), (-1,))
    u_rep_d = jnp.reshape(jnp.broadcast_to(u_flat[None, :], (n_states, n_asset_events)), (-1,))
    idx_rep_d = jnp.reshape(jnp.broadcast_to(idx[None, :], (n_states, n_asset_events)), (-1,))
    alpha_rep_d = jnp.reshape(jnp.broadcast_to(alpha[None, :], (n_states, n_asset_events)), (-1,))

    f_flat = jnp.reshape(F, (-1,))
    flat_lower = u_rep_d * (n_states * n_points) + d_flat * n_points + idx_rep_d
    flat_upper = flat_lower + 1
    f_lo = f_flat[flat_lower]
    f_hi = f_flat[flat_upper]
    exceed_flat = (1.0 - alpha_rep_d) * f_lo + alpha_rep_d * f_hi
    exceed_flat = jnp.clip(exceed_flat, 0.0, 1.0)

    exceed_all = jnp.transpose(jnp.reshape(exceed_flat, (n_states, n_asset_events)))
    p_no_damage = 1.0 - exceed_all[:, 0:1]
    p_intermediate = exceed_all[:, :-1] - exceed_all[:, 1:]
    p_complete = exceed_all[:, -1:]
    damage_probs_flat = jnp.concatenate([p_no_damage, p_intermediate, p_complete], axis=1)
    damage_probs_flat = jnp.maximum(damage_probs_flat, jnp.zeros_like(damage_probs_flat))
    return jnp.reshape(damage_probs_flat, (n_assets, n_events, n_states + 1))


def consequence_loss(
    damage_probs: jax.Array,
    consequence_ratios: jax.Array,
    v: jax.Array,
    u: jax.Array,
) -> jax.Array:
    """Compute losses from damage-state probabilities and consequence ratios."""
    cr_per_asset = consequence_ratios[u.astype(jnp.int32)]
    mdr = jnp.sum(damage_probs * cr_per_asset[:, None, :], axis=2)
    return v[:, None] * mdr


def classical_damage(
    u: jax.Array,
    F: jax.Array,
    x_grid: jax.Array,
    hazard_poes: jax.Array,
    hazard_imls: jax.Array,
) -> jax.Array:
    """Compute expected damage-state fractions per asset from hazard curves."""
    n_assets = hazard_poes.shape[0]
    n_levels = hazard_imls.shape[0]
    damage_probs = fragility_damage_distribution(
        u,
        F,
        x_grid,
        jnp.broadcast_to(hazard_imls[None, :], (n_assets, n_levels)),
    )
    poe_shifted = jnp.concatenate(
        [hazard_poes[:, 1:], jnp.zeros((n_assets, 1), dtype=hazard_poes.dtype)],
        axis=1,
    )
    delta_poe = hazard_poes - poe_shifted
    return jnp.sum(damage_probs * delta_poe[:, :, None], axis=1)


def benefit_cost_ratio(
    aal_original: jax.Array,
    aal_retrofitted: jax.Array,
    retrofit_cost: jax.Array,
    interest_rate: float = 0.05,
    asset_life_expectancy: float = 50.0,
) -> jax.Array:
    """Compute benefit-cost ratio for retrofitting."""
    r = jnp.asarray(interest_rate, dtype=aal_original.dtype)
    t = jnp.asarray(asset_life_expectancy, dtype=aal_original.dtype)
    bpvf = (1.0 - jnp.power(1.0 + r, -t)) / (r + jnp.asarray(1e-10, dtype=aal_original.dtype))
    benefit = (aal_original - aal_retrofitted) * bpvf
    return benefit / (retrofit_cost + jnp.asarray(1e-10, dtype=aal_original.dtype))
