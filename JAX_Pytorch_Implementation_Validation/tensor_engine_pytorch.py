"""
PyTorch implementation of the Tensorial Risk Engine.

This module mirrors the public API of ``tensor_engine.py`` while leaving the
canonical TensorFlow implementation untouched. It is intended for backend
validation and hardware comparisons using the same NumPy inputs and comparable
tensor outputs.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


TensorDict = Dict[str, torch.Tensor]


def _as_float_tensor(x, *, device: Optional[Union[torch.device, str]] = None) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _as_index_tensor(x, *, device: Optional[Union[torch.device, str]] = None) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.long, device=device)


def generate_synthetic_portfolio(
    n_assets: int,
    n_events: int,
    n_typologies: int = 5,
    n_curve_points: int = 20,
    lambdas: Optional[np.ndarray] = None,
    lambda_distribution: str = "exponential",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic portfolio data matching the TensorFlow implementation.
    """
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
    v: torch.Tensor,
    u: torch.Tensor,
    C: torch.Tensor,
    x_grid: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """
    Compute deterministic loss for a single hazard scenario.
    """
    u = u.long()
    M = x_grid.shape[0]

    h_eval = torch.clamp(h, min=x_grid[0], max=x_grid[-1])
    valid_iml = h >= x_grid[0]

    idx = torch.searchsorted(x_grid, h_eval, right=True) - 1
    idx = torch.clamp(idx, 0, M - 2).long()

    x_lower = x_grid[idx]
    x_upper = x_grid[idx + 1]
    alpha = (h_eval - x_lower) / (x_upper - x_lower + 1e-8)

    c_flat = C.reshape(-1)
    flat_idx_lower = u * M + idx
    flat_idx_upper = u * M + idx + 1
    c_lower = c_flat[flat_idx_lower]
    c_upper = c_flat[flat_idx_upper]

    mdr = (1.0 - alpha) * c_lower + alpha * c_upper
    mdr = torch.where(valid_iml, mdr, torch.zeros_like(mdr))
    return torch.sum(v * mdr)


def probabilistic_loss_matrix(
    v: torch.Tensor,
    u: torch.Tensor,
    C: torch.Tensor,
    x_grid: torch.Tensor,
    H: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the event loss matrix for all assets and events.
    """
    u = u.long()
    n_assets = H.shape[0]
    n_events = H.shape[1]
    n_points = x_grid.shape[0]

    H_flat = H.reshape(-1)
    H_eval = torch.clamp(H_flat, min=x_grid[0], max=x_grid[-1])
    valid_iml = H_flat >= x_grid[0]

    idx = torch.searchsorted(x_grid, H_eval, right=True) - 1
    idx = torch.clamp(idx, 0, n_points - 2).long()

    x_lower = x_grid[idx]
    x_upper = x_grid[idx + 1]
    alpha = (H_eval - x_lower) / (x_upper - x_lower + 1e-8)

    u_flat = u.unsqueeze(1).expand(n_assets, n_events).reshape(-1)
    c_flat = C.reshape(-1)
    c_lower = c_flat[u_flat * n_points + idx]
    c_upper = c_flat[u_flat * n_points + idx + 1]

    mdr_flat = (1.0 - alpha) * c_lower + alpha * c_upper
    mdr_flat = torch.where(valid_iml, mdr_flat, torch.zeros_like(mdr_flat))
    mdr_matrix = mdr_flat.reshape(n_assets, n_events)

    return v.unsqueeze(1) * mdr_matrix


def compute_risk_metrics(
    J_matrix: torch.Tensor,
    lambdas: Optional[torch.Tensor] = None,
) -> TensorDict:
    """
    Compute rate-weighted risk metrics from the loss matrix.
    """
    if lambdas is None:
        q = J_matrix.shape[1]
        lambdas = torch.ones(q, dtype=J_matrix.dtype, device=J_matrix.device) / float(q)
    else:
        lambdas = lambdas.to(dtype=J_matrix.dtype, device=J_matrix.device)

    total_rate = torch.sum(lambdas)
    weights = lambdas / (total_rate + 1e-10)

    aal_per_asset = torch.sum(J_matrix * lambdas.unsqueeze(0), dim=1)
    aal_portfolio = torch.sum(aal_per_asset)
    mean_per_event_per_asset = aal_per_asset / (total_rate + 1e-10)
    deviations_sq = torch.square(J_matrix - mean_per_event_per_asset.unsqueeze(1))
    variance_per_asset = torch.sum(deviations_sq * weights.unsqueeze(0), dim=1)
    std_per_asset = torch.sqrt(variance_per_asset)
    loss_per_event = torch.sum(J_matrix, dim=0)

    return {
        "aal_per_asset": aal_per_asset,
        "aal_portfolio": aal_portfolio,
        "mean_per_event_per_asset": mean_per_event_per_asset,
        "variance_per_asset": variance_per_asset,
        "std_per_asset": std_per_asset,
        "loss_per_event": loss_per_event,
        "total_rate": total_rate,
    }


class TensorialRiskEngine:
    """
    PyTorch backend with the same high-level API as ``tensor_engine.TensorialRiskEngine``.
    """

    def __init__(
        self,
        v: np.ndarray,
        u: np.ndarray,
        C: np.ndarray,
        x_grid: np.ndarray,
        H: np.ndarray,
        lambdas: Optional[np.ndarray] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.v = _as_float_tensor(v, device=self.device).clone().detach().requires_grad_(True)
        self.u = _as_index_tensor(u, device=self.device)
        self.C = _as_float_tensor(C, device=self.device).clone().detach().requires_grad_(True)
        self.x_grid = _as_float_tensor(x_grid, device=self.device)
        self.H = _as_float_tensor(H, device=self.device).clone().detach().requires_grad_(True)

        self.n_assets = int(v.shape[0])
        self.n_events = int(H.shape[1])
        self.n_typologies = int(C.shape[0])

        if lambdas is None:
            lambdas = np.ones(self.n_events, dtype=np.float32) / self.n_events
        self.lambdas = _as_float_tensor(lambdas, device=self.device).clone().detach().requires_grad_(True)

    def compute_loss_and_metrics(self) -> Tuple[torch.Tensor, TensorDict]:
        J_matrix = probabilistic_loss_matrix(self.v, self.u, self.C, self.x_grid, self.H)
        metrics = compute_risk_metrics(J_matrix, self.lambdas)
        return J_matrix, metrics

    def gradient_wrt_vulnerability(self) -> Tuple[torch.Tensor, TensorDict]:
        _, metrics = self.compute_loss_and_metrics()
        grad_C = torch.autograd.grad(metrics["aal_portfolio"], self.C, retain_graph=False)[0]
        return grad_C, metrics

    def gradient_wrt_exposure(self) -> Tuple[torch.Tensor, TensorDict]:
        _, metrics = self.compute_loss_and_metrics()
        grad_v = torch.autograd.grad(metrics["aal_portfolio"], self.v, retain_graph=False)[0]
        return grad_v, metrics

    def gradient_wrt_hazard(self) -> Tuple[torch.Tensor, TensorDict]:
        _, metrics = self.compute_loss_and_metrics()
        grad_H = torch.autograd.grad(metrics["aal_portfolio"], self.H, retain_graph=False)[0]
        return grad_H, metrics

    def gradient_wrt_lambdas(self) -> Tuple[torch.Tensor, TensorDict]:
        _, metrics = self.compute_loss_and_metrics()
        grad_lambdas = torch.autograd.grad(metrics["aal_portfolio"], self.lambdas, retain_graph=False)[0]
        return grad_lambdas, metrics

    def full_gradient_analysis(self) -> Dict[str, Union[torch.Tensor, TensorDict]]:
        J_matrix, metrics = self.compute_loss_and_metrics()
        grad_H, grad_C, grad_v, grad_lambdas = torch.autograd.grad(
            metrics["aal_portfolio"],
            (self.H, self.C, self.v, self.lambdas),
            retain_graph=False,
        )
        return {
            "grad_hazard": grad_H,
            "grad_vulnerability": grad_C,
            "grad_exposure": grad_v,
            "grad_lambdas": grad_lambdas,
            "metrics": metrics,
            "loss_matrix": J_matrix,
        }


def classical_loss(
    v: torch.Tensor,
    u: torch.Tensor,
    C: torch.Tensor,
    x_grid: torch.Tensor,
    hazard_poes: torch.Tensor,
    hazard_imls: torch.Tensor,
) -> torch.Tensor:
    """
    Compute average annual loss per asset from hazard curves.
    """
    u = u.long()
    n_assets = v.shape[0]
    n_levels = hazard_imls.shape[0]
    n_points = x_grid.shape[0]

    imls_flat = hazard_imls.unsqueeze(0).expand(n_assets, n_levels).reshape(-1)
    imls_eval = torch.clamp(imls_flat, min=x_grid[0], max=x_grid[-1])
    valid_iml = imls_flat >= x_grid[0]

    idx = torch.searchsorted(x_grid, imls_eval, right=True) - 1
    idx = torch.clamp(idx, 0, n_points - 2).long()
    x_lower = x_grid[idx]
    x_upper = x_grid[idx + 1]
    alpha = (imls_eval - x_lower) / (x_upper - x_lower + 1e-8)

    u_flat = u.unsqueeze(1).expand(n_assets, n_levels).reshape(-1)
    c_flat = C.reshape(-1)
    c_lower = c_flat[u_flat * n_points + idx]
    c_upper = c_flat[u_flat * n_points + idx + 1]
    mdr_flat = (1.0 - alpha) * c_lower + alpha * c_upper
    mdr_flat = torch.where(valid_iml, mdr_flat, torch.zeros_like(mdr_flat))
    mdr_matrix = mdr_flat.reshape(n_assets, n_levels)

    poe_shifted = torch.cat(
        [hazard_poes[:, 1:], torch.zeros((n_assets, 1), dtype=hazard_poes.dtype, device=hazard_poes.device)],
        dim=1,
    )
    delta_poe = hazard_poes - poe_shifted
    return v * torch.sum(mdr_matrix * delta_poe, dim=1)


def fragility_damage_distribution(
    u: torch.Tensor,
    F: torch.Tensor,
    x_grid: torch.Tensor,
    H: torch.Tensor,
) -> torch.Tensor:
    """
    Compute damage-state probability distributions from fragility curves.
    """
    u = u.long()
    n_assets = H.shape[0]
    n_events = H.shape[1]
    n_states = F.shape[1]
    n_points = F.shape[2]

    H_flat = H.reshape(-1)
    n_asset_events = H_flat.shape[0]
    H_eval = torch.clamp(H_flat, min=x_grid[0], max=x_grid[-1])

    idx = torch.searchsorted(x_grid, H_eval, right=True) - 1
    idx = torch.clamp(idx, 0, n_points - 2).long()
    x_lower = x_grid[idx]
    x_upper = x_grid[idx + 1]
    alpha = (H_eval - x_lower) / (x_upper - x_lower + 1e-8)

    u_flat = u.unsqueeze(1).expand(n_assets, n_events).reshape(-1)
    d_flat = torch.arange(n_states, device=H.device).unsqueeze(1).expand(n_states, n_asset_events).reshape(-1)
    u_rep_d = u_flat.unsqueeze(0).expand(n_states, n_asset_events).reshape(-1)
    idx_rep_d = idx.unsqueeze(0).expand(n_states, n_asset_events).reshape(-1)
    alpha_rep_d = alpha.unsqueeze(0).expand(n_states, n_asset_events).reshape(-1)

    f_flat = F.reshape(-1)
    flat_lower = u_rep_d * (n_states * n_points) + d_flat * n_points + idx_rep_d
    flat_upper = u_rep_d * (n_states * n_points) + d_flat * n_points + idx_rep_d + 1
    f_lo = f_flat[flat_lower]
    f_hi = f_flat[flat_upper]
    exceed_flat = (1.0 - alpha_rep_d) * f_lo + alpha_rep_d * f_hi
    exceed_flat = torch.clamp(exceed_flat, 0.0, 1.0)

    exceed_all = exceed_flat.reshape(n_states, n_asset_events).transpose(0, 1)
    p_no_damage = 1.0 - exceed_all[:, 0:1]
    p_intermediate = exceed_all[:, :-1] - exceed_all[:, 1:]
    p_complete = exceed_all[:, -1:]
    damage_probs_flat = torch.cat([p_no_damage, p_intermediate, p_complete], dim=1)
    damage_probs_flat = torch.maximum(damage_probs_flat, torch.zeros_like(damage_probs_flat))
    return damage_probs_flat.reshape(n_assets, n_events, n_states + 1)


def consequence_loss(
    damage_probs: torch.Tensor,
    consequence_ratios: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """
    Compute losses from damage-state probabilities and consequence ratios.
    """
    cr_per_asset = consequence_ratios[u.long()]
    mdr = torch.sum(damage_probs * cr_per_asset.unsqueeze(1), dim=2)
    return v.unsqueeze(1) * mdr


def classical_damage(
    u: torch.Tensor,
    F: torch.Tensor,
    x_grid: torch.Tensor,
    hazard_poes: torch.Tensor,
    hazard_imls: torch.Tensor,
) -> torch.Tensor:
    """
    Compute expected damage-state fractions per asset from hazard curves.
    """
    u = u.long()
    n_assets = hazard_poes.shape[0]
    n_levels = hazard_imls.shape[0]
    n_states = F.shape[1]
    n_points = F.shape[2]

    imls_flat = hazard_imls.unsqueeze(0).expand(n_assets, n_levels).reshape(-1)
    n_asset_levels = imls_flat.shape[0]
    imls_eval = torch.clamp(imls_flat, min=x_grid[0], max=x_grid[-1])

    idx = torch.searchsorted(x_grid, imls_eval, right=True) - 1
    idx = torch.clamp(idx, 0, n_points - 2).long()
    x_lower = x_grid[idx]
    x_upper = x_grid[idx + 1]
    alpha = (imls_eval - x_lower) / (x_upper - x_lower + 1e-8)

    u_flat = u.unsqueeze(1).expand(n_assets, n_levels).reshape(-1)
    d_flat = torch.arange(n_states, device=hazard_poes.device).unsqueeze(1).expand(n_states, n_asset_levels).reshape(-1)
    u_rep_d = u_flat.unsqueeze(0).expand(n_states, n_asset_levels).reshape(-1)
    idx_rep_d = idx.unsqueeze(0).expand(n_states, n_asset_levels).reshape(-1)
    alpha_rep_d = alpha.unsqueeze(0).expand(n_states, n_asset_levels).reshape(-1)

    f_flat = F.reshape(-1)
    flat_lower = u_rep_d * (n_states * n_points) + d_flat * n_points + idx_rep_d
    flat_upper = u_rep_d * (n_states * n_points) + d_flat * n_points + idx_rep_d + 1
    f_lo = f_flat[flat_lower]
    f_hi = f_flat[flat_upper]
    exceed_flat = (1.0 - alpha_rep_d) * f_lo + alpha_rep_d * f_hi
    exceed_flat = torch.clamp(exceed_flat, 0.0, 1.0)

    exceed_all = exceed_flat.reshape(n_states, n_asset_levels).transpose(0, 1)
    p_no_damage = 1.0 - exceed_all[:, 0:1]
    p_intermediate = exceed_all[:, :-1] - exceed_all[:, 1:]
    p_complete = exceed_all[:, -1:]
    damage_probs_flat = torch.cat([p_no_damage, p_intermediate, p_complete], dim=1)
    damage_probs_flat = torch.maximum(damage_probs_flat, torch.zeros_like(damage_probs_flat))
    damage_probs = damage_probs_flat.reshape(n_assets, n_levels, n_states + 1)

    poe_shifted = torch.cat(
        [hazard_poes[:, 1:], torch.zeros((n_assets, 1), dtype=hazard_poes.dtype, device=hazard_poes.device)],
        dim=1,
    )
    delta_poe = hazard_poes - poe_shifted
    return torch.sum(damage_probs * delta_poe.unsqueeze(2), dim=1)


def benefit_cost_ratio(
    aal_original: torch.Tensor,
    aal_retrofitted: torch.Tensor,
    retrofit_cost: torch.Tensor,
    interest_rate: float = 0.05,
    asset_life_expectancy: float = 50.0,
) -> torch.Tensor:
    """
    Compute benefit-cost ratio for retrofitting.
    """
    r = torch.tensor(interest_rate, dtype=aal_original.dtype, device=aal_original.device)
    t = torch.tensor(asset_life_expectancy, dtype=aal_original.dtype, device=aal_original.device)
    bpvf = (1.0 - torch.pow(1.0 + r, -t)) / (r + 1e-10)
    benefit = (aal_original - aal_retrofitted) * bpvf
    return benefit / (retrofit_cost + 1e-10)
