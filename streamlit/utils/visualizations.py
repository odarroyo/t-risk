"""
Visualization Module for Tensor Risk Engine
============================================
Plotly-based interactive visualizations for risk analysis results.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict


def create_vulnerability_curves_plot(C: np.ndarray, x_grid: np.ndarray,
                                     H: Optional[np.ndarray] = None,
                                     typology_names: Optional[list] = None) -> go.Figure:
    """
    Create interactive vulnerability curves plot.
    
    Parameters
    ----------
    C : np.ndarray, shape (K, M)
        Vulnerability matrix
    x_grid : np.ndarray, shape (M,)
        Intensity grid
    H : np.ndarray, shape (N, Q), optional
        Hazard matrix to overlay intensity ranges
    typology_names : list, optional
        Names for each typology
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    K = C.shape[0]
    
    if typology_names is None:
        typology_names = [f'Typology {k}' for k in range(K)]
    
    fig = go.Figure()
    
    # Plot each vulnerability curve
    colors = px.colors.qualitative.Set2
    for k in range(K):
        fig.add_trace(go.Scatter(
            x=x_grid,
            y=C[k, :],
            mode='lines+markers',
            name=typology_names[k],
            line=dict(width=2.5, color=colors[k % len(colors)]),
            marker=dict(size=6),
            hovertemplate='Intensity: %{x:.3f}g<br>MDR: %{y:.3f}<extra></extra>'
        ))
    
    # Add shaded regions for hazard intensity ranges if provided
    if H is not None:
        h_min, h_max = np.min(H), np.max(H)
        h_p50 = np.median(H)
        
        # Add vertical shaded region for typical hazard range
        fig.add_vrect(
            x0=h_min, x1=h_p50,
            fillcolor="lightblue", opacity=0.2,
            layer="below", line_width=0,
            annotation_text="Frequent Events", annotation_position="top left"
        )
        fig.add_vrect(
            x0=h_p50, x1=h_max,
            fillcolor="lightyellow", opacity=0.2,
            layer="below", line_width=0,
            annotation_text="Rare Events", annotation_position="top right"
        )
    
    fig.update_layout(
        title='Vulnerability Curves by Building Typology',
        xaxis_title='Ground Motion Intensity (g)',
        yaxis_title='Mean Damage Ratio (MDR)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        height=500
    )
    
    fig.update_xaxes(gridcolor='lightgray')
    fig.update_yaxes(gridcolor='lightgray', range=[0, 1.05])
    
    return fig


def create_aal_vs_exposure_scatter(aal_per_asset: np.ndarray, v: np.ndarray,
                                   u: np.ndarray, typology_names: Optional[list] = None) -> go.Figure:
    """
    Create AAL vs Exposure scatter plot colored by typology.
    
    Parameters
    ----------
    aal_per_asset : np.ndarray, shape (N,)
        AAL per asset
    v : np.ndarray, shape (N,)
        Exposure values
    u : np.ndarray, shape (N,)
        Typology indices
    typology_names : list, optional
        Names for typologies
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    K = int(np.max(u)) + 1
    
    if typology_names is None:
        typology_names = [f'Type {k}' for k in range(K)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Exposure': v,
        'AAL': aal_per_asset,
        'Typology': [typology_names[int(u[i])] for i in range(len(u))],
        'AAL_Ratio': aal_per_asset / (v + 1e-6)
    })
    
    fig = px.scatter(
        df, x='Exposure', y='AAL', color='Typology',
        hover_data={'AAL_Ratio': ':.4f'},
        title='Average Annual Loss vs. Exposure by Building Type',
        labels={'Exposure': 'Exposure ($)', 'AAL': 'AAL ($)'},
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Add diagonal reference line (AAL = Exposure)
    max_val = max(v.max(), aal_per_asset.max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='AAL = Exposure',
        line=dict(dash='dash', color='gray', width=1),
        showlegend=True
    ))
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(height=500)
    
    return fig


def create_exposure_distribution(v: np.ndarray, u: np.ndarray,
                                 typology_names: Optional[list] = None) -> go.Figure:
    """
    Create exposure distribution histogram by typology.
    
    Parameters
    ----------
    v : np.ndarray, shape (N,)
        Exposure values
    u : np.ndarray, shape (N,)
        Typology indices
    typology_names : list, optional
        Names for typologies
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    K = int(np.max(u)) + 1
    
    if typology_names is None:
        typology_names = [f'Type {k}' for k in range(K)]
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    for k in range(K):
        mask = u == k
        fig.add_trace(go.Histogram(
            x=v[mask],
            name=typology_names[k],
            opacity=0.7,
            marker_color=colors[k % len(colors)],
            nbinsx=30
        ))
    
    fig.update_layout(
        title='Exposure Distribution by Building Typology',
        xaxis_title='Exposure ($)',
        yaxis_title='Number of Assets',
        template='plotly_white',
        barmode='overlay',
        height=500
    )
    
    return fig


def create_aal_distribution(aal_per_asset: np.ndarray, u: np.ndarray,
                            typology_names: Optional[list] = None) -> go.Figure:
    """
    Create AAL distribution box plot by typology.
    
    Parameters
    ----------
    aal_per_asset : np.ndarray, shape (N,)
        AAL per asset
    u : np.ndarray, shape (N,)
        Typology indices
    typology_names : list, optional
        Names for typologies
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    K = int(np.max(u)) + 1
    
    if typology_names is None:
        typology_names = [f'Type {k}' for k in range(K)]
    
    df = pd.DataFrame({
        'AAL': aal_per_asset,
        'Typology': [typology_names[int(u[i])] for i in range(len(u))]
    })
    
    fig = px.box(
        df, x='Typology', y='AAL',
        title='AAL Distribution by Building Typology',
        labels={'AAL': 'Average Annual Loss ($)'},
        template='plotly_white',
        color='Typology',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(showlegend=False, height=500)
    
    return fig


def create_event_loss_distribution(loss_per_event: np.ndarray) -> go.Figure:
    """
    Create event loss distribution histogram.
    
    Parameters
    ----------
    loss_per_event : np.ndarray, shape (Q,)
        Total loss per event
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    mean_loss = np.mean(loss_per_event)
    median_loss = np.median(loss_per_event)
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=loss_per_event,
        nbinsx=50,
        marker_color='steelblue',
        opacity=0.7,
        name='Event Losses'
    ))
    
    # Add vertical lines for mean and median
    fig.add_vline(x=mean_loss, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: ${mean_loss:,.0f}",
                  annotation_position="top right")
    fig.add_vline(x=median_loss, line_dash="dash", line_color="green",
                  annotation_text=f"Median: ${median_loss:,.0f}",
                  annotation_position="bottom right")
    
    fig.update_layout(
        title='Portfolio Loss Distribution Across Events',
        xaxis_title='Total Portfolio Loss per Event ($)',
        yaxis_title='Number of Events',
        template='plotly_white',
        showlegend=False,
        height=500
    )
    
    return fig


def create_vulnerability_gradient_heatmap(grad_C: np.ndarray, x_grid: np.ndarray,
                                          typology_names: Optional[list] = None) -> go.Figure:
    """
    Create vulnerability gradient heatmap.
    
    Parameters
    ----------
    grad_C : np.ndarray, shape (K, M)
        Gradient of AAL w.r.t. vulnerability curves
    x_grid : np.ndarray, shape (M,)
        Intensity grid
    typology_names : list, optional
        Names for typologies
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    K = grad_C.shape[0]
    
    if typology_names is None:
        typology_names = [f'Type {k}' for k in range(K)]
    
    fig = go.Figure(data=go.Heatmap(
        z=grad_C,
        x=x_grid,
        y=typology_names,
        colorscale='RdYlGn_r',
        colorbar=dict(title='∂AAL/∂C'),
        hovertemplate='Typology: %{y}<br>Intensity: %{x:.3f}g<br>Gradient: %{z:.2e}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Vulnerability Gradient Sensitivity Heatmap',
        xaxis_title='Ground Motion Intensity (g)',
        yaxis_title='Building Typology',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_exposure_gradient_chart(grad_v: np.ndarray, v: np.ndarray, u: np.ndarray,
                                   top_n: int = 100, typology_names: Optional[list] = None) -> go.Figure:
    """
    Create exposure gradient bar chart for top N assets.
    
    Parameters
    ----------
    grad_v : np.ndarray, shape (N,)
        Gradient of AAL w.r.t. exposure
    v : np.ndarray, shape (N,)
        Exposure values
    u : np.ndarray, shape (N,)
        Typology indices
    top_n : int
        Number of top assets to show
    typology_names : list, optional
        Names for typologies
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    # Get top N assets (limit to actual number of assets)
    actual_n = min(top_n, len(grad_v))
    top_indices = np.argsort(grad_v)[-actual_n:][::-1]
    
    K = int(np.max(u)) + 1
    if typology_names is None:
        typology_names = [f'Type {k}' for k in range(K)]
    
    df = pd.DataFrame({
        'Asset': [f'Asset {i}' for i in top_indices],
        'Gradient': grad_v[top_indices],
        'Exposure': v[top_indices],
        'Typology': [typology_names[int(u[i])] for i in top_indices]
    })
    
    fig = px.bar(
        df, x='Gradient', y='Asset', color='Typology',
        orientation='h',
        title=f'Top {actual_n} Assets by Exposure Sensitivity (∂AAL/∂v)',
        labels={'Gradient': '∂AAL/∂v ($/$ of exposure)'},
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data={'Exposure': ':$,.0f'}
    )
    
    fig.update_layout(height=max(500, actual_n * 8), showlegend=True)
    fig.update_yaxes(showticklabels=False)
    
    return fig


def create_hazard_sensitivity_vs_return_period(grad_H: np.ndarray, lambdas: np.ndarray,
                                                sample_size: int = 1000) -> go.Figure:
    """
    Create hazard sensitivity vs return period scatter plot.
    
    Parameters
    ----------
    grad_H : np.ndarray, shape (N, Q)
        Gradient of AAL w.r.t. hazard
    lambdas : np.ndarray, shape (Q,), optional
        Occurrence rates (if None, uses uniform rates)
    sample_size : int
        Number of points to sample for visualization
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    N, Q = grad_H.shape
    
    # Handle None lambdas (uniform rates)
    if lambdas is None:
        lambdas = np.ones(Q, dtype=np.float32) / Q
    
    # Calculate return periods
    return_periods = 1.0 / (lambdas + 1e-10)
    
    # Sample points for visualization (limit to actual size)
    total_points = N * Q
    actual_sample_size = min(sample_size, total_points)
    
    if total_points > actual_sample_size:
        # Random sampling
        indices = np.random.choice(total_points, actual_sample_size, replace=False)
        grad_flat = grad_H.flatten()[indices]
        rp_flat = np.repeat(return_periods, N)[indices]
        lambda_flat = np.repeat(lambdas, N)[indices]
    else:
        grad_flat = grad_H.flatten()
        rp_flat = np.repeat(return_periods, N)
        lambda_flat = np.repeat(lambdas, N)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rp_flat,
        y=np.abs(grad_flat),
        mode='markers',
        marker=dict(
            size=4,
            color=lambda_flat,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='λ (1/yr)'),
            opacity=0.6
        ),
        hovertemplate='Return Period: %{x:.0f} yr<br>|∂AAL/∂H|: %{y:.2e}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Hazard Sensitivity vs. Return Period',
        xaxis_title='Return Period (years)',
        yaxis_title='|∂AAL/∂H| ($/g)',
        xaxis_type='log',
        yaxis_type='log',
        template='plotly_white',
        height=500
    )
    
    return fig


def create_hazard_gradient_heatmap(grad_H: np.ndarray, max_assets: int = 50,
                                    max_events: int = 100) -> go.Figure:
    """
    Create hazard gradient heatmap (sampled if too large).
    
    Parameters
    ----------
    grad_H : np.ndarray, shape (N, Q)
        Gradient of AAL w.r.t. hazard
    max_assets : int
        Maximum number of assets to display
    max_events : int
        Maximum number of events to display
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    N, Q = grad_H.shape
    
    # Sample if too large
    if N > max_assets:
        asset_indices = np.linspace(0, N-1, max_assets, dtype=int)
    else:
        asset_indices = np.arange(N)
    
    if Q > max_events:
        event_indices = np.linspace(0, Q-1, max_events, dtype=int)
    else:
        event_indices = np.arange(Q)
    
    grad_sample = grad_H[np.ix_(asset_indices, event_indices)]
    
    fig = go.Figure(data=go.Heatmap(
        z=grad_sample,
        x=[f'E{i}' for i in event_indices],
        y=[f'A{i}' for i in asset_indices],
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title='∂AAL/∂H ($/g)'),
        hovertemplate='Asset: %{y}<br>Event: %{x}<br>Gradient: %{z:.2e}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Hazard Gradient Heatmap (Sampled: {len(asset_indices)} × {len(event_indices)})',
        xaxis_title='Event Index',
        yaxis_title='Asset Index',
        template='plotly_white',
        height=600
    )
    
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig


def create_event_contribution_plot(loss_per_event: np.ndarray, lambdas: np.ndarray) -> go.Figure:
    """
    Create event contribution to AAL scatter plot.
    
    Parameters
    ----------
    loss_per_event : np.ndarray, shape (Q,)
        Total loss per event
    lambdas : np.ndarray, shape (Q,), optional
        Occurrence rates (if None, uses uniform rates)
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    # Handle None lambdas (uniform rates)
    if lambdas is None:
        Q = len(loss_per_event)
        lambdas = np.ones(Q, dtype=np.float32) / Q
    
    # Calculate contribution to AAL
    contribution = loss_per_event * lambdas
    return_periods = 1.0 / (lambdas + 1e-10)
    
    # Sort by contribution
    sort_idx = np.argsort(contribution)[::-1]
    
    df = pd.DataFrame({
        'Event': range(len(loss_per_event)),
        'Return Period': return_periods,
        'Contribution': contribution,
        'Loss': loss_per_event,
        'Lambda': lambdas
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Return Period'],
        y=df['Contribution'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['Loss'],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title='Event Loss ($)'),
            opacity=0.7
        ),
        hovertemplate='RP: %{x:.0f} yr<br>Contribution: $%{y:,.0f}<br>λ: %{customdata[0]:.6f}<extra></extra>',
        customdata=np.column_stack([df['Lambda']])
    ))
    
    fig.update_layout(
        title='Event Contribution to AAL (λ × Loss)',
        xaxis_title='Return Period (years)',
        yaxis_title='Contribution to AAL ($)',
        xaxis_type='log',
        template='plotly_white',
        height=500
    )
    
    return fig


def create_scenario_loss_vs_rate_plot(loss_per_event: np.ndarray, lambdas: np.ndarray) -> go.Figure:
    """
    Create scenario loss vs occurrence rate log-log plot.
    
    Parameters
    ----------
    loss_per_event : np.ndarray, shape (Q,)
        Total loss per event
    lambdas : np.ndarray, shape (Q,), optional
        Occurrence rates (if None, uses uniform rates)
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    # Handle None lambdas (uniform rates)
    if lambdas is None:
        Q = len(loss_per_event)
        lambdas = np.ones(Q, dtype=np.float32) / Q
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lambdas,
        y=loss_per_event,
        mode='markers',
        marker=dict(
            size=8,
            color=lambdas,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='λ (1/yr)'),
            opacity=0.7
        ),
        hovertemplate='λ: %{x:.6f} /yr<br>Loss: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Scenario Loss vs. Occurrence Rate',
        xaxis_title='Occurrence Rate λ (events/year)',
        yaxis_title='Portfolio Loss ($)',
        xaxis_type='log',
        yaxis_type='log',
        template='plotly_white',
        height=500
    )
    
    return fig


def create_top_assets_table(aal_per_asset: np.ndarray, v: np.ndarray, u: np.ndarray,
                            top_n: int = 20, typology_names: Optional[list] = None) -> pd.DataFrame:
    """
    Create top assets table sorted by AAL.
    
    Parameters
    ----------
    aal_per_asset : np.ndarray, shape (N,)
        AAL per asset
    v : np.ndarray, shape (N,)
        Exposure values
    u : np.ndarray, shape (N,)
        Typology indices
    top_n : int
        Number of top assets to include
    typology_names : list, optional
        Names for typologies
    
    Returns
    -------
    pd.DataFrame
        Table with top assets
    """
    K = int(np.max(u)) + 1
    if typology_names is None:
        typology_names = [f'Type {k}' for k in range(K)]
    
    # Get top N assets (limit to actual number of assets)
    actual_n = min(top_n, len(aal_per_asset))
    top_indices = np.argsort(aal_per_asset)[-actual_n:][::-1]
    
    df = pd.DataFrame({
        'Rank': range(1, actual_n + 1),
        'Asset ID': top_indices,
        'AAL ($)': aal_per_asset[top_indices],
        'Exposure ($)': v[top_indices],
        'AAL/Exposure': aal_per_asset[top_indices] / v[top_indices],
        'Typology': [typology_names[int(u[i])] for i in top_indices]
    })
    
    return df


def create_portfolio_summary_metrics(metrics: Dict) -> Dict[str, str]:
    """
    Format portfolio summary metrics for display.
    
    Parameters
    ----------
    metrics : dict
        Metrics from engine
    
    Returns
    -------
    dict
        Formatted metrics for display
    """
    return {
        'Portfolio AAL': f"${metrics['aal_portfolio']:,.2f}",
        'Total Rate (Λ)': f"{metrics['total_rate']:.6f} events/year",
        'Mean Loss per Event': f"${np.mean(metrics['loss_per_event']):,.2f}",
        'Max Event Loss': f"${np.max(metrics['loss_per_event']):,.2f}",
        'Portfolio Std Dev': f"${np.sum(metrics['std_per_asset']):,.2f}"
    }
