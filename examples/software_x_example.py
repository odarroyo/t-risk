#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 09:08:17 2026

@author: odarroyo
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tensor_engine import generate_synthetic_portfolio, TensorialRiskEngine

# 1. Generate synthetic portfolio (1,000 assets, 5,000 events)
v, u, C, x_grid, H, lambdas = generate_synthetic_portfolio(n_assets=10, n_events=5, n_typologies=5)

# 2. Initialize the Tensorial Risk Engine
engine = TensorialRiskEngine(v, u, C, x_grid, H, lambdas)

# 3. Compute Loss Matrix and Risk Metrics
J_matrix, metrics = engine.compute_loss_and_metrics()

print(f"Portfolio AAL: ${metrics['aal_portfolio'].numpy():,.2f}")
print(f"Max event loss: ${tf.reduce_max(metrics['loss_per_event']).numpy():,.2f}")

# 4. Plot J matrix
plt.figure(figsize=(12, 6))
plt.imshow(J_matrix.numpy(), aspect='auto', cmap='viridis')
colorbar = plt.colorbar(label='Loss')
colorbar.ax.tick_params(labelsize=14)
colorbar.set_label('Loss', fontsize=16)
plt.title('J Matrix (Asset x Event Losses)', fontsize=20)
plt.xlabel('Event Index', fontsize=16)
plt.ylabel('Asset Index', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

output_dir = Path('examples')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'J_matrix.png'
plt.savefig(output_path, dpi=300)
print(f"J matrix plot saved to: {output_path}")

plt.close()

if not output_path.exists():
    raise RuntimeError(f"Expected output file was not created: {output_path}")
