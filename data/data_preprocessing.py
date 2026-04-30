import numpy as np
import pandas as pd
from data.data_loading import real_data_loading
from models.sbts_multi import simulateSB_multi
from metrics.eval_functions import get_stats

# Load
data = pd.read_csv("data/sp500_top10_prices.csv")
numeric_cols = data.select_dtypes(include=[np.number]).columns
data = data[numeric_cols].ffill().bfill()

# Log-returns
log_returns = np.log(data / data.shift(1)).dropna().values

print(f"Log-returns shape: {log_returns.shape}")

# ========== USE THEIR PREPROCESSING ==========
print("Preprocessing (MinMax + Windows + Shuffle)...")

X_all, max_, min_ = real_data_loading(log_returns, seq_len=252)

print(f"X_all shape: {X_all.shape}")
print(f"X_all normalized [0,1]: min={X_all.min():.4f}, max={X_all.max():.4f}")

# Add initial point
X = np.zeros((X_all.shape[0], X_all.shape[1] + 1, X_all.shape[2]))
X[:, 1:, :] = X_all

N_windows = X.shape[0]
d = X.shape[2]

print(f"X shape: {X.shape}")

# ========== GENERATE ==========
print("\nGenerating...")

X_synth = simulateSB_multi(
    N=252,
    M=N_windows,
    d=d,
    X=X,
    N_pi=100,   # Élevé comme eux
    h=0.2,      # Comme eux
    deltati=1.0,
    M_simu=200
)

print(f"✓ Generated: {X_synth.shape}")

# ========== DENORMALIZE ==========
print("Denormalizing...")

from data.data_loading import invert_back

X_synth_denorm = invert_back(X_synth, max_, min_)

print(f"X_synth_denorm shape: {X_synth_denorm.shape}")

# ========== EVALUATE ==========
stats = get_stats(X_all, X_synth_denorm)
print("\nStatistics:")
print(stats)

# ========== SAVE ==========
np.save('X_synth_returns.npy', X_synth_denorm)
np.save('X_synth_prices.npy', np.exp(X_synth_denorm.cumsum(axis=1)) * 100)

print("\n✓ Done!")