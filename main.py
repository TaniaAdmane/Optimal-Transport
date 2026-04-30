import numpy as np
import pandas as pd
from models.sbts_multi_markov import simulateSB_multi_mark
from models.hyperparams_selection.markovian_optimal_multi import get_optimal_order_multi
from metrics.eval_functions import get_stats

# ========== LOAD & PREPARE ==========
data = pd.read_csv("data/sp500_top10_prices.csv")
numeric_cols = data.select_dtypes(include=[np.number]).columns
data = data[numeric_cols].ffill().bfill()

log_returns = np.log(data / data.shift(1)).dropna().values
X_all = np.array([log_returns[i:i+252] for i in range(len(log_returns) - 252)])
d = X_all.shape[2]
N_windows = X_all.shape[0]

X = np.zeros((N_windows, 253, d))
X[:, 1:, :] = X_all

# ========== SPLIT TRAIN/VAL ==========
split_idx = int(0.8 * N_windows)
X_train = X[:split_idx]
X_val = X[split_idx:]

N = 252
M = X_train.shape[0]

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# ========== GRID SEARCH ==========
print("\nSearching for optimal h and K...")

h_candidates = np.linspace(0.2, 0.8, 13)  # [0.2, 0.25, ..., 0.8]
K_candidates = [0, 1, 2, 3, 5]

mse_table = get_optimal_order_multi(
    N=N,
    M=M,
    d=d,
    K_markov=K_candidates,
    X=X_train,
    x_past=X_val[:, :-1],          # Past (251 points)
    x_target=X_val[:, -1],         # Target (last point)
    N_pi=3,
    h=h_candidates,
    deltati=1.0,
    itter=50
)

# ========== ANALYZE RESULTS ==========
df = pd.DataFrame(
    mse_table[1],
    index=[f"h={h:.3f}" for h in h_candidates],
    columns=[f"K={k}" for k in K_candidates]
)
print("\nMSE Table (h × K):")
print(df)

# Find best
best_idx = np.unravel_index(np.argmin(mse_table[1]), mse_table[1].shape)
best_h = h_candidates[best_idx[0]]
best_K = K_candidates[best_idx[1]]

print(f"\n✓ OPTIMAL: h={best_h:.4f}, K={best_K}")

# ========== GENERATE WITH OPTIMAL PARAMS ==========
print(f"\nGenerating with optimal params (h={best_h}, K={best_K})...")

X_synth = simulateSB_multi_mark(
    N=252,
    M=N_windows,
    d=d,
    K=best_K,           # ← Optimal K
    X=X,                # Use ALL data
    N_pi=5,
    h=best_h,           # ← Optimal h
    deltati=1.0,
    M_simu=2000
)

print(f"✓ Generated: {X_synth.shape}")

# ========== EVALUATE ==========
stats = get_stats(X[:, 1:, :], X_synth)
print("\nStatistics:")
print(stats)

# ========== SAVE ==========
np.save('X_synth_returns.npy', X_synth)
np.save('X_synth_prices.npy', np.exp(X_synth.cumsum(axis=1)) * 100)
print("\n✓ Done!")