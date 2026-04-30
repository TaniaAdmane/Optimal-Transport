import numpy as np
import pandas as pd
from models.sbts_multi import simulateSB_multi
from models.sbts_multi_markov import simulateSB_multi_mark
from metrics.eval_functions import get_stats

# ========== LOAD & PREPARE ==========
data = pd.read_csv("data/sp500_top10_prices.csv")
numeric_cols = data.select_dtypes(include=[np.number]).columns
data = data[numeric_cols].ffill().bfill()

log_returns = np.log(data / data.shift(1)).dropna().values

window_size = 10
d = log_returns.shape[1]

# ========== CREATE WINDOWS ==========
X_all = np.array([log_returns[i:i+window_size] for i in range(len(log_returns) - window_size)])
N_windows = X_all.shape[0]
N = window_size

X = np.zeros((N_windows, window_size+1, d))
X[:, 1:, :] = X_all

print(f"Data shape: {X.shape}")
print(f"Window size: {window_size}\n")

# ========== APPROACH 1: NON-MARKOVIAN (K=0) ==========
print("\n" + "="*60)
print("APPROACH 1: NON-MARKOVIAN (simulateSB_multi)")
print("="*60)

print("\nGenerating with K=0 (no Markov)...")

X_synth_k0 = simulateSB_multi(
    N=N,
    M=N_windows,
    d=d,
    X=X,
    N_pi=100,
    h=0.1,
    deltati=1/252,
    M_simu=1000
)

print(f"✓ Generated: {X_synth_k0.shape}")

stats_k0 = get_stats(X_all, X_synth_k0)
print(f"\nK=0 Statistics:")
print(f"  Mean Std Real: {stats_k0['Std Data'].mean():.6f}")
print(f"  Mean Std SBTS: {stats_k0['Std SBTS'].mean():.6f}")
error_k0 = abs(stats_k0['Std SBTS'].mean() - stats_k0['Std Data'].mean())
print(f"  Error: {error_k0:.6f}")

# Save K=0
np.save('X_synth_returns_k0.npy', X_synth_k0)
np.save('X_synth_prices_k0.npy', np.exp(X_synth_k0.cumsum(axis=1)) * 100)

# ========== APPROACH 2: MARKOVIAN (Test K=1 with different N_pi) ==========
print("\n" + "="*60)
print("APPROACH 2: MARKOVIAN (K=1 with varying N_pi)")
print("="*60)

print("\nTesting K=1 with different N_pi values...\n")

results_k1 = []

for N_pi_test in [10, 20, 50]:
    print(f"Testing K=1 with N_pi={N_pi_test}...")
    
    X_test = simulateSB_multi_mark(
        N=N,
        M=N_windows,
        d=d,
        K=1,
        X=X,
        N_pi=N_pi_test,
        h=0.1,
        deltati=1/252,
        M_simu=100
    )
    
    stats = get_stats(X_all, X_test)
    
    mean_std_real = stats['Std Data'].mean()
    mean_std_synth = stats['Std SBTS'].mean()
    error = abs(mean_std_synth - mean_std_real)
    
    results_k1.append({
        'N_pi': N_pi_test,
        'std_real': mean_std_real,
        'std_synth': mean_std_synth,
        'error': error
    })
    
    print(f"  Std Real: {mean_std_real:.6f}, Std SBTS: {mean_std_synth:.6f}, Error: {error:.6f}\n")

# Find best N_pi for K=1
results_k1_df = pd.DataFrame(results_k1)
best_n_pi_idx = results_k1_df['error'].idxmin()
best_n_pi = int(results_k1_df.loc[best_n_pi_idx, 'N_pi'])
best_error_k1 = results_k1_df['error'].min()

print(f"BEST N_pi for K=1 = {best_n_pi} (Error: {best_error_k1:.6f})\n")

# ========== GENERATE FULL WITH BEST K=1 CONFIG ==========
print(f"Generating full dataset with K=1, N_pi={best_n_pi}...")

X_synth_k1_best = simulateSB_multi_mark(
    N=N,
    M=N_windows,
    d=d,
    K=1,
    X=X,
    N_pi=best_n_pi,        # Optimal N_pi
    h=0.1,
    deltati=1/252,
    M_simu=1000
)

print(f"✓ Generated: {X_synth_k1_best.shape}")

stats_k1_best = get_stats(X_all, X_synth_k1_best)
print(f"\nK=1 (N_pi={best_n_pi}) Statistics:")
print(f"  Mean Std Real: {stats_k1_best['Std Data'].mean():.6f}")
print(f"  Mean Std SBTS: {stats_k1_best['Std SBTS'].mean():.6f}")
error_k1_best = abs(stats_k1_best['Std SBTS'].mean() - stats_k1_best['Std Data'].mean())
print(f"  Error: {error_k1_best:.6f}")

# Save K=1 best
np.save('X_synth_returns_k1_best.npy', X_synth_k1_best)
np.save('X_synth_prices_k1_best.npy', np.exp(X_synth_k1_best.cumsum(axis=1)) * 100)

# ========== FINAL COMPARISON ==========
print("\n" + "="*60)
print("FINAL COMPARISON: K=0 vs K=1 (optimized)")
print("="*60)

comparison = pd.DataFrame({
    'Method': [
        'Non-Markovian (K=0)',
        f'Markovian (K=1, N_pi={best_n_pi})'
    ],
    'Std Real': [
        stats_k0['Std Data'].mean(),
        stats_k1_best['Std Data'].mean()
    ],
    'Std SBTS': [
        stats_k0['Std SBTS'].mean(),
        stats_k1_best['Std SBTS'].mean()
    ],
    'Error': [error_k0, error_k1_best]
})

print("\n" + comparison.to_string(index=False))

# Determine best method
if error_k0 < error_k1_best:
    best_method = 'Non-Markovian (K=0)'
    best_synth = X_synth_k0
else:
    best_method = f'Markovian (K=1, N_pi={best_n_pi})'
    best_synth = X_synth_k1_best

print(f"\n✓ BEST METHOD: {best_method}")
print(f"  Error: {min(error_k0, error_k1_best):.6f}")

# Show improvement
improvement = ((error_k1_best - error_k0) / error_k0) * 100
if improvement < 0:
    print(f"  K=1 improved by {abs(improvement):.1f}% over K=0 ✓")
else:
    print(f"  K=0 remained {improvement:.1f}% better than K=1")

# ========== SAVE BEST ==========
print("\nSaving best results...")

np.save('X_synth_returns.npy', best_synth)
np.save('X_synth_prices.npy', np.exp(best_synth.cumsum(axis=1)) * 100)

print("\n✓ Done!")
print("Saved:")
print(f"  - X_synth_returns_k0.npy (Non-Markovian)")
print(f"  - X_synth_prices_k0.npy")
print(f"  - X_synth_returns_k1_best.npy (Markovian optimized)")
print(f"  - X_synth_prices_k1_best.npy")
print(f"  - X_synth_returns.npy (BEST: {best_method})")
print(f"  - X_synth_prices.npy")