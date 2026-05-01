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
deltati = 1/252

# ========== CREATE WINDOWS ==========
X_all = np.array([log_returns[i:i+window_size] for i in range(len(log_returns) - window_size)])
N_windows = X_all.shape[0]
N = window_size

# ========== RESCALING (Section 6 du papier) ==========
# R_tilde = R * sqrt(deltati) / sigma(R)  (per-feature)
# On calcule sigma feature-wise sur l'ensemble des log-returns
print("="*60)
print("RESCALING (Section 6)")
print("="*60)

sigma_features = log_returns.std(axis=0)  # shape (d,)
scale_factor = np.sqrt(deltati) / sigma_features  # shape (d,)

print(f"Sigma per feature (original): {sigma_features}")
print(f"Scale factor (sqrt(dt)/sigma): {scale_factor}")
print(f"Target std after rescaling: {np.sqrt(deltati):.6f}\n")

# Apply rescaling
X_all_rescaled = X_all * scale_factor[np.newaxis, np.newaxis, :]

print(f"Std before rescaling: {X_all.std():.6f}")
print(f"Std after rescaling:  {X_all_rescaled.std():.6f}\n")

# Build X (with leading zero) for SBTS input
X = np.zeros((N_windows, window_size+1, d))
X[:, 1:, :] = X_all_rescaled  # ← on passe les données RESCALÉES au modèle

print(f"Data shape: {X.shape}")
print(f"Window size: {window_size}\n")

# Helper to invert rescaling
def unrescale(X_synth_rescaled):
    """R_synth = R_synth_rescaled * sigma / sqrt(deltati)"""
    return X_synth_rescaled / scale_factor[np.newaxis, np.newaxis, :]

# ========== BANDWIDTH SWEEP (sur données rescalées) ==========
# Après rescaling, std ≈ sqrt(1/252) ≈ 0.063
# Donc h doit être réajusté — on teste un range plus large
print("="*60)
print("BANDWIDTH SWEEP on rescaled data (K=0)")
print("="*60)

h_candidates = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
sweep_results = []

for h_test in h_candidates:
    print(f"\nTesting h={h_test}...")
    X_test = simulateSB_multi(
        N=N,
        M=N_windows,
        d=d,
        X=X,
        N_pi=100,
        h=h_test,
        deltati=deltati,
        M_simu=300  # samples réduit pour le sweep, on veut juste tendance
    )
    # Ramène en échelle originale avant comparaison
    X_test_orig = unrescale(X_test)
    stats = get_stats(X_all, X_test_orig)
    
    std_real = stats['Std Data'].mean()
    std_synth = stats['Std SBTS'].mean()
    error = abs(std_synth - std_real)
    
    sweep_results.append({'h': h_test, 'std_real': std_real, 'std_synth': std_synth, 'error': error})
    print(f"  Std Real: {std_real:.6f}, Std SBTS: {std_synth:.6f}, Error: {error:.6f}")

sweep_df = pd.DataFrame(sweep_results)
print("\n" + sweep_df.to_string(index=False))

best_h_idx = sweep_df['error'].idxmin()
best_h_k0 = sweep_df.loc[best_h_idx, 'h']
print(f"\n✓ BEST h for K=0 = {best_h_k0}")

# ========== APPROACH 1: NON-MARKOVIAN (K=0) avec best h ==========
print("\n" + "="*60)
print(f"APPROACH 1: NON-MARKOVIAN (K=0, h={best_h_k0}) on RESCALED data")
print("="*60)

X_synth_k0_rescaled = simulateSB_multi(
    N=N, M=N_windows, d=d, X=X,
    N_pi=100, h=best_h_k0, deltati=deltati,
    M_simu=1000
)
X_synth_k0 = unrescale(X_synth_k0_rescaled)
print(f"✓ Generated: {X_synth_k0.shape}")

stats_k0 = get_stats(X_all, X_synth_k0)
print(f"  Mean Std Real: {stats_k0['Std Data'].mean():.6f}")
print(f"  Mean Std SBTS: {stats_k0['Std SBTS'].mean():.6f}")
error_k0 = abs(stats_k0['Std SBTS'].mean() - stats_k0['Std Data'].mean())
print(f"  Error: {error_k0:.6f}")

np.save('X_synth_returns_k0.npy', X_synth_k0)
np.save('X_synth_prices_k0.npy', np.exp(X_synth_k0.cumsum(axis=1)) * 100)

# ========== APPROACH 2: MARKOVIAN (K=1, sweep h + N_pi) ==========
print("\n" + "="*60)
print("APPROACH 2: MARKOVIAN (K=1) on RESCALED data")
print("="*60)

# Pour K=1 le bon h est typiquement plus petit (cf. Figure 2 du papier)
# On teste un range adapté
h_candidates_k1 = [0.005,0.01,0.05,0.1, 0.2]
n_pi_candidates = [20, 50, 100]

results_k1 = []
print("\nGrid search h × N_pi for K=1...\n")

for h_test in h_candidates_k1:
    for N_pi_test in n_pi_candidates:
        print(f"  h={h_test}, N_pi={N_pi_test}...")
        X_test = simulateSB_multi_mark(
            N=N, M=N_windows, d=d, K=1, X=X,
            N_pi=N_pi_test, h=h_test, deltati=deltati,
            M_simu=300
        )
        X_test_orig = unrescale(X_test)
        stats = get_stats(X_all, X_test_orig)
        std_real = stats['Std Data'].mean()
        std_synth = stats['Std SBTS'].mean()
        error = abs(std_synth - std_real)
        results_k1.append({'h': h_test, 'N_pi': N_pi_test,
                          'std_real': std_real, 'std_synth': std_synth, 'error': error})
        print(f"    Error: {error:.6f}")

results_k1_df = pd.DataFrame(results_k1)
print("\n" + results_k1_df.to_string(index=False))

best_idx = results_k1_df['error'].idxmin()
best_h_k1 = results_k1_df.loc[best_idx, 'h']
best_n_pi = int(results_k1_df.loc[best_idx, 'N_pi'])
print(f"\n✓ BEST K=1: h={best_h_k1}, N_pi={best_n_pi}")

# Final K=1 generation
print(f"\nGenerating full K=1 with h={best_h_k1}, N_pi={best_n_pi}...")
X_synth_k1_rescaled = simulateSB_multi_mark(
    N=N, M=N_windows, d=d, K=1, X=X,
    N_pi=best_n_pi, h=best_h_k1, deltati=deltati,
    M_simu=2000
)
X_synth_k1_best = unrescale(X_synth_k1_rescaled)

stats_k1_best = get_stats(X_all, X_synth_k1_best)
error_k1_best = abs(stats_k1_best['Std SBTS'].mean() - stats_k1_best['Std Data'].mean())
print(f"  Error: {error_k1_best:.6f}")

np.save('X_synth_returns_k1_best.npy', X_synth_k1_best)
np.save('X_synth_prices_k1_best.npy', np.exp(X_synth_k1_best.cumsum(axis=1)) * 100)

# ========== FINAL COMPARISON ==========
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)

comparison = pd.DataFrame({
    'Method': [f'K=0 (h={best_h_k0})', f'K=1 (h={best_h_k1}, N_pi={best_n_pi})'],
    'Std Real': [stats_k0['Std Data'].mean(), stats_k1_best['Std Data'].mean()],
    'Std SBTS': [stats_k0['Std SBTS'].mean(), stats_k1_best['Std SBTS'].mean()],
    'Error': [error_k0, error_k1_best]
})
print("\n" + comparison.to_string(index=False))

if error_k0 < error_k1_best:
    best_method, best_synth = 'K=0', X_synth_k0
else:
    best_method, best_synth = 'K=1', X_synth_k1_best

print(f"\n✓ BEST: {best_method}")
np.save('X_synth_returns.npy', best_synth)
np.save('X_synth_prices.npy', np.exp(best_synth.cumsum(axis=1)) * 100)
print("✓ Done!")