from models.flow_matching import FlowMatchingOT  # nouvelle classe
from metrics.eval_functions import get_stats, get_scores, plot_sample_multi
import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))


def main():
    print("="*70)
    print("FLOW MATCHING FOR FINANCIAL TIME SERIES")
    print("(SBTS-paper-style rescaling for fair comparison)")
    print("="*70 + "\n")

    # ========== LOAD DATA ==========
    print("Loading data...")
    data = pd.read_csv("data/sp500_top10_prices.csv")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data = data[numeric_cols].ffill().bfill()
    asset_names = numeric_cols.tolist()

    log_returns = np.log(data / data.shift(1)).dropna().values

    window_size = 10
    deltati = 1 / 252  # daily data
    n_assets = log_returns.shape[1]

    X_all = np.array([
        log_returns[i:i+window_size]
        for i in range(len(log_returns) - window_size)
    ])
    print(f"✓ Data: {X_all.shape}\n")

    # ========== RESCALING (SBTS paper, Section 6) ==========
    # R_tilde = R * sqrt(deltati) / sigma(R)  per-feature
    # Ensures variance of scaled increments matches sqrt(deltati),
    # which is the "natural scale" both for SBTS SDE and for FM
    # (where prior is N(0, I) and we want data on a comparable scale).
    print("="*70)
    print("RESCALING (per-feature, sqrt(dt)/sigma)")
    print("="*70)

    sigma_features = log_returns.std(axis=0)              # shape (d,)
    scale_factor = np.sqrt(deltati) / sigma_features       # shape (d,)

    print(f"Sigma per feature (original): {sigma_features}")
    print(f"Scale factor (sqrt(dt)/sigma): {scale_factor}")
    print(f"Target std after rescaling: {np.sqrt(deltati):.6f}")

    X_all_rescaled = X_all * scale_factor[np.newaxis, np.newaxis, :]

    print(f"\nStd before rescaling: {X_all.std():.6f}")
    print(f"Std after rescaling:  {X_all_rescaled.std():.6f}\n")

    def unrescale(X_synth_rescaled):
        """Inverse: R = R_rescaled * sigma / sqrt(dt)"""
        return X_synth_rescaled / scale_factor[np.newaxis, np.newaxis, :]

    # ========== TRAIN FLOW MATCHING ==========
    fm = FlowMatchingOT(
        window_size=window_size,
        n_assets=n_assets,
        sigma_min=1e-4,    # standard from Lipman et al.
        hidden_dim=256,    # reduced for input_dim=90
        lr=1e-3,
    )

    print("="*70)
    print("TRAINING")
    print("="*70)
    losses = fm.train(X_all_rescaled, epochs=200, batch_size=128)
    print()

    # ========== GENERATE ==========
    X_synth_rescaled = fm.generate(n_samples=1000)

    # ========== UNRESCALE ==========
    print("Unrescaling to original log-return space...")
    X_synth = unrescale(X_synth_rescaled)
    print(f"✓ Std synthetic (original scale): {X_synth.std():.6f}")
    print(f"  Std real (original scale):     {X_all.std():.6f}\n")

    # ========== SAVE ==========
    print("Saving...")
    np.save('X_synth_fm_returns.npy', X_synth)
    X_synth_prices = np.exp(X_synth.cumsum(axis=1)) * 100
    np.save('X_synth_fm_prices.npy', X_synth_prices)
    print("✓ Saved: X_synth_fm_returns.npy, X_synth_fm_prices.npy\n")

    # ========== STATISTICAL EVALUATION ==========
    print("="*70)
    print("STATISTICAL EVALUATION (in original log-return space)")
    print("="*70 + "\n")

    stats = get_stats(X_all, X_synth, col=asset_names)
    print("Statistics (Real vs Flow Matching):")
    print(stats)
    print()

    # ========== DISCRIMINATIVE & PREDICTIVE SCORES ==========
    print("="*70)
    print("DISCRIMINATIVE & PREDICTIVE SCORES")
    print("="*70 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        print("Computing scores (this may take several minutes)...\n")
        disc_scores, pred_scores = get_scores(
            X_all,
            X_synth,
            col_pred=None,
            itt=1000,
            n_temp=10,        # 10 runs as in the SBTS paper (Section B.1)
            min_max=False,
            device=device
        )

        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"\nDiscriminative Score (lower = better):")
        print(f"  Mean ± Std: {disc_scores.mean():.4f} ± {disc_scores.std():.4f}")

        print(f"\nPredictive Score (lower = better):")
        print(f"  Mean ± Std: {pred_scores.mean():.4f} ± {pred_scores.std():.4f}")
        print()

    except Exception as e:
        print(f"⚠️  Scores computation skipped: {e}\n")

    # ========== VISUALIZATION ==========
    print("="*70)
    print("VISUALIZATION")
    print("="*70 + "\n")

    try:
        plot_sample_multi(X_all, X_synth, col=asset_names, x0=0)
        print("✓ Plot displayed\n")
    except Exception as e:
        print(f"⚠️  Visualization skipped: {e}\n")

    print("="*70)
    print("✓ DONE!")
    print("="*70)


if __name__ == "__main__":
    main()