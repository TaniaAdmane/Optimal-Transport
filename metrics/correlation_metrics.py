# metrics/correlation_metrics.py
import numpy as np
import pandas as pd


def autocorrelation_score(X_real, X_synth, max_lag=5):
    """
    Auto-correlation score per feature, averaged.

    Computes ACF up to max_lag for each (sample, feature) pair,
    then averages over samples to get per-feature ACF.
    The score is the mean absolute difference between real and synth ACF.

    Following SBTS paper Table 3 convention.

    Parameters
    ----------
    X_real, X_synth : np.ndarray
        Shape (n_samples, window_size, n_features)
    max_lag : int
        Maximum lag to consider.

    Returns
    -------
    score : float
        Mean absolute difference of ACF over all lags and features.
    per_feature : np.ndarray
        Shape (n_features,), per-feature score.
    """
    n_features = X_real.shape[2]
    acf_real = compute_acf(X_real, max_lag)   # (n_features, max_lag)
    acf_synth = compute_acf(X_synth, max_lag)

    diff = np.abs(acf_real - acf_synth)
    per_feature = diff.mean(axis=1)             # mean over lags
    score = per_feature.mean()                  # mean over features
    return score, per_feature, acf_real, acf_synth


def compute_acf(X, max_lag):
    """
    Compute autocorrelation function per feature, averaged over samples.

    For each sample s and feature f, compute ACF(lag) on the window.
    Then average ACF over samples.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, window_size, n_features)
    max_lag : int

    Returns
    -------
    acf : np.ndarray, shape (n_features, max_lag)
    """
    n_samples, T, n_features = X.shape
    acf = np.zeros((n_features, max_lag))

    for f in range(n_features):
        sample_acfs = []
        for s in range(n_samples):
            x = X[s, :, f]
            x_centered = x - x.mean()
            var = np.dot(x_centered, x_centered)
            if var < 1e-12:
                continue
            for lag in range(1, max_lag + 1):
                if T - lag <= 0:
                    continue
                cov = np.dot(x_centered[:-lag], x_centered[lag:])
                acf_lag = cov / var
                if len(sample_acfs) <= lag - 1:
                    sample_acfs.append([])
                sample_acfs[lag - 1].append(acf_lag)

        for lag in range(max_lag):
            if lag < len(sample_acfs) and len(sample_acfs[lag]) > 0:
                acf[f, lag] = np.mean(sample_acfs[lag])
    return acf


def cross_correlation_score(X_real, X_synth):
    """
    Cross-correlation score: distance between cross-correlation matrices.

    Computes the mean cross-correlation matrix over samples for both
    real and synthetic data, then returns the Frobenius distance
    (excluding diagonal, since diagonal is always 1).

    Parameters
    ----------
    X_real, X_synth : np.ndarray, shape (n_samples, window_size, n_features)

    Returns
    -------
    score : float
        Frobenius norm of off-diagonal difference, normalized by n_features.
    corr_real, corr_synth : np.ndarray, shape (n_features, n_features)
    """
    corr_real = compute_cross_corr(X_real)
    corr_synth = compute_cross_corr(X_synth)

    diff = corr_real - corr_synth
    # Off-diagonal only
    n = diff.shape[0]
    mask = ~np.eye(n, dtype=bool)
    score = np.sqrt((diff[mask] ** 2).mean())
    return score, corr_real, corr_synth


def compute_cross_corr(X):
    """
    Mean cross-correlation matrix over samples.

    For each sample, flatten time dimension (or compute per-sample cross-corr
    over the window) and average.
    """
    n_samples, T, n_features = X.shape
    corr_sum = np.zeros((n_features, n_features))
    count = 0
    for s in range(n_samples):
        sample = X[s]                  # shape (T, n_features)
        if T < 2:
            continue
        c = np.corrcoef(sample, rowvar=False)  # (n_features, n_features)
        if not np.any(np.isnan(c)):
            corr_sum += c
            count += 1
    return corr_sum / max(count, 1)


def abs_returns_acf_score(X_real, X_synth, max_lag=5):
    """
    Auto-correlation of absolute returns (volatility clustering).

    This metric captures whether the model reproduces volatility clustering,
    a key stylized fact of financial returns that ACF on raw returns misses.
    """
    score, per_feature, acf_real, acf_synth = autocorrelation_score(
        np.abs(X_real), np.abs(X_synth), max_lag=max_lag
    )
    return score, per_feature, acf_real, acf_synth


def correlation_report(X_real, X_synth, asset_names=None, max_lag=5):
    """
    Pretty-print all correlation metrics for SBTS vs FM comparison.
    """
    print("="*70)
    print("CORRELATION METRICS (lower = better fit)")
    print("="*70)

    # Auto-correlation on raw returns
    auto_score, auto_per_feat, _, _ = autocorrelation_score(X_real, X_synth, max_lag)
    print(f"\nAuto-correlation score (raw returns, lags 1-{max_lag}):")
    print(f"  Mean: {auto_score:.4f}")
    if asset_names:
        for name, val in zip(asset_names, auto_per_feat):
            print(f"  {name}: {val:.4f}")

    # Auto-correlation on absolute returns (vol clustering)
    abs_score, abs_per_feat, _, _ = abs_returns_acf_score(X_real, X_synth, max_lag)
    print(f"\nAbs-returns ACF score (volatility clustering, lags 1-{max_lag}):")
    print(f"  Mean: {abs_score:.4f}")
    if asset_names:
        for name, val in zip(asset_names, abs_per_feat):
            print(f"  {name}: {val:.4f}")

    # Cross-correlation
    cross_score, corr_real, corr_synth = cross_correlation_score(X_real, X_synth)
    print(f"\nCross-correlation score (off-diagonal Frobenius RMSE):")
    print(f"  Score: {cross_score:.4f}")

    return {
        'auto_corr': auto_score,
        'abs_corr': abs_score,
        'cross_corr': cross_score,
        'corr_real': corr_real,
        'corr_synth': corr_synth,
    }