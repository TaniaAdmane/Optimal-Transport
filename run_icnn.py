import argparse
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

from metrics.eval_functions import get_stats
from models.icnn import (
    ICNNMongeGenerator,
    LossWeights,
    TimeSeriesDistributionLoss,
    sample_source,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sp500_log_returns(data_path: str) -> Tuple[np.ndarray, pd.Index]:
    data = pd.read_csv(data_path)

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data = data[numeric_columns].ffill().bfill()

    log_returns = np.log(data / data.shift(1)).dropna().values.astype(np.float32)

    return log_returns, numeric_columns


def make_windows(log_returns: np.ndarray, window_size: int) -> np.ndarray:
    return np.array(
        [log_returns[i : i + window_size] for i in range(len(log_returns) - window_size)],
        dtype=np.float32,
    )


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, x_flat: np.ndarray, eps: float = 1e-6) -> "Standardizer":
        mean = x_flat.mean(axis=0)
        std = np.maximum(x_flat.std(axis=0), eps)

        return cls(mean, std)

    def transform(self, x_flat: np.ndarray) -> np.ndarray:
        return ((x_flat - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x_flat_standardized: np.ndarray) -> np.ndarray:
        return (x_flat_standardized * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> Dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def train_val_split(
    x: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))

    n_val = int(len(x) * val_ratio)

    return x[indices[n_val:]], x[indices[:n_val]]


def feature_correlation(x: np.ndarray) -> np.ndarray:
    batch_size, time_steps, n_features = x.shape
    flat = x.reshape(batch_size * time_steps, n_features)

    return np.corrcoef(flat.T)


def flat_covariance(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(len(x), -1)
    flat = flat - flat.mean(axis=0, keepdims=True)

    return flat.T @ flat / max(len(flat) - 1, 1)


def temporal_autocovariance_np(x: np.ndarray, lag: int) -> np.ndarray:
    x0 = x[:, :-lag, :]
    x1 = x[:, lag:, :]

    x0 = x0 - x0.mean(axis=(0, 1), keepdims=True)
    x1 = x1 - x1.mean(axis=(0, 1), keepdims=True)

    return (x0 * x1).mean(axis=(0, 1))


def distribution_metrics(real: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
    real_flat = real.reshape(len(real), -1)
    generated_flat = generated.reshape(len(generated), -1)

    out = {}

    out["mean_mse"] = float(np.mean((generated_flat.mean(axis=0) - real_flat.mean(axis=0)) ** 2))
    out["std_mse"] = float(np.mean((generated_flat.std(axis=0) - real_flat.std(axis=0)) ** 2))
    out["flat_cov_mse"] = float(np.mean((flat_covariance(generated) - flat_covariance(real)) ** 2))
    out["feature_corr_mse"] = float(np.nanmean((feature_correlation(generated) - feature_correlation(real)) ** 2))

    for lag in [1, 2, 3]:
        if lag < real.shape[1]:
            out[f"autocov_lag{lag}_mse"] = float(
                np.mean(
                    (
                        temporal_autocovariance_np(generated, lag)
                        - temporal_autocovariance_np(real, lag)
                    )
                    ** 2
                )
            )

    real_feature_flat = real.reshape(-1, real.shape[-1])
    generated_feature_flat = generated.reshape(-1, generated.shape[-1])

    for q in [0.01, 0.05, 0.5, 0.95, 0.99]:
        real_q = np.quantile(real_feature_flat, q, axis=0)
        generated_q = np.quantile(generated_feature_flat, q, axis=0)
        out[f"q{int(q * 100):02d}_mse"] = float(np.mean((generated_q - real_q) ** 2))

    real_q01 = np.quantile(real_feature_flat, 0.01, axis=0)
    real_q05 = np.quantile(real_feature_flat, 0.05, axis=0)
    real_q95 = np.quantile(real_feature_flat, 0.95, axis=0)
    real_q99 = np.quantile(real_feature_flat, 0.99, axis=0)

    out["coverage_below_q01"] = float(np.mean(generated_feature_flat < real_q01))
    out["coverage_below_q05"] = float(np.mean(generated_feature_flat < real_q05))
    out["coverage_above_q95"] = float(np.mean(generated_feature_flat > real_q95))
    out["coverage_above_q99"] = float(np.mean(generated_feature_flat > real_q99))

    real_abs_min = np.abs(real_feature_flat.min(axis=0))
    generated_abs_min = np.abs(generated_feature_flat.min(axis=0))
    real_abs_max = np.abs(real_feature_flat.max(axis=0))
    generated_abs_max = np.abs(generated_feature_flat.max(axis=0))

    out["min_abs_ratio"] = float(np.mean(generated_abs_min / np.maximum(real_abs_min, 1e-12)))
    out["max_abs_ratio"] = float(np.mean(generated_abs_max / np.maximum(real_abs_max, 1e-12)))

    return out


def scale_by_real_minmax(real: np.ndarray, generated: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    minimum = real.min(axis=(0, 1), keepdims=True)
    maximum = real.max(axis=(0, 1), keepdims=True)
    denominator = np.maximum(maximum - minimum, 1e-8)

    return (real - minimum) / denominator, (generated - minimum) / denominator


class ScoreDiscriminator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        hidden_dim = max(dim // 2, 1)

        self.rnn = nn.GRU(
            input_size=dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, state = self.rnn(x)
        return self.head(state[-1])


class ScorePredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        hidden_dim = max(input_dim // 2, 1)

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(x)
        return self.head(h)


def random_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    indices = torch.randperm(len(x), device=x.device)[:batch_size]
    return x[indices]


def discriminative_score(
    real: np.ndarray,
    generated: np.ndarray,
    iterations: int,
    device: torch.device,
    seed: int,
) -> float:
    set_seed(seed)

    n = min(len(real), len(generated))
    real = real[:n]
    generated = generated[:n]

    real_tensor = torch.tensor(real, dtype=torch.float32, device=device)
    generated_tensor = torch.tensor(generated, dtype=torch.float32, device=device)

    indices_real = torch.randperm(n, device=device)
    indices_generated = torch.randperm(n, device=device)

    split = int(0.8 * n)

    train_real = real_tensor[indices_real[:split]]
    test_real = real_tensor[indices_real[split:]]

    train_generated = generated_tensor[indices_generated[:split]]
    test_generated = generated_tensor[indices_generated[split:]]

    model = ScoreDiscriminator(real.shape[-1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    batch_size = min(128, split)

    model.train()

    for _ in range(iterations):
        x_real = random_batch(train_real, batch_size)
        x_generated = random_batch(train_generated, batch_size)

        logits_real = model(x_real)
        logits_generated = model(x_generated)

        loss = criterion(logits_real, torch.ones_like(logits_real))
        loss = loss + criterion(logits_generated, torch.zeros_like(logits_generated))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        pred_real = torch.sigmoid(model(test_real)).detach().cpu().numpy().reshape(-1)
        pred_generated = torch.sigmoid(model(test_generated)).detach().cpu().numpy().reshape(-1)

    predictions = np.concatenate([pred_real, pred_generated])
    labels = np.concatenate([np.ones_like(pred_real), np.zeros_like(pred_generated)])

    accuracy = accuracy_score(labels, predictions > 0.5)

    return float(abs(accuracy - 0.5))


def predictive_score(
    real: np.ndarray,
    generated: np.ndarray,
    iterations: int,
    device: torch.device,
    seed: int,
    col_pred: int = -1,
) -> float:
    set_seed(seed)

    if col_pred < 0:
        col_pred = real.shape[-1] - 1

    real_scaled, generated_scaled = scale_by_real_minmax(real, generated)

    real_tensor = torch.tensor(real_scaled, dtype=torch.float32, device=device)
    generated_tensor = torch.tensor(generated_scaled, dtype=torch.float32, device=device)

    feature_indices = [i for i in range(real.shape[-1]) if i != col_pred]

    model = ScorePredictor(len(feature_indices)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    batch_size = min(128, len(generated_tensor))

    model.train()

    for _ in range(iterations):
        batch = random_batch(generated_tensor, batch_size)

        x = batch[:, :-1, feature_indices]
        y = batch[:, 1:, col_pred].unsqueeze(-1)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()

    errors = []

    with torch.no_grad():
        for i in range(len(real_tensor)):
            x = real_tensor[i : i + 1, :-1, feature_indices]
            y = real_tensor[i : i + 1, 1:, col_pred].detach().cpu().numpy().reshape(-1)

            pred = model(x).detach().cpu().numpy().reshape(-1)
            errors.append(mean_absolute_error(y, pred))

    return float(np.mean(errors))


def neural_scores(
    real: np.ndarray,
    generated: np.ndarray,
    runs: int,
    iterations: int,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    disc = []
    pred = []

    for run in range(runs):
        current_seed = seed + 10_000 + run
        disc.append(discriminative_score(real, generated, iterations, device, current_seed))
        pred.append(predictive_score(real, generated, iterations, device, current_seed))

    return {
        "disc_mean": float(np.mean(disc)),
        "disc_std": float(np.std(disc)),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
    }


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_training_history(history, path: Path) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def load_existing_methods(real_windows: np.ndarray) -> Dict[str, np.ndarray]:
    paths = {
        "SBTS_K0": "X_synth_returns_k0.npy",
        "SBTS_K1": "X_synth_returns_k1_best.npy",
        "SBTS_best": "X_synth_returns.npy",
    }

    out = {}

    for name, path in paths.items():
        if os.path.exists(path):
            data = np.load(path)
            if data.shape[1:] == real_windows.shape[1:]:
                out[name] = data.astype(np.float32)

    return out


def train_icnn(args) -> None:
    set_seed(args.seed)

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    log_returns, feature_names = load_sp500_log_returns(args.data_path)
    real_windows = make_windows(log_returns, args.window_size)

    n_windows, time_steps, n_features = real_windows.shape
    flat_dim = time_steps * n_features

    real_flat = real_windows.reshape(n_windows, flat_dim)

    train_flat, val_flat = train_val_split(real_flat, val_ratio=args.val_ratio, seed=args.seed)

    standardizer = Standardizer.fit(train_flat)

    train_standardized = standardizer.transform(train_flat)
    val_standardized = standardizer.transform(val_flat)

    train_tensor = torch.tensor(train_standardized, dtype=torch.float32)
    val_tensor = torch.tensor(val_standardized, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    hidden_dims = tuple(int(hidden_dim) for hidden_dim in args.hidden_dims.split(","))

    model = ICNNMongeGenerator(
        input_dim=flat_dim,
        hidden_dims=hidden_dims,
        activation=args.activation,
        strong_convexity=args.strong_convexity,
        residual_scale=args.residual_scale,
        psd_rank=args.psd_rank,
        psd_init_scale=args.psd_init_scale,
    ).to(device)

    if args.init_gaussian_brenier:
        init_samples = torch.tensor(train_standardized, dtype=torch.float32, device=device)
        model.initialize_psd_from_covariance(samples=init_samples, shrinkage=args.linear_init_shrinkage)

    weights = LossWeights(
        sinkhorn=args.w_sinkhorn,
        sliced_wasserstein=args.w_swd,
        mmd=args.w_mmd,
        mean=args.w_mean,
        std=args.w_std,
        covariance=args.w_covariance,
        feature_correlation=args.w_feature_corr,
        autocovariance=args.w_autocov,
        quantile=args.w_quantile,
        tail=args.w_tail,
        feature_tail=args.w_feature_tail,
        output_norm=args.w_output_norm,
    )

    criterion = TimeSeriesDistributionLoss(
        time_steps=time_steps,
        n_features=n_features,
        weights=weights,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
        n_projections=args.n_projections,
        autocov_lags=tuple(args.autocov_lags),
        tail_alpha=args.tail_alpha,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )

    n_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    config = vars(args).copy()
    config["flat_dim"] = flat_dim
    config["n_features"] = n_features
    config["time_steps"] = time_steps
    config["feature_names"] = list(feature_names)
    config["n_train"] = len(train_standardized)
    config["n_val"] = len(val_standardized)
    config["n_parameters"] = n_parameters
    config["standardizer"] = standardizer.to_dict()

    save_json(config, output_dir / "icnn_config.json")

    print(
        f"device={device} "
        f"data={real_windows.shape} "
        f"dim={flat_dim} "
        f"params={n_parameters} "
        f"source={args.source}"
    )

    best_val_loss = float("inf")
    best_path = output_dir / "icnn_best.pt"
    last_path = output_dir / "icnn_last.pt"

    patience_count = 0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_components = {}
        n_batches = 0

        for (real_batch,) in train_loader:
            real_batch = real_batch.to(device)
            current_batch_size = real_batch.shape[0]

            z = sample_source(
                current_batch_size,
                flat_dim,
                device,
                source=args.source,
                df=args.df,
            )

            z.requires_grad_(True)
            generated_batch = model(z)

            loss, components = criterion(generated_batch, real_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            for key, value in components.items():
                epoch_components[key] = epoch_components.get(key, 0.0) + float(value.detach().cpu())

            n_batches += 1

        scheduler.step()

        train_log = {
            f"train_{key}": value / max(n_batches, 1)
            for key, value in epoch_components.items()
        }

        model.eval()

        val_batch_size = min(args.eval_batch_size, len(val_tensor))

        val_indices = torch.randperm(len(val_tensor))[:val_batch_size]
        real_val = val_tensor[val_indices].to(device)

        z_val = sample_source(
            val_batch_size,
            flat_dim,
            device,
            source=args.source,
            df=args.df,
        )

        z_val.requires_grad_(True)

        with torch.enable_grad():
            generated_val = model.transport(z_val, create_graph=False)

        val_loss, val_components = criterion(generated_val, real_val)

        val_log = {
            f"val_{key}": float(value.detach().cpu())
            for key, value in val_components.items()
        }

        current_val_loss = float(val_loss.detach().cpu())

        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            **train_log,
            **val_log,
        }

        history.append(row)

        if epoch % args.print_every == 0 or epoch == 1:
            elapsed = time.time() - start_time
            seconds_per_epoch = elapsed / epoch
            eta_seconds = seconds_per_epoch * (args.epochs - epoch)
            eta_minutes = int(eta_seconds // 60)

            print(
                f"{epoch}/{args.epochs} "
                f"val={current_val_loss:.4f} "
                f"sink={val_log['val_sinkhorn']:.4f} "
                f"swd={val_log['val_sliced_wasserstein']:.4f} "
                f"corr={val_log['val_feature_correlation']:.5f} "
                f"tail={val_log['val_tail']:.5f} "
                f"q={val_log['val_quantile']:.5f} "
                f"eta={eta_minutes}m"
            )

        if current_val_loss < best_val_loss - args.min_delta:
            best_val_loss = current_val_loss
            patience_count = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "config": config,
                },
                best_path,
            )
        else:
            patience_count += 1

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "config": config,
                },
                last_path,
            )

            save_training_history(history, output_dir / "icnn_training_history.csv")

        if args.early_stopping and patience_count >= args.patience:
            print(f"early_stop epoch={epoch} best_val={best_val_loss:.6f}")
            break

    save_training_history(history, output_dir / "icnn_training_history.csv")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generated_standardized = model.generate(
        n_samples=args.n_synth,
        device=device,
        batch_size=args.generate_batch_size,
        source=args.source,
        df=args.df,
    ).numpy()

    generated_flat = standardizer.inverse_transform(generated_standardized)
    synthetic_returns = generated_flat.reshape(args.n_synth, time_steps, n_features)
    synthetic_prices = np.exp(synthetic_returns.cumsum(axis=1)) * 100.0

    returns_path = output_dir / "X_synth_returns_icnn.npy"
    prices_path = output_dir / "X_synth_prices_icnn.npy"

    np.save(returns_path, synthetic_returns)
    np.save(prices_path, synthetic_prices)

    if args.save_to_root:
        np.save("X_synth_returns_icnn.npy", synthetic_returns)
        np.save("X_synth_prices_icnn.npy", synthetic_prices)

    stats = get_stats(real_windows, synthetic_returns, col=list(feature_names))
    stats.to_csv(output_dir / "icnn_stats.csv")

    evaluation = {
        "ICNN": distribution_metrics(real_windows, synthetic_returns),
    }

    for method_name, method_data in load_existing_methods(real_windows).items():
        evaluation[method_name] = distribution_metrics(real_windows, method_data)

    if args.eval_neural:
        evaluation["ICNN_neural"] = neural_scores(
            real_windows,
            synthetic_returns,
            runs=args.neural_runs,
            iterations=args.neural_iters,
            device=device,
            seed=args.seed,
        )

        for method_name, method_data in load_existing_methods(real_windows).items():
            evaluation[f"{method_name}_neural"] = neural_scores(
                real_windows,
                method_data,
                runs=args.neural_runs,
                iterations=args.neural_iters,
                device=device,
                seed=args.seed + 100,
            )

    save_json(evaluation, output_dir / "icnn_evaluation.json")

    summary = {
        "out_dir": str(output_dir),
        "best_val_loss": best_val_loss,
        "returns": str(returns_path),
        "prices": str(prices_path),
        "evaluation": evaluation,
    }

    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/sp500_top10_prices.csv")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    parser.add_argument("--hidden_dims", type=str, default="512,512,512,256")
    parser.add_argument("--activation", type=str, default="softplus")
    parser.add_argument("--strong_convexity", type=float, default=0.0)
    parser.add_argument("--residual_scale", type=float, default=0.05)

    parser.add_argument("--psd_rank", type=int, default=90)
    parser.add_argument("--psd_init_scale", type=float, default=1e-3)
    parser.add_argument("--linear_init_shrinkage", type=float, default=0.02)
    parser.add_argument("--init_gaussian_brenier", action="store_true", default=True)
    parser.add_argument("--no_init_gaussian_brenier", dest="init_gaussian_brenier", action="store_false")

    parser.add_argument("--source", type=str, default="student_t", choices=["gaussian", "student_t"])
    parser.add_argument("--df", type=float, default=5.0)

    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.5)
    parser.add_argument("--sinkhorn_iters", type=int, default=80)
    parser.add_argument("--n_projections", type=int, default=512)
    parser.add_argument("--autocov_lags", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--tail_alpha", type=float, default=0.05)

    parser.add_argument("--w_sinkhorn", type=float, default=0.5)
    parser.add_argument("--w_swd", type=float, default=1.0)
    parser.add_argument("--w_mmd", type=float, default=0.2)
    parser.add_argument("--w_mean", type=float, default=5.0)
    parser.add_argument("--w_std", type=float, default=5.0)
    parser.add_argument("--w_covariance", type=float, default=20.0)
    parser.add_argument("--w_feature_corr", type=float, default=30.0)
    parser.add_argument("--w_autocov", type=float, default=10.0)
    parser.add_argument("--w_quantile", type=float, default=3.0)
    parser.add_argument("--w_tail", type=float, default=10.0)
    parser.add_argument("--w_feature_tail", type=float, default=5.0)
    parser.add_argument("--w_output_norm", type=float, default=1e-4)

    parser.add_argument("--n_synth", type=int, default=1000)
    parser.add_argument("--generate_batch_size", type=int, default=512)

    parser.add_argument("--eval_neural", action="store_true")
    parser.add_argument("--neural_runs", type=int, default=3)
    parser.add_argument("--neural_iters", type=int, default=500)

    parser.add_argument("--out_dir", type=str, default="outputs_icnn_final")
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=600)
    parser.add_argument("--min_delta", type=float, default=1e-5)

    parser.add_argument("--save_to_root", action="store_true", default=True)
    parser.add_argument("--no_save_to_root", dest="save_to_root", action="store_false")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    train_icnn(parse_args())