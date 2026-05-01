import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus_inverse(x: float) -> float:
    return math.log(math.expm1(x))


def pairwise_squared_distances(x: torch.Tensor, y: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    cost = torch.cdist(x, y, p=2).pow(2)
    if normalize:
        cost = cost / x.shape[1]
    return cost


def sample_source(
    batch_size: int,
    dim: int,
    device: torch.device,
    source: str = "student_t",
    df: float = 5.0,
) -> torch.Tensor:
    if source == "gaussian":
        return torch.randn(batch_size, dim, device=device)

    if source == "student_t":
        distribution = torch.distributions.StudentT(df=df)
        z = distribution.sample((batch_size, dim)).to(device)
        if df > 2:
            z = z * math.sqrt((df - 2.0) / df)
        return z

    raise ValueError("source must be either 'gaussian' or 'student_t'")


class PositiveLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.raw_weight, mean=softplus_inverse(0.02), std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @property
    def weight(self) -> torch.Tensor:
        return F.softplus(self.raw_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ICNNPotential(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (512, 512, 512, 256),
        activation: str = "softplus",
        strong_convexity: float = 0.0,
        residual_scale: float = 0.05,
        psd_rank: int = 0,
        psd_init_scale: float = 1e-3,
    ):
        super().__init__()

        if activation not in {"softplus", "elu_softplus"}:
            raise ValueError("activation must be either 'softplus' or 'elu_softplus'")

        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.activation_name = activation
        self.strong_convexity = strong_convexity
        self.residual_scale = residual_scale
        self.psd_rank = int(psd_rank)

        if self.psd_rank > 0:
            self.psd_factor = nn.Parameter(psd_init_scale * torch.randn(input_dim, self.psd_rank))
        else:
            self.register_parameter("psd_factor", None)

        self.x_layers = nn.ModuleList()
        self.z_layers = nn.ModuleList()

        previous_hidden_dim = None

        for hidden_dim in self.hidden_dims:
            self.x_layers.append(nn.Linear(input_dim, hidden_dim, bias=True))
            if previous_hidden_dim is not None:
                self.z_layers.append(PositiveLinear(previous_hidden_dim, hidden_dim, bias=False))
            previous_hidden_dim = hidden_dim

        self.final_z = PositiveLinear(previous_hidden_dim, 1, bias=False)
        self.final_x = nn.Linear(input_dim, 1, bias=True)
        self.raw_diag_quad = nn.Parameter(torch.full((input_dim,), softplus_inverse(1e-3)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.x_layers:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)

        nn.init.zeros_(self.final_x.weight)
        nn.init.zeros_(self.final_x.bias)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "softplus":
            return F.softplus(x, beta=1.0)
        return F.softplus(F.elu(x) + 1.0)

    @property
    def diag_quad(self) -> torch.Tensor:
        return F.softplus(self.raw_diag_quad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = None

        for layer_index, x_layer in enumerate(self.x_layers):
            x_part = x_layer(x)

            if layer_index == 0:
                pre_activation = x_part
            else:
                z_part = self.z_layers[layer_index - 1](z)
                pre_activation = z_part + x_part

            z = self.activation(pre_activation)

        convex_residual = self.final_z(z).squeeze(-1)
        linear_term = self.final_x(x).squeeze(-1)
        base_quad = 0.5 * self.strong_convexity * x.pow(2).sum(dim=1)
        diag_quad = 0.5 * (self.diag_quad * x.pow(2)).sum(dim=1)

        if self.psd_factor is not None:
            psd_quad = 0.5 * (x @ self.psd_factor).pow(2).sum(dim=1)
        else:
            psd_quad = 0.0

        return base_quad + diag_quad + psd_quad + linear_term + self.residual_scale * convex_residual

    @torch.no_grad()
    def initialize_psd_from_covariance(
        self,
        samples: torch.Tensor,
        shrinkage: float = 0.02,
        eps: float = 1e-5,
    ) -> None:
        if self.psd_factor is None:
            return

        x = samples.detach()
        x = x - x.mean(dim=0, keepdim=True)

        covariance = x.T @ x / max(x.shape[0] - 1, 1)
        dim = covariance.shape[0]

        identity = torch.eye(dim, device=covariance.device, dtype=covariance.dtype)
        covariance = (1.0 - shrinkage) * covariance + shrinkage * identity

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)

        eigenvalues = eigenvalues[order].clamp_min(eps)
        eigenvectors = eigenvectors[:, order]

        rank = min(self.psd_rank, dim)
        factor = eigenvectors[:, :rank] * eigenvalues[:rank].pow(0.25).unsqueeze(0)

        new_factor = torch.zeros(
            dim,
            self.psd_rank,
            device=covariance.device,
            dtype=self.psd_factor.dtype,
        )

        new_factor[:, :rank] = factor.to(self.psd_factor.dtype)
        self.psd_factor.copy_(new_factor)


class ICNNMongeGenerator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (512, 512, 512, 256),
        activation: str = "softplus",
        strong_convexity: float = 0.0,
        residual_scale: float = 0.05,
        psd_rank: int = 0,
        psd_init_scale: float = 1e-3,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.potential = ICNNPotential(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            strong_convexity=strong_convexity,
            residual_scale=residual_scale,
            psd_rank=psd_rank,
            psd_init_scale=psd_init_scale,
        )

    def transport(self, z: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        if not z.requires_grad:
            z = z.requires_grad_(True)

        phi = self.potential(z)

        transported = torch.autograd.grad(
            outputs=phi.sum(),
            inputs=z,
            create_graph=create_graph,
            retain_graph=create_graph,
            only_inputs=True,
        )[0]

        return transported

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.transport(z, create_graph=self.training)

    def initialize_psd_from_covariance(
        self,
        samples: torch.Tensor,
        shrinkage: float = 0.02,
    ) -> None:
        self.potential.initialize_psd_from_covariance(samples=samples, shrinkage=shrinkage)

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        device: torch.device,
        batch_size: int = 512,
        source: str = "student_t",
        df: float = 5.0,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()

        generated_batches = []
        produced = 0

        while produced < n_samples:
            current_batch_size = min(batch_size, n_samples - produced)
            z = sample_source(current_batch_size, self.input_dim, device, source=source, df=df)
            z.requires_grad_(True)

            with torch.enable_grad():
                x = self.transport(z, create_graph=False)

            generated_batches.append(x.detach().cpu())
            produced += current_batch_size

        if was_training:
            self.train()

        return torch.cat(generated_batches, dim=0)


def sliced_wasserstein_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 256,
    p: int = 2,
) -> torch.Tensor:
    if x.shape[0] != y.shape[0]:
        n = min(x.shape[0], y.shape[0])
        x = x[:n]
        y = y[:n]

    dim = x.shape[1]

    directions = torch.randn(dim, n_projections, device=x.device)
    directions = directions / (directions.norm(dim=0, keepdim=True) + 1e-12)

    x_projection = x @ directions
    y_projection = y @ directions

    x_sorted, _ = torch.sort(x_projection, dim=0)
    y_sorted, _ = torch.sort(y_projection, dim=0)

    if p == 1:
        return (x_sorted - y_sorted).abs().mean()

    return (x_sorted - y_sorted).pow(2).mean()


def sinkhorn_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.5,
    n_iters: int = 50,
    normalize_cost: bool = True,
) -> torch.Tensor:
    n = x.shape[0]
    m = y.shape[0]

    cost = pairwise_squared_distances(x, y, normalize=normalize_cost)

    log_a = -math.log(n) * torch.ones(n, device=x.device)
    log_b = -math.log(m) * torch.ones(m, device=x.device)

    u = torch.zeros(n, device=x.device)
    v = torch.zeros(m, device=x.device)

    for _ in range(n_iters):
        u = epsilon * (log_a - torch.logsumexp((v[None, :] - cost) / epsilon, dim=1))
        v = epsilon * (log_b - torch.logsumexp((u[:, None] - cost) / epsilon, dim=0))

    log_plan = (u[:, None] + v[None, :] - cost) / epsilon
    plan = torch.exp(log_plan)

    return torch.sum(plan * cost)


def sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.5,
    n_iters: int = 50,
    normalize_cost: bool = True,
) -> torch.Tensor:
    xy = sinkhorn_cost(x, y, epsilon, n_iters, normalize_cost)
    xx = sinkhorn_cost(x, x, epsilon, n_iters, normalize_cost)
    yy = sinkhorn_cost(y, y, epsilon, n_iters, normalize_cost)

    return xy - 0.5 * xx - 0.5 * yy


def gaussian_mmd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    cost_xx = pairwise_squared_distances(x, x, normalize=True)
    cost_yy = pairwise_squared_distances(y, y, normalize=True)
    cost_xy = pairwise_squared_distances(x, y, normalize=True)

    loss = 0.0

    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma**2)

        kernel_xx = torch.exp(-gamma * cost_xx)
        kernel_yy = torch.exp(-gamma * cost_yy)
        kernel_xy = torch.exp(-gamma * cost_xy)

        loss = loss + kernel_xx.mean() + kernel_yy.mean() - 2.0 * kernel_xy.mean()

    return loss / len(sigmas)


def covariance_matrix(x: torch.Tensor) -> torch.Tensor:
    centered = x - x.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(centered.shape[0] - 1, 1)


def correlation_matrix_from_sequences(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    batch_size, time_steps, n_features = x.shape

    flat = x.reshape(batch_size * time_steps, n_features)
    flat = flat - flat.mean(dim=0, keepdim=True)

    covariance = flat.T @ flat / max(flat.shape[0] - 1, 1)

    std = torch.sqrt(torch.diag(covariance).clamp_min(eps))
    correlation = covariance / (std[:, None] * std[None, :] + eps)

    return correlation


def temporal_autocovariance(x: torch.Tensor, lag: int) -> torch.Tensor:
    if lag <= 0 or lag >= x.shape[1]:
        raise ValueError("lag must satisfy 1 <= lag < time_length")

    x0 = x[:, :-lag, :]
    x1 = x[:, lag:, :]

    x0 = x0 - x0.mean(dim=(0, 1), keepdim=True)
    x1 = x1 - x1.mean(dim=(0, 1), keepdim=True)

    return (x0 * x1).mean(dim=(0, 1))


def quantile_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    quantiles: Tuple[float, ...] = (0.01, 0.05, 0.50, 0.95, 0.99),
) -> torch.Tensor:
    q = torch.tensor(quantiles, device=x.device)

    x_quantiles = torch.quantile(x, q, dim=0)
    y_quantiles = torch.quantile(y, q, dim=0)

    return F.mse_loss(x_quantiles, y_quantiles)


def tail_mean_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.05,
) -> torch.Tensor:
    n = min(x.shape[0], y.shape[0])
    k = max(1, int(alpha * n))

    x_sorted, _ = torch.sort(x[:n], dim=0)
    y_sorted, _ = torch.sort(y[:n], dim=0)

    lower = F.mse_loss(x_sorted[:k].mean(dim=0), y_sorted[:k].mean(dim=0))
    upper = F.mse_loss(x_sorted[-k:].mean(dim=0), y_sorted[-k:].mean(dim=0))

    return lower + upper


def feature_tail_mean_loss(
    generated_seq: torch.Tensor,
    real_seq: torch.Tensor,
    alpha: float = 0.05,
) -> torch.Tensor:
    generated_flat = generated_seq.reshape(-1, generated_seq.shape[-1])
    real_flat = real_seq.reshape(-1, real_seq.shape[-1])
    return tail_mean_loss(generated_flat, real_flat, alpha=alpha)


@dataclass
class LossWeights:
    sinkhorn: float = 0.5
    sliced_wasserstein: float = 1.0
    mmd: float = 0.2
    mean: float = 5.0
    std: float = 5.0
    covariance: float = 20.0
    feature_correlation: float = 30.0
    autocovariance: float = 10.0
    quantile: float = 3.0
    tail: float = 10.0
    feature_tail: float = 5.0
    output_norm: float = 1e-4


class TimeSeriesDistributionLoss(nn.Module):
    def __init__(
        self,
        time_steps: int,
        n_features: int,
        weights: Optional[LossWeights] = None,
        sinkhorn_epsilon: float = 0.5,
        sinkhorn_iters: int = 50,
        n_projections: int = 256,
        autocov_lags: Tuple[int, ...] = (1, 2, 3),
        tail_alpha: float = 0.05,
    ):
        super().__init__()

        self.time_steps = time_steps
        self.n_features = n_features
        self.flat_dim = time_steps * n_features

        self.weights = weights or LossWeights()

        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.n_projections = n_projections
        self.autocov_lags = tuple(lag for lag in autocov_lags if 1 <= lag < time_steps)
        self.tail_alpha = tail_alpha

    def forward(
        self,
        generated_flat: torch.Tensor,
        real_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        generated = generated_flat
        real = real_flat

        generated_seq = generated.reshape(-1, self.time_steps, self.n_features)
        real_seq = real.reshape(-1, self.time_steps, self.n_features)

        components = {}

        components["sinkhorn"] = sinkhorn_divergence(
            generated,
            real,
            epsilon=self.sinkhorn_epsilon,
            n_iters=self.sinkhorn_iters,
            normalize_cost=True,
        )

        components["sliced_wasserstein"] = sliced_wasserstein_loss(
            generated,
            real,
            n_projections=self.n_projections,
            p=2,
        )

        components["mmd"] = gaussian_mmd_loss(generated, real)

        components["mean"] = F.mse_loss(generated.mean(dim=0), real.mean(dim=0))
        components["std"] = F.mse_loss(generated.std(dim=0), real.std(dim=0))
        components["covariance"] = F.mse_loss(covariance_matrix(generated), covariance_matrix(real))

        components["feature_correlation"] = F.mse_loss(
            correlation_matrix_from_sequences(generated_seq),
            correlation_matrix_from_sequences(real_seq),
        )

        autocovariance_loss = 0.0

        for lag in self.autocov_lags:
            autocovariance_loss = autocovariance_loss + F.mse_loss(
                temporal_autocovariance(generated_seq, lag),
                temporal_autocovariance(real_seq, lag),
            )

        if len(self.autocov_lags) > 0:
            autocovariance_loss = autocovariance_loss / len(self.autocov_lags)
        else:
            autocovariance_loss = torch.tensor(0.0, device=generated.device)

        components["autocovariance"] = autocovariance_loss
        components["quantile"] = quantile_loss(generated, real)
        components["tail"] = tail_mean_loss(generated, real, alpha=self.tail_alpha)
        components["feature_tail"] = feature_tail_mean_loss(generated_seq, real_seq, alpha=self.tail_alpha)
        components["output_norm"] = generated.pow(2).mean()

        total = (
            self.weights.sinkhorn * components["sinkhorn"]
            + self.weights.sliced_wasserstein * components["sliced_wasserstein"]
            + self.weights.mmd * components["mmd"]
            + self.weights.mean * components["mean"]
            + self.weights.std * components["std"]
            + self.weights.covariance * components["covariance"]
            + self.weights.feature_correlation * components["feature_correlation"]
            + self.weights.autocovariance * components["autocovariance"]
            + self.weights.quantile * components["quantile"]
            + self.weights.tail * components["tail"]
            + self.weights.feature_tail * components["feature_tail"]
            + self.weights.output_norm * components["output_norm"]
        )

        components["total"] = total

        return total, components