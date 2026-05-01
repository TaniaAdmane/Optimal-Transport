import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchdiffeq import odeint
import time
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from metrics.eval_functions import get_stats, get_scores, plot_sample_multi

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========== SINUSOIDAL TIME EMBEDDING (cf. Transformer/diffusion models) ==========
class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for continuous t ∈ [0, 1].
    Used in DDPM, Score Matching, and FM implementations.
    Provides a structured time signal rather than learning it from scratch.
    """
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Embedding dim must be even"
        self.dim = dim

    def forward(self, t):
        # t shape: (batch,) or (batch, 1)
        if t.dim() == 2:
            t = t.squeeze(-1)
        device = t.device
        half_dim = self.dim // 2
        # log-spaced frequencies, scale chosen so that the lowest freq covers [0, 1]
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb  # shape: (batch, dim)


# ========== VELOCITY NETWORK (simplified, closer to Lipman et al.) ==========
class VelocityNetwork(nn.Module):
    """
    MLP velocity field for Flow Matching.

    Changes vs previous version:
    - Sinusoidal time embedding instead of deep MLP from scratch
    - Reduced depth (4 hidden blocks) and width (256) — appropriate for input_dim ~90
    - Kept LayerNorm for stability on noisy financial data
    """
    def __init__(self, input_dim, hidden_dim=256, time_emb_dim=128, n_blocks=4):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Stack of residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_blocks)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        if t.dim() == 1:
            t_input = t
        else:
            t_input = t.squeeze(-1)
        t_emb = self.time_embed(t_input)

        h = self.input_proj(torch.cat([x, t_emb], dim=-1))

        # Residual blocks
        for block in self.blocks:
            h = h + block(h)

        h = self.out_norm(h)
        return self.out_proj(h)


# ========== FLOW MATCHING WITH OT PATH ==========
class FlowMatchingOT:
    """
    Flow Matching with Optimal Transport conditional path,
    following Lipman et al. (2023).

    Key parameters from the paper:
    - sigma_min: small positive value (1e-4 to 1e-2 in the paper)
      so that p_1(x|x_1) is concentrated near x_1.
      Previous default of 0.5-0.6 was way too large and prevented the model
      from generating data close to the true distribution.
    """
    def __init__(self, window_size, n_assets, sigma_min=1e-4,
                 hidden_dim=256, lr=1e-3, weight_decay=0.0):
        self.window_size = window_size
        self.n_assets = n_assets
        self.input_dim = window_size * n_assets
        self.device = DEVICE
        self.sigma_min = sigma_min

        self.model = VelocityNetwork(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.optimizer = Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = None  # set in train()

    def ot_path(self, x0, x1, t):
        """ψ_t(x0) = (1 - (1 - σ_min) t) x0 + t x1  — eq. (22) of Lipman et al."""
        return (1 - (1 - self.sigma_min) * t) * x0 + t * x1

    def ot_velocity(self, x0, x1):
        """u_t = x1 - (1 - σ_min) x0  — eq. (23) of Lipman et al."""
        return x1 - (1 - self.sigma_min) * x0

    def train_step(self, x_batch):
        batch_size = x_batch.shape[0]
        x1 = x_batch.reshape(batch_size, -1)
        x0 = torch.randn_like(x1, device=self.device)
        t = torch.rand(batch_size, 1, device=self.device)

        psi_t = self.ot_path(x0, x1, t)
        u_target = self.ot_velocity(x0, x1)
        u_pred = self.model(psi_t, t.squeeze(-1))

        loss = ((u_pred - u_target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train(self, X_train, epochs=200, batch_size=128, verbose_every=10):
        X_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        dataloader = DataLoader(
            TensorDataset(X_tensor),
            batch_size=batch_size,
            shuffle=True,
        )

        # Cosine schedule with warmup-like behavior
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)

        losses = []
        self.model.train()
        for epoch in range(epochs):
            epoch_loss, n_batches = 0.0, 0
            for (batch,) in dataloader:
                loss = self.train_step(batch)
                epoch_loss += loss
                n_batches += 1
            self.scheduler.step()

            epoch_loss /= n_batches
            losses.append(epoch_loss)

            if (epoch + 1) % verbose_every == 0:
                lr_now = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs} | Loss {epoch_loss:.6f} | LR {lr_now:.2e}")
        return losses

    def generate(self, n_samples=1000, method='dopri5', rtol=1e-5, atol=1e-5):
        """
        Generate samples by solving the ODE from t=0 to t=1.
        With dopri5, n_steps in t_span only controls output grid; the solver
        adapts internally. We just request t=0 and t=1.
        """
        self.model.eval()
        with torch.no_grad():
            z0 = torch.randn(n_samples, self.input_dim, device=self.device)
            t_span = torch.tensor([0.0, 1.0], device=self.device)

            def velocity_fn(t, x):
                t_batch = torch.full((x.shape[0],), t.item(), device=self.device)
                return self.model(x, t_batch)

            print(f"Generating {n_samples} samples (solver={method})...")
            start = time.time()
            X_traj = odeint(
                velocity_fn, z0, t_span,
                method=method, rtol=rtol, atol=atol,
            )
            X_synth_flat = X_traj[-1]
            elapsed = time.time() - start
            print(f"✓ Generated in {elapsed:.2f}s\n")

            X_synth = X_synth_flat.cpu().numpy().reshape(
                n_samples, self.window_size, self.n_assets
            )
            return X_synth