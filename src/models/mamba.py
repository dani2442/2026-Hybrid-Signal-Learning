"""Mamba (Selective State Space Model) for time series forecasting.

Pure-PyTorch implementation of the S6 / Mamba architecture:

    x'(t) = A x(t) + B u(t)
    y(t)  = C x(t) + D u(t)

where A, B, C are *input-dependent* (selective scan) and the continuous
parameters are discretised with a learned time-step Δ.

Reference:
    Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
    Spaces*, 2023.  arXiv:2312.00752
"""

from __future__ import annotations

from typing import Callable
import numpy as np
from tqdm.auto import tqdm

from .base import BaseModel


class Mamba(BaseModel):
    """
    Mamba (Selective State Space Model) for system identification.

    Uses a stack of Mamba blocks, each containing:
      - Linear projection to expand the channel dimension
      - 1-D causal convolution
      - Selective SSM core (discretised A, B, C with learned Δ)
      - Gated output path (SiLU activation)

    Args:
        nu: Number of input lags (sequence length for inputs)
        ny: Number of output lags (sequence length for outputs)
        d_model: Model / embedding dimension
        d_state: SSM state expansion factor (N in the paper)
        d_conv: Width of the local 1-D convolution
        n_layers: Number of stacked Mamba blocks
        expand_factor: Channel expansion inside each block
        dropout: Dropout rate
        learning_rate: Learning rate for Adam
        epochs: Number of training epochs
        batch_size: Mini-batch size
    """

    def __init__(
        self,
        nu: int = 10,
        ny: int = 10,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        n_layers: int = 2,
        expand_factor: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        lr: float | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        max_lag: int | None = None,
    ):
        if max_lag is not None:
            max_lag = int(max_lag)
            if max_lag < 0:
                raise ValueError("max_lag must be non-negative")
            if (nu != 10 or ny != 10) and (nu != max_lag or ny != max_lag):
                raise ValueError("Use either max_lag or nu/ny, not conflicting values")
            nu = max_lag
            ny = max_lag

        if lr is not None:
            learning_rate = float(lr)

        super().__init__(nu=nu, ny=ny)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.n_layers = n_layers
        self.expand_factor = expand_factor
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model_ = None
        self._device = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._u_mean = 0.0
        self._u_std = 1.0

    # ------------------------------------------------------------------
    def _build_model(self, input_size: int):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        d_model = self.d_model
        d_state = self.d_state
        d_conv = self.d_conv
        n_layers = self.n_layers
        expand = self.expand_factor
        drop = self.dropout

        class SelectiveSSM(nn.Module):
            """Core selective scan (S6) — vectorised implementation."""

            def __init__(self, d_inner, d_state):
                super().__init__()
                self.d_inner = d_inner
                self.d_state = d_state

                # A is initialised as a diagonal matrix via log-space
                self.A_log = nn.Parameter(
                    torch.log(
                        torch.arange(1, d_state + 1, dtype=torch.float32)
                        .unsqueeze(0)
                        .expand(d_inner, -1)
                        .clone()
                    )
                )
                # D is a residual skip
                self.D = nn.Parameter(torch.ones(d_inner))

                # Input-dependent projections → Δ, B, C
                self.proj_delta = nn.Linear(d_inner, d_inner, bias=True)
                self.proj_B = nn.Linear(d_inner, d_state, bias=False)
                self.proj_C = nn.Linear(d_inner, d_state, bias=False)

            def forward(self, x):
                """x: (B, L, D_inner) → (B, L, D_inner)"""
                B_batch, L, D = x.shape

                A = -torch.exp(self.A_log)            # (D, N)

                delta = F.softplus(self.proj_delta(x))  # (B, L, D)
                B_mat = self.proj_B(x)                   # (B, L, N)
                C_mat = self.proj_C(x)                   # (B, L, N)

                # Discretise: A_bar = exp(Δ · A),  B_bar = Δ · B
                deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
                deltaB = delta.unsqueeze(-1) * B_mat.unsqueeze(2)  # (B, L, D, N)

                # Sequential scan (over L which is small, typically 10)
                ys = torch.zeros(B_batch, L, D, device=x.device, dtype=x.dtype)
                h = torch.zeros(B_batch, D, self.d_state, device=x.device, dtype=x.dtype)
                for t in range(L):
                    h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)
                    ys[:, t] = (h * C_mat[:, t].unsqueeze(1)).sum(dim=-1)

                return ys + x * self.D

        class MambaBlock(nn.Module):
            def __init__(self, d_model, d_state, d_conv, expand, dropout):
                super().__init__()
                d_inner = d_model * expand
                self.norm = nn.LayerNorm(d_model)

                # Two linear projections (like a gated path)
                self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

                # Causal depth-wise conv on the SSM path
                self.conv1d = nn.Conv1d(
                    d_inner, d_inner, kernel_size=d_conv,
                    padding=d_conv - 1, groups=d_inner, bias=True,
                )

                self.ssm = SelectiveSSM(d_inner, d_state)

                self.out_proj = nn.Linear(d_inner, d_model, bias=False)
                self.drop = nn.Dropout(dropout)

            def forward(self, x):
                """x: (B, L, D)"""
                residual = x
                x = self.norm(x)

                xz = self.in_proj(x)                         # (B, L, 2*D_inner)
                x_ssm, z = xz.chunk(2, dim=-1)               # each (B, L, D_inner)

                # Causal conv1d
                x_ssm = x_ssm.permute(0, 2, 1)               # (B, D_inner, L)
                x_ssm = self.conv1d(x_ssm)[:, :, :x.size(1)] # causal trim
                x_ssm = x_ssm.permute(0, 2, 1)               # (B, L, D_inner)
                x_ssm = F.silu(x_ssm)

                # Selective SSM
                x_ssm = self.ssm(x_ssm)

                # Gate
                out = x_ssm * F.silu(z)
                out = self.out_proj(self.drop(out))

                return out + residual

        class MambaModel(nn.Module):
            def __init__(self, input_size, d_model, d_state, d_conv, n_layers,
                         expand, dropout):
                super().__init__()
                self.embed = nn.Linear(input_size, d_model)
                self.blocks = nn.ModuleList([
                    MambaBlock(d_model, d_state, d_conv, expand, dropout)
                    for _ in range(n_layers)
                ])
                self.norm = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, 1)

            def forward(self, x):
                """x: (B, L, input_size)"""
                x = self.embed(x)
                for block in self.blocks:
                    x = block(x)
                x = self.norm(x)
                return self.head(x[:, -1, :])

        return MambaModel(
            input_size, d_model, d_state, d_conv, n_layers, expand, drop,
        )

    # ------------------------------------------------------------------
    def _create_sequences(self, y: np.ndarray, u: np.ndarray):
        """Create sequences identical to GRU / LSTM."""
        seq_len = self.max_lag
        n_samples = len(y) - seq_len

        X = np.zeros((n_samples, seq_len, 2))
        Y = np.zeros(n_samples)

        for i in range(n_samples):
            X[i, :, 0] = y[i : i + seq_len]
            X[i, :, 1] = u[i : i + seq_len]
            Y[i] = y[i + seq_len]

        return X, Y

    # ------------------------------------------------------------------
    def fit(
        self,
        u: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
        eval_callback: Callable | None = None,
        eval_every: int = 1,
    ) -> "Mamba":
        """Train the Mamba network."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

        u = np.asarray(u, dtype=float)
        y = np.asarray(y, dtype=float)

        # Normalize
        self._y_mean, self._y_std = y.mean(), y.std()
        self._u_mean, self._u_std = u.mean(), u.std()

        y_norm = (y - self._y_mean) / (self._y_std + 1e-8)
        u_norm = (u - self._u_mean) / (self._u_std + 1e-8)

        X, Y = self._create_sequences(y_norm, u_norm)
        if len(Y) == 0:
            raise ValueError("Not enough data for given lag orders")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = self._build_model(input_size=2).to(self._device)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(self._device)

        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        epoch_iter = range(self.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training Mamba", unit="epoch")

        eval_every = max(1, int(eval_every))

        for epoch in epoch_iter:
            self.model_.train()
            epoch_loss = 0.0

            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                pred = self.model_(batch_X)
                loss = criterion(pred.squeeze(), batch_Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=avg_loss)
            if eval_callback is not None and (epoch + 1) % eval_every == 0:
                eval_callback(model=self, epoch=epoch + 1, train_loss=float(avg_loss))

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction."""
        import torch

        u = np.asarray(u, dtype=float)
        y = np.asarray(y, dtype=float)

        y_norm = (y - self._y_mean) / (self._y_std + 1e-8)
        u_norm = (u - self._u_mean) / (self._u_std + 1e-8)

        X, _ = self._create_sequences(y_norm, u_norm)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)

        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(X_tensor)
            pred = pred.squeeze().cpu().numpy()

        return pred * self._y_std + self._y_mean

    def predict_free_run(
        self, u: np.ndarray, y_initial: np.ndarray, show_progress: bool = True,
    ) -> np.ndarray:
        """Free-run simulation using predicted outputs recursively."""
        import torch

        u = np.asarray(u, dtype=float)
        y_init = np.asarray(y_initial, dtype=float)

        if len(y_init) < self.max_lag:
            raise ValueError(
                f"Need {self.max_lag} initial conditions, got {len(y_init)}"
            )

        u_norm = (u - self._u_mean) / (self._u_std + 1e-8)
        y_init_norm = (y_init - self._y_mean) / (self._y_std + 1e-8)

        n_total = len(u)
        y_hat_norm = np.zeros(n_total)
        y_hat_norm[: self.max_lag] = y_init_norm[: self.max_lag]

        self.model_.eval()

        sim_range = range(self.max_lag, n_total)
        if show_progress:
            sim_range = tqdm(sim_range, desc="Mamba Free-run simulation", unit="step")

        with torch.no_grad():
            for k in sim_range:
                seq_y = y_hat_norm[k - self.max_lag : k]
                seq_u = u_norm[k - self.max_lag : k]

                x = np.stack([seq_y, seq_u], axis=1)
                x_tensor = (
                    torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self._device)
                )

                pred = self.model_(x_tensor)
                y_hat_norm[k] = pred.squeeze().cpu().numpy()

        y_hat = y_hat_norm * self._y_std + self._y_mean
        return y_hat[self.max_lag :]

    # ------------------------------------------------------------------
    def summary(self) -> str:
        if not self._is_fitted:
            return "Model not fitted"

        total_params = sum(p.numel() for p in self.model_.parameters())
        trainable_params = sum(
            p.numel() for p in self.model_.parameters() if p.requires_grad
        )

        return (
            f"Mamba Model:\n"
            f"  d_model: {self.d_model}\n"
            f"  d_state: {self.d_state}\n"
            f"  n_layers: {self.n_layers}\n"
            f"  Total parameters: {total_params:,}\n"
            f"  Trainable parameters: {trainable_params:,}"
        )

    def __repr__(self) -> str:
        return (
            f"Mamba(nu={self.nu}, ny={self.ny}, d_model={self.d_model}, "
            f"d_state={self.d_state}, n_layers={self.n_layers})"
        )
