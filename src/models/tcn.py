"""TCN (Temporal Convolutional Network) model for time series forecasting."""

from typing import List, Optional
import numpy as np
from tqdm.auto import tqdm

from .base import BaseModel


class TCN(BaseModel):
    """
    Temporal Convolutional Network for sequence-to-sequence system identification.

    Uses causal dilated 1-D convolutions with residual connections.
    Dilation grows exponentially (1, 2, 4, …) so the receptive field
    covers the full input sequence with relatively few layers.

    Args:
        nu: Number of input lags (sequence length for inputs)
        ny: Number of output lags (sequence length for outputs)
        num_channels: List of channel sizes per residual block
        kernel_size: Convolution kernel width (applied causally)
        dropout: Spatial dropout between blocks
        learning_rate: Learning rate for Adam
        epochs: Number of training epochs
        batch_size: Mini-batch size
    """

    def __init__(
        self,
        nu: int = 10,
        ny: int = 10,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        super().__init__(nu=nu, ny=ny)
        self.num_channels = num_channels or [64, 64, 64, 64]
        self.kernel_size = kernel_size
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

    def _build_model(self, input_channels: int):
        """Build PyTorch TCN model with causal dilated convolutions."""
        import torch
        import torch.nn as nn

        class CausalConv1d(nn.Module):
            """Conv1d with left-side zero-padding to enforce causality."""

            def __init__(self, in_ch, out_ch, kernel_size, dilation):
                super().__init__()
                self.pad = (kernel_size - 1) * dilation
                self.conv = nn.Conv1d(
                    in_ch, out_ch, kernel_size, dilation=dilation,
                )

            def forward(self, x):
                # x: (batch, channels, seq_len)
                x = nn.functional.pad(x, (self.pad, 0))
                return self.conv(x)

        class ResidualBlock(nn.Module):
            """Two causal convolutions + residual skip."""

            def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
                super().__init__()
                self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
                self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.skip = (
                    nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
                )

            def forward(self, x):
                out = self.dropout(self.relu(self.conv1(x)))
                out = self.dropout(self.relu(self.conv2(out)))
                return self.relu(out + self.skip(x))

        class TCNModel(nn.Module):
            def __init__(self, input_channels, num_channels, kernel_size, dropout):
                super().__init__()
                blocks = []
                for i, out_ch in enumerate(num_channels):
                    in_ch = input_channels if i == 0 else num_channels[i - 1]
                    dilation = 2 ** i
                    blocks.append(
                        ResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout)
                    )
                self.network = nn.Sequential(*blocks)
                self.fc = nn.Linear(num_channels[-1], 1)

            def forward(self, x):
                # x: (batch, seq_len, input_channels) -> permute for Conv1d
                x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
                out = self.network(x)  # (batch, channels, seq_len)
                out = out[:, :, -1]  # take last time step
                return self.fc(out)

        return TCNModel(input_channels, self.num_channels, self.kernel_size, self.dropout)

    def _create_sequences(self, y: np.ndarray, u: np.ndarray):
        """Create sequences for TCN training (identical to GRU/LSTM)."""
        seq_len = self.max_lag
        n_samples = len(y) - seq_len

        X = np.zeros((n_samples, seq_len, 2))
        Y = np.zeros(n_samples)

        for i in range(n_samples):
            X[i, :, 0] = y[i : i + seq_len]
            X[i, :, 1] = u[i : i + seq_len]
            Y[i] = y[i + seq_len]

        return X, Y

    def fit(self, u: np.ndarray, y: np.ndarray, verbose: bool = True) -> "TCN":
        """Train the TCN network."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

        u = np.asarray(u, dtype=float)
        y = np.asarray(y, dtype=float)

        # Normalize data
        self._y_mean, self._y_std = y.mean(), y.std()
        self._u_mean, self._u_std = u.mean(), u.std()

        y_norm = (y - self._y_mean) / (self._y_std + 1e-8)
        u_norm = (u - self._u_mean) / (self._u_std + 1e-8)

        # Create sequences
        X, Y = self._create_sequences(y_norm, u_norm)

        if len(Y) == 0:
            raise ValueError("Not enough data for given lag orders")

        # Set device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model_ = self._build_model(input_channels=2).to(self._device)

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).to(self._device)

        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        epoch_iter = range(self.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training TCN", unit="epoch")

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

            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=epoch_loss / len(loader))

        self._is_fitted = True
        return self

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction."""
        import torch

        u = np.asarray(u, dtype=float)
        y = np.asarray(y, dtype=float)

        # Normalize
        y_norm = (y - self._y_mean) / (self._y_std + 1e-8)
        u_norm = (u - self._u_mean) / (self._u_std + 1e-8)

        X, _ = self._create_sequences(y_norm, u_norm)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)

        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(X_tensor)
            pred = pred.squeeze().cpu().numpy()

        # Denormalize
        return pred * self._y_std + self._y_mean

    def predict_free_run(
        self, u: np.ndarray, y_initial: np.ndarray, show_progress: bool = True
    ) -> np.ndarray:
        """Free-run simulation using predicted outputs recursively."""
        import torch

        u = np.asarray(u, dtype=float)
        y_init = np.asarray(y_initial, dtype=float)

        if len(y_init) < self.max_lag:
            raise ValueError(
                f"Need {self.max_lag} initial conditions, got {len(y_init)}"
            )

        # Normalize
        u_norm = (u - self._u_mean) / (self._u_std + 1e-8)
        y_init_norm = (y_init - self._y_mean) / (self._y_std + 1e-8)

        n_total = len(u)
        y_hat_norm = np.zeros(n_total)
        y_hat_norm[: self.max_lag] = y_init_norm[: self.max_lag]

        self.model_.eval()

        sim_range = range(self.max_lag, n_total)
        if show_progress:
            sim_range = tqdm(sim_range, desc="TCN Free-run simulation", unit="step")

        with torch.no_grad():
            for k in sim_range:
                seq_y = y_hat_norm[k - self.max_lag : k]
                seq_u = u_norm[k - self.max_lag : k]

                x = np.stack([seq_y, seq_u], axis=1)  # (seq_len, 2)
                x_tensor = (
                    torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self._device)
                )

                pred = self.model_(x_tensor)
                y_hat_norm[k] = pred.squeeze().cpu().numpy()

        # Denormalize and return
        y_hat = y_hat_norm * self._y_std + self._y_mean
        return y_hat[self.max_lag :]

    def summary(self) -> str:
        """Print model summary."""
        if not self._is_fitted:
            return "Model not fitted"

        total_params = sum(p.numel() for p in self.model_.parameters())
        trainable_params = sum(
            p.numel() for p in self.model_.parameters() if p.requires_grad
        )

        return (
            f"TCN Model:\n"
            f"  Channels: {self.num_channels}\n"
            f"  Kernel size: {self.kernel_size}\n"
            f"  Total parameters: {total_params:,}\n"
            f"  Trainable parameters: {trainable_params:,}"
        )

    def __repr__(self) -> str:
        return (
            f"TCN(nu={self.nu}, ny={self.ny}, "
            f"num_channels={self.num_channels}, kernel_size={self.kernel_size})"
        )
