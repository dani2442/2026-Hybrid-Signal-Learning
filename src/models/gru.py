"""GRU (Gated Recurrent Unit) model for time series forecasting."""

from typing import List, Optional
import numpy as np
from tqdm.auto import tqdm

from .base import BaseModel


class GRU(BaseModel):
    """
    GRU (Gated Recurrent Unit) Network for sequence-to-sequence system identification.
    
    Args:
        nu: Number of input lags (sequence length for inputs)
        ny: Number of output lags (sequence length for outputs)
        hidden_size: Size of GRU hidden state
        num_layers: Number of stacked GRU layers
        dropout: Dropout rate between GRU layers
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        batch_size: Mini-batch size
    """

    def __init__(
        self,
        nu: int = 10,
        ny: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        super().__init__(nu=nu, ny=ny)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model_ = None
        self._device = None
        self._scaler_y = None
        self._scaler_u = None

    def _build_model(self, input_size: int):
        """Build PyTorch GRU model."""
        import torch
        import torch.nn as nn

        class GRUModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x, hidden=None):
                # x: (batch, seq_len, input_size)
                out, hidden = self.gru(x, hidden)
                # Take only the last output
                out = self.fc(out[:, -1, :])
                return out, hidden

        return GRUModel(input_size, self.hidden_size, self.num_layers, self.dropout)

    def _create_sequences(self, y: np.ndarray, u: np.ndarray):
        """Create sequences for GRU training."""
        seq_len = self.max_lag
        n_samples = len(y) - seq_len
        
        # Each sample: sequence of [y, u] pairs
        X = np.zeros((n_samples, seq_len, 2))
        Y = np.zeros(n_samples)
        
        for i in range(n_samples):
            X[i, :, 0] = y[i:i + seq_len]
            X[i, :, 1] = u[i:i + seq_len]
            Y[i] = y[i + seq_len]
        
        return X, Y

    def fit(self, u: np.ndarray, y: np.ndarray, verbose: bool = True) -> "GRU":
        """Train the GRU network."""
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
        self.model_ = self._build_model(input_size=2).to(self._device)

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
            epoch_iter = tqdm(epoch_iter, desc="Training GRU", unit="epoch")

        for epoch in epoch_iter:
            self.model_.train()
            epoch_loss = 0.0
            
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                pred, _ = self.model_(batch_X)
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
            pred, _ = self.model_(X_tensor)
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
        hidden = None

        sim_range = range(self.max_lag, n_total)
        if show_progress:
            sim_range = tqdm(sim_range, desc="GRU Free-run simulation", unit="step")

        with torch.no_grad():
            for k in sim_range:
                # Build sequence from past predictions
                seq_y = y_hat_norm[k - self.max_lag:k]
                seq_u = u_norm[k - self.max_lag:k]
                
                x = np.stack([seq_y, seq_u], axis=1)  # (seq_len, 2)
                x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self._device)
                
                pred, hidden = self.model_(x_tensor, hidden)
                y_hat_norm[k] = pred.squeeze().cpu().numpy()
                
                # Detach hidden state
                hidden = hidden.detach()

        # Denormalize and return
        y_hat = y_hat_norm * self._y_std + self._y_mean
        return y_hat[self.max_lag:]

    def summary(self) -> str:
        """Print model summary."""
        if not self._is_fitted:
            return "Model not fitted"
        
        total_params = sum(p.numel() for p in self.model_.parameters())
        trainable_params = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        
        return (
            f"GRU Model:\n"
            f"  Hidden size: {self.hidden_size}\n"
            f"  Num layers: {self.num_layers}\n"
            f"  Total parameters: {total_params:,}\n"
            f"  Trainable parameters: {trainable_params:,}"
        )

    def __repr__(self) -> str:
        return f"GRU(nu={self.nu}, ny={self.ny}, hidden_size={self.hidden_size}, num_layers={self.num_layers})"
