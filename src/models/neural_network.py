"""Neural Network model for system identification."""

from typing import List, Optional
import numpy as np
from tqdm.auto import tqdm

from .base import BaseModel
from ..utils.regression import create_lagged_features


class NeuralNetwork(BaseModel):
    """
    Feedforward Neural Network for NARX-like system identification.
    
    Args:
        nu: Number of input lags
        ny: Number of output lags  
        hidden_layers: List of hidden layer sizes
        activation: Activation function ('relu', 'tanh', 'selu')
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        batch_size: Mini-batch size
    """

    def __init__(
        self,
        nu: int = 5,
        ny: int = 5,
        hidden_layers: List[int] = [80, 80, 80],
        activation: str = "selu",
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        super().__init__(nu=nu, ny=ny)
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model_ = None
        self._device = None

    def _build_model(self, n_inputs: int):
        """Build PyTorch model."""
        import torch
        import torch.nn as nn

        layers = []
        prev_size = n_inputs

        # Activation mapping
        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "selu": nn.SELU,
            "leaky_relu": nn.LeakyReLU,
        }
        act_fn = act_map.get(self.activation, nn.SELU)

        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(act_fn())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        
        model = nn.Sequential(*layers)
        
        # Initialize weights
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        return model

    def fit(self, u: np.ndarray, y: np.ndarray, verbose: bool = True) -> "NeuralNetwork":
        """Train the neural network."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

        # Prepare data
        features, target = create_lagged_features(y, u, self.ny, self.nu)
        
        if len(target) == 0:
            raise ValueError("Not enough data for given lag orders")

        # Set device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build model
        n_inputs = features.shape[1]
        self.model_ = self._build_model(n_inputs).to(self._device)

        # Convert to tensors
        X = torch.tensor(features, dtype=torch.float32).to(self._device)
        Y = torch.tensor(target, dtype=torch.float32).to(self._device)

        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training
        optimizer = optim.NAdam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        epoch_iter = range(self.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training", unit="epoch")

        for epoch in epoch_iter:
            self.model_.train()
            epoch_loss = 0.0
            
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                pred = self.model_(batch_X).squeeze()
                loss = criterion(pred, batch_Y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if verbose and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=epoch_loss / len(loader))

        self._is_fitted = True
        return self

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction."""
        import torch

        features, _ = create_lagged_features(y, u, self.ny, self.nu)
        X = torch.tensor(features, dtype=torch.float32).to(self._device)

        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(X).squeeze().cpu().numpy()

        return pred

    def predict_free_run(
        self, u: np.ndarray, y_initial: np.ndarray, show_progress: bool = True
    ) -> np.ndarray:
        """Free-run simulation."""
        import torch

        u = np.asarray(u, dtype=float)
        y_init = np.asarray(y_initial, dtype=float)

        if len(y_init) < self.max_lag:
            raise ValueError(f"Need {self.max_lag} initial conditions")

        n_total = len(u)
        y_hat = np.zeros(n_total)
        y_hat[: self.max_lag] = y_init[: self.max_lag]

        self.model_.eval()
        sim_range = range(self.max_lag, n_total)
        if show_progress:
            sim_range = tqdm(sim_range, desc="Free-run simulation", unit="step")

        with torch.no_grad():
            for k in sim_range:
                # Build feature vector
                features = []
                for j in range(1, self.ny + 1):
                    features.append(y_hat[k - j])
                for j in range(1, self.nu + 1):
                    features.append(u[k - j])

                X = torch.tensor([features], dtype=torch.float32).to(self._device)
                y_hat[k] = self.model_(X).squeeze().cpu().item()

        return y_hat[self.max_lag:]

    def __repr__(self) -> str:
        return (
            f"NeuralNetwork(nu={self.nu}, ny={self.ny}, "
            f"hidden_layers={self.hidden_layers})"
        )
