"""Neural Controlled Differential Equation (CDE) model for time series.

Follows the architecture from:
    Kidger et al., "Neural Controlled Differential Equations for Irregular
    Time Series", NeurIPS 2020.

Reference implementation: https://github.com/patrick-kidger/torchcde
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .base import BaseModel


# ---------------------------------------------------------------------------
# Lazy nn.Module imports — keeps the top of the file free from torch
# ---------------------------------------------------------------------------


def _build_cde_func(hidden_dim: int, input_channels: int, hidden_layers: list):
    """Build the CDE vector-field f_theta as a proper nn.Module.

    Returns a module whose ``forward(t, z)`` outputs a tensor of shape
    ``(batch, hidden_dim, input_channels)`` — the matrix that multiplies
    ``dX/dt`` in the CDE formulation.
    """
    import torch.nn as nn

    class CDEFunc(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = int(hidden_dim)
            self.input_channels = int(input_channels)

            layers: list[nn.Module] = []
            prev = self.hidden_dim
            for h in hidden_layers:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                prev = h
            layers.append(nn.Linear(prev, self.hidden_dim * self.input_channels))
            # Final tanh — "easy-to-forget gotcha" from the torchcde docs.
            layers.append(nn.Tanh())
            self.net = nn.Sequential(*layers)

            # Xavier init
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, t, z):
            # z: (batch, hidden_dim)
            out = self.net(z)
            return out.view(*z.shape[:-1], self.hidden_dim, self.input_channels)

    return CDEFunc()


def _build_initial_network(input_channels: int, hidden_dim: int):
    """Return an nn.Linear that maps the first observation X(t_0) → z_0."""
    import torch.nn as nn

    initial = nn.Linear(input_channels, hidden_dim)
    nn.init.xavier_uniform_(initial.weight)
    nn.init.zeros_(initial.bias)
    return initial


def _build_readout(hidden_dim: int, output_dim: int = 1):
    """Return an nn.Linear that maps the hidden state z → y_pred."""
    import torch.nn as nn

    readout = nn.Linear(hidden_dim, output_dim)
    nn.init.xavier_uniform_(readout.weight)
    nn.init.zeros_(readout.bias)
    return readout


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class NeuralCDE(BaseModel):
    """
    Neural Controlled Differential Equation for continuous-time system identification.

    Solves:
        dz(t) = f_theta(z(t)) dX(t),   z(t_0) = initial(X(t_0))

    where X(t) is a continuous path built from discrete (u, y) observations via
    Hermite cubic spline (or linear) interpolation.

    Key design choices (matching the official torchcde examples):
    * **z0 is a learned projection** of the first observation X(t_0), not a
      fixed zero vector.
    * Training uses **subsequence batching** to keep the computation graph
      short enough for stable gradient flow.
    * A final **tanh** is applied inside the vector field f_theta.
    """

    _VALID_SOLVERS = {"dopri5", "rk4", "euler", "midpoint"}
    _VALID_INTERPOLATIONS = {"cubic", "linear"}

    def __init__(
        self,
        hidden_dim: int = 32,
        input_dim: int = 2,
        hidden_layers: List[int] = [64, 64],
        interpolation: str = "cubic",
        solver: str = "rk4",
        learning_rate: float = 1e-3,
        lr: float | None = None,
        epochs: int = 100,
        rtol: float = 1e-4,
        atol: float = 1e-5,
        sequence_length: int = 50,
        windows_per_epoch: int = 24,
    ):
        super().__init__(nu=input_dim, ny=1)
        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim)
        self.hidden_layers = list(hidden_layers)
        self.interpolation = interpolation
        self.solver = solver
        if lr is not None:
            learning_rate = float(lr)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.sequence_length = int(sequence_length)
        self.windows_per_epoch = int(windows_per_epoch)

        if self.solver not in self._VALID_SOLVERS:
            raise ValueError(
                f"Unknown solver: {self.solver}. "
                f"Supported: {', '.join(sorted(self._VALID_SOLVERS))}"
            )
        if self.interpolation not in self._VALID_INTERPOLATIONS:
            raise ValueError(
                f"Unknown interpolation: {self.interpolation}. "
                f"Supported: {', '.join(sorted(self._VALID_INTERPOLATIONS))}"
            )

        # Will be initialised in fit()
        self.cde_func_ = None
        self.initial_net_ = None
        self.readout_layer_ = None
        self._device = None
        self._dtype = None
        self.training_loss_: list[float] = []

        # Normalization statistics (set during fit)
        self._y_mean = 0.0
        self._y_std = 1.0
        self._u_mean = 0.0
        self._u_std = 1.0

    # ------------------------------------------------------------------
    # Interpolation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_interpolation_from_tensor(x, interpolation: str):
        """Create a torchcde interpolation object from a data tensor.

        Args:
            x: Tensor of shape (batch, length, channels)
            interpolation: "cubic" or "linear"
        """
        import torchcde

        if interpolation == "cubic":
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            return torchcde.CubicSpline(coeffs)
        else:
            coeffs = torchcde.linear_interpolation_coeffs(x)
            return torchcde.LinearInterpolation(coeffs)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _normalise_and_build_tensor(
        self,
        u: np.ndarray,
        y: np.ndarray,
        *,
        fit_stats: bool = False,
    ):
        """Return a torch tensor of shape ``(1, length, 3)`` with channels [t, u, y].

        When ``fit_stats=True`` the normalisation statistics are (re-)computed
        from the supplied data; otherwise the existing stats are used.
        """
        import torch

        u = np.asarray(u, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        if fit_stats:
            self._u_mean = float(u.mean())
            self._u_std = float(u.std()) + 1e-8
            self._y_mean = float(y.mean())
            self._y_std = float(y.std()) + 1e-8

        u_norm = (u - self._u_mean) / self._u_std
        y_norm = (y - self._y_mean) / self._y_std
        t_vec = np.linspace(0.0, 1.0, len(u)).reshape(-1, 1)

        data = np.concatenate([t_vec, u_norm, y_norm], axis=1)  # (length, 3)
        return torch.tensor(
            data[np.newaxis], dtype=self._dtype, device=self._device,
        )

    # ------------------------------------------------------------------
    # Core CDE solve (shared by training and inference)
    # ------------------------------------------------------------------

    def _solve_cde(self, X, *, adjoint: bool = False):
        """Solve the CDE over interpolation *X* and return ``(z_path, y_pred)``.

        ``z_path`` has shape ``(batch, length, hidden_dim)``.
        ``y_pred`` has shape ``(batch, length, 1)`` (normalised).
        """
        import torchcde

        # z0 from the first observation — the "official" pattern
        X0 = X.evaluate(X.interval[0])  # (batch, input_channels)
        z0 = self.initial_net_(X0)      # (batch, hidden_dim)

        # Evaluate at all grid points for a dense prediction
        t_eval = X.grid_points

        # For fixed-step solvers, pass step_size
        extra = {}
        if self.solver in ("rk4", "euler", "midpoint"):
            step = (X.grid_points[1:] - X.grid_points[:-1]).min().item()
            extra["options"] = {"step_size": step}

        z_path = torchcde.cdeint(
            X=X,
            func=self.cde_func_,
            z0=z0,
            t=t_eval,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
            adjoint=adjoint,
        )
        # torchcde output: (batch, length, hidden_dim)

        y_pred = self.readout_layer_(z_path)  # (batch, length, 1)
        return z_path, y_pred

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        u: np.ndarray | Sequence[np.ndarray],
        y: np.ndarray | Sequence[np.ndarray],
        verbose: bool = True,
        wandb_run=None,
        wandb_log_every: int = 1,
    ) -> "NeuralCDE":
        """
        Train the Neural CDE model using subsequence batching.

        The full time series is split into overlapping windows of
        ``sequence_length`` steps. Each epoch randomly selects windows
        so that the computation graph stays short and gradients flow
        well (same idea as NeuralODE's ``train_sequence_batches``).

        Args:
            u: Input signal array of shape ``(N,)``
            y: Output signal array of shape ``(N,)``
            verbose: Show tqdm progress bar
            sequence_length: Length of training subsequences
            wandb_run: Optional W&B run for logging
            wandb_log_every: Log to W&B every N epochs
        """
        try:
            import torch
            import torch.optim as optim
        except ImportError as exc:
            raise ImportError(
                "PyTorch and torchcde are required. "
                "Install with: pip install torch torchcde"
            ) from exc

        if verbose:
            from tqdm.auto import tqdm

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32

        is_multi = isinstance(u, Sequence) and not isinstance(u, np.ndarray)
        if is_multi != (isinstance(y, Sequence) and not isinstance(y, np.ndarray)):
            raise ValueError("u and y must both be arrays or both be dataset lists")

        if is_multi:
            if len(u) != len(y):
                raise ValueError("u and y dataset lists must have the same length")
            u_data = [np.asarray(u_ds, dtype=np.float64).flatten() for u_ds in u]
            y_data = [np.asarray(y_ds, dtype=np.float64).flatten() for y_ds in y]
            for u_ds, y_ds in zip(u_data, y_data):
                if len(u_ds) != len(y_ds):
                    raise ValueError("Each dataset pair must have u and y with equal length")
            u_concat = np.concatenate([u_ds.reshape(-1, 1) for u_ds in u_data], axis=0)
            y_concat = np.concatenate([y_ds.reshape(-1, 1) for y_ds in y_data], axis=0)
            self._u_mean = float(u_concat.mean())
            self._u_std = float(u_concat.std()) + 1e-8
            self._y_mean = float(y_concat.mean())
            self._y_std = float(y_concat.std()) + 1e-8
            x_datasets = [
                self._normalise_and_build_tensor(u_ds, y_ds, fit_stats=False)
                for u_ds, y_ds in zip(u_data, y_data)
            ]
        else:
            u_data = np.asarray(u, dtype=np.float64).flatten()
            y_data = np.asarray(y, dtype=np.float64).flatten()
            if len(u_data) != len(y_data):
                raise ValueError("u and y must have the same length")
            x_datasets = [self._normalise_and_build_tensor(u_data, y_data, fit_stats=True)]

        input_channels = x_datasets[0].shape[2]  # 3 = [t, u, y]

        # --- build sub-modules ---
        self.cde_func_ = _build_cde_func(
            hidden_dim=self.hidden_dim,
            input_channels=input_channels,
            hidden_layers=self.hidden_layers,
        ).to(self._device)

        self.initial_net_ = _build_initial_network(
            input_channels=input_channels,
            hidden_dim=self.hidden_dim,
        ).to(self._device)

        self.readout_layer_ = _build_readout(
            hidden_dim=self.hidden_dim, output_dim=1,
        ).to(self._device)

        # --- optimiser ---
        all_params = (
            list(self.cde_func_.parameters())
            + list(self.initial_net_.parameters())
            + list(self.readout_layer_.parameters())
        )
        optimizer = optim.Adam(all_params, lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=200, factor=0.5, min_lr=1e-6,
        )
        criterion = torch.nn.MSELoss()

        # --- subsequence indices (single or multi-dataset) ---
        dataset_windows = []
        for x_ds in x_datasets:
            n_total_ds = int(x_ds.shape[1])
            if n_total_ds <= 0:
                continue
            seq_len_ds = min(self.sequence_length, n_total_ds)
            n_windows_ds = max(1, n_total_ds - seq_len_ds)
            dataset_windows.append((x_ds, seq_len_ds, n_windows_ds))

        if not dataset_windows:
            raise ValueError("No non-empty datasets available for NeuralCDE training")

        ds_weights = np.asarray([n_windows for _, _, n_windows in dataset_windows], dtype=float)
        ds_weights = ds_weights / np.sum(ds_weights)
        windows_per_epoch = max(
            1,
            int(
                sum(
                    max(1, int(x_ds.shape[1]) // max(1, seq_len_ds))
                    for x_ds, seq_len_ds, _ in dataset_windows
                )
            ),
        )
        windows_per_epoch = min(windows_per_epoch, self.windows_per_epoch)

        # --- best model checkpointing ---
        import copy
        best_loss = float("inf")
        best_state = None

        # --- training loop ---
        epoch_iter = range(self.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training NeuralCDE", unit="epoch")

        self.training_loss_ = []
        for epoch in epoch_iter:
            self.cde_func_.train()
            self.initial_net_.train()
            self.readout_layer_.train()

            epoch_loss = 0.0
            for _ in range(windows_per_epoch):
                # Random dataset + random start window
                ds_idx = int(np.random.choice(len(dataset_windows), p=ds_weights))
                x_source, seq_len, n_windows = dataset_windows[ds_idx]
                start = int(torch.randint(0, max(1, n_windows), (1,)).item())
                end = start + seq_len

                x_window = x_source[:, start:end, :].clone()

                # Re-normalise time within the window to [0, 1]
                x_window[:, :, 0] = torch.linspace(
                    0, 1, x_window.shape[1],
                    dtype=self._dtype, device=self._device,
                )

                X = self._create_interpolation_from_tensor(x_window, self.interpolation)
                y_true = x_window[0, :, 2:3]  # (seq_len, 1) — normalised y

                optimizer.zero_grad()

                _, y_pred = self._solve_cde(X, adjoint=False)
                # y_pred: (1, seq_len, 1)
                loss = criterion(y_pred[0], y_true)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / windows_per_epoch
            scheduler.step(avg_loss)
            self.training_loss_.append(avg_loss)

            # Checkpoint best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {
                    "cde_func": copy.deepcopy(self.cde_func_.state_dict()),
                    "initial_net": copy.deepcopy(self.initial_net_.state_dict()),
                    "readout": copy.deepcopy(self.readout_layer_.state_dict()),
                }

            if verbose and isinstance(epoch_iter, tqdm):
                epoch_iter.set_postfix({"loss": f"{avg_loss:.6f}", "best": f"{best_loss:.6f}"})

            if wandb_run is not None and (epoch + 1) % wandb_log_every == 0:
                wandb_run.log({"train/loss": avg_loss, "train/epoch": epoch + 1})

        # Restore best model
        if best_state is not None:
            self.cde_func_.load_state_dict(best_state["cde_func"])
            self.initial_net_.load_state_dict(best_state["initial_net"])
            self.readout_layer_.load_state_dict(best_state["readout"])

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _predict_full(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve CDE over a full (u, y) sequence, return denormalised predictions."""
        import torch

        x = self._normalise_and_build_tensor(u, y, fit_stats=False)
        X = self._create_interpolation_from_tensor(x, self.interpolation)

        self.cde_func_.eval()
        self.initial_net_.eval()
        self.readout_layer_.eval()

        with torch.no_grad():
            _, y_pred = self._solve_cde(X, adjoint=False)
        # y_pred: (1, n, 1) — normalised
        y_out = y_pred[0, :, 0].cpu().numpy() * self._y_std + self._y_mean
        return y_out

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction (uses true y in the driving path)."""
        preds = self._predict_full(u, y)
        return preds[1:]  # shifted by one

    def predict_free_run(
        self,
        u: np.ndarray,
        y_initial: np.ndarray,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Free-run simulation: y is generated autoregressively.

        The model feeds its own predictions back as the y channel,
        keeping only the measured u and time.
        """
        import torch

        del show_progress

        u = np.asarray(u, dtype=np.float64).reshape(-1, 1)
        y_init = np.asarray(y_initial, dtype=np.float64).flatten()
        n = len(u)

        self.cde_func_.eval()
        self.initial_net_.eval()
        self.readout_layer_.eval()

        y_sim = np.zeros(n)
        y_sim[0] = y_init[0]

        # We propagate the hidden state across the whole sequence instead of
        # resetting z0 at every step.  We process small windows and carry the
        # last hidden state forward.

        with torch.no_grad():
            # Use the initial observation to get z0
            u_norm_0 = (u[0, 0] - self._u_mean) / self._u_std
            y_norm_0 = (y_sim[0] - self._y_mean) / self._y_std
            x0_vec = torch.tensor(
                [[0.0, u_norm_0, y_norm_0]],
                dtype=self._dtype, device=self._device,
            )
            z = self.initial_net_(x0_vec)  # (1, hidden_dim)

            for i in range(n - 1):
                # Build 2-point path for one step
                u_pair = np.array([u[i, 0], u[min(i + 1, n - 1), 0]]).reshape(-1, 1)
                y_pair = np.array([y_sim[i], y_sim[i]]).reshape(-1, 1)

                u_norm = (u_pair - self._u_mean) / self._u_std
                y_norm = (y_pair - self._y_mean) / self._y_std
                t_pair = np.array([[0.0], [1.0]])

                data = np.concatenate([t_pair, u_norm, y_norm], axis=1)[np.newaxis]
                x_t = torch.tensor(data, dtype=self._dtype, device=self._device)
                X = self._create_interpolation_from_tensor(x_t, self.interpolation)

                import torchcde

                extra = {}
                if self.solver in ("rk4", "euler", "midpoint"):
                    extra["options"] = {"step_size": 1.0}

                z_out = torchcde.cdeint(
                    X=X, func=self.cde_func_, z0=z,
                    t=X.interval,
                    method=self.solver,
                    rtol=self.rtol, atol=self.atol,
                    adjoint=False,
                    **extra,
                )
                # z_out: (1, 2, hidden_dim) — take the terminal state
                z = z_out[:, -1, :]  # carry hidden state forward

                y_pred_norm = self.readout_layer_(z)  # (1, 1)
                y_sim[i + 1] = float(
                    y_pred_norm.cpu().numpy().flatten()[0] * self._y_std + self._y_mean
                )

        return y_sim

    def __repr__(self) -> str:
        return (
            f"NeuralCDE(hidden_dim={self.hidden_dim}, "
            f"input_dim={self.input_dim}, "
            f"hidden_layers={self.hidden_layers}, "
            f"interpolation='{self.interpolation}', "
            f"solver='{self.solver}')"
        )
