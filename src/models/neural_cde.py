"""Neural Controlled Differential Equation (CDE) model.

Architecture from Kidger et al., "Neural CDEs for Irregular Time Series",
NeurIPS 2020.  Reference: https://github.com/patrick-kidger/torchcde
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import BaseModel, resolve_device, DEFAULT_GRAD_CLIP


# ── helper builders ───────────────────────────────────────────────────

def _build_cde_func(hidden_dim: int, input_channels: int, hidden_layers: list):
    """CDE vector-field f_θ: (batch, hidden) -> (batch, hidden, channels)."""
    import torch.nn as nn

    class CDEFunc(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = int(hidden_dim)
            self.input_channels = int(input_channels)
            layers: list[nn.Module] = []
            prev = self.hidden_dim
            for h in hidden_layers:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers += [nn.Linear(prev, self.hidden_dim * self.input_channels), nn.Tanh()]
            self.net = nn.Sequential(*layers)
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, t, z):
            return self.net(z).view(*z.shape[:-1], self.hidden_dim, self.input_channels)

    return CDEFunc()


def _build_initial_network(input_channels: int, hidden_dim: int):
    import torch.nn as nn
    net = nn.Linear(input_channels, hidden_dim)
    nn.init.xavier_uniform_(net.weight); nn.init.zeros_(net.bias)
    return net


def _build_readout(hidden_dim: int, output_dim: int = 1):
    import torch.nn as nn
    net = nn.Linear(hidden_dim, output_dim)
    nn.init.xavier_uniform_(net.weight); nn.init.zeros_(net.bias)
    return net


# ── public model ──────────────────────────────────────────────────────

class NeuralCDE(BaseModel):
    """Neural CDE: dz(t) = f_θ(z(t)) dX(t), z(t₀) = initial(X(t₀))."""

    _VALID_SOLVERS = {"dopri5", "rk4", "euler", "midpoint"}
    _VALID_INTERPOLATIONS = {"cubic", "linear"}

    def __init__(self, config=None):
        from ..config import NeuralCDEConfig
        if config is None:
            config = NeuralCDEConfig()
        super().__init__(config)
        c = self.config
        if c.solver not in self._VALID_SOLVERS:
            raise ValueError(f"Unknown solver: {c.solver}")
        if c.interpolation not in self._VALID_INTERPOLATIONS:
            raise ValueError(f"Unknown interpolation: {c.interpolation}")

        self.cde_func_ = None
        self.initial_net_ = None
        self.readout_layer_ = None
        self._device = None
        self._dtype = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._u_mean = 0.0
        self._u_std = 1.0

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _create_interpolation(x, interpolation: str):
        import torchcde
        if interpolation == "cubic":
            coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)
            return torchcde.CubicSpline(coeffs)
        coeffs = torchcde.linear_interpolation_coeffs(x)
        return torchcde.LinearInterpolation(coeffs)

    def _norm_tensor(self, u, y, *, fit_stats=False):
        """Build (1, T, 3) tensor [t, u_norm, y_norm]."""
        import torch
        u = np.asarray(u, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if fit_stats:
            self._u_mean, self._u_std = float(u.mean()), float(u.std()) + 1e-8
            self._y_mean, self._y_std = float(y.mean()), float(y.std()) + 1e-8
        data = np.concatenate([
            np.linspace(0, 1, len(u)).reshape(-1, 1),
            (u - self._u_mean) / self._u_std,
            (y - self._y_mean) / self._y_std,
        ], axis=1)
        return torch.tensor(data[np.newaxis], dtype=self._dtype, device=self._device)

    def _solve_cde(self, X, adjoint=False):
        import torchcde
        c = self.config
        z0 = self.initial_net_(X.evaluate(X.interval[0]))
        extra = {}
        if c.solver in ("rk4", "euler", "midpoint"):
            step = (X.grid_points[1:] - X.grid_points[:-1]).min().item()
            extra["options"] = {"step_size": step}
        z_path = torchcde.cdeint(
            X=X, func=self.cde_func_, z0=z0, t=X.grid_points,
            method=c.solver, rtol=c.rtol, atol=c.atol, adjoint=adjoint, **extra)
        return z_path, self.readout_layer_(z_path)

    def _build_networks(self):
        c = self.config
        self.cde_func_ = _build_cde_func(
            c.hidden_dim, 3, c.hidden_layers).to(self._device)
        self.initial_net_ = _build_initial_network(3, c.hidden_dim).to(self._device)
        self.readout_layer_ = _build_readout(c.hidden_dim, 1).to(self._device)

    # ── training ──────────────────────────────────────────────────────

    def _fit(self, u, y, *, val_data=None, logger=None):
        import torch, torch.optim as optim, copy
        from tqdm.auto import tqdm
        c = self.config

        self._device = resolve_device(c.device)
        self._dtype = torch.float32

        u = np.asarray(u, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()
        x_data = self._norm_tensor(u, y, fit_stats=True)

        self._build_networks()
        all_params = (list(self.cde_func_.parameters())
                      + list(self.initial_net_.parameters())
                      + list(self.readout_layer_.parameters()))
        optimizer = optim.Adam(all_params, lr=c.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=200, factor=0.5, min_lr=1e-6)
        criterion = torch.nn.MSELoss()

        n_total = int(x_data.shape[1])
        seq_len = min(c.sequence_length, n_total)
        n_windows = max(1, n_total - seq_len)
        wpe = min(c.sequences_per_epoch, max(1, n_total // max(1, seq_len)))

        best_loss, best_state = float("inf"), None
        bad_epochs = 0
        it = range(c.epochs)
        if c.verbose:
            it = tqdm(it, desc="Training NeuralCDE", unit="epoch")

        self.training_loss_ = []
        for epoch in it:
            self.cde_func_.train(); self.initial_net_.train(); self.readout_layer_.train()
            epoch_loss = 0.0
            for _ in range(wpe):
                start = int(torch.randint(0, max(1, n_windows), (1,)).item())
                x_w = x_data[:, start:start + seq_len, :].clone()
                x_w[:, :, 0] = torch.linspace(0, 1, x_w.shape[1],
                                               dtype=self._dtype, device=self._device)
                X = self._create_interpolation(x_w, c.interpolation)
                y_true = x_w[0, :, 2:3]
                optimizer.zero_grad()
                _, y_pred = self._solve_cde(X)
                loss = criterion(y_pred[0], y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, DEFAULT_GRAD_CLIP)
                optimizer.step()
                epoch_loss += loss.item()

            avg = epoch_loss / wpe
            scheduler.step(avg)
            self.training_loss_.append(avg)

            if avg < best_loss:
                best_loss = avg
                best_state = {
                    "cde": copy.deepcopy(self.cde_func_.state_dict()),
                    "init": copy.deepcopy(self.initial_net_.state_dict()),
                    "read": copy.deepcopy(self.readout_layer_.state_dict()),
                }
                bad_epochs = 0
            else:
                bad_epochs += 1

            if c.verbose and hasattr(it, "set_postfix"):
                it.set_postfix(loss=f"{avg:.6f}", best=f"{best_loss:.6f}")
            if logger and c.wandb_log_every > 0 and (epoch + 1) % c.wandb_log_every == 0:
                logger.log_metrics({"train/loss": avg, "train/epoch": epoch + 1}, step=epoch + 1)
            if c.early_stopping_patience and bad_epochs >= c.early_stopping_patience:
                if c.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state:
            self.cde_func_.load_state_dict(best_state["cde"])
            self.initial_net_.load_state_dict(best_state["init"])
            self.readout_layer_.load_state_dict(best_state["read"])

    # ── predict ───────────────────────────────────────────────────────

    def _predict_full(self, u, y):
        import torch
        x = self._norm_tensor(u, y)
        X = self._create_interpolation(x, self.config.interpolation)
        self.cde_func_.eval(); self.initial_net_.eval(); self.readout_layer_.eval()
        with torch.no_grad():
            _, yp = self._solve_cde(X)
        return yp[0, :, 0].cpu().numpy() * self._y_std + self._y_mean

    def predict_osa(self, u, y):
        return self._predict_full(u, y)[1:]

    def predict_free_run(self, u, y_initial):
        import torch, torchcde
        c = self.config
        u = np.asarray(u, dtype=float).reshape(-1, 1)
        y0 = np.asarray(y_initial, dtype=float).flatten()
        n = len(u)

        self.cde_func_.eval(); self.initial_net_.eval(); self.readout_layer_.eval()
        y_sim = np.zeros(n)
        y_sim[0] = y0[0]

        with torch.no_grad():
            u0n = (u[0, 0] - self._u_mean) / self._u_std
            y0n = (y_sim[0] - self._y_mean) / self._y_std
            x0v = torch.tensor([[0.0, u0n, y0n]], dtype=self._dtype, device=self._device)
            z = self.initial_net_(x0v)

            for i in range(n - 1):
                up = np.array([u[i, 0], u[min(i + 1, n - 1), 0]]).reshape(-1, 1)
                yp = np.array([y_sim[i], y_sim[i]]).reshape(-1, 1)
                data = np.concatenate([
                    np.array([[0.0], [1.0]]),
                    (up - self._u_mean) / self._u_std,
                    (yp - self._y_mean) / self._y_std,
                ], axis=1)[np.newaxis]
                x_t = torch.tensor(data, dtype=self._dtype, device=self._device)
                X = self._create_interpolation(x_t, c.interpolation)
                extra = {}
                if c.solver in ("rk4", "euler", "midpoint"):
                    extra["options"] = {"step_size": 1.0}
                z_out = torchcde.cdeint(
                    X=X, func=self.cde_func_, z0=z, t=X.interval,
                    method=c.solver, rtol=c.rtol, atol=c.atol,
                    adjoint=False, **extra)
                z = z_out[:, -1, :]
                ypn = self.readout_layer_(z)
                y_sim[i + 1] = float(ypn.cpu().numpy().flatten()[0]) * self._y_std + self._y_mean

        return y_sim

    # ── save / load hooks ─────────────────────────────────────────────

    def _collect_state(self) -> Dict[str, Any]:
        if self.cde_func_ is None:
            return {}
        return {
            "cde_func": self.cde_func_.state_dict(),
            "initial_net": self.initial_net_.state_dict(),
            "readout": self.readout_layer_.state_dict(),
        }

    def _restore_state(self, state):
        self.cde_func_.load_state_dict(state["cde_func"])
        self.initial_net_.load_state_dict(state["initial_net"])
        self.readout_layer_.load_state_dict(state["readout"])

    def _collect_extra_state(self) -> Dict[str, Any]:
        return {"y_mean": self._y_mean, "y_std": self._y_std,
                "u_mean": self._u_mean, "u_std": self._u_std}

    def _restore_extra_state(self, extra):
        self._y_mean = extra["y_mean"]; self._y_std = extra["y_std"]
        self._u_mean = extra["u_mean"]; self._u_std = extra["u_std"]

    def _build_for_load(self):
        import torch
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._build_networks()

    def __repr__(self):
        c = self.config
        return (f"NeuralCDE(hidden={c.hidden_dim}, input={c.input_dim}, "
                f"solver='{c.solver}', interp='{c.interpolation}')")
