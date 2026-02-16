"""Neural SDE model with learned drift and diffusion."""

from __future__ import annotations

from itertools import chain
from typing import Any, Dict, List

import numpy as np

from ..config import NeuralSDEConfig
from .base import BaseModel
from .torchsde_utils import (
    ControlledPathMixin,
    simulate_controlled_sde,
    train_sequence_batches,
    _euler_ode_integrate,
)


# ── internal SDE dynamics ─────────────────────────────────────────────

class _ControlledNeuralSDEFunc(ControlledPathMixin):
    """SDE function with piecewise-constant control input."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, state_dim, input_dim, hidden_layers, diffusion_hidden_layers, dt):
        import torch

        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)

        self.drift_net = self._build_net(
            self.state_dim + self.input_dim, self.state_dim, hidden_layers, 0.0)
        self.diffusion_net = self._build_net(
            self.state_dim + self.input_dim, self.state_dim, diffusion_hidden_layers, -3.0)
        self._init_control_path(dt, self.input_dim, torch.device("cpu"), torch.float32)

    @staticmethod
    def _build_net(in_sz, out_sz, hidden, final_bias):
        import torch.nn as nn
        layers, prev = [], in_sz
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, out_sz))
        net = nn.Sequential(*layers)
        lins = [m for m in net.modules() if isinstance(m, nn.Linear)]
        for i, m in enumerate(lins):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, final_bias if i == len(lins) - 1 else 0.0)
        return net

    def to(self, device):
        self.drift_net = self.drift_net.to(device)
        self.diffusion_net = self.diffusion_net.to(device)
        self._u_path = self._u_path.to(device)
        return self

    def parameters(self):
        return chain(self.drift_net.parameters(), self.diffusion_net.parameters())

    def train(self):
        self.drift_net.train(); self.diffusion_net.train()

    def eval(self):
        self.drift_net.eval(); self.diffusion_net.eval()

    def f(self, t, y):
        import torch
        u_t = self._u_at(t, y.shape[0])
        return self.drift_net(torch.cat([y, u_t], dim=-1))

    def g(self, t, y):
        import torch, torch.nn.functional as F
        u_t = self._u_at(t, y.shape[0])
        return F.softplus(self.diffusion_net(torch.cat([y, u_t], dim=-1))) + 1e-6


# ── public model ──────────────────────────────────────────────────────

class NeuralSDE(BaseModel):
    """Neural SDE: dx = f_θ(x,u)dt + g_φ(x,u)dW_t."""

    _SUPPORTED_SOLVERS = {"euler", "milstein", "srk"}

    def __init__(self, config: NeuralSDEConfig | None = None, **kwargs):
        if config is None:
            config = NeuralSDEConfig(**kwargs)
        super().__init__(config)
        c = self.config
        self.state_dim = c.state_dim
        self.input_dim = c.input_dim
        self.dt = c.dt
        self.solver = c.solver
        if self.solver not in self._SUPPORTED_SOLVERS:
            raise ValueError(f"Unknown solver: {self.solver}. Use {sorted(self._SUPPORTED_SOLVERS)}")
        self.sde_func_ = None
        self._device = None
        self._dtype = None

    def _build_sde_func(self):
        c = self.config
        self.sde_func_ = _ControlledNeuralSDEFunc(
            state_dim=c.state_dim, input_dim=c.input_dim,
            hidden_layers=c.hidden_layers,
            diffusion_hidden_layers=c.diffusion_hidden_layers,
            dt=c.dt,
        ).to(self._device)

    def _simulate_trajectory(self, u_path, x0, deterministic=False):
        if deterministic:
            self.sde_func_.set_control(u_path)
            return _euler_ode_integrate(self.sde_func_, u_path, x0, self.dt, self.dt)
        return simulate_controlled_sde(self.sde_func_, u_path, x0, self.dt, self.solver)

    # ── training ──────────────────────────────────────────────────────

    def _fit(self, u, y, *, val_data=None, logger=None):
        import torch
        c = self.config
        self._device = self._resolve_torch_device()
        self._dtype = torch.float32
        self._build_sde_func()

        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y = np.asarray(y, dtype=float).reshape(-1, self.state_dim)

        self.training_loss_ = train_sequence_batches(
            sde_func=self.sde_func_,
            simulate_fn=lambda u_seq, x0: self._simulate_trajectory(u_seq, x0),
            u=u, y=y,
            input_dim=self.input_dim, state_dim=self.state_dim,
            sequence_length=c.sequence_length, epochs=c.epochs,
            learning_rate=c.learning_rate,
            device=self._device, dtype=self._dtype,
            verbose=c.verbose, progress_desc="Training NeuralSDE",
            logger=logger, log_every=c.wandb_log_every,
            sequences_per_epoch=c.sequences_per_epoch,
            early_stopping_patience=c.early_stopping_patience,
        )

    # ── predict ───────────────────────────────────────────────────────

    def predict_osa(self, u, y):
        import torch
        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y = np.asarray(y, dtype=float).reshape(-1, self.state_dim)
        preds = []
        self.sde_func_.eval()
        with torch.no_grad():
            for t in range(len(y) - 1):
                x_t = torch.tensor(y[t], dtype=self._dtype, device=self._device)
                u_t = torch.tensor(u[t:t+1], dtype=self._dtype, device=self._device)
                u_p = torch.cat([u_t, u_t], dim=0)
                x_next = self._simulate_trajectory(u_p, x_t, deterministic=True)[-1]
                preds.append(x_next.cpu().numpy())
        return np.asarray(preds).reshape(-1)

    def predict_free_run(self, u, y_initial):
        import torch
        u = np.asarray(u, dtype=float).reshape(-1, self.input_dim)
        y0 = np.asarray(y_initial, dtype=float).reshape(-1, self.state_dim)
        self.sde_func_.eval()
        with torch.no_grad():
            u_t = torch.tensor(u, dtype=self._dtype, device=self._device)
            x0 = torch.tensor(y0[0], dtype=self._dtype, device=self._device)
            pred = self._simulate_trajectory(u_t, x0, deterministic=True)
        return pred.cpu().numpy().reshape(-1)

    # ── save / load hooks ─────────────────────────────────────────────

    def _collect_state(self) -> Dict[str, Any]:
        if self.sde_func_ is None:
            return {}
        return {
            "drift_net": self.sde_func_.drift_net.state_dict(),
            "diffusion_net": self.sde_func_.diffusion_net.state_dict(),
        }

    def _restore_state(self, state):
        self.sde_func_.drift_net.load_state_dict(state["drift_net"])
        self.sde_func_.diffusion_net.load_state_dict(state["diffusion_net"])

    def _build_for_load(self):
        import torch
        self._device = self._resolve_torch_device("cpu")
        self._dtype = torch.float32
        self._build_sde_func()

    def __repr__(self):
        c = self.config
        return (f"NeuralSDE(state={c.state_dim}, input={c.input_dim}, "
                f"hidden={c.hidden_layers}, solver='{c.solver}')")
