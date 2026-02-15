"""NARX (Nonlinear AutoRegressive with eXogenous inputs) model."""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import NARXConfig
from ..utils.frols import frols
from ..utils.regression import reg_mat_narx
from .base import BaseModel


class NARX(BaseModel):
    """NARX with polynomial basis functions and FROLS term selection."""

    def __init__(self, config: NARXConfig | None = None, **kwargs):
        if config is None:
            config = NARXConfig(**kwargs)
        super().__init__(config)

        self.theta_: Optional[np.ndarray] = None
        self.selected_colnames_: List[str] = []
        self.selected_indices_: List[int] = []
        self.candidate_colnames_: List[str] = []
        self.fit_results_: Optional[Dict] = None

        # Base column names for free-run reconstruction
        self._p0_colnames: List[str] = []
        if self.ny > 0:
            self._p0_colnames.extend([f"y(k-{i})" for i in range(1, self.ny + 1)])
        if self.nu > 0:
            self._p0_colnames.extend([f"u(k-{i})" for i in range(1, self.nu + 1)])
        self._n_base = len(self._p0_colnames)

    # ── training ──────────────────────────────────────────────────────

    def _fit(
        self,
        u: np.ndarray,
        y: np.ndarray,
        *,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        logger: Any = None,
    ) -> None:
        cfg = self.config
        P, self.candidate_colnames_, target = reg_mat_narx(
            u, y, self.nu, self.ny, cfg.poly_order
        )
        results = frols(P, target, cfg.selection_criteria, self.candidate_colnames_)

        self.theta_ = results["theta"]
        self.selected_colnames_ = results["selected_colnames"]
        self.selected_indices_ = results["selected_indices"]
        self.fit_results_ = results

        if len(self.theta_) == 0:
            raise RuntimeError("FROLS did not select any terms.")

    # ── prediction ────────────────────────────────────────────────────

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        P, _, _ = reg_mat_narx(u, y, self.nu, self.ny, self.config.poly_order)
        if P.shape[0] == 0:
            return np.array([])
        return P[:, self.selected_indices_] @ self.theta_

    def predict_free_run(self, u: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        y_init = np.asarray(y_initial, dtype=float)
        if len(y_init) < self.max_lag:
            raise ValueError(f"Need {self.max_lag} initial conditions, got {len(y_init)}")

        n_total = len(u)
        y_hat = np.zeros(n_total)
        y_hat[: self.max_lag] = y_init[: self.max_lag]

        for k in range(self.max_lag, n_total):
            y_lags = [y_hat[k - j] for j in range(1, self.ny + 1)] if self.ny > 0 else []
            u_lags = [u[k - j] for j in range(1, self.nu + 1)] if self.nu > 0 else []
            row = self._build_candidate_row(y_lags, u_lags)
            y_hat[k] = row[self.selected_indices_] @ self.theta_

        return y_hat[self.max_lag:]

    def _build_candidate_row(self, y_lags: List[float], u_lags: List[float]) -> np.ndarray:
        p0_values = list(y_lags) + list(u_lags)
        terms = {"constant": 1.0}
        for i, name in enumerate(self._p0_colnames):
            terms[name] = p0_values[i]
        if self.config.poly_order >= 2 and self._n_base > 0:
            for order in range(2, self.config.poly_order + 1):
                for indices in combinations_with_replacement(range(self._n_base), order):
                    name = "".join(self._p0_colnames[j] for j in indices)
                    terms[name] = np.prod([p0_values[j] for j in indices])
        return np.array([terms.get(n, 0.0) for n in self.candidate_colnames_])

    # ── save / load hooks ─────────────────────────────────────────────

    def _collect_extra_state(self) -> Dict[str, Any]:
        return {
            "theta": self.theta_,
            "selected_colnames": self.selected_colnames_,
            "selected_indices": self.selected_indices_,
            "candidate_colnames": self.candidate_colnames_,
            "fit_results": self.fit_results_,
            "p0_colnames": self._p0_colnames,
            "n_base": self._n_base,
        }

    def _restore_extra_state(self, extra: Dict[str, Any]) -> None:
        self.theta_ = extra["theta"]
        self.selected_colnames_ = extra["selected_colnames"]
        self.selected_indices_ = extra["selected_indices"]
        self.candidate_colnames_ = extra["candidate_colnames"]
        self.fit_results_ = extra["fit_results"]
        self._p0_colnames = extra["p0_colnames"]
        self._n_base = extra["n_base"]
