"""NARX (Nonlinear AutoRegressive with eXogenous inputs) model."""

from itertools import combinations_with_replacement
from typing import Dict, List, Optional
import numpy as np
from tqdm.auto import tqdm

from .base import BaseModel
from ..utils.regression import reg_mat_narx
from ..utils.frols import frols


class NARX(BaseModel):
    """
    NARX model with polynomial basis functions and FROLS term selection.
    
    Args:
        nu: Number of input lags
        ny: Number of output lags
        poly_order: Polynomial order for nonlinear terms
        selection_criteria: ERR threshold (0-1) or number of terms (int)
    """

    def __init__(
        self,
        nu: int = 1,
        ny: int = 1,
        poly_order: int = 2,
        selection_criteria: float = 0.01,
    ):
        super().__init__(nu=nu, ny=ny)
        self.poly_order = poly_order
        self.selection_criteria = selection_criteria

        # Fitted parameters
        self.theta_: Optional[np.ndarray] = None
        self.selected_colnames_: List[str] = []
        self.selected_indices_: List[int] = []
        self.candidate_colnames_: List[str] = []
        self.fit_results_: Optional[Dict] = None

        # Base column names for reconstruction
        self._p0_colnames: List[str] = []
        if ny > 0:
            self._p0_colnames.extend([f"y(k-{i})" for i in range(1, ny + 1)])
        if nu > 0:
            self._p0_colnames.extend([f"u(k-{i})" for i in range(1, nu + 1)])
        self._n_base = len(self._p0_colnames)

    def fit(self, u: np.ndarray, y: np.ndarray) -> "NARX":
        """Fit NARX model using FROLS algorithm."""
        u = np.asarray(u, dtype=float)
        y = np.asarray(y, dtype=float)

        # Build candidate matrix
        P, self.candidate_colnames_, target = reg_mat_narx(
            u, y, self.nu, self.ny, self.poly_order
        )

        # Run FROLS
        results = frols(P, target, self.selection_criteria, self.candidate_colnames_)

        self.theta_ = results["theta"]
        self.selected_colnames_ = results["selected_colnames"]
        self.selected_indices_ = results["selected_indices"]
        self.fit_results_ = results
        self._is_fitted = len(self.theta_) > 0

        if not self._is_fitted:
            print("Warning: FROLS did not select any terms.")

        return self

    def _build_candidate_row(
        self, y_lags: List[float], u_lags: List[float]
    ) -> np.ndarray:
        """Build single row of candidate regressors for free-run prediction."""
        p0_values = []
        if self.ny > 0:
            p0_values.extend(y_lags)
        if self.nu > 0:
            p0_values.extend(u_lags)

        terms = {"constant": 1.0}
        for i, name in enumerate(self._p0_colnames):
            terms[name] = p0_values[i]

        # Polynomial terms
        if self.poly_order >= 2 and self._n_base > 0:
            for order in range(2, self.poly_order + 1):
                for indices in combinations_with_replacement(range(self._n_base), order):
                    name = "".join(self._p0_colnames[j] for j in indices)
                    terms[name] = np.prod([p0_values[j] for j in indices])

        return np.array([terms.get(n, 0.0) for n in self.candidate_colnames_])

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction using actual past outputs."""
        u = np.asarray(u, dtype=float)
        y = np.asarray(y, dtype=float)

        P, _, _ = reg_mat_narx(u, y, self.nu, self.ny, self.poly_order)
        if P.shape[0] == 0:
            return np.array([])

        P_sel = P[:, self.selected_indices_]
        return P_sel @ self.theta_

    def predict_free_run(
        self, u: np.ndarray, y_initial: np.ndarray, show_progress: bool = True
    ) -> np.ndarray:
        """Free-run simulation using predicted outputs recursively."""
        u = np.asarray(u, dtype=float)
        y_init = np.asarray(y_initial, dtype=float)

        if len(y_init) < self.max_lag:
            raise ValueError(
                f"Need {self.max_lag} initial conditions, got {len(y_init)}"
            )

        n_total = len(u)
        y_hat = np.zeros(n_total)
        y_hat[: self.max_lag] = y_init[: self.max_lag]

        sim_range = range(self.max_lag, n_total)
        if show_progress:
            sim_range = tqdm(sim_range, desc="Free-run simulation", unit="step")

        for k in sim_range:
            y_lags = [y_hat[k - j] for j in range(1, self.ny + 1)] if self.ny > 0 else []
            u_lags = [u[k - j] for j in range(1, self.nu + 1)] if self.nu > 0 else []

            candidate_row = self._build_candidate_row(y_lags, u_lags)
            selected = candidate_row[self.selected_indices_]
            y_hat[k] = selected @ self.theta_

        return y_hat[self.max_lag:]

    def summary(self) -> str:
        """Print model summary."""
        if not self._is_fitted:
            return "Model not fitted"

        lines = [
            f"NARX Model (ny={self.ny}, nu={self.nu}, poly_order={self.poly_order})",
            "-" * 50,
            "Selected terms and parameters:",
        ]

        for i, (name, theta) in enumerate(zip(self.selected_colnames_, self.theta_)):
            err = self.fit_results_["err_values"][i] * 100
            lines.append(f"  {i+1}. {name}: {theta:.6f} (ERR: {err:.4f}%)")

        total_err = self.fit_results_["err_values"].sum() * 100
        lines.append(f"\nTotal ERR: {total_err:.4f}%")
        
        summary = "\n".join(lines)
        print(summary)
        return summary

    def __repr__(self) -> str:
        return (
            f"NARX(nu={self.nu}, ny={self.ny}, "
            f"poly_order={self.poly_order}, "
            f"selection_criteria={self.selection_criteria})"
        )
