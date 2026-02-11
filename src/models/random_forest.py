"""Random Forest model for time series forecasting."""

import contextlib
import importlib
import io
import sys
from typing import Optional
import numpy as np
from tqdm.auto import tqdm

from .base import BaseModel
from ._dependency_utils import is_binary_incompatibility_error
from ..utils.regression import create_lagged_features


_MISSING = object()


def _binary_incompatibility_message() -> str:
    return (
        "Binary incompatibility between NumPy and compiled dependencies "
        "(scikit-learn/pandas/pyarrow). Reinstall compatible versions, e.g.:\n"
        "  python -m pip install --upgrade --force-reinstall "
        "\"numpy<2\" pandas pyarrow scikit-learn"
    )


def _restore_module(name: str, previous):
    if previous is _MISSING:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


def _import_random_forest_regressor():
    """Import sklearn RandomForestRegressor with a pandas-free fallback."""
    def _safe_import(root_exc: BaseException | None = None):
        # Force sklearn to treat pandas/pyarrow as unavailable so optional
        # imports do not crash on ABI-mismatched wheels.
        previous_pandas = sys.modules.get("pandas", _MISSING)
        previous_pyarrow = sys.modules.get("pyarrow", _MISSING)
        for modname in list(sys.modules):
            if modname == "sklearn" or modname.startswith("sklearn."):
                del sys.modules[modname]

        sys.modules["pandas"] = None
        sys.modules["pyarrow"] = None
        try:
            ensemble = importlib.import_module("sklearn.ensemble")
            return (
                ensemble.RandomForestRegressor,
                "Warning: loaded scikit-learn without pandas/pyarrow due to NumPy ABI mismatch.",
            )
        except Exception as exc:
            if is_binary_incompatibility_error(exc):
                if root_exc is None:
                    root_exc = exc
                raise RuntimeError(_binary_incompatibility_message()) from root_exc
            raise RuntimeError(
                "Failed to import scikit-learn RandomForestRegressor."
            ) from exc
        finally:
            _restore_module("pandas", previous_pandas)
            _restore_module("pyarrow", previous_pyarrow)

    # With NumPy>=2 environments, prefer the safe import path directly to
    # avoid optional pandas/pyarrow ABI crashes during sklearn import.
    try:
        numpy_major = int(str(np.__version__).split(".", maxsplit=1)[0])
    except Exception:
        numpy_major = 0
    if numpy_major >= 2:
        return _safe_import()

    try:
        # NumPy may dump ABI traceback to stderr before raising; keep notebook output clean.
        with contextlib.redirect_stderr(io.StringIO()):
            from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor, None
    except ImportError as exc:
        if not is_binary_incompatibility_error(exc):
            raise ImportError(
                "scikit-learn required. Install with: pip install scikit-learn"
            ) from exc
        root_exc = exc
    except Exception as exc:
        if not is_binary_incompatibility_error(exc):
            raise
        root_exc = exc

    return _safe_import(root_exc=root_exc)


class RandomForest(BaseModel):
    """
    Random Forest Regressor for NARX-like system identification.
    
    Uses sklearn RandomForestRegressor with lagged features.
    
    Args:
        nu: Number of input lags
        ny: Number of output lags
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None for unlimited)
        min_samples_split: Minimum samples required to split
        min_samples_leaf: Minimum samples required at leaf
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        nu: int = 5,
        ny: int = 5,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        max_lag: int | None = None,
    ):
        if max_lag is not None:
            max_lag = int(max_lag)
            if max_lag < 0:
                raise ValueError("max_lag must be non-negative")
            if (nu != 5 or ny != 5) and (nu != max_lag or ny != max_lag):
                raise ValueError("Use either max_lag or nu/ny, not conflicting values")
            nu = max_lag
            ny = max_lag

        super().__init__(nu=nu, ny=ny)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model_ = None

    def fit(self, u: np.ndarray, y: np.ndarray, verbose: bool = True) -> "RandomForest":
        """Train the Random Forest model."""
        RandomForestRegressor, import_warning = _import_random_forest_regressor()
        if verbose and import_warning:
            print(import_warning)

        # Prepare data with lagged features
        features, target = create_lagged_features(y, u, self.ny, self.nu)
        
        if len(target) == 0:
            raise ValueError("Not enough data for given lag orders")

        # Build and train model
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1 if verbose else 0,
        )
        
        if verbose:
            print(f"Training RandomForest with {features.shape[0]} samples...")
        
        self.model_.fit(features, target)
        self._is_fitted = True
        
        return self

    def predict_osa(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction using actual past outputs."""
        features, _ = create_lagged_features(y, u, self.ny, self.nu)
        return self.model_.predict(features)

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
            sim_range = tqdm(sim_range, desc="RF Free-run simulation", unit="step")

        for k in sim_range:
            # Build feature vector from past predictions and inputs
            y_lags = [y_hat[k - j] for j in range(1, self.ny + 1)] if self.ny > 0 else []
            u_lags = [u[k - j] for j in range(1, self.nu + 1)] if self.nu > 0 else []
            
            features = np.array(y_lags + u_lags).reshape(1, -1)
            y_hat[k] = self.model_.predict(features)[0]

        return y_hat[self.max_lag:]

    def feature_importances(self) -> dict:
        """Get feature importances."""
        if not self._is_fitted:
            return {}
        
        feature_names = []
        for i in range(1, self.ny + 1):
            feature_names.append(f"y(k-{i})")
        for i in range(1, self.nu + 1):
            feature_names.append(f"u(k-{i})")
        
        return dict(zip(feature_names, self.model_.feature_importances_))

    def summary(self) -> str:
        """Print model summary."""
        if not self._is_fitted:
            return "Model not fitted"
        
        lines = [
            f"RandomForest (n_estimators={self.n_estimators}, max_depth={self.max_depth})",
            f"Number of features: {self.nu + self.ny}",
            "Top 5 feature importances:",
        ]
        
        importances = self.feature_importances()
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_imp[:5]:
            lines.append(f"  {name}: {imp:.4f}")
        
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"RandomForest(nu={self.nu}, ny={self.ny}, n_estimators={self.n_estimators})"
