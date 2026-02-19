"""W&B logging wrapper for standardised experiment tracking."""

from __future__ import annotations

from typing import Any, Dict, Optional


class WandbLogger:
    """Thin wrapper around ``wandb`` for consistent logging.

    If ``wandb`` is not installed or *project* is ``None`` the logger
    silently becomes a no-op so callers never need to guard.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._run = None
        if project is None:
            return
        try:
            import wandb

            self._run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                reinit=True,
            )
        except Exception:
            self._run = None

    @property
    def active(self) -> bool:
        return self._run is not None

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Merge *updates* into the W&B run config (e.g. model hyperparams
        discovered after the run is initialised)."""
        if self._run is None:
            return
        try:
            self._run.config.update(updates, allow_val_change=True)
        except Exception:
            pass

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._run is None:
            return
        self._run.log(metrics, step=step)

    def log_summary(self, metrics: Dict[str, Any]) -> None:
        if self._run is None:
            return
        for k, v in metrics.items():
            self._run.summary[k] = v

    def finish(self) -> None:
        if self._run is None:
            return
        self._run.finish()
        self._run = None
