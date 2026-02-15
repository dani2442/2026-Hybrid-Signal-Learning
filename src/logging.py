"""W&B logging wrapper for standardised experiment tracking."""

from __future__ import annotations

from typing import Any, Dict, Optional


class WandbLogger:
    """Thin wrapper around ``wandb`` for consistent logging.

    Usage::

        logger = WandbLogger(project="my-project", run_name="lstm-run-1",
                              config=cfg.to_dict())
        logger.log_metrics({"train/loss": 0.05, "val/loss": 0.06}, step=10)
        logger.finish()

    If ``wandb`` is not installed or *project* is ``None`` the logger
    silently becomes a no-op so callers never need to guard with
    ``if logger:``.
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
            # wandb import or init failure → degrade gracefully
            self._run = None

    @property
    def active(self) -> bool:
        return self._run is not None

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
