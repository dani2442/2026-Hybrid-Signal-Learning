"""Shared supervised training loops for torch models."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence


def _clone_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {k: v.detach().clone() for k, v in state_dict.items()}


def train_supervised_torch_model(
    model,
    optimizer,
    criterion,
    train_loader,
    epochs: int,
    verbose: bool,
    progress_desc: str,
    forward_fn: Callable[[Any], Any],
    val_loader=None,
    grad_clip_norm: Optional[float] = None,
    early_stopping_patience: Optional[int] = None,
    logger: Any = None,
    log_every: int = 1,
) -> Sequence[float]:
    """Train a torch model with optional validation and early stopping."""
    import torch
    from tqdm.auto import tqdm

    history: list[float] = []
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = early_stopping_patience or float("inf")

    epoch_iter = range(int(epochs))
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc=progress_desc, unit="epoch")

    for epoch in epoch_iter:
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = forward_fn(bx)
            loss = criterion(pred.squeeze(), by)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
            epoch_loss += float(loss.item())
        avg_train = epoch_loss / len(train_loader)
        history.append(avg_train)

        avg_val = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    pred = forward_fn(bx)
                    val_loss += float(criterion(pred.squeeze(), by).item())
            avg_val = val_loss / len(val_loader)

        monitor = avg_val if avg_val is not None else avg_train
        if monitor < best_loss:
            best_loss = monitor
            best_state = _clone_state_dict(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and hasattr(epoch_iter, "set_postfix"):
            postfix = {"loss": avg_train}
            if avg_val is not None:
                postfix["val"] = avg_val
            epoch_iter.set_postfix(postfix)

        if logger and logger.active and (epoch + 1) % int(log_every) == 0:
            payload = {"train/loss": avg_train, "train/epoch": epoch + 1}
            if avg_val is not None:
                payload["val/loss"] = avg_val
            logger.log_metrics(payload, step=epoch + 1)

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
