"""Training loop for the BAB autoencoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from .models import BABAutoencoder

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


def train_autoencoder(
    model: BABAutoencoder,
    dataset,
    *,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cpu",
    freeze_encoder_backbone: bool = True,
    show_progress: bool = True,
) -> tuple[BABAutoencoder, list[float]]:
    """Train the autoencoder with MSE reconstruction loss.

    Parameters
    ----------
    model : BABAutoencoder
    dataset : Dataset returning (3, H, W) tensors in [0, 1]
    epochs, batch_size, lr : training hyperparameters
    device : "cpu", "mps", or "cuda"
    freeze_encoder_backbone : if True, freeze ResNet backbone and only
        train the encoder FC head + full decoder.  Dramatically faster.

    Returns
    -------
    (trained_model, loss_history)
    """
    model.to(device)

    # Freeze backbone if requested (only train fc + decoder)
    if freeze_encoder_backbone:
        backbone_params = set()
        for name, p in model.encoder.named_parameters():
            if name.startswith("fc"):
                continue  # keep FC trainable
            p.requires_grad_(False)
            backbone_params.add(name)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Frozen {len(backbone_params)} backbone params, "
              f"training {trainable:,} parameters")

    model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.MSELoss()

    losses: list[float] = []

    epoch_iter = range(epochs)
    if show_progress and tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="Autoencoder")

    for epoch in epoch_iter:
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            x_recon, z = model(batch)
            loss = criterion(x_recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)

        avg_loss = epoch_loss / len(dataset)
        losses.append(avg_loss)

        if show_progress and tqdm is not None:
            epoch_iter.set_postfix(loss=f"{avg_loss:.5f}")

    model.eval()
    return model, losses
