"""TCN (Temporal Convolutional Network) model for time series forecasting."""

from __future__ import annotations

from ..config import TCNConfig
from .sequence_base import SequenceModel


class TCN(SequenceModel):
    """Temporal Convolutional Network with causal dilated convolutions.

    Dilation grows exponentially (1, 2, 4, …) so the receptive field
    covers the full input sequence with relatively few layers.
    All shared logic lives in :class:`SequenceModel`.
    """

    def __init__(self, config: TCNConfig | None = None, **kwargs):
        if config is None:
            config = TCNConfig(**kwargs)
        super().__init__(config)

    def _build_model(self, input_size: int):
        import torch.nn as nn

        cfg = self.config

        class CausalConv1d(nn.Module):
            def __init__(self, in_ch, out_ch, kernel_size, dilation):
                super().__init__()
                self.pad = (kernel_size - 1) * dilation
                self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

            def forward(self, x):
                x = nn.functional.pad(x, (self.pad, 0))
                return self.conv(x)

        class ResidualBlock(nn.Module):
            def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
                super().__init__()
                self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
                self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

            def forward(self, x):
                out = self.dropout(self.relu(self.conv1(x)))
                out = self.dropout(self.relu(self.conv2(out)))
                return self.relu(out + self.skip(x))

        class TCNModel(nn.Module):
            def __init__(self):
                super().__init__()
                blocks = []
                for i, out_ch in enumerate(cfg.num_channels):
                    in_ch = input_size if i == 0 else cfg.num_channels[i - 1]
                    blocks.append(ResidualBlock(in_ch, out_ch, cfg.kernel_size, 2 ** i, cfg.dropout))
                self.network = nn.Sequential(*blocks)
                self.fc = nn.Linear(cfg.num_channels[-1], 1)

            def forward(self, x):
                x = x.permute(0, 2, 1)
                out = self.network(x)
                return self.fc(out[:, :, -1])

        return TCNModel()

