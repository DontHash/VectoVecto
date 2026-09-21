"""
model.py — CRNN (CNN + BiLSTM + CTC) for Devanagari line crops.

Input: grayscale 32 x W (padded to 256), output: log-probabilities over the
charset per timestep. Small on purpose: trains on a single T4 in minutes and
runs on CPU for inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 256, in_h: int = 32):
        super().__init__()
        # GroupNorm, not BatchNorm: small-batch CTC training diverged to NaN
        # with BN (measured), GroupNorm is stable at batch 8 and 64 alike.
        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1),
                    nn.GroupNorm(8, cout), nn.ReLU()]

        self.cnn = nn.Sequential(
            *block(1, 32), nn.MaxPool2d(2),        # 16 x W/2
            *block(32, 64), nn.MaxPool2d(2),       # 8 x W/4
            *block(64, 128), nn.MaxPool2d((2, 1)),  # 4 x W/4
            *block(128, 128), nn.MaxPool2d((2, 1)),  # 2 x W/4
            *block(128, 256), nn.MaxPool2d((2, 1)),  # 1 x W/4
        )
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True,
                           batch_first=True)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) -> log-probs (T, B, C) for CTCLoss."""
        f = self.cnn(x)                            # B, 256, 1, W/4
        b, c, h, w = f.shape
        f = f.squeeze(2).permute(0, 2, 1)          # B, W/4, 256
        out, _ = self.rnn(f)
        logits = self.head(out)                    # B, T, C
        return logits.log_softmax(-1).permute(1, 0, 2)
