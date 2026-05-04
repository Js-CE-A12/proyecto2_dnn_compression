"""
models.py — Arquitecturas para detección de apnea del sueño.
SleepApneaCNN: CNN 1D basada en "Exploring the efficacy of CNNs in sleep apnea".
Input: (B, 1, 3000) float32  →  logits (B, 2)
"""

import torch
import torch.nn as nn


class SleepApneaCNN(nn.Module):
    """
    1-D CNN para clasificación binaria de apnea del sueño.

    Tamaños internos (entrada = 3000 muestras):
      Block 1: Conv(k=50) + BN + ReLU + MaxPool(8) → 375
      Block 2: Conv(k=8)  + BN + ReLU + MaxPool(4) → 93
      Block 3: Conv(k=8)  + BN + ReLU + MaxPool(4) → 23
      Flatten: 128 * 23 = 2944
      FC: 2944 → 64 → num_classes
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — kernel grande para capturar ritmos EEG lentos
            nn.Conv1d(1, 32, kernel_size=50, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(dropout),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=8, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout),
            # Block 3
            nn.Conv1d(64, 128, kernel_size=8, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 23, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
