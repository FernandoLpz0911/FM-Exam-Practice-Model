"""P5.2 — DKT LSTM model (Piech et al., NeurIPS 2015)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DKT(nn.Module):
    def __init__(
        self,
        n_concepts: int,
        hidden: int = 128,
        layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n = n_concepts  # NOTE: stored but never read elsewhere — dead field
        # Input is a 2*n_concepts-wide one-hot vector per step (concept index,
        # offset by n_concepts if the answer was correct — see
        # engine/tracing/dataset.py:encode_sequence for the matching encoder).
        self.lstm = nn.LSTM(
            2 * n_concepts,
            hidden,
            num_layers=layers,
            batch_first=True,
            # Dropout between LSTM layers only makes sense with >1 layer;
            # PyTorch warns if dropout is set with layers=1.
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden, n_concepts)

    def forward(self, one_hot_input: torch.Tensor) -> torch.Tensor:
        """one_hot_input: (B, T, 2M) → (B, T, M) predicted P(correct) per concept.

        Sigmoid squashes each concept's output independently to (0,1), since
        P(correct) per concept isn't mutually exclusive across concepts.
        """
        lstm_output, _ = self.lstm(one_hot_input)
        return torch.sigmoid(self.out(lstm_output))
