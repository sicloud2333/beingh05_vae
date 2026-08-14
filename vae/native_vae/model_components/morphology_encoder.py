from __future__ import annotations

import torch
import torch.nn as nn

from .gesture_encoder import ResMlp
from ..morphology import MORPHOLOGY_DIM


class MorphologyEncoder(nn.Module):
    """
    VAE-style encoder for fixed-length morphology descriptors.

    Input:
        h_morphology: [B, 127] by default
            This is expected to be the dataset field `batch["morphology_vec"]`,
            i.e. the normalized `full_features` descriptor.

    Output:
        mu:      [B, latent_dim]
        log_var: [B, latent_dim]
    """

    def __init__(
        self,
        input_dim: int = MORPHOLOGY_DIM,
        hidden_dim: int = 256,
        latent_dim: int = 16,
        n_lyr: int = 3,
        drop: float = 0.0,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            ResMlp(self.hidden_dim, n_lyr=n_lyr, drop=drop),
            nn.LayerNorm(self.hidden_dim),
        )
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, h_morphology: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_morphology: [B, input_dim]

        Returns:
            mu:      [B, latent_dim]
            log_var: [B, latent_dim]
        """
        if h_morphology.dim() != 2:
            raise ValueError(
                f"Expected h_morphology shape [B, D], got {tuple(h_morphology.shape)}"
            )
        if h_morphology.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {h_morphology.shape[-1]}"
            )

        hidden = self.net(h_morphology)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        return mu, log_var
