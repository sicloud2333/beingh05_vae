from __future__ import annotations

import torch
import torch.nn as nn


class ResMlp(nn.Module):
    """
    Residual MLP block stack.

    Structure of each block:
        LayerNorm -> GELU -> Dropout -> Linear
    with residual connection.
    """

    def __init__(self, in_features: int, n_lyr: int, drop: float = 0.0):
        super().__init__()
        self.model = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_features),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(in_features, in_features),
            )
            for _ in range(n_lyr)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lyr in self.model:
            x = lyr(x) + x
        return x


class GestureEncoder(nn.Module):
    """
    VAE-style encoder for unified single-frame gesture representation.

    Input:
        x_gesture: [B, 60]

    Output:
        mu:      [B, latent_dim]
        log_var: [B, latent_dim]
    """

    def __init__(
        self,
        input_dim: int = 60,
        hidden_dim: int = 256,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        latent_dim: int = 16,
        n_lyr: int = 3,
        drop: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dims = tuple(int(v) for v in hidden_dims) if hidden_dims is not None else None
        self.latent_dim = latent_dim

        if self.hidden_dims is None:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                ResMlp(hidden_dim, n_lyr=n_lyr, drop=drop),
                nn.LayerNorm(hidden_dim),
            )
            out_dim = hidden_dim
        else:
            if len(self.hidden_dims) == 0:
                raise ValueError("hidden_dims must contain at least one layer size.")
            layers: list[nn.Module] = []
            in_features = input_dim
            for h_dim in self.hidden_dims:
                layers.extend(
                    [
                        nn.Linear(in_features, h_dim),
                        nn.LayerNorm(h_dim),
                        nn.GELU(),
                    ]
                )
                if drop > 0.0:
                    layers.append(nn.Dropout(drop))
                in_features = h_dim
            self.net = nn.Sequential(*layers)
            out_dim = in_features

        self.fc_mu = nn.Linear(out_dim, latent_dim)
        self.fc_var = nn.Linear(out_dim, latent_dim)

    def forward(self, x_gesture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_gesture: [B, 60]

        Returns:
            mu:      [B, latent_dim]
            log_var: [B, latent_dim]
        """
        if x_gesture.dim() != 2:
            raise ValueError(f"Expected x_gesture shape [B, D], got {tuple(x_gesture.shape)}")
        if x_gesture.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {x_gesture.shape[-1]}"
            )

        h = self.net(x_gesture)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
