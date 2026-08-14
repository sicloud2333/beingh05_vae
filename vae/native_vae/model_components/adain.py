from __future__ import annotations

import torch
import torch.nn as nn


class AdaINLayer(nn.Module):
    def __init__(self, input_dim: int, style_dim: int):
        super().__init__()
        self.style_scale = nn.Linear(style_dim, input_dim)
        self.style_shift = nn.Linear(style_dim, input_dim)

        nn.init.ones_(self.style_scale.weight)
        nn.init.zeros_(self.style_scale.bias)
        nn.init.zeros_(self.style_shift.weight)
        nn.init.zeros_(self.style_shift.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-8
        x_norm = (x - x_mean) / x_std

        gamma = self.style_scale(style)
        beta = self.style_shift(style)
        return x_norm * gamma + beta


class AdaINResBlock(nn.Module):
    def __init__(self, in_features: int, style_dim: int, drop: float = 0.0):
        super().__init__()
        self.norm = AdaINLayer(in_features, style_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.linear = nn.Linear(in_features, in_features)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x, style)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear(x)
        return x + residual


class AdaINResMlp(nn.Module):
    def __init__(self, in_features: int, n_lyr: int, style_dim: int, drop: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            AdaINResBlock(in_features, style_dim, drop=drop)
            for _ in range(n_lyr)
        ])

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, style)
        return x

