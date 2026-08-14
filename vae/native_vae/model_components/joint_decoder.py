from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

from .gesture_encoder import ResMlp

from .adain import AdaINResMlp


def build_joint_queries_from_limits(
    q_lower: torch.Tensor,
    q_upper: torch.Tensor,
    q_mask: torch.Tensor,
    angle_scale: float = torch.pi,
) -> torch.Tensor:
    """
    Build simple per-joint queries for the universal decoder.

    Query layout:
        [lower/pi, upper/pi, center/pi, range/pi, mask]
    """
    center = 0.5 * (q_lower + q_upper)
    joint_range = q_upper - q_lower
    return torch.stack(
        [
            q_lower / float(angle_scale),
            q_upper / float(angle_scale),
            center / float(angle_scale),
            joint_range / float(angle_scale),
            q_mask,
        ],
        dim=-1,
    )


class PolyHandMultiHeadDecoder(nn.Module):
    """
    Adapted from PolyHands `core/cvae.py::MultiHeadDecoder`.

    We keep the same structure:
    1. shared trunk from action latent
    2. AdaIN modulation from morphology latent
    3. per-hand output heads
    """

    def __init__(
        self,
        latent_dim: int = 16,
        style_dim: int = 16,
        hidden_dim: int = 256,
        trunk_layers: int = 3,
        head_layers: int = 1,
        hand_configs: Dict[str, int] | None = None,
        drop: float = 0.0,
    ):
        super().__init__()
        if not hand_configs:
            raise ValueError("hand_configs is required for multihead decoder.")

        self.latent_dim = int(latent_dim)
        self.style_dim = int(style_dim)
        self.hidden_dim = int(hidden_dim)
        self.hand_configs = dict(hand_configs)
        self.max_output_dim = max(self.hand_configs.values())

        self.fc_in = nn.Linear(self.latent_dim, self.hidden_dim)
        self.trunk_model = AdaINResMlp(
            in_features=self.hidden_dim,
            n_lyr=trunk_layers,
            style_dim=self.style_dim,
            drop=drop,
        )
        self.trunk_norm = nn.LayerNorm(self.hidden_dim)

        self.heads = nn.ModuleDict()
        for name, joint_dim in self.hand_configs.items():
            self.heads[name] = nn.Sequential(
                ResMlp(self.hidden_dim, n_lyr=head_layers, drop=drop),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, joint_dim),
            )

    def _decode_group(
        self,
        common_features: torch.Tensor,
        hand_name: str,
    ) -> torch.Tensor:
        if hand_name not in self.heads:
            raise ValueError(f"Hand {hand_name} not registered in decoder heads.")
        return self.heads[hand_name](common_features)

    def forward(
        self,
        z: torch.Tensor,
        hand_name: str | Sequence[str],
        cond_vec: torch.Tensor,
    ) -> torch.Tensor:
        if z.dim() != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected z [B,{self.latent_dim}], got {tuple(z.shape)}")
        if cond_vec.dim() != 2 or cond_vec.shape[0] != z.shape[0] or cond_vec.shape[-1] != self.style_dim:
            raise ValueError(
                f"Expected cond_vec [B,{self.style_dim}], got {tuple(cond_vec.shape)}"
            )

        x = self.fc_in(z)
        x = self.trunk_model(x, cond_vec)
        common_features = self.trunk_norm(x)

        if isinstance(hand_name, str):
            return self._decode_group(common_features, hand_name)

        if len(hand_name) != z.shape[0]:
            raise ValueError("hand_name sequence length must match batch size.")

        out = torch.zeros(
            (z.shape[0], self.max_output_dim),
            dtype=common_features.dtype,
            device=common_features.device,
        )
        unique_names = sorted(set(hand_name))
        for name in unique_names:
            indices = [idx for idx, item in enumerate(hand_name) if item == name]
            head_out = self._decode_group(common_features[indices], name)
            out[indices, : head_out.shape[1]] = head_out
        return out


class PolyHandUniversalDecoder(nn.Module):
    """
    Adapted from PolyHands `core/universal_decoder.py`.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        style_dim: int = 16,
        query_dim: int = 5,
        hidden_dim: int = 256,
        n_layers: int = 4,
        drop: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.style_dim = int(style_dim)
        self.query_dim = int(query_dim)
        self.hidden_dim = int(hidden_dim)

        self.trunk_fc_in = nn.Linear(self.latent_dim, self.hidden_dim)
        trunk_layers = max(1, n_layers - 1)
        self.trunk_backbone = AdaINResMlp(
            in_features=self.hidden_dim,
            n_lyr=trunk_layers,
            style_dim=self.style_dim,
            drop=drop,
        )
        self.trunk_norm = nn.LayerNorm(self.hidden_dim)

        branch_input_dim = self.hidden_dim + self.query_dim
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            ResMlp(self.hidden_dim, n_lyr=1, drop=drop),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        cond_vec: torch.Tensor,
        joint_queries: torch.Tensor,
    ) -> torch.Tensor:
        if z.dim() != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected z [B,{self.latent_dim}], got {tuple(z.shape)}")
        if cond_vec.dim() != 2 or cond_vec.shape[0] != z.shape[0] or cond_vec.shape[-1] != self.style_dim:
            raise ValueError(
                f"Expected cond_vec [B,{self.style_dim}], got {tuple(cond_vec.shape)}"
            )
        if joint_queries.dim() != 3 or joint_queries.shape[0] != z.shape[0] or joint_queries.shape[-1] != self.query_dim:
            raise ValueError(
                f"Expected joint_queries [B,N,{self.query_dim}], got {tuple(joint_queries.shape)}"
            )

        batch_size = z.shape[0]
        num_joints = joint_queries.shape[1]

        global_ctx = self.trunk_fc_in(z)
        global_ctx = self.trunk_backbone(global_ctx, cond_vec)
        global_ctx = self.trunk_norm(global_ctx)

        global_ctx_expanded = global_ctx.unsqueeze(1).expand(-1, num_joints, -1)
        decoder_input = torch.cat([global_ctx_expanded, joint_queries], dim=2)

        flat_input = decoder_input.reshape(-1, decoder_input.shape[-1])
        # Keep normalized joint predictions inside [-1, 1] so decoded
        # joint angles stay within the URDF limits.
        flat_angles = torch.tanh(self.branch_net(flat_input))
        return flat_angles.reshape(batch_size, num_joints)


class DirectURDFJointDecoder(nn.Module):
    """
    Fixed-URDF decoder for the current 22-joint canonical hands.

    Once outputs follow URDF joint order, a direct vector decoder is simpler and
    easier to optimize than per-query scalar decoding.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        style_dim: int = 16,
        output_dim: int = 22,
        hidden_dim: int = 256,
        n_layers: int = 3,
        drop: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.style_dim = int(style_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)

        layers: list[nn.Module] = []
        in_features = self.latent_dim + self.style_dim
        for _ in range(max(1, int(n_layers))):
            layers.extend(
                [
                    nn.Linear(in_features, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.GELU(),
                ]
            )
            if drop > 0.0:
                layers.append(nn.Dropout(drop))
            in_features = self.hidden_dim
        layers.append(nn.Linear(in_features, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected z [B,{self.latent_dim}], got {tuple(z.shape)}")
        if cond_vec.dim() != 2 or cond_vec.shape[0] != z.shape[0] or cond_vec.shape[-1] != self.style_dim:
            raise ValueError(
                f"Expected cond_vec [B,{self.style_dim}], got {tuple(cond_vec.shape)}"
            )
        return torch.tanh(self.net(torch.cat([z, cond_vec], dim=-1)))


class JointDecoder(nn.Module):
    """
    Wrapper that exposes PolyHands-style multihead/universal decoders.
    """

    def __init__(
        self,
        decoder_type: str = "multihead",
        latent_dim: int = 16,
        style_dim: int = 16,
        hidden_dim: int = 256,
        trunk_layers: int = 3,
        head_layers: int = 1,
        hand_configs: Dict[str, int] | None = None,
        output_dim: int = 22,
        query_dim: int = 5,
        drop: float = 0.0,
    ):
        super().__init__()
        self.decoder_type = str(decoder_type).lower()

        if self.decoder_type == "multihead":
            self.decoder = PolyHandMultiHeadDecoder(
                latent_dim=latent_dim,
                style_dim=style_dim,
                hidden_dim=hidden_dim,
                trunk_layers=trunk_layers,
                head_layers=head_layers,
                hand_configs=hand_configs,
                drop=drop,
            )
        elif self.decoder_type == "universal":
            self.decoder = PolyHandUniversalDecoder(
                latent_dim=latent_dim,
                style_dim=style_dim,
                query_dim=query_dim,
                hidden_dim=hidden_dim,
                n_layers=trunk_layers + head_layers,
                drop=drop,
            )
        elif self.decoder_type in {"direct", "direct_urdf"}:
            self.decoder = DirectURDFJointDecoder(
                latent_dim=latent_dim,
                style_dim=style_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                n_layers=trunk_layers + head_layers,
                drop=drop,
            )
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")

    def forward(
        self,
        z: torch.Tensor,
        cond_vec: torch.Tensor,
        hand_name: str | Sequence[str] | None = None,
        joint_queries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.decoder_type == "multihead":
            if hand_name is None:
                raise ValueError("hand_name is required for multihead decoder.")
            return self.decoder(z, hand_name, cond_vec)

        if self.decoder_type in {"direct", "direct_urdf"}:
            return self.decoder(z, cond_vec)

        if joint_queries is None:
            raise ValueError("joint_queries is required for universal decoder.")
        return self.decoder(z, cond_vec, joint_queries)
