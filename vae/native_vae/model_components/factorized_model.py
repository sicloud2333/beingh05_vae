from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..morphology import JOINT_QUERY_DIM

from .gesture_encoder import GestureEncoder
from .joint_decoder import JointDecoder
from .morphology_encoder import MorphologyEncoder


@dataclass
class FactorizedModelOutput:
    q_norm_hat: torch.Tensor
    z_action: torch.Tensor
    z_morphology: torch.Tensor
    action_mu: torch.Tensor
    action_log_var: torch.Tensor
    morphology_mu: torch.Tensor
    morphology_log_var: torch.Tensor


class FactorizedActionMorphologyModel(nn.Module):
    """
    Minimal factorized model:
        x_gesture_norm -> z_action
        morphology_vec -> z_morphology
        (z_action, z_morphology) -> q_norm_hat
    """

    def __init__(
        self,
        x_dim: int = 60,
        h_dim: int = 127,
        q_dim: int = 22,
        hidden_dim: int = 256,
        action_latent_dim: int = 16,
        morphology_latent_dim: int = 16,
        encoder_layers: int = 3,
        action_hidden_dims: list[int] | tuple[int, ...] | None = None,
        morphology_encoder_type: str = "res_mlp",
        ohra_hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        decoder_type: str = "universal",
        decoder_trunk_layers: int = 3,
        decoder_head_layers: int = 1,
        hand_configs: dict[str, int] | None = None,
        query_dim: int = JOINT_QUERY_DIM,
        drop: float = 0.0,
    ):
        super().__init__()
        self.decoder_type = str(decoder_type).lower()
        self.morphology_encoder_type = str(morphology_encoder_type).lower()
        self.gesture_encoder = GestureEncoder(
            input_dim=x_dim,
            hidden_dim=hidden_dim,
            hidden_dims=action_hidden_dims,
            latent_dim=action_latent_dim,
            n_lyr=encoder_layers,
            drop=drop,
        )
        if self.morphology_encoder_type != "res_mlp":
            raise ValueError(
                "This inference package only supports morphology_encoder_type='res_mlp'."
            )
        self.morphology_encoder = MorphologyEncoder(
            input_dim=h_dim,
            hidden_dim=hidden_dim,
            latent_dim=morphology_latent_dim,
            n_lyr=encoder_layers,
            drop=drop,
        )
        self.joint_decoder = JointDecoder(
            decoder_type=self.decoder_type,
            latent_dim=action_latent_dim,
            style_dim=morphology_latent_dim,
            hidden_dim=hidden_dim,
            trunk_layers=decoder_trunk_layers,
            head_layers=decoder_head_layers,
            hand_configs=hand_configs,
            output_dim=q_dim,
            query_dim=query_dim,
            drop=drop,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        log_var = log_var.clamp(min=-12.0, max=12.0)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_action(self, x_gesture_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gesture_encoder(x_gesture_norm)

    def encode_morphology(self, morphology_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.morphology_encoder(morphology_vec)

    def decode_joints(
        self,
        z_action: torch.Tensor,
        z_morphology: torch.Tensor,
        hand_name: str | list[str] | None = None,
        joint_queries: torch.Tensor | None = None,
        q_lower: torch.Tensor | None = None,
        q_upper: torch.Tensor | None = None,
        q_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.decoder_type == "universal":
            if joint_queries is None:
                raise ValueError("Universal decoder requires joint_queries.")
            return self.joint_decoder(
                z_action,
                z_morphology,
                joint_queries=joint_queries,
            ) * (q_mask if q_mask is not None else 1.0)

        return self.joint_decoder(
            z_action,
            z_morphology,
            hand_name=hand_name,
        )

    def forward(
        self,
        x_gesture_norm: torch.Tensor,
        morphology_vec: torch.Tensor,
        hand_name: str | list[str] | None = None,
        joint_queries: torch.Tensor | None = None,
        q_lower: torch.Tensor | None = None,
        q_upper: torch.Tensor | None = None,
        q_mask: torch.Tensor | None = None,
    ) -> FactorizedModelOutput:
        action_mu, action_log_var = self.encode_action(x_gesture_norm)
        morphology_mu, morphology_log_var = self.encode_morphology(morphology_vec)
        action_log_var = action_log_var.clamp(min=-12.0, max=12.0)
        morphology_log_var = morphology_log_var.clamp(min=-12.0, max=12.0)

        z_action = self.reparameterize(action_mu, action_log_var)
        z_morphology = self.reparameterize(morphology_mu, morphology_log_var)
        q_norm_hat = self.decode_joints(
            z_action=z_action,
            z_morphology=z_morphology,
            hand_name=hand_name,
            joint_queries=joint_queries,
            q_lower=q_lower,
            q_upper=q_upper,
            q_mask=q_mask,
        )

        return FactorizedModelOutput(
            q_norm_hat=q_norm_hat,
            z_action=z_action,
            z_morphology=z_morphology,
            action_mu=action_mu,
            action_log_var=action_log_var,
            morphology_mu=morphology_mu,
            morphology_log_var=morphology_log_var,
        )
