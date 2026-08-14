from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import yaml

from .hand_runtime import NativeHandRuntime
from .hand_spec import load_native_hand_specs
from .model_components import FactorizedActionMorphologyModel


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RetargetResult:
    z_gesture: torch.Tensor
    target_q: torch.Tensor
    source_fingerpads: torch.Tensor
    target_fingerpads: torch.Tensor
    fingerpad_error: torch.Tensor


def _build_model(cfg: Mapping[str, object], device: torch.device) -> FactorizedActionMorphologyModel:
    return FactorizedActionMorphologyModel(
        x_dim=int(cfg["x_dim"]),
        h_dim=int(cfg["h_dim"]),
        q_dim=int(cfg["q_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        action_latent_dim=int(cfg["action_latent_dim"]),
        morphology_latent_dim=int(cfg["morphology_latent_dim"]),
        encoder_layers=int(cfg["encoder_layers"]),
        morphology_encoder_type=str(cfg["morphology_encoder_type"]),
        decoder_type=str(cfg["decoder_type"]),
        decoder_trunk_layers=int(cfg["decoder_trunk_layers"]),
        decoder_head_layers=int(cfg["decoder_head_layers"]),
        query_dim=int(cfg["query_dim"]),
        drop=float(cfg.get("dropout", 0.0)),
    ).to(device)


class NativeVAE:
    """Inference-only interface for the three-hand Native-URDF VAE."""

    def __init__(
        self,
        model: FactorizedActionMorphologyModel,
        runtimes: Mapping[str, NativeHandRuntime],
        device: torch.device,
    ) -> None:
        self.model = model
        self.runtimes = dict(runtimes)
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = PACKAGE_ROOT / "checkpoints/native_n2_epoch800_inference.pt",
        config: str | Path = PACKAGE_ROOT / "configs/model.yaml",
        hand_config: str | Path = PACKAGE_ROOT / "configs/right_hands.yaml",
        device: str | torch.device = "cuda",
    ) -> "NativeVAE":
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")

        model_cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
        specs = load_native_hand_specs(hand_config)
        runtimes = {
            name: NativeHandRuntime.build(spec, device=requested)
            for name, spec in specs.items()
        }
        model = _build_model(model_cfg, requested)
        payload = torch.load(checkpoint, map_location=requested)
        state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
        model.load_state_dict(state, strict=True)
        model.eval()
        return cls(model=model, runtimes=runtimes, device=requested)

    @property
    def hand_names(self) -> tuple[str, ...]:
        return tuple(self.runtimes)

    def joint_names(self, hand: str) -> tuple[str, ...]:
        return self._runtime(hand).spec.active_joint_names

    def _runtime(self, hand: str) -> NativeHandRuntime:
        if hand not in self.runtimes:
            raise KeyError(f"Unknown hand {hand!r}; available={self.hand_names}")
        return self.runtimes[hand]

    def _q_tensor(self, q: torch.Tensor | object, hand: str) -> torch.Tensor:
        value = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        expected = len(self.joint_names(hand))
        if value.ndim != 2 or value.shape[-1] != expected:
            raise ValueError(f"{hand}: expected q [B,{expected}], got {tuple(value.shape)}")
        return value

    @torch.inference_mode()
    def fingerpads(self, q: torch.Tensor | object, hand: str) -> torch.Tensor:
        runtime = self._runtime(hand)
        return runtime.kinematic_chain_gesture(self._q_tensor(q, hand))["tips"]

    @torch.inference_mode()
    def gesture(self, q: torch.Tensor | object, hand: str) -> torch.Tensor:
        runtime = self._runtime(hand)
        raw = runtime.kinematic_chain_gesture(self._q_tensor(q, hand))["x_gesture"]
        return raw / float(runtime.palm_radius)

    @torch.inference_mode()
    def encode(self, q: torch.Tensor | object, hand: str) -> torch.Tensor:
        z_mu, _ = self.model.encode_action(self.gesture(q, hand))
        return z_mu

    @torch.inference_mode()
    def decode(self, z_gesture: torch.Tensor | object, hand: str) -> torch.Tensor:
        runtime = self._runtime(hand)
        z = torch.as_tensor(z_gesture, dtype=torch.float32, device=self.device)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.ndim != 2 or z.shape[-1] != self.model.gesture_encoder.latent_dim:
            raise ValueError(
                f"Expected z_gesture [B,{self.model.gesture_encoder.latent_dim}], "
                f"got {tuple(z.shape)}"
            )
        batch = z.shape[0]
        morphology = runtime.morphology_vec.to(self.device).unsqueeze(0).expand(batch, -1)
        z_morphology, _ = self.model.encode_morphology(morphology)
        joint_queries = runtime.joint_queries.to(self.device).unsqueeze(0).expand(batch, -1, -1)
        q_mask = runtime.q_mask.to(self.device).unsqueeze(0).expand(batch, -1)
        q_norm = self.model.decode_joints(
            z,
            z_morphology,
            joint_queries=joint_queries,
            q_mask=q_mask,
        )
        lower = runtime.q_lower.to(self.device).unsqueeze(0)
        upper = runtime.q_upper.to(self.device).unsqueeze(0)
        q = torch.where(
            q_mask > 0.5,
            0.5 * (q_norm + 1.0) * (upper - lower) + lower,
            torch.zeros_like(q_norm),
        )
        return q[:, : len(runtime.spec.active_joint_names)]

    @torch.inference_mode()
    def reconstruct(self, q: torch.Tensor | object, hand: str) -> RetargetResult:
        return self.retarget(q, source_hand=hand, target_hand=hand)

    @torch.inference_mode()
    def retarget(
        self,
        q: torch.Tensor | object,
        source_hand: str,
        target_hand: str,
    ) -> RetargetResult:
        source_q = self._q_tensor(q, source_hand)
        z = self.encode(source_q, source_hand)
        target_q = self.decode(z, target_hand)
        source_tips = self._runtime(source_hand).kinematic_chain_gesture(source_q)["tips"]
        target_tips = self._runtime(target_hand).kinematic_chain_gesture(target_q)["tips"]
        error = torch.linalg.norm(target_tips - source_tips, dim=-1)
        return RetargetResult(
            z_gesture=z,
            target_q=target_q,
            source_fingerpads=source_tips,
            target_fingerpads=target_tips,
            fingerpad_error=error,
        )
