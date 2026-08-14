from __future__ import annotations

import numpy as np

from native_vae import NativeVAE
from sim import (
    CallablePolicy,
    GesturePolicyAdapter,
    GraspEnv,
    GraspEnvConfig,
    PolicyEvaluationClient,
)


def main() -> None:
    hand = "sharpa_hand_right"
    vae = NativeVAE.from_pretrained(device="cuda")

    # Replace this callable with model.forward(). It receives wrist6+z24 state,
    # RGB image, object pose and other observation fields.
    def latent_policy(observation):
        return np.asarray(observation["state"], dtype=np.float32)

    policy = GesturePolicyAdapter(
        CallablePolicy(latent_policy),
        vae=vae,
        target_hand=hand,
        encode_observation=True,
    )
    with GraspEnv(GraspEnvConfig(hand=hand, max_steps=30)) as env:
        result = PolicyEvaluationClient(env, policy).run(max_steps=30)
    print(f"success={result.success}, max_lift={result.max_lift_m:.3f}m")


if __name__ == "__main__":
    main()
