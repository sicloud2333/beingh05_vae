#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from native_vae import NativeVAE  # noqa: E402
from native_vae.morphology import FINGER_ORDER  # noqa: E402
from scripts.infer import load_q  # noqa: E402


HAND_COLORS = {
    "shadow_hand_right": "#4f81bd",
    "gaia_hand_right": "#c76b5a",
    "sharpa_hand_right": "#5b9279",
}


def chain_points(model: NativeVAE, q: torch.Tensor, hand: str) -> np.ndarray:
    runtime = model.runtimes[hand]
    x = runtime.kinematic_chain_gesture(q)["x_gesture"]
    batch = x.shape[0]
    root = x[:, 0:15].reshape(batch, 5, 3)
    joint1 = root + x[:, 15:30].reshape(batch, 5, 3)
    tail = x[:, 30:60].reshape(batch, 10, 3)
    joint2 = joint1 + tail[:, 0:5]
    tip = joint2 + tail[:, 5:10]
    return torch.stack((root, joint1, joint2, tip), dim=2).detach().cpu().numpy()


def add_hand(
    figure: go.Figure,
    points: np.ndarray,
    *,
    name: str,
    color: str,
    row: int = 1,
    col: int = 1,
    dash: str = "solid",
) -> None:
    for finger_index, finger in enumerate(FINGER_ORDER):
        value = points[finger_index]
        figure.add_trace(
            go.Scatter3d(
                x=value[:, 0],
                y=value[:, 1],
                z=value[:, 2],
                mode="lines+markers",
                line={"color": color, "width": 7, "dash": dash},
                marker={"color": color, "size": [4, 4, 4, 8]},
                name=name,
                legendgroup=name,
                showlegend=finger_index == 0,
                hovertemplate=f"{finger}<extra>{name}</extra>",
            ),
            row=row,
            col=col,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Native-URDF VAE finger chains and pads.")
    parser.add_argument("--mode", choices=("fingerpads", "reconstruct", "retarget"), required=True)
    parser.add_argument("--source_hand", default="shadow_hand_right")
    parser.add_argument("--target_hand", default=None)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/native_n2_epoch800_inference.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = NativeVAE.from_pretrained(checkpoint=args.checkpoint, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "fingerpads":
        figure = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "scene"}] * 3],
            subplot_titles=[name.replace("_hand_right", "").title() for name in model.hand_names],
        )
        for column, hand in enumerate(model.hand_names, start=1):
            q = torch.zeros((1, len(model.joint_names(hand))), device=model.device)
            add_hand(
                figure,
                chain_points(model, q, hand)[0],
                name=hand,
                color=HAND_COLORS[hand],
                col=column,
            )
        title = "Native-URDF finger chains and finger-pad points"
    else:
        if args.input is None:
            parser.error("--input is required for reconstruct and retarget")
        q_array = load_q(args.input)
        q = model._q_tensor(q_array, args.source_hand)
        if not 0 <= args.frame < len(q):
            raise IndexError(f"frame must be in [0, {len(q) - 1}]")
        q = q[args.frame : args.frame + 1]
        target = args.source_hand if args.mode == "reconstruct" else args.target_hand
        if target is None:
            parser.error("--target_hand is required for retarget")
        result = model.retarget(q, args.source_hand, target)
        figure = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
        add_hand(
            figure,
            chain_points(model, q, args.source_hand)[0],
            name=f"source: {args.source_hand}",
            color=HAND_COLORS[args.source_hand],
        )
        add_hand(
            figure,
            chain_points(model, result.target_q, target)[0],
            name=f"target: {target}",
            color=HAND_COLORS[target],
            dash="dash",
        )
        title = (
            f"{args.source_hand} -> {target} | frame {args.frame} | "
            f"finger-pad error {float(result.fingerpad_error.mean() * 1000.0):.2f} mm"
        )

    figure.update_layout(
        title=title,
        template="plotly_white",
        scene={"aspectmode": "data", "xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        legend={"orientation": "h"},
    )
    figure.write_html(args.output, include_plotlyjs="cdn")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
