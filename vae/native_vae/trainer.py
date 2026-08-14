from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import yaml

from .api import _build_model
from .dataset import NativeTensorBundleDataset
from .losses import NativeHandBank, compute_native_vae_loss


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LOSS_KEYS = (
    "loss",
    "loss_q",
    "loss_tip",
    "loss_cross_tip_abs",
    "loss_cross_pair_vector",
    "loss_cross_pair_distance",
    "loss_kl_action",
    "loss_kl_morphology",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return value


def package_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PACKAGE_ROOT / value


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def active_cross_weights(config: Mapping[str, Any], epoch: int) -> tuple[float, float, float]:
    if epoch < int(config["cross_loss_start_epoch"]):
        return 0.0, 0.0, 0.0
    return (
        float(config["lambda_cross_abs"]),
        float(config["lambda_cross_pair_vector"]),
        float(config["lambda_cross_pair_distance"]),
    )


def loss_kwargs(config: Mapping[str, Any], epoch: int) -> dict[str, Any]:
    cross_abs, pair_vector, pair_distance = active_cross_weights(config, epoch)
    return {
        "lambda_q": float(config["lambda_q"]),
        "lambda_tip": float(config["lambda_tip"]),
        "lambda_cross_abs": cross_abs,
        "lambda_cross_pair_vector": pair_vector,
        "lambda_cross_pair_distance": pair_distance,
        "beta_action": float(config["beta_action"]),
        "beta_morphology": float(config["beta_morphology"]),
        "finger_weights_cross_abs": tuple(config["finger_weights_cross_abs"]),
    }


def build_loader(
    path: Path,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> tuple[NativeTensorBundleDataset, DataLoader]:
    dataset = NativeTensorBundleDataset(path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return dataset, loader


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_val: float,
    config: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val": float(best_val),
            "config": dict(config),
        },
        path,
    )


def run_epoch(
    *,
    model,
    loader: DataLoader,
    hand_bank: NativeHandBank,
    device: torch.device,
    config: Mapping[str, Any],
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in LOSS_KEYS}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = to_device(batch, device)
            output = model(
                batch["x_gesture_norm"],
                batch["morphology_vec"],
                hand_name=batch["hand_name"],
                joint_queries=batch["joint_queries"],
                q_lower=batch["q_lower"],
                q_upper=batch["q_upper"],
                q_mask=batch["q_mask"],
            )
            losses = compute_native_vae_loss(
                model=model,
                output=output,
                batch=batch,
                hand_bank=hand_bank,
                **loss_kwargs(config, epoch),
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                grad_clip = float(config.get("grad_clip_norm", 0.0))
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            for key in LOSS_KEYS:
                totals[key] += float(losses[key].detach())
            count += 1
            if max_batches > 0 and count >= max_batches:
                break
    if count == 0:
        raise RuntimeError("DataLoader produced no batches")
    return {key: value / count for key, value in totals.items()}


def maybe_wandb(config: Mapping[str, Any], run_dir: Path):
    if not bool(config.get("wandb_enabled", False)):
        return None
    try:
        import wandb
    except ImportError as error:
        raise ImportError("wandb_enabled=true, but wandb is not installed") from error
    return wandb.init(
        project=str(config.get("project_name", "Native-URDF-VAE")),
        name=str(config["run_name"]),
        group=config.get("wandb_group"),
        tags=config.get("wandb_tags"),
        config=dict(config),
        dir=str(run_dir),
    )


def train(
    config_path: str | Path,
    *,
    device_override: str | None = None,
    epochs_override: int | None = None,
    max_train_batches_override: int | None = None,
    max_val_batches_override: int | None = None,
    wandb_enabled_override: bool | None = None,
) -> Path:
    config = load_yaml(config_path)
    if device_override is not None:
        config["device"] = device_override
    if epochs_override is not None:
        config["num_epochs"] = int(epochs_override)
    if max_train_batches_override is not None:
        config["max_train_batches"] = int(max_train_batches_override)
    if max_val_batches_override is not None:
        config["max_val_batches"] = int(max_val_batches_override)
    if wandb_enabled_override is not None:
        config["wandb_enabled"] = bool(wandb_enabled_override)

    set_seed(int(config["seed"]), bool(config.get("deterministic", True)))
    requested = torch.device(str(config.get("device", "cuda")))
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = package_path(config.get("output_root", "runs")) / f"{config['run_name']}_{timestamp}"
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    train_dataset, train_loader = build_loader(
        package_path(config["train_data"]),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", True)),
    )
    val_dataset, val_loader = build_loader(
        package_path(config["val_data"]),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", True)),
    )
    if set(train_dataset.hand_names) != set(val_dataset.hand_names):
        raise ValueError("Train and validation bundles must contain the same hands")

    model_config = load_yaml(package_path(config["model_config"]))
    model = _build_model(model_config, device)
    hand_bank = NativeHandBank.build(
        package_path(config["hand_config"]),
        device,
        train_dataset.hand_names,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.get("lr_decay_factor", 0.5)),
        patience=int(config.get("lr_patience", 10)),
        min_lr=float(config.get("min_lr", 1e-6)),
    )

    start_epoch = 1
    best_val = float("inf")
    resume_path = config.get("resume_path")
    if resume_path:
        payload = torch.load(package_path(resume_path), map_location=device)
        model.load_state_dict(payload["model_state"], strict=True)
        if bool(config.get("resume_optimizer", True)) and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
            scheduler.load_state_dict(payload["scheduler_state"])
            start_epoch = int(payload.get("epoch", 0)) + 1
            best_val = float(payload.get("best_val", best_val))

    wandb_run = maybe_wandb(config, run_dir)
    metrics_path = log_dir / "metrics.jsonl"
    num_epochs = int(config["num_epochs"])
    print(f"[train] run={run_dir} device={device} params={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(start_epoch, num_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            hand_bank=hand_bank,
            device=device,
            config=config,
            epoch=epoch,
            optimizer=optimizer,
            max_batches=int(config.get("max_train_batches", 0)),
        )
        record: dict[str, Any] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train/{key}": value for key, value in train_metrics.items()},
        }
        print(
            f"[train] epoch={epoch:04d} loss={train_metrics['loss']:.6f} "
            f"q={train_metrics['loss_q']:.6f} tip={train_metrics['loss_tip']:.6f}"
        )

        if epoch % int(config["val_interval"]) == 0 or epoch == num_epochs:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                hand_bank=hand_bank,
                device=device,
                config=config,
                epoch=epoch,
                optimizer=None,
                max_batches=int(config.get("max_val_batches", 0)),
            )
            scheduler.step(val_metrics["loss"])
            record.update({f"val/{key}": value for key, value in val_metrics.items()})
            print(
                f"[val]   epoch={epoch:04d} loss={val_metrics['loss']:.6f} "
                f"q={val_metrics['loss_q']:.6f} tip={val_metrics['loss_tip']:.6f}"
            )
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val=best_val,
                    config=config,
                )

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if wandb_run is not None:
            wandb_run.log(record, step=epoch)
        if epoch % int(config["save_interval"]) == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val=best_val,
                config=config,
            )

    save_checkpoint(
        checkpoint_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=num_epochs,
        best_val=best_val,
        config=config,
    )
    torch.save(
        {"model_state": model.state_dict(), "model_config": model_config, "epoch": num_epochs},
        checkpoint_dir / "inference.pt",
    )
    if wandb_run is not None:
        wandb_run.finish()
    print(f"[done] {run_dir}")
    return run_dir
