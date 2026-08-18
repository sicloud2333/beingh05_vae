#!/usr/bin/env python3
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def main() -> None:
    device_type = os.environ.get(
        "BEINGH_DEVICE_TYPE",
        "cuda" if torch.cuda.is_available() else "npu",
    )
    backend = os.environ.get(
        "BEINGH_DIST_BACKEND", "hccl" if device_type == "npu" else "nccl"
    )
    dist.init_process_group(backend)
    local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
    accelerator = getattr(torch, device_type)
    accelerator.set_device(local_rank)
    device = torch.device(device_type, local_rank)

    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 32),
    )
    model = FSDP(
        model,
        device_id=device,
        use_orig_params=True,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=False)
    inputs = torch.randn(8, 64, device=device)
    targets = torch.randn(8, 32, device=device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type, dtype=torch.bfloat16):
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    model.clip_grad_norm_(1.0)
    optimizer.step()
    reduced = loss.detach().to(torch.float32)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= dist.get_world_size()
    accelerator.synchronize()
    if dist.get_rank() == 0:
        print(
            f"[ok] backend={backend} device={device_type} "
            f"world_size={dist.get_world_size()} loss={reduced.item():.6f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
