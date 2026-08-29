"""Minimal torchrun / DDP helpers for the two tower trainers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistInfo:
    """Init DDP when launched by ``torchrun``; otherwise stay single-process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or not torch.cuda.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistInfo(False, 0, 0, 1, device)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return DistInfo(True, rank, local_rank, world_size, torch.device("cuda", local_rank))


def wrap_ddp(model: torch.nn.Module, info: DistInfo, *, find_unused_parameters: bool) -> torch.nn.Module:
    if not info.enabled:
        return model
    return DDP(
        model,
        device_ids=[info.local_rank],
        output_device=info.local_rank,
        find_unused_parameters=find_unused_parameters,
    )


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def reduce_mean(value: float, info: DistInfo) -> float:
    if not info.enabled:
        return value
    tensor = torch.tensor(value, device=info.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / info.world_size).item()


def broadcast_bool(value: bool, info: DistInfo) -> bool:
    """Broadcast a rank-0 boolean so every process takes the same branch."""
    if not info.enabled:
        return value
    tensor = torch.tensor([int(value)], device=info.device)
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())


def barrier(info: DistInfo) -> None:
    if info.enabled:
        dist.barrier()


def cleanup(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()
