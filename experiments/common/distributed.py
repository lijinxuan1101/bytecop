"""Minimal torchrun / DDP helpers for the two tower trainers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Sampler


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


class StridedSampler(Sampler[int]):
    """Split ``length`` indices across ranks with no padding or shuffle."""

    def __init__(self, length: int, *, rank: int, world_size: int) -> None:
        self.indices = list(range(rank, length, world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def init_distributed() -> DistInfo:
    """Init DDP when launched by ``torchrun``; otherwise stay single-process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or not torch.cuda.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistInfo(False, 0, 0, 1, device)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    # Bind the rank to its device up front. Without device_id NCCL guesses the
    # rank->GPU mapping when it builds the first communicator, which races and
    # deadlocks the first collective (observed intermittently on this 8×A40 box).
    # TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC alone does not change WorkNCCL timeout;
    # that comes from init_process_group(timeout=), default 10 minutes.
    timeout_s = int(
        os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC")
        or os.environ.get("NCCL_TIMEOUT")
        or "1800"
    )
    dist.init_process_group(
        backend="nccl",
        device_id=device,
        timeout=timedelta(seconds=max(timeout_s, 60)),
    )
    return DistInfo(True, rank, local_rank, world_size, device)


def broadcast_module(model: torch.nn.Module, info: DistInfo, *, src: int = 0) -> None:
    """Copy parameters and buffers from ``src`` to every other rank."""
    if not info.enabled:
        return
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param.data, src=src)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=src)


def broadcast_object(obj: Any, info: DistInfo, *, src: int = 0) -> Any:
    """Broadcast a picklable Python object from ``src``."""
    if not info.enabled:
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


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


def eval_sampler(length: int, info: DistInfo) -> StridedSampler | None:
    if not info.enabled:
        return None
    return StridedSampler(length, rank=info.rank, world_size=info.world_size)


def gather_cat(tensor: torch.Tensor, info: DistInfo) -> torch.Tensor:
    """Concatenate 1-D (or [N, ...]) tensors from every rank, no padding leftovers."""
    if not info.enabled:
        return tensor
    tensor = tensor.contiguous().to(info.device)
    local_n = torch.tensor([tensor.shape[0]], device=info.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_n) for _ in range(info.world_size)]
    dist.all_gather(sizes, local_n)
    counts = [int(s.item()) for s in sizes]
    max_n = max(counts)
    if tensor.shape[0] < max_n:
        pad_shape = (max_n - tensor.shape[0],) + tensor.shape[1:]
        tensor = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)
    gathered = [torch.empty_like(tensor) for _ in range(info.world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat([chunk[:n] for chunk, n in zip(gathered, counts)], dim=0)


def reduce_sum(value: float, info: DistInfo) -> float:
    if not info.enabled:
        return value
    tensor = torch.tensor(value, device=info.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def broadcast_bool(value: bool, info: DistInfo) -> bool:
    """Broadcast a rank-0 boolean so every process takes the same branch."""
    if not info.enabled:
        return value
    tensor = torch.tensor([int(value)], device=info.device)
    dist.broadcast(tensor, src=0)
    return bool(tensor.item())


def barrier(info: DistInfo) -> None:
    if info.enabled:
        dist.barrier(device_ids=[info.local_rank])


def cleanup(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()
