"""Micro-batching scheduler: a stream of single requests -> GPU-sized batches.

Measured on one A40 (bf16, CLIP ViT-H/14):
    batch   1 ->  93 img/s   (11 ms/batch)
    batch  32 -> 218 img/s  (147 ms/batch)   <- chosen: 99% of peak, half the latency
    batch  64 -> 220 img/s  (291 ms/batch)
    batch 256 -> 221 img/s  (1160 ms/batch)

A batch leaves when it is full OR when the oldest request has waited MAX_WAIT.
The queue is bounded; a full queue means 503 rather than an unbounded backlog.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np


class Overloaded(Exception):
    """Queue is full — shed load instead of melting down."""


@dataclass
class _Item:
    array: np.ndarray
    future: asyncio.Future
    queued_at: float = field(default_factory=time.perf_counter)


class MicroBatcher:
    def __init__(self, detector, *, max_batch: int = 32, max_wait_ms: float = 10.0,
                 max_queue: int = 512) -> None:
        self.det = detector
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self.q: asyncio.Queue[_Item] = asyncio.Queue(maxsize=max_queue)
        self.stats = {"batches": 0, "images": 0, "batch_sizes": [], "gpu_ms": 0.0,
                      "rejected": 0}
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, array: np.ndarray) -> float:
        item = _Item(array=array, future=asyncio.get_running_loop().create_future())
        try:
            self.q.put_nowait(item)
        except asyncio.QueueFull:
            self.stats["rejected"] += 1
            raise Overloaded("inference queue full")
        return await item.future

    async def _collect(self) -> list[_Item]:
        first = await self.q.get()
        batch = [first]
        # Greedy drain: take everything already waiting before starting the clock.
        # Without this a closed-loop client can lock the scheduler into batch=1:
        # GPU takes 11 ms at bs=1, so arrivals space out to ~11 ms, which is wider
        # than max_wait, so every window catches exactly one request.
        while len(batch) < self.max_batch:
            try:
                batch.append(self.q.get_nowait())
            except asyncio.QueueEmpty:
                break
        deadline = first.queued_at + self.max_wait
        while len(batch) < self.max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self.q.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        return batch

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect()
            stacked = np.stack([b.array for b in batch])
            t0 = time.perf_counter()
            try:
                # Run the blocking CUDA call off the event loop so new requests
                # keep arriving and batching up while this batch is on the GPU.
                logits = await loop.run_in_executor(None, self.det.infer, stacked)
            except Exception as exc:                      # noqa: BLE001
                for b in batch:
                    if not b.future.done():
                        b.future.set_exception(exc)
                continue
            gpu_ms = (time.perf_counter() - t0) * 1000
            self.stats["batches"] += 1
            self.stats["images"] += len(batch)
            self.stats["batch_sizes"].append(len(batch))
            self.stats["gpu_ms"] += gpu_ms
            for b, lg in zip(batch, logits):
                if not b.future.done():
                    b.future.set_result(float(lg))

    def snapshot(self) -> dict:
        sizes = self.stats["batch_sizes"]
        n = self.stats["images"] or 1
        return {
            "batches": self.stats["batches"],
            "images": self.stats["images"],
            "avg_batch_size": round(sum(sizes) / len(sizes), 2) if sizes else 0,
            "gpu_ms_per_img": round(self.stats["gpu_ms"] / n, 3),
            "rejected": self.stats["rejected"],
            "queue_depth": self.q.qsize(),
        }
