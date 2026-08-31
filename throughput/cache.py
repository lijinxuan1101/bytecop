"""Content-hash dedup. Re-uploads and reposts are a large slice of real traffic;
a hit costs a dict lookup instead of 19.7 ms of decode + inference."""
from __future__ import annotations

import hashlib
from collections import OrderedDict


class HashCache:
    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self._d: OrderedDict[str, float] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    def get(self, k: str) -> float | None:
        if k in self._d:
            self._d.move_to_end(k)
            self.hits += 1
            return self._d[k]
        self.misses += 1
        return None

    def put(self, k: str, logit: float) -> None:
        self._d[k] = logit
        self._d.move_to_end(k)
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)

    def snapshot(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "size": len(self._d),
                "hit_rate": round(self.hits / total, 4) if total else 0.0}
