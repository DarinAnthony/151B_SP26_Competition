"""Stage-level wall-clock timing for experiment runs.

Usage:

    registry = TimingsRegistry()
    with use_registry(registry):
        with Timer("generate.forward", cuda_sync=True):
            model.generate(...)
    print(registry.render_table("timings"))

`Timer` writes into the registry bound to the current `contextvars` scope, so
shared/ code can be instrumented without threading a registry parameter through
every call site. When no registry is active, `Timer` is a no-op — other
experiments importing instrumented modules pay nothing.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, Optional

_active: ContextVar[Optional["TimingsRegistry"]] = ContextVar("telemetry_active", default=None)


@dataclass
class _StageStats:
    count: int = 0
    total_seconds: float = 0.0


@dataclass
class TimingsRegistry:
    stages: dict[str, _StageStats] = field(default_factory=dict)

    def add(self, name: str, seconds: float) -> None:
        s = self.stages.setdefault(name, _StageStats())
        s.count += 1
        s.total_seconds += seconds

    def merge(self, other: "TimingsRegistry") -> None:
        for name, s in other.stages.items():
            mine = self.stages.setdefault(name, _StageStats())
            mine.count += s.count
            mine.total_seconds += s.total_seconds

    def render_table(self, title: str) -> str:
        if not self.stages:
            return f"\n[{title}] (no timings recorded)\n"
        total = sum(s.total_seconds for s in self.stages.values())
        rows = sorted(self.stages.items(), key=lambda kv: kv[1].total_seconds, reverse=True)
        header = f"{'stage':<28} {'count':>7} {'total(s)':>10} {'mean(ms)':>10} {'% sum':>8}"
        lines = [
            "",
            f"[{title}]",
            "-" * len(header),
            header,
            "-" * len(header),
        ]
        for name, s in rows:
            mean_ms = (s.total_seconds / s.count) * 1000.0 if s.count else 0.0
            pct = (s.total_seconds / total * 100.0) if total else 0.0
            lines.append(
                f"{name:<28} {s.count:>7d} {s.total_seconds:>10.2f} {mean_ms:>10.1f} {pct:>7.1f}%"
            )
        lines.append("-" * len(header))
        lines.append(f"sum of stage times: {total:.2f}s "
                     f"(stages overlap if nested; use this as a relative breakdown, not wall-clock)")
        return "\n".join(lines) + "\n"


@contextmanager
def use_registry(registry: TimingsRegistry) -> Iterator[TimingsRegistry]:
    token = _active.set(registry)
    try:
        yield registry
    finally:
        _active.reset(token)


class Timer:
    """Context manager that records elapsed seconds into the active registry.

    `cuda_sync=True` calls `torch.cuda.synchronize()` before stopping the clock,
    so GPU forward-pass timings reflect actual execution rather than kernel
    launch. Torch is imported lazily inside __exit__ so this module stays
    torch-free for callers that don't need it.
    """

    __slots__ = ("name", "cuda_sync", "_t0", "_registry")

    def __init__(self, name: str, cuda_sync: bool = False):
        self.name = name
        self.cuda_sync = cuda_sync
        self._t0 = 0.0
        self._registry: Optional[TimingsRegistry] = None

    def __enter__(self) -> "Timer":
        self._registry = _active.get()
        if self._registry is not None:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._registry is None:
            return
        if self.cuda_sync:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass
        self._registry.add(self.name, time.perf_counter() - self._t0)
