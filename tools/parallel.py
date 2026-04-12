"""Parallel execution helpers for the generation pipeline.

All generation steps in this pipeline are I/O-bound — they sit in HTTP polling
loops waiting for someone else's GPU. ThreadPoolExecutor is the right tool:
threads are cheap, the GIL releases on network I/O, and we avoid rewriting
every tool as async.

Usage:
    results = run_parallel(
        segments,
        lambda seg, idx: generate_one_image(seg),
        max_workers=5,
        label="images",
    )

Results preserve input order. Exceptions are captured per-item and surfaced
either as a list (strict=False) or re-raised at the end (strict=True).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Single global print lock so concurrent log lines don't interleave mid-line.
_print_lock = threading.Lock()


def safe_print(msg: str) -> None:
    """Thread-safe print. Use this from inside parallel workers."""
    with _print_lock:
        print(msg, flush=True)


@dataclass
class ParallelResult(Generic[R]):
    """Outcome of running a single item through run_parallel."""

    index: int
    value: R | None
    error: Exception | None
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        return self.error is None


def run_parallel(
    items: list[T],
    fn: Callable[[T, int], R],
    max_workers: int,
    label: str = "parallel",
    strict: bool = False,
) -> list[ParallelResult[R]]:
    """Run `fn(item, index)` across `items` using a thread pool.

    Args:
        items: Input list. Order is preserved in the output list.
        fn: Worker function. Receives (item, index). Should do its own logging
            via `safe_print` so lines don't interleave.
        max_workers: Maximum concurrent workers. Cap around provider rate limits.
        label: Short name for logs (e.g. "images", "avatars").
        strict: If True, re-raise the first exception after all workers finish
            (still lets every item attempt). If False, return errors in results.

    Returns:
        List of ParallelResult in the same order as `items`.
    """
    if not items:
        return []

    n = len(items)
    workers = max(1, min(max_workers, n))
    safe_print(f"  ⟳ {label}: running {n} job(s) across {workers} worker(s)")

    results: list[ParallelResult[R] | None] = [None] * n
    started = time.monotonic()

    def _wrap(index: int, item: T) -> ParallelResult[R]:
        t0 = time.monotonic()
        try:
            value = fn(item, index)
            return ParallelResult(index=index, value=value, error=None,
                                  elapsed_seconds=time.monotonic() - t0)
        except Exception as exc:  # noqa: BLE001
            return ParallelResult(index=index, value=None, error=exc,
                                  elapsed_seconds=time.monotonic() - t0)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_wrap, i, item): i
            for i, item in enumerate(items)
        }
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            results[res.index] = res
            completed += 1
            status = "✓" if res.ok else "✗"
            tag = f"[{completed}/{n}]"
            if res.ok:
                safe_print(f"    {status} {label} {tag} item {res.index + 1} "
                           f"({res.elapsed_seconds:.1f}s)")
            else:
                safe_print(f"    {status} {label} {tag} item {res.index + 1} "
                           f"failed: {res.error}")

    elapsed_total = time.monotonic() - started
    ok_count = sum(1 for r in results if r and r.ok)
    safe_print(f"  ⟳ {label}: {ok_count}/{n} succeeded in {elapsed_total:.1f}s")

    final: list[ParallelResult[R]] = [r for r in results if r is not None]
    if strict:
        first_err = next((r.error for r in final if r.error is not None), None)
        if first_err is not None:
            raise first_err
    return final


def run_stages_in_parallel(stages: list[tuple[str, Callable[[], None]]]) -> None:
    """Run multiple top-level stages concurrently.

    Used by the orchestrator to fan out "images+voices" or "avatars+animations"
    stages that have no mutual dependency. Each stage is a zero-arg callable
    that does its own logging. Exceptions from one stage are re-raised after
    all stages finish so we don't partially corrupt run state.
    """
    if not stages:
        return

    safe_print(f"  ⟳ Running {len(stages)} stage(s) concurrently: "
               f"{', '.join(name for name, _ in stages)}")

    errors: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=len(stages)) as pool:
        futures = {
            pool.submit(fn): name for name, fn in stages
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                errors.append((name, exc))
                safe_print(f"  ✗ Stage '{name}' failed: {exc}")

    if errors:
        first_name, first_err = errors[0]
        raise RuntimeError(
            f"{len(errors)} stage(s) failed; first: {first_name}: {first_err}"
        ) from first_err
