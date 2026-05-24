"""Lightweight step timing and RAM instrumentation for the TriTopic pipeline."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

try:
    import psutil as _psutil
    _PROC = _psutil.Process(os.getpid())
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@contextmanager
def step_timer(label: str | None = None, verbose: bool = True, indent: int = 6):
    """Print elapsed wall time and process RSS delta on exit when verbose=True.

    Usage::

        with step_timer("tfidf", verbose=self.verbose):
            matrix = vectorizer.fit_transform(docs)

    Output (indent=6)::

        [tfidf] → 3.2 s  |  ΔRAM +890 MB  (proc: 3.5 GB)
    """
    if not verbose:
        yield
        return

    rss_before = _PROC.memory_info().rss if _HAS_PSUTIL else 0
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0

    pad = " " * indent
    if _HAS_PSUTIL:
        rss_after = _PROC.memory_info().rss
        delta_mb = (rss_after - rss_before) / 1024**2
        total_gb = rss_after / 1024**3
        sign = "+" if delta_mb >= 0 else ""
        suffix = f"ΔRAM {sign}{delta_mb:.0f} MB  (proc: {total_gb:.1f} GB)"
    else:
        suffix = "(install psutil for RAM tracking)"

    tag = f"[{label}] " if label else ""
    print(f"{pad}{tag}→ {elapsed:.1f} s  |  {suffix}")
