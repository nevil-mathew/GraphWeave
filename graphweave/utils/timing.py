"""Lightweight step timing and RAM instrumentation for the GraphWeave pipeline."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

try:
    import psutil as _psutil
    _PROC = _psutil.Process(os.getpid())
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import resource as _resource
    _HAS_RESOURCE = True
except ImportError:  # Windows has no resource module
    _HAS_RESOURCE = False


def _peak_rss_gb() -> float:
    """All-time peak RSS for this process in GB (monotone high-water mark)."""
    if not _HAS_RESOURCE:
        return 0.0
    maxrss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    # Linux: ru_maxrss in KB; macOS: in bytes
    if sys.platform == "darwin":
        return maxrss / 1024 ** 3
    return maxrss / 1024 ** 2  # KB → GB (1 GB = 1024*1024 KB)


@contextmanager
def step_timer(label: str | None = None, verbose: bool = True, indent: int = 6):
    """Print elapsed wall time and memory stats on exit when verbose=True.

    Tracks three memory signals:
    - ``Δproc``: delta of Python-process RSS (current allocations, can drop if temporaries freed)
    - ``rss``:   current process RSS at step exit
    - ``peak``:  all-time max RSS the process has *ever* reached (monotone — captures spikes
                 even if the step freed memory before exiting); ``↑`` marks a new high
    - ``Δsys`` / ``sys``: system-wide used RAM (includes OS, GPU drivers, other processes)

    Output (indent=6)::

        [tfidf] → 3.2 s  |  Δproc +890 MB  (rss: 3.5 GB  peak: 8.5 GB)  |  Δsys +1.2 GB  (sys: 11.8 GB)
    """
    if not verbose:
        yield
        return

    if _HAS_PSUTIL:
        rss_before = _PROC.memory_info().rss
        sys_before = _psutil.virtual_memory().used
    else:
        rss_before = sys_before = 0
    peak_before = _peak_rss_gb()
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0

    pad = " " * indent
    if _HAS_PSUTIL:
        rss_after = _PROC.memory_info().rss
        sys_after = _psutil.virtual_memory().used
        peak_after = _peak_rss_gb()

        delta_proc_mb = (rss_after - rss_before) / 1024 ** 2
        delta_sys_gb = (sys_after - sys_before) / 1024 ** 3
        rss_gb = rss_after / 1024 ** 3
        sys_gb = sys_after / 1024 ** 3

        sign_proc = "+" if delta_proc_mb >= 0 else ""
        sign_sys = "+" if delta_sys_gb >= 0 else ""
        peak_marker = " ↑" if peak_after > peak_before + 0.05 else ""

        suffix = (
            f"Δproc {sign_proc}{delta_proc_mb:.0f} MB"
            f"  (rss: {rss_gb:.1f} GB  peak: {peak_after:.1f} GB{peak_marker})"
            f"  |  Δsys {sign_sys}{delta_sys_gb:.1f} GB  (sys: {sys_gb:.1f} GB)"
        )
    elif _HAS_RESOURCE:
        peak_after = _peak_rss_gb()
        peak_marker = " ↑" if peak_after > peak_before + 0.05 else ""
        suffix = f"peak: {peak_after:.1f} GB{peak_marker}  (install psutil for full RAM stats)"
    else:
        suffix = "(install psutil for RAM tracking)"

    tag = f"[{label}] " if label else ""
    print(f"{pad}{tag}→ {elapsed:.1f} s  |  {suffix}")
