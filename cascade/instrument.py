"""Process-level measurements: elapsed (wall-clock) time, CPU time and peak memory.

CP-SAT allocates in C++, invisible to ``tracemalloc``, so memory is read from
the operating system.  On Windows the exact per-process peaks come from
``NtQueryInformationProcess(ProcessVmCounters)``:

* ``peak_virtual``  PeakVirtualSize    -- reserved address space
* ``peak_commit``   PeakPagefileUsage  -- committed private bytes ("Commit size")
* ``peak_wset``     PeakWorkingSetSize -- physical RAM in use

Windows never resets these peaks, which is why every measured request runs in
a fresh process (see ``experiments/isolated.py``).  On other systems the
peaks fall back to ``resource.getrusage`` / psutil.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass

MB = 1024 * 1024


@dataclass
class MemorySnapshot:
    virtual: int = 0
    peak_virtual: int = 0
    commit: int = 0
    peak_commit: int = 0
    wset: int = 0
    peak_wset: int = 0

    def as_mb(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}_mb": round(v / MB, 2) for k, v in asdict(self).items()}


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _VM_COUNTERS(ctypes.Structure):
        _fields_ = [("PeakVirtualSize", ctypes.c_size_t),
                    ("VirtualSize", ctypes.c_size_t),
                    ("PageFaultCount", wintypes.ULONG),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    _ntdll = ctypes.WinDLL("ntdll")
    _kernel32 = ctypes.WinDLL("kernel32")
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _ntdll.NtQueryInformationProcess.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                 wintypes.ULONG, ctypes.POINTER(wintypes.ULONG)]
    _ntdll.NtQueryInformationProcess.restype = ctypes.c_long
    _PROCESS_VM_COUNTERS = 3

    def memory_snapshot() -> MemorySnapshot:
        c = _VM_COUNTERS()
        ret = wintypes.ULONG()
        status = _ntdll.NtQueryInformationProcess(_kernel32.GetCurrentProcess(), _PROCESS_VM_COUNTERS,
                                                  ctypes.byref(c), ctypes.sizeof(c), ctypes.byref(ret))
        if status != 0:
            return _psutil_snapshot()
        return MemorySnapshot(virtual=c.VirtualSize, peak_virtual=c.PeakVirtualSize,
                              commit=c.PagefileUsage, peak_commit=c.PeakPagefileUsage,
                              wset=c.WorkingSetSize, peak_wset=c.PeakWorkingSetSize)
else:
    def memory_snapshot() -> MemorySnapshot:
        return _psutil_snapshot()


def _psutil_snapshot() -> MemorySnapshot:
    try:
        import psutil
        mi = psutil.Process(os.getpid()).memory_info()
        vms, rss = getattr(mi, "vms", 0), getattr(mi, "rss", 0)
    except Exception:          # psutil missing: best effort
        vms = rss = 0
    peak = rss
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:
        pass
    return MemorySnapshot(virtual=vms, peak_virtual=vms, commit=vms, peak_commit=vms,
                          wset=rss, peak_wset=max(rss, peak))


def private_bytes(pid: int) -> int:
    """Current committed private bytes of another process (for the memory guard)."""
    import psutil
    mi = psutil.Process(pid).memory_info()
    return int(getattr(mi, "private", 0) or getattr(mi, "rss", 0))


class Stopwatch:
    """Wall-clock (elapsed) and process CPU time of a block."""

    def __enter__(self) -> "Stopwatch":
        self.t0 = time.perf_counter()
        self.c0 = time.process_time()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self.t0
        self.cpu = time.process_time() - self.c0
