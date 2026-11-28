"""Adaptive memory estimates, separated from archive and decoded-pixel limits."""
from __future__ import annotations
from dataclasses import dataclass
import os
import sys
from pathlib import Path

MIB=1024**2

def _windows_peak_rss_bytes():
    """PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize from the current process."""
    import ctypes as C
    from ctypes import wintypes as W

    class MemoryCounters(C.Structure):
        _fields_ = [("cb", W.DWORD), ("PageFaultCount", W.DWORD)] + [
            (name, C.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
            )
        ]

    get_process = C.windll.kernel32.GetCurrentProcess
    get_process.argtypes = []
    get_process.restype = W.HANDLE
    get_memory = C.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [W.HANDLE, C.POINTER(MemoryCounters), W.DWORD]
    get_memory.restype = W.BOOL
    counters = MemoryCounters()
    counters.cb = C.sizeof(counters)
    if not get_memory(get_process(), C.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize)


def peak_rss_mib() -> float | None:
    """Process high-water resident memory, in MiB; None when unavailable.

    Linux reports KiB via getrusage, macOS reports bytes, and Windows
    reports peak working-set bytes. No fabricated metric on WebAssembly.
    """
    try:
        if sys.platform == "win32":
            size = _windows_peak_rss_bytes()
            return size / MIB if size is not None else None
        if sys.platform not in ("linux", "darwin"):
            return None
        import resource
        scale = MIB if sys.platform == "darwin" else 1024
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / scale
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def available_memory():
    estimates=[]
    if sys.platform=="win32":
        try:
            import ctypes as C
            class MemoryStatus(C.Structure):
                _fields_=[("length",C.c_uint32),("load",C.c_uint32)]+[(key,C.c_uint64) for key in
                    ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","avail_extended")]
            status=MemoryStatus();status.length=C.sizeof(status)
            if C.windll.kernel32.GlobalMemoryStatusEx(C.byref(status)):
                estimates.append(int(status.avail_phys))
        except (OSError,AttributeError):pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                estimates.append(int(line.split()[1])*1024);break
    except (OSError,ValueError,IndexError):pass
    try:estimates.append(os.sysconf("SC_AVPHYS_PAGES")*os.sysconf("SC_PAGE_SIZE"))
    except (OSError,ValueError,AttributeError):pass
    for cap,used in (("/sys/fs/cgroup/memory.max","/sys/fs/cgroup/memory.current"),
                     ("/sys/fs/cgroup/memory/memory.limit_in_bytes","/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            limit=int(Path(cap).read_text());usage=int(Path(used).read_text())
            if 0<limit<2**60:estimates.append(max(0,limit-usage))
        except (OSError,ValueError):pass
    return min((v for v in estimates if v>=0),default=None)


@dataclass(frozen=True)
class ResourceBudget:
    memory_bytes: int
    source: str
    available_bytes: int|None
    browser: bool=False

    @property
    def working_pixels(self):return max(1,self.memory_bytes//18)
    @property
    def canvas_pixels(self):return max(1,self.memory_bytes//24)
    @property
    def arc_records(self):return max(1024,min(2**30,self.memory_bytes//(3*48)))
    @property
    def vote_entries(self):return max(1024,min(2**30,self.memory_bytes//(3*72)))
    @property
    def packed_bytes(self):return max(1,self.memory_bytes//3)
    @property
    def gap_pair_budget(self):return max(16,self.memory_bytes//(128*1024))
    def describe(self):
        return {"source":self.source,"memory_budget_mib":round(self.memory_bytes/MIB,1),
                "available_memory_mib":round(self.available_bytes/MIB,1) if self.available_bytes is not None else None,
                "working_pixels":self.working_pixels,"canvas_pixels":self.canvas_pixels,
                "arc_records_per_shard":self.arc_records,"vote_entries":self.vote_entries,
                "gap_pair_budget":self.gap_pair_budget}


def resource_budget(*,browser=False,override_mib=None):
    if browser:
        available=None;default=256*MIB;source="browser-worker-budget"
    else:
        available=available_memory()
        default=int(.55*available) if available is not None else 512*MIB
        source="available-host-memory" if available is not None else "portable-fallback"
    raw=override_mib if override_mib is not None else os.getenv("SHARDELIX_MEMORY_MIB")
    if raw is not None:
        try:value=int(raw)*MIB
        except (TypeError,ValueError):raise ValueError("SHARDELIX_MEMORY_MIB must be an integer >= 32") from None
        if value<32*MIB:raise ValueError("SHARDELIX_MEMORY_MIB must be >= 32")
        default=min(value,default) if available is not None or browser else value
        source+="; explicit lower budget"
    if default<32*MIB:
        raise MemoryError("Insufficient available memory for scan reconstruction (need at least 32 MiB headroom)")
    return ResourceBudget(default,source,available,browser)
