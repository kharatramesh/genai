"""Read-only system monitoring tools (psutil)."""
import platform
import time

import psutil
from langchain_core.tools import tool


@tool
def cpu_usage() -> str:
    """Current CPU utilisation (overall + per core) and core count."""
    per_core = psutil.cpu_percent(interval=1, percpu=True)
    return (f"CPU overall: {sum(per_core) / len(per_core):.1f}% | cores: {psutil.cpu_count(logical=False)} physical / "
            f"{psutil.cpu_count()} logical | per-core: {per_core}")


@tool
def memory_usage() -> str:
    """RAM and swap usage."""
    m, s = psutil.virtual_memory(), psutil.swap_memory()

    def gb(b):
        return f"{b / 1024**3:.1f} GB"

    return f"RAM: {gb(m.used)} / {gb(m.total)} ({m.percent}%) | Swap: {gb(s.used)} / {gb(s.total)} ({s.percent}%)"


@tool
def disk_usage() -> str:
    """Usage of every mounted disk partition."""
    out = []
    for part in psutil.disk_partitions():
        try:
            u = psutil.disk_usage(part.mountpoint)
            out.append(f"{part.mountpoint}: {u.used / 1024**3:.0f}/{u.total / 1024**3:.0f} GB ({u.percent}%)")
        except OSError:
            continue
    return "\n".join(out)


@tool
def top_processes(count: int = 5, sort_by: str = "memory") -> str:
    """Top processes by 'memory' or 'cpu'."""
    procs = list(psutil.process_iter(["pid", "name"]))
    for p in procs:  # the first cpu_percent() call always returns 0.0, so prime it, wait, then measure
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    time.sleep(0.5)
    rows = []
    for p in procs:
        try:
            rows.append((p.info["pid"], p.info["name"] or "?", p.memory_percent(), p.cpu_percent(None)))
        except psutil.Error:  # process ended or access denied
            continue
    rows.sort(key=lambda r: r[3] if sort_by == "cpu" else r[2], reverse=True)
    return "\n".join(f"{pid:>6} {name:<28} mem {mem:.1f}%  cpu {cpu:.1f}%" for pid, name, mem, cpu in rows[:count])


@tool
def system_info() -> str:
    """OS, hostname and uptime."""
    up = (time.time() - psutil.boot_time()) / 3600
    return f"{platform.system()} {platform.release()} | host {platform.node()} | uptime {up:.1f} h"


@tool
def network_stats() -> str:
    """Bytes sent/received since boot."""
    n = psutil.net_io_counters()
    return f"sent {n.bytes_sent / 1024**2:.0f} MB, received {n.bytes_recv / 1024**2:.0f} MB"


TOOLS = [system_info, cpu_usage, memory_usage, disk_usage, top_processes, network_stats]
