"""MCP server #2: summary of the local system hardware."""
import platform
import shutil

import psutil
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hardware", log_level="WARNING")

GB = 1024 ** 3


def _root() -> str:
    return "C:\\" if platform.system() == "Windows" else "/"


@mcp.tool()
def get_cpu_info() -> str:
    """CPU model, core counts and current usage."""
    return (f"CPU: {platform.processor()}; physical cores={psutil.cpu_count(logical=False)}, "
            f"logical cores={psutil.cpu_count()}, usage={psutil.cpu_percent(interval=0.5)}%")


@mcp.tool()
def get_memory_info() -> str:
    """Total and used RAM."""
    m = psutil.virtual_memory()
    return f"RAM: {m.total / GB:.1f} GB total, {m.used / GB:.1f} GB used ({m.percent}%)"


@mcp.tool()
def get_disk_info() -> str:
    """Total and free space of the system drive."""
    d = shutil.disk_usage(_root())
    return f"Disk {_root()}: {d.total / GB:.0f} GB total, {d.free / GB:.0f} GB free"


@mcp.tool()
def get_system_summary() -> str:
    """One-shot summary: OS, CPU, RAM and disk."""
    return "\n".join([
        f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
        get_cpu_info(), get_memory_info(), get_disk_info(),
    ])


if __name__ == "__main__":
    mcp.run()
