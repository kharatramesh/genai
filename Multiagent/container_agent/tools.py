"""Docker lifecycle tools via the docker CLI. No shell is used; names and images are validated."""
import re
import shutil
import subprocess

from langchain_core.tools import tool

from common.hitl import rejected, require_approval

_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
_IMAGE = re.compile(r"^[a-z0-9][a-z0-9_./:@-]{0,127}$")
_PORTS = re.compile(r"^\d{1,5}:\d{1,5}$")


def _docker(*args: str, timeout: int = 120) -> str:
    if not shutil.which("docker"):
        return "ERROR: docker CLI not found. Install Docker Desktop and make sure it is running."
    try:
        r = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "ERROR: docker command timed out"
    return (r.stdout + r.stderr).strip()[:4000] or "(no output)"


def _check(name: str):
    if not _NAME.match(name):
        raise ValueError(f"invalid container name '{name}'")


@tool
def list_containers(all_containers: bool = True) -> str:
    """List containers (running and stopped by default)."""
    args = ["ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"]
    return _docker(*(args + ["-a"] if all_containers else args))


@tool
def list_images() -> str:
    """List local Docker images."""
    return _docker("images", "--format", "{{.Repository}}:{{.Tag}}\t{{.Size}}")


@tool
def run_container(image: str, name: str, port_mapping: str = "") -> str:
    """Create and start a detached container, e.g. image='nginx:alpine', name='web', port_mapping='8080:80'. Needs approval."""
    _check(name)
    if not _IMAGE.match(image):
        return f"invalid image '{image}'"
    if port_mapping and not _PORTS.match(port_mapping):
        return f"invalid port mapping '{port_mapping}' (use HOST:CONTAINER)"
    cmd = ["run", "-d", "--name", name] + (["-p", port_mapping] if port_mapping else []) + [image]
    ok, why = require_approval("run_container", "docker " + " ".join(cmd))
    return _docker(*cmd, timeout=300) if ok else rejected(why)


@tool
def start_container(name: str) -> str:
    """Start a stopped container."""
    _check(name)
    return _docker("start", name)


@tool
def stop_container(name: str) -> str:
    """Stop a running container. Needs approval."""
    _check(name)
    ok, why = require_approval("stop_container", f"docker stop {name}")
    return _docker("stop", name) if ok else rejected(why)


@tool
def restart_container(name: str) -> str:
    """Restart a container. Needs approval."""
    _check(name)
    ok, why = require_approval("restart_container", f"docker restart {name}")
    return _docker("restart", name) if ok else rejected(why)


@tool
def remove_container(name: str) -> str:
    """Force-remove a container. Destructive; needs approval."""
    _check(name)
    ok, why = require_approval("remove_container", f"docker rm -f {name}  (container and its writable layer are lost)")
    return _docker("rm", "-f", name) if ok else rejected(why)


@tool
def container_logs(name: str, tail: int = 30) -> str:
    """Last N log lines of a container."""
    _check(name)
    return _docker("logs", "--tail", str(min(tail, 200)), name)


@tool
def container_stats(name: str) -> str:
    """One-shot CPU / memory usage of a container."""
    _check(name)
    return _docker("stats", "--no-stream", "--format", "cpu {{.CPUPerc}} | mem {{.MemUsage}}", name)


TOOLS = [list_containers, list_images, run_container, start_container, stop_container,
         restart_container, remove_container, container_logs, container_stats]
