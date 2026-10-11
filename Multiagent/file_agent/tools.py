"""File tools - sandboxed to Multiagent/workspace. Overwrites and deletes need human approval."""
from pathlib import Path

from langchain_core.tools import tool

from common.hitl import rejected, require_approval

WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)


def _safe(name: str) -> Path:
    """Resolve inside the workspace; refuse '../' escapes."""
    p = (WORKSPACE / name).resolve()
    if p != WORKSPACE and WORKSPACE not in p.parents:
        raise ValueError(f"Path '{name}' is outside the workspace")
    return p


@tool
def list_files(subdir: str = "") -> str:
    """List files and folders in the workspace (optionally inside a subfolder)."""
    base = _safe(subdir)
    items = sorted(f"{p.name}{'/' if p.is_dir() else ''}" for p in base.iterdir() if p.name != ".gitkeep")
    return "\n".join(items) or "(empty)"


@tool
def read_file(filename: str) -> str:
    """Read a text file from the workspace."""
    return _safe(filename).read_text(encoding="utf-8")[:8000]


@tool
def write_file(filename: str, content: str) -> str:
    """Create a text file (or overwrite it) in the workspace."""
    p = _safe(filename)
    if p.exists():
        ok, why = require_approval("overwrite_file", f"Overwrite existing file '{filename}' ({p.stat().st_size} bytes)")
        if not ok:
            return rejected(why)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {p.name}"


@tool
def delete_file(filename: str) -> str:
    """Delete a file from the workspace. Requires human approval."""
    p = _safe(filename)
    if not p.is_file():
        return f"'{filename}' does not exist"
    ok, why = require_approval("delete_file", f"Permanently delete '{filename}'")
    if not ok:
        return rejected(why)
    p.unlink()
    return f"Deleted {filename}"


@tool
def search_in_files(text: str) -> str:
    """Find workspace files whose content contains the given text."""
    hits = [str(p.relative_to(WORKSPACE)) for p in WORKSPACE.rglob("*")
            if p.is_file() and text.lower() in p.read_text(encoding="utf-8", errors="ignore").lower()]
    return "\n".join(hits) or "no matches"


TOOLS = [list_files, read_file, write_file, delete_file, search_in_files]
