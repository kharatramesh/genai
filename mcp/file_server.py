"""MCP server #3: file IO, restricted to the ./sandbox folder (safety!)."""
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("files", log_level="WARNING")

ROOT = (Path(__file__).parent / "sandbox").resolve()
ROOT.mkdir(exist_ok=True)


def _safe(name: str) -> Path:
    """Resolve a name inside ROOT; refuse '..' escapes."""
    p = (ROOT / name).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Access outside the sandbox is not allowed")
    return p


@mcp.tool()
def list_files() -> str:
    """List files in the sandbox folder."""
    names = sorted(p.name for p in ROOT.iterdir() if p.is_file())
    return "\n".join(names) or "(empty)"


@mcp.tool()
def read_file(name: str) -> str:
    """Read a text file from the sandbox."""
    return _safe(name).read_text(encoding="utf-8")


@mcp.tool()
def write_file(name: str, content: str) -> str:
    """Create or overwrite a text file in the sandbox."""
    _safe(name).write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {name}"


@mcp.tool()
def append_file(name: str, content: str) -> str:
    """Append text to a file in the sandbox."""
    with _safe(name).open("a", encoding="utf-8") as f:
        f.write(content)
    return f"Appended {len(content)} chars to {name}"


@mcp.tool()
def delete_file(name: str) -> str:
    """Delete a file from the sandbox."""
    _safe(name).unlink()
    return f"Deleted {name}"


if __name__ == "__main__":
    mcp.run()
