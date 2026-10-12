"""No-LLM smoke test: talk to each MCP server directly, to see raw MCP."""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).parent
CASES = [
    ("weather_server.py", "get_current_weather", {"city": "Pune"}),
    ("hardware_server.py", "get_system_summary", {}),
    ("file_server.py", "write_file", {"name": "hello.txt", "content": "Hello MCP!\n"}),
    ("file_server.py", "read_file", {"name": "hello.txt"}),
    ("file_server.py", "list_files", {}),
    ("file_server.py", "read_file", {"name": "../client.py"}),  # must be blocked
]


async def main():
    for script, tool, args in CASES:
        params = StdioServerParameters(command=sys.executable, args=[str(HERE / script)])
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool(tool, args)
            flag = "ERROR " if res.isError else ""
            print(f"{script}:{tool}{args}\n  -> {flag}{res.content[0].text}\n")


asyncio.run(main())
