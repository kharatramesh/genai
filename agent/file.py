# Standard library imports only: no third-party dependencies needed
import json              # encode/decode the HTTP request and response bodies
import urllib.request    # talk to the local Ollama server
import shutil            # copy files
import zipfile           # compress files into .zip archives
from pathlib import Path # cross-platform file paths

# All file operations are confined to this folder: ~/hp-ai/agent
workspace_dir = Path.home()/"hp-ai"/"agent"
# Create the workspace if it does not exist yet (no error if it does)
workspace_dir.mkdir(parents=True,exist_ok=True)


# function is tool in agentic AI
def create_file(filename: str , content: str) -> str:
  """Create a file (and any parent folders) in the workspace with the given text."""
  # Build the full path inside the workspace
  file_path = workspace_dir/filename
  # Make sure any sub-folders in the filename exist
  file_path.parent.mkdir(parents=True,exist_ok=True)
  # Write the text as UTF-8 (overwrites an existing file)
  file_path.write_text(content, encoding="utf-8")
  # Return a success message that the model sees as the tool result
  return f"Successfully create file at : {file_path.resolve()}"

def rename_file(old_name: str , new_name: str) -> str:
  """Rename/move a file inside the workspace, refusing paths outside it or existing targets."""
  # resolve() collapses things like ".." so we can check the real location
  old_path = (workspace_dir/old_name).resolve()
  new_path = (workspace_dir/new_name).resolve()
  root = workspace_dir.resolve()
  # Safety check: both paths must be inside the workspace (blocks path traversal)
  if root not in old_path.parents or root not in new_path.parents:
    raise ValueError("Paths must stay inside the workspace")
  # The source must be an existing file
  if not old_path.is_file():
    raise ValueError(f"File not found: {old_name}")
  # Never overwrite an existing file
  if new_path.exists():
    raise ValueError(f"Target already exists: {new_name}")
  # Create destination folders if the new name includes new sub-folders
  new_path.parent.mkdir(parents=True,exist_ok=True)
  old_path.rename(new_path)
  return f"Successfully renamed file to : {new_path}"

def delete_file(filename: str) -> str:
  """Delete a file inside the workspace, refusing paths outside it."""
  file_path = (workspace_dir/filename).resolve()
  # Safety check: must be inside the workspace (blocks path traversal)
  if workspace_dir.resolve() not in file_path.parents:
    raise ValueError("Path must stay inside the workspace")
  if not file_path.is_file():
    raise ValueError(f"File not found: {filename}")
  file_path.unlink()
  return f"Successfully deleted file : {file_path}"

def copy_file(source: str , destination: str) -> str:
  """Copy a file inside the workspace, refusing paths outside it or existing targets."""
  src_path = (workspace_dir/source).resolve()
  dst_path = (workspace_dir/destination).resolve()
  root = workspace_dir.resolve()
  # Safety check: both paths must be inside the workspace (blocks path traversal)
  if root not in src_path.parents or root not in dst_path.parents:
    raise ValueError("Paths must stay inside the workspace")
  if not src_path.is_file():
    raise ValueError(f"File not found: {source}")
  # Never overwrite an existing file
  if dst_path.exists():
    raise ValueError(f"Target already exists: {destination}")
  dst_path.parent.mkdir(parents=True,exist_ok=True)
  # copy2 keeps timestamps and permissions
  shutil.copy2(src_path, dst_path)
  return f"Successfully copied file to : {dst_path}"

def compress_file(filename: str , archive_name: str = "") -> str:
  """Compress a file inside the workspace into a .zip archive (default: <filename>.zip)."""
  file_path = (workspace_dir/filename).resolve()
  root = workspace_dir.resolve()
  if root not in file_path.parents:
    raise ValueError("Path must stay inside the workspace")
  if not file_path.is_file():
    raise ValueError(f"File not found: {filename}")
  archive_path = (workspace_dir/archive_name).resolve() if archive_name else file_path.with_name(file_path.name + ".zip")
  if root not in archive_path.parents:
    raise ValueError("Archive path must stay inside the workspace")
  if archive_path.exists():
    raise ValueError(f"Archive already exists: {archive_path.name}")
  archive_path.parent.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    # Store only the file name so the zip has no absolute paths
    archive.write(file_path, arcname=file_path.name)
  return f"Successfully compressed file to : {archive_path}"

def change_permission(filename: str , mode: str) -> str:
  """Change a file's permissions inside the workspace; mode is an octal string like '644'."""
  file_path = (workspace_dir/filename).resolve()
  if workspace_dir.resolve() not in file_path.parents:
    raise ValueError("Path must stay inside the workspace")
  if not file_path.is_file():
    raise ValueError(f"File not found: {filename}")
  try:
    mode_value = int(str(mode), 8)
  except ValueError:
    raise ValueError(f"Mode must be an octal string like '644', got: {mode}")
  if not 0 <= mode_value <= 0o777:
    raise ValueError("Mode must be between 000 and 777")
  # Note: on Windows only the owner write bit has an effect (read-only flag)
  file_path.chmod(mode_value)
  return f"Successfully changed permission of {file_path} to {mode_value:03o}"



# Registry mapping the tool name the model uses to the real Python function
TOOLS_MAP = {
    "create_file": create_file,
    "rename_file": rename_file,
    "delete_file": delete_file,
    "copy_file": copy_file,
    "compress_file": compress_file,
    "change_permission": change_permission,
}


# JSON tool schema: tells the model which tools exist and what arguments they take
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a file in the workspace with the given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Relative file path"},
                    "content": {"type": "string", "description": "Text to write into the file"}
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rename_file",
            "description": "Rename or move a file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_name": {"type": "string", "description": "Current relative path"},
                    "new_name": {"type": "string", "description": "New relative path"}
                },
                "required": ["old_name", "new_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Relative file path"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "copy_file",
            "description": "Copy a file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Relative path of the file to copy"},
                    "destination": {"type": "string", "description": "Relative path of the copy"}
                },
                "required": ["source", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compress_file",
            "description": "Compress a file in the workspace into a .zip archive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Relative path of the file to compress"},
                    "archive_name": {"type": "string", "description": "Optional relative path of the .zip (default: <filename>.zip)"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "change_permission",
            "description": "Change the permissions of a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Relative file path"},
                    "mode": {"type": "string", "description": "Octal permission string, e.g. '644' or '755'"}
                },
                "required": ["filename", "mode"]
            }
        }
    }
]


# call model (Ollama's HTTP API itself only accepts JSON)
def call_local_ollama(messages, model_name="qwen3:8b"):
  """Send the chat messages to the local Ollama server and return its JSON response."""
  # Ollama's OpenAI-compatible chat endpoint
  url = "http://localhost:11434/v1/chat/completions"
  payload = {
    "model": model_name,
    "messages": messages,
    "tools": TOOLS,   # let the model call our functions
    "stream": False   # get the full reply in one response
}
  request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

  try:
        # Local models can be slow, so allow up to 3 minutes
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
  except urllib.error.URLError as error:
        # Give a friendlier message when the server is not reachable
        raise RuntimeError(
            f"Could not connect to Ollama. Check that Ollama is running: {error}"
        ) from error

# Agent loop: ask the model, execute tools, and continue
def run_agent(user_prompt, model_name="qwen3:8b"):
    """Run the agent loop: query the model, execute any requested tool, feed back the result."""
    # Conversation history: system instructions first, then the user's request
    messages = [
        {
            "role": "system",
            "content": (
                "You are a local file assistant. "
                "Use relative filenames only. Do not claim a file was created "
                "unless the tool reports success."
            )
        },
        {"role": "user", "content": user_prompt}
    ]

    # Cap the number of model/tool round trips to avoid infinite loops
    max_iterations = 5

    for _ in range(max_iterations):
        # Ask the model what to do next
        result = call_local_ollama(messages, model_name)
        message = result["choices"][0]["message"]
        # Keep the model's message in the history for context
        messages.append(message)

        # No tool calls: this is the final answer, so print it and stop
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            print(message["content"])
            return

        # The model may request several tools at once; run each one
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            if tool_name not in TOOLS_MAP:
                tool_result = f"Error: Unknown tool {tool_name}"
            else:
                try:
                    # Arguments arrive as a JSON string, e.g. '{"filename": "a.txt", ...}'
                    arguments = json.loads(tool_call["function"]["arguments"])
                    tool_result = TOOLS_MAP[tool_name](**arguments)
                except (ValueError, TypeError, OSError) as error:
                    # Report failures back to the model so it can correct itself
                    tool_result = f"Tool execution failed: {error}"

            # Feed the tool result back, linked to the call it answers
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(tool_result)
            })

    print("Stopped: maximum agent iterations reached.")


def format_tools() -> str:
    """Return every registered tool with its description and arguments as text."""
    lines = []
    for tool in TOOLS:
        function = tool["function"]
        params = function["parameters"]
        required = params.get("required", [])
        # Mark optional arguments with a trailing '?'
        args = ", ".join(
            name if name in required else f"{name}?"
            for name in params["properties"]
        )
        lines.append(f"  {function['name']}({args})\n      {function['description']}")
    return "\n".join(lines)


def list_tools():
    """Print the tool list (the /tools command)."""
    print(format_tools())


# Run only when executed directly (not when imported)
if __name__ == "__main__":
    print("Type a request, /tools to list tools, or /exit to quit.")
    while True:
        prompt = input("What should the agent do? ").strip()
        if not prompt:
            continue
        # Slash commands are handled locally and never sent to the model
        if prompt == "/tools":
            list_tools()
        elif prompt in ("/exit", "/quit"):
            break
        else:
            run_agent(prompt)
