# Standard library imports only: no third-party dependencies needed
import json              # encode/decode the HTTP request and response bodies
import urllib.request    # talk to the local Ollama server
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



# Registry mapping the tool name the model uses to the real Python function
TOOLS_MAP = {"create_file": create_file, "rename_file": rename_file, "delete_file": delete_file}


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


# Run only when executed directly (not when imported)
if __name__ == "__main__":
    prompt = input("What should the agent do? ")
    run_agent(prompt)
