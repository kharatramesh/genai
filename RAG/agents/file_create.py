import os
import json
import urllib.request
from pathlib import Path

workspace_dir = Path.home()/"hp-ai"/"agent"
workspace_dir.mkdir(parents=True,exist_ok=True)

# function is tool in agentic AI
def create_file(filename: str , content: str) -> str:
  file_path = workspace_dir/filename
  file_path.parent.mkdir(parents=True,exist_ok=True)
  file_path.write_text(content, encoding="utf-8")
  return f"Successfully create file at : {file_path.resolve()}"
# def rename_file(fname: str , content: str) -> str:
#   file_path = workspace_dir/fname
#   file_path.parent.mkdir(parents=True,exists_ok=True)
#   file_path.write_text(content, encoding="utf-8")
#   return f"Successfully create file at : {file_path.resolve()}"
TOOLS_MAP = {
  "create_file": create_file
}
# OpenAI / ollama- compitable tool schema
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                  "filename":{
                    "type": "string",
                    "descripiton": "use relative path"
                  },
                  "content":{}
                },
                "required": ["filename","content"],
            },
        },
    }]



# call model
def call_local_ollama(messages, model_name="qwen3:8b"):
  url = "http://localhost:11434/v1/chat/completions"
  payload = {
    "model": model_name,
    "messages": messages,
    "tools": tools_schema,
    "stream": False
}
  request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

  try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
  except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not connect to Ollama. Check that Ollama is running: {error}"
        ) from error

# Agent loop: ask the model, execute tools, and continue
def run_agent(user_prompt, model_name="qwen3:8b"):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a local file assistant. "
                "When asked to create a file, call the create_file tool. "
                "Use relative filenames only. Do not claim a file was created "
                "unless the tool reports success."
            )
        },
        {"role": "user", "content": user_prompt}
    ]

    max_iterations = 5

    for _ in range(max_iterations):
        result = call_local_ollama(messages, model_name)
        assistant_message = result["choices"][0]["message"]

        messages.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls", [])

        if not tool_calls:
            print(assistant_message.get("content", ""))
            return

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]

            if function_name not in TOOLS_MAP:
                tool_result = f"Error: Unknown tool {function_name}"
            else:
                try:
                    arguments = json.loads(
                        tool_call["function"]["arguments"]
                    )
                    tool_result = TOOLS_MAP[function_name](**arguments)
                except (ValueError, TypeError, OSError) as error:
                    tool_result = f"Tool execution failed: {error}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result
            })

    print("Stopped: maximum agent iterations reached.")


if __name__ == "__main__":
    prompt = input("What should the agent do? ")
    run_agent(prompt)
