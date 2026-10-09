# CrewAI: a two-agent crew (researcher -> writer) using local Ollama
# Install first: pip install crewai
import sys

# Windows consoles default to cp1252 and crash on CrewAI's emoji logging
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(model="ollama/qwen3:8b", base_url="http://localhost:11434")

researcher = Agent(
    role="Researcher",
    goal="Collect the key facts about {topic}",
    backstory="You are a careful analyst who lists accurate, concise facts.",
    llm=llm,
    verbose=True,
)
writer = Agent(
    role="Writer",
    goal="Turn research notes into a short, clear summary",
    backstory="You write plain-English summaries for beginners.",
    llm=llm,
    verbose=True,
)

research_task = Task(
    description="Research the topic: {topic}. List 5 key facts.",
    expected_output="A bullet list of 5 facts.",
    agent=researcher,
)
write_task = Task(
    description="Write a 3-paragraph summary of {topic} from the research notes.",
    expected_output="A 3-paragraph summary.",
    agent=writer,
    context=[research_task],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
)

if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "agentic AI"
    result = crew.kickoff(inputs={"topic": topic})
    print("\n=== Final result ===\n", result)
