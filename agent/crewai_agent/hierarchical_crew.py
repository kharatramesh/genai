# CrewAI hierarchical process: a MANAGER agent plans and delegates to specialists
# Concepts: Process.hierarchical, manager_llm, allow_delegation, no fixed task order
import sys

# Windows consoles default to cp1252 and crash on CrewAI's emoji logging
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(model="ollama/qwen3:8b", base_url="http://localhost:11434")

planner = Agent(
    role="Trip Planner",
    goal="Design a day-by-day itinerary for {city}",
    backstory="You build realistic, well-paced travel plans.",
    llm=llm,
    allow_delegation=False,
)
budgeter = Agent(
    role="Budget Analyst",
    goal="Estimate the cost of a trip to {city}",
    backstory="You give simple per-day cost estimates for travel, food and stay.",
    llm=llm,
    allow_delegation=False,
)

# One broad task: the manager decides who does which part and in what order
trip_task = Task(
    description="Plan a 3-day trip to {city} and include an estimated total budget.",
    expected_output="An itinerary for 3 days followed by a budget summary.",
)

crew = Crew(
    agents=[planner, budgeter],
    tasks=[trip_task],
    process=Process.hierarchical,
    manager_llm=llm,  # the manager is created automatically from this model
)

if __name__ == "__main__":
    city = " ".join(sys.argv[1:]) or "Jaipur"
    result = crew.kickoff(inputs={"city": city})
    print("\n=== Final result ===\n", result)
