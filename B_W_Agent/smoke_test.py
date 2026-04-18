import os
from crewai import Crew, Agent, Task, LLM

# ---------------------------
# LLM - Ultra-light configuration for low RAM
# ---------------------------
llm = LLM(
    model="ollama/llama3.2:1b",      # Smallest reliable model
    base_url="http://localhost:11434",
    temperature=0.3,
    # Low-memory settings
    num_ctx=1024,
    num_gpu=0,
)

# ---------------------------
# Agents
# ---------------------------
researcher = Agent(
    role="Researcher",
    goal="Fetch background information on the given topic",
    backstory="You are an expert web researcher.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

writer = Agent(
    role="Writer",
    goal="Draft an engaging blog post based on research",
    backstory="You are a skilled technical writer.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# ---------------------------
# Tasks
# ---------------------------
research_task = Task(
    description="Research the topic 'Why local AI matters' and provide key findings.",
    agent=researcher,
    expected_output="A bullet-point summary of key facts and insights about local AI.",
)

write_task = Task(
    description="Write a short blog draft about 'Why local AI matters' using the research provided.",
    agent=writer,
    expected_output="A 2-3 paragraph markdown draft introducing the benefits of local AI.",
)

# ---------------------------
# Crew
# ---------------------------
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    verbose=True,
    memory=False,
)

# ---------------------------
# Execute
# ---------------------------
print("🚀 Starting ultra-light CrewAI smoke test with llama3.2:1b...")
result = crew.kickoff()
print("\n" + "=" * 60 + " FINAL RESULT " + "=" * 60)
print(result)
print("=" * 120)