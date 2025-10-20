from crewai import Crew

from src.agents_src.agents.QA_agent import qa_agent
from src.agents_src.tasks.QA_task import qa_task

crew = Crew(
    agents=[qa_agent],
    tasks=[qa_task],
    verbose=True,
)