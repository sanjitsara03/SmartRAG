from crewai import Agent

from src.agents_src.tools.rag_qa import rag_query_tool
from src.agents_src.tools.web_research import web_search_tool
from src.agents_src.llm.get_llm import get_llm_for_agent


name = "Question Answer Agent"
llm = get_llm_for_agent(name)

qa_agent = Agent(
    role = "Question Answering Agent that uses retrieved documents to answer user queries accurately.",
    llm = llm,
    tools = [rag_query_tool, web_search_tool],
    goal="Answer user queries accurately using internal document retrieval first. "
        "If no sufficient evidence is found internally, perform a targeted web search to gather additional information. "
        "Ensure that all responses are concise, factual, and clearly indicate the source (internal RAG or web).",

    backstory="You are an intelligent research assistant capable of blending internal and external knowledge sources. "
        "You always start with internal document retrieval via RAG to ensure answers are grounded in verified context. "
        "If RAG cannot provide an adequate answer, you independently perform a web search to supplement your response. "
        "Your answers are always transparent, well-reasoned, and backed by credible evidence.",

    verbose=True,
)