import logging
from crewai.tools import tool
from tavily import TavilyClient
from src.agents_src.config.agent_settings import AgentSettings

# Logger
logger = logging.getLogger(__name__)

@tool
def web_search_tool(query: str) -> dict:
    """
    Performs a web search using the Tavily API and returns summarized search results.

    Args:
        query (str): The search query.

    Returns:
        dict: {
            "answer": str - summarized web content,
            "source_links": list[str] - URLs of top sources
        }
    """
    try:
        settings = AgentSettings()
        tavily_api_key = settings.TAVILY_API_KEY
        client = TavilyClient(api_key=tavily_api_key)

        logger.info(f"Performing Tavily web search for query: {query}")
        response = client.search(query=query, max_results=5)

        results = response.get("results", [])
        if not results:
            logger.warning("No web search results found.")
            return {"answer": "No relevant web results found.", "source_links": []}

        snippets = []
        links = []

        for r in results:
            title = r.get("title", "Untitled")
            content = r.get("content", "")
            url = r.get("url", "")
            links.append(url)
            snippets.append(f"🔹 **{title}**\n{content}\n(Source: {url})")

        summarized = "\n\n".join(snippets)
        logger.info(f"Returning {len(results)} web results.")

        return {"answer": summarized, "source_links": links}

    except Exception as e:
        logger.exception(f"Error in web_search_tool: {e}")
        return {"answer": "", "source_links": [], "error": str(e)}
