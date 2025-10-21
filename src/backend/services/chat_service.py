import logging
from src.agents_src.crew import crew

logger = logging.getLogger(__name__)

def get_answer(chat_history: list) -> dict:
    logger.info(f"Received chat history: {chat_history}")
    
    query = chat_history[-1]["content"]
    logger.info(f"Extracted query: {query}")
    prev_history = chat_history[:-1]
    input_data = {
        "user_query": query,
        "chat_history": prev_history
    }
    logger.debug(f"Input data for crew: {input_data}")
    response = crew.kickoff(input_data)
    response_dict = response.to_dict()
    logger.info(f"Generated response: {response_dict}")
    return response_dict
    

