from crewai import Task 
from pydantic import BaseModel

from src.agents_src.agents.QA_agent import qa_agent

class AnswerStruct(BaseModel):
    answer: str
    sources: list[str]
    tool_used: str
    rationale: str

qa_task = Task(
    agent=qa_agent,
    name="Question Answering Task",
    description="""
    Address the user query "{user_query}" using a Retrieval-Augmented Generation (RAG) approach.
    chat_history: "{chat_history}"
    
    Instructions:
    - Retrieve relevant information from the document store or chat history.
    - Focus on evidence that directly supports the question.
    - Generate a concise, factually accurate response based on the retrieved context.
    - If the answer cannot be derived from the documents or chat history, do not fabricate a response.
      Instead, clearly state that the available knowledge sources do not contain the required information.
    - Maintain transparency by including references, tool usage, and reasoning steps in the output.
    """,
    expected_output="""
    A structured JSON object with the following fields:
    {
      "answer": "Direct response to the query (1–3 paragraphs, clear and accurate). 
                 If no answer is found, return: 'The knowledge source does not contain the required information.'",
      "sources": ["List of document titles, sections, or citations used (empty list if none)"],
      "tool_used": "Name of the retrieval/analysis tool invoked (e.g., RAG Retriever, VectorDB, ChatHistory, etc.)",
      "rationale": "Brief explanation of why this answer was chosen, or why no relevant information was found"
    }
    """,
    output_pydantic=AnswerStruct,
)