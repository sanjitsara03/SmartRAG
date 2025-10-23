import logging
from crewai.tools import tool

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq

from src.agents_src.config.agent_settings import AgentSettings

#Logger
logger = logging.getLogger(__name__)


logger.info("Loading HuggingFace embedding model...")
embed_model =  OpenAIEmbeddings(
    model="text-embedding-3-small"
)


@tool
def rag_query_tool(query: str) -> dict:
    """
    Answers a query by retrieving relevant documents and generating a response.
    Returns both the generated answer and the source file names from which the information was retrieved.

    Args:
        query (str): The input query string to be processed.

    Returns:
        dict: A dictionary with the following keys:
            - 'answer': The generated answer string.
            - 'source_files': List of source file names used for retrieval.
    """
    try:
        settings = AgentSettings()
        vector_store_path = settings.VECTOR_STORE_DIR
        collection_name = settings.COLLECTION_NAME

        #Initialize LLM 
        llm = ChatGroq(
            model=settings.MODEL_NAME,
            temperature=settings.MODEL_TEMPERATURE,
        )

        
        logger.info(f"Loading Chroma collection: {collection_name} from {vector_store_path}")
        vectordb = Chroma(
            persist_directory=vector_store_path,
            collection_name=collection_name,
            embedding_function=embed_model,
        )

        #Retrieve top matches
        logger.info(f"Performing similarity search for query: {query}")
        retrieved_docs = vectordb.similarity_search(query, k=3)

        if not retrieved_docs:
            logger.warning("No relevant documents found.")
            return {"answer": "No relevant documents found.", "source_files": []}

        #Combine retrieved chunks into context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        
        source_files = list({
            doc.metadata.get("file_name")
            or doc.metadata.get("source")
            or doc.metadata.get("path")
            or doc.metadata.get("file_path")
            for doc in retrieved_docs
            if doc.metadata
        })

        return {"answer": answer, "source_files": source_files}

    except Exception as e:
        logger.exception(f"Error in rag_query_tool: {e}")
        return {"answer": "", "source_files": [], "error": str(e)}



