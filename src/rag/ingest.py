import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


from src.rag.config.rag_settings import DocIngestionSettings

#Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = DocIngestionSettings()

# Initialize embedding model
logger.info("Loading HuggingFace embedding model...")

embed_model = HuggingFaceEmbeddings()


def load_documents(docs_dir_path: str) -> List:
    docs_path = Path(docs_dir_path)
    if not docs_path.exists() or not docs_path.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir_path}")

    logger.info(f"Loading documents from directory: {docs_dir_path}")

    documents = []

    text_loader = DirectoryLoader(
    docs_dir_path,
    glob="**/*.*",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
    show_progress=True,
    use_multithreading=True,
    )
    documents.extend(text_loader.load())

    return documents


def build_vector_store():
    logger.info("Starting vector store ingestion process.")
    try:
        docs_dir_path = settings.DOCUMENTS_DIR
        vector_store_path = settings.VECTOR_STORE_DIR
        collection_name = settings.COLLECTION_NAME

        #Load documents
        documents = load_documents(docs_dir_path)
        if not documents:
            logger.warning("No documents found to ingest.")
            return 1
        logger.info(f"Loaded {len(documents)} documents.")

        #Chunking documents
        logger.info("Parsing documents into chunks (nodes).")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""],
        )
        splits = splitter.split_documents(documents)
        logger.info(f"Parsed {len(splits)} chunks.")

       
        logger.info(f"Initializing Chroma persistent store at: {vector_store_path}")
        logger.info(f"Creating/using Chroma collection: {collection_name}")

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embed_model,
            persist_directory=vector_store_path,
            collection_name=collection_name,
        )

        
        vectordb.persist()
        logger.info("Vector store build successfully.")
        return 0

    except Exception as e:
        logger.error(f"Error during vector store build: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    build_vector_store()
