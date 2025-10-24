from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class AgentSettings(BaseSettings):
    GROQ_API_KEY: str
    MODEL_NAME: str
    MODEL_TEMPERATURE: float
    COLLECTION_NAME: str
    DOCUMENTS_DIR: str
    VECTOR_STORE_DIR: str
    TAVILY_API_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"