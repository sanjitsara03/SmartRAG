import logging
from fastapi import FastAPI
from src.backend.api.chat import router as chat_router
from src.backend.config.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = FastAPI()
app.include_router(chat_router)

settings = Settings()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.backend.main:app", 
        host=settings.API_HOST, 
        port=settings.API_PORT,
    )
