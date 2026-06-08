from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME: str = "ML-Tutor"
    APP_VERSION: str = "0.1.0"
    DEBUG:bool = False
    API_PREFIX:str = "/api/v1"

    LLM_API_BASE_URL:str
    LLM_API_KEY:str
    LLM_MODEL_NAME:str = "gpt-3.5-turbo"
    LLM_TEMPERATURE:float = 0.1

    EMBEDDING_MODEL:str = "text-embedding-3-small"

    PDF_PATH:str = "./dataset/dataset.pdf"

    CHROMA_PERSIST_DIR:str = "./chroma_db"
    CHROMA_COLLECTION_NAME:str = "Hands on Machine Learning"

    CHUNK_SIZE:int = 1000
    CHUNK_OVERLAP:int = 200

    DEFAULT_TOP_K:int = 4
    MAX_TOP_K:int = 10

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache
def get_settings() -> Settings:
    return Settings()