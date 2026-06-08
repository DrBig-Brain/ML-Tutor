from langchain_openai import OpenAIEmbeddings
from typing import List
from app.core.config import get_settings
from app.core.exceptions import LLMServiceError
from app.core.logging_config import get_logger
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = get_logger(__name__)
settings = get_settings()

class EmbeddingService:
    """Handles Embedding generation for both documents and queries"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model =settings.EMBEDDING_MODEL,
            api_key = settings.LLM_API_KEY,
            chunk_size = 100
        )
        self._executor = ThreadPoolExecutor(max_workers = 2)
    async def embed_query(self, query:str) -> List[float]:
        """Generate embedding for a query string"""
        try:
            loop = asyncio.get_envent_loop()
            embedding = await loop.run_in_executor(
                self._executor,
                self.embeddings.embed_query,
                query
            )
            return embedding
        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            raise LLMServiceError(f"Failed to generate query embedding: {str(e)}")
    async def embed_document(self,texts: List[str]) -> List[List[float]]:
        """Generate Query for multiple texts"""
        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self._executor,
                self.embeddings.embed_documents
            )
            return embeddings
        except Exception as e:
            logger.error(f"Document embedding error: {str(e)}")
            raise LLMServiceError(f"Failed to generate document embeddings: {str(e)}")