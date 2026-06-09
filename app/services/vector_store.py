from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional, Tuple
from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.services.embedding_service import EmbeddingService
from app.core.logging_config import get_logger
import asyncio
from pathlib import Path

logger = get_logger(__name__)
settings = get_settings()

class VectorStoreService:
    """Manage ChromaDB for a single, pre-loaded document"""
    def __init__(self,embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.persist_dir = Path(settings.CHROMA_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok = True)

        self.vector_store = self._initialize_or_load_store()
        
    def _initialize_or_load_store(self) -> Chroma:
        """Initialize Chroma, creating or loading existing collection"""
        try:
            return Chroma(
                collection_name = settings.CHROMA_COLLECTION_NAME,
                embedding_function = self.embedding_service.embed_document,
                persist_directory = str(self.persist_dir)
            )
        except Exception as e:
            logger.error(f"Vector store initialization failed: {str(e)}")
            raise VectorStoreError(f"Failed to initialize vector store: {str(e)}")
    
    async def add_document(self, chunks: List[Document]) -> int:
        """Add document chunk to vector store"""
        try:
            await asyncio.to_thread(
                self.vector_store.add_documents,
                chunks
            )
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise VectorStoreError(f"Failed to store document chunks: {str(e)}")
        
    async def similarity_search(
            self,
            query: str,
            k:int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar chunks
        Return list of (document, similarity score)
        """
        try:
            results = await asyncio.to_thread(
                self.vector_store.similarity_search_with_score,
                query,
                k=k
            )

            processed_results = []
            for doc, distance in results:
                similarity = 1.0/(1.0+distance)
                processed_results.append((doc,similarity))

            return processed_results
        
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            raise VectorStoreError(f"Search Failed: {str(e)}")
        
    async def get_chunk_count(self) -> int:
        """Get total number of chunks in vector store"""
        try:
            collection = self.vector_store._collection
            return collection.count()
        except Exception as e:
            logger.error(f"Failed to get chunk count: {str(e)}")
            return 0
    async def document_exists(self) -> bool:
        """Check if a document is already loaded in vector store"""
        return await self.get_chunk_count() > 0
    
    async def clear_store(self) -> None:
        """Clear all the chunks (use for reinitialization)"""
        try:
            await asyncio.to_thread(self.vector_store._collection.delete)
            self.vector_store = self._initialize_or_load_store()
            logger.warning("Vector store cleared")

        except Exception as e:
            logger.error(f"Failed to clear store: {str(e)}")
            raise VectorStoreError(f"Clear operations failed: {str(e)}")