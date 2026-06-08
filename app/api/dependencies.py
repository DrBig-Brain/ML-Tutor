from functools import lru_cache
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.rag_chain import RAGChainService
from app.services.document_loader import DocumentLoader

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Singleton Embedding Service"""
    return EmbeddingService()

@lru_cache()
def get_vector_store() -> VectorStoreService:
    """Singleton vector store service"""
    return VectorStoreService(embedding_service=EmbeddingService())

@lru_cache()
def get_rag_chain() -> RAGChainService:
    """Sinlgeton Rag chain service"""
    return RAGChainService()

@lru_cache()
def get_document_loader() -> DocumentLoader:
    """Singleton Document Loader"""
    return DocumentLoader()