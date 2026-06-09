from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import QueryRequest, QueryResponse, DocumentInfoResponse
from app.services.rag_chain import RAGChainService
from app.api.dependencies import get_rag_chain
from app.core.exceptions import LLMServiceError, VectorStoreError
from app.core.logging_config import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/query",tags=["query"])

@router.post("/",response_model = QueryResponse)
async def query_documents(
    request: QueryRequest,
    rag_chain: RAGChainService = Depends(get_rag_chain)
):
    """
    Query the document using RAG
    Send a question and get an answer based on the documnet content.
    """
    try:
        result = await rag_chain.answer_question(
            question=request.query,
            top_k = request.top_k,
            include_source = request.include_source
        )
        return QueryResponse(
            query = request.query,
            answer = result["answer"],
            source=result.get("sources"),
            processing_time_ms=result["processing_time_ms"],
            mode_used=result["model_used"]
        )
    except LLMServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except VectorStoreError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in query endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@router.get("/info",response_model=DocumentInfoResponse)
async def get_document_info(
    rag_chain: RAGChainService = Depends(get_rag_chain)
):
    """Get information about the loaded document"""
    try:
        chunk_count = await rag_chain.vector_store_service.get_chunk_count()

        return DocumentInfoResponse(
            filename=settings.PDF_PATH.split("/")[-1],
            total_chunks=chunk_count,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            embedding_model=settings.EMBEDDING_MODEL,
            llm_model=settings.LLM_MODEL_NAME
        )
    except Exception as e:
        logger.error(f"Failed to get document info: {str(e)}")
        raise HTTPException(status_code = 500, detail = "Failed to retrieve document information")