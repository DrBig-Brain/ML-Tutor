from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from app.api.routes import query
from app.core.config import get_settings
from app.core.logging_config import setup_logging, get_logger
from app.core.exceptions import RAGException, DocumentProcessingError
from app.models.schemas import HealthResponse
from app.services.document_loader import DocumentLoader
from app.services.vector_store import VectorStoreService
from app.services.rag_chain import RAGChainService
from app.api.dependencies import get_document_loader, get_embedding_service, get_rag_chain,get_vector_store

load_dotenv()

settings = get_settings()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events.
    Loads the pdf into vector store when the application starts.
    """

    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"PDF_Path: {settings.PDF_PATH}")
    logger.info(f"LLM_API: {settings.LLM_API_BASE_URL}")

    try:
        pdf_path = Path(settings.PDF_PATH)
        if not pdf_path.exists():
            raise DocumentProcessingError(
                f"PDF file not found at {settings.PDF_PATH}."
                f"Please ensure your pdf is placed in the dataset/ directory as dataset.pdf"
            )
        document_loader = DocumentLoader()
        embedding_service =get_embedding_service()
        vector_store = get_vector_store()

        doc_exists = await vector_store.document_exists()

        if not doc_exists:
            logger.info("loading and indexing pdf document")

            doc_hash, chunks = document_loader.load_and_chunk()
            chunk_count = await vector_store.add_document(chunks)

            logger.info(f"Successfully loaded PDF: {chunk_count} chunks indexed")
            logger.info(f"Document Hash: {doc_hash}")

        else:
            chunk_count = await vector_store.get_chunk_count()
            logger.info(f"Document already loaded: {chunk_count} chunks found in vector store")

        rag_chain = get_rag_chain()
        llm_ready =await rag_chain.check_llm_health()

        if llm_ready:
            logger.info("LLM API connections successful")
        else:
            logger.warning("LLM API health check failed - service may be degraded")

    except DocumentProcessingError as e:
        logger.error(f"Document loading failed: {str(e)}")
        app.state.document_loaded = False
    except Exception as e:
        logger.error(f"Starup initialization failed: {str(e)}",exc_info=True)
        app.state.document_loaded = False
    
    app.state.document_loaded = True
    
    yield

    logger.info(f"Shutting down {settings.APP_NAME}")

def create_app() -> FastAPI:
    """Application Factory Pattern"""
    setup_logging(settings.DEBUG)

    app = FastAPI(
        title= settings.APP_NAME,
        version = settings.APP_VERSION,
        description = "RAG pipeline for querying a single pdf documnet using LLM API",
        lifespan = lifespan,
        doc_url = "/docs" if settings.DEBUG else None,
        redoc_url = "/redoc" if settings.DEBUG else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins = ["*"],
        allow_credentials = True,
        allow_methods = ["*"],
        allow_headers = ["*"]
    )

    app.include_router(query.router,prefix = settings.API_PREFIX)

    @app.exception_handler(RAGException)
    async def rag_exception_handler(request, exc: RAGException):
        return JSONResponse(
            status_code = exc.status_code,
            content ={"error":exc.message,"detail":exc.detail}
        )
    
    @app.get("/")
    async def root():
        return {
            "name":settings.APP_NAME,
            "version":settings.APP_VERSION,
            "status":"running",
            "document_loaded": getattr(app, "document_loaded",False),
            "endpoints":{
                "query":f"{settings.API_PREFIX}/query",
                "info":f"{settings.API_PREFIX}/query/info",
                "health":"/health"
            }
        }
    
    @app.get("/health",response_model = HealthResponse)
    async def health_check():
        """Comprehensive health check"""
        try:
            vector_store = get_vector_store()
            rag_chain = get_rag_chain()

            chunk_count = await vector_store.get_chunk_count()
            llm_ready = await rag_chain.check_llm_health()
            document_loaded = chunk_count > 0

            status = "health"

            if not document_loaded:
                status = "degraded"
            elif not llm_ready:
                status = "degraded"
            
            return HealthResponse(
                status = status,
                version = settings.APP_VERSION,
                document_loaded=document_loaded,
                vector_store_ready=True,
                llm_api_ready= llm_ready,
                total_chunks = chunk_count
            )
        except Exception as e:
            logger.error(f"Health Check Failed: {str(e)}")
            return HealthResponse(
                status = "unhealthy",
                version = settings.APP_VERSION,
                document_loaded = False,
                vector_store_ready = False,
                llm_api_ready = False,
                total_chunks = 0
            )
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host = "0.0.0.0",
        port = 8000,
        reload=settings.DEBUG,
        log_level="info"
    )