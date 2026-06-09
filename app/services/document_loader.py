from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Tuple
from pathlib import Path
import hashlib
from app.core.config import get_settings
from app.core.exceptions import DocumentProcessingError
from app.core.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

class DocumentLoader:
    """
    Loads and chunks the single PDF Document.
    This runs once at startup and the chunks are stored in a vector DB.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = settings.CHUNK_SIZE,
            chunk_overlap = settings.CHUNK_OVERLAP,
            length_function = len,
            separators = ["\n\n","\n",". "," ",""],
            add_start_index = True
        )
    def load_and_chunk(self)->Tuple[str,List[Document]]:
        """
        Load PDF from configured path and split into chunks.
        Return (document_hash, list of chunks)
        """
        pdf_path = Path(settings.PDF_PATH)

        if not pdf_path:
            raise DocumentProcessingError(
                f"PDF file not found at {settings.PDF_PATH}."
                f"Please place your PDF in the dataset/ directory as 'dataset.pdf'"
            )
        
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            if not documents:
                raise DocumentProcessingError("No extraction from pdff")
            logger.info(f"Loaded PDF: {pdf_path.name} ({len(documents)}pages)")

            for doc in documents:
                doc.page_content = (
                    doc.page_content
                    .encode("utf-8",errors="ignore")
                    .decode("utf-8")
                )

            for i,doc in enumerate(documents):
                doc.metadata["chunk_type"] = "origial_page"
                doc.metadata["source_file"] = pdf_path.name

            chunks = self.text_splitter.split_documents(documents)

            for idx, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = idx
                chunk.metadata["total_chunks"] = len(chunks)

                if 'page' in chunk.metadata:
                    chunk.metadata["page_number"] = chunk.metadata['page']+1


            doc_hash = self._compute_hash(documents)

            logger.info(f"Created {len(chunks)} chunks from {pdf_path.name}")
            return doc_hash, chunks
        except Exception as e:
            logger.error(f"failed to load PDF: {str(e)}")
            raise DocumentProcessingError(f"PDF loading failed {str(e)}")
        
    @staticmethod
    def _compute_hash(documents: List[Document]) -> str:
        """compute hash of document content to detect changes"""
        content = "".join([doc.page_content for doc in documents])
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_document_info(self) -> dict:
        """Get information about the loaded document"""
        pdf_path = Path(settings.PDF_PATH)
        return{
            "filename":pdf_path.name,
            "file_size_mb":pdf_path.stat().st_size / (1024*1024) if pdf_path.exists() else 0,
            "exists":pdf_path.exists()
        }