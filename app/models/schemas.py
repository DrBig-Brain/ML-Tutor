from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1,max_length=500, description="User_question")
    top_k: int = Field(default = 1,ge=1,le=10, description="Number of relevant chunks to retrieve")
    include_source: bool = Field(default = True,description="Include source text chunks in response")


class SourceChunk(BaseModel):
    text:str = Field(..., description="Relevant text chunk from pdf")
    page_number: Optional[int] = Field(None, description="Page number if available")
    relevance_score: float = Field(..., description="Similarity score (0-1)")
    chunk_index: int = Field(..., description="Chunk position in document")

class QueryResponse(BaseModel):
    query:str
    answer:str
    source:Optional[List[SourceChunk]]= None
    processing_time_ms:float
    model_use:str
    timestamp:datetime = Field(default_factory=datetime.now)
class HealthResponse(BaseModel):
    status:str
    version:str
    document_loaded:bool
    vector_store_ready:bool
    llm_api_ready:bool
    total_chunks: int

class DocumentInfoResponse(BaseModel):
    filename:str
    total_chunks:int
    chunk_size:int
    chunk_overlap:int
    embedding_model:str
    llm_model:str