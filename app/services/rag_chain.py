from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List,Dict,Any,Tuple
from langchain_core.documents import Document
from app.core.config import get_settings
from app.services.vector_store import VectorStoreService
from app.core.exceptions import LLMServiceError
from app.core.logging_config import get_logger
import time
import asyncio
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)
settings = get_settings()

class RAGChainService:
    """RAG pipeline for answer question from a single pdf"""
    def __init__(self,vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service(EmbeddingService)
        self.llm = self._initialize_llm()
        self.prompt_template = self._create_prompt_template()

    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the open ai llm"""
        try:
            return ChatOpenAI(
                model = settings.LLM_MODEL_NAME,
                api_key = settings.LLM_API_KEY,
                temperature = settings.LLM_TEMPERATURE,
                max_tokens = 500,
                timeout = 60
            )
        except Exception as e:
            logger.error(f"LLM Initialization failed: {str(e)}")
            return LLMServiceError(f"Failed to initialize LLM: {str(e)}")
        
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for QA"""
        return ChatPromptTemplate([("system","""You are a precise AI assistant answering question based on a specific document.
                                    CONTEXT (from the document):
                                    {context}
                                    INSTRUCTIONS:
                                    1. Answer ONLY using the information provided from the given context.
                                    2. If the context doesn't contain the answer, say "I cannot find information about this in the document".
                                    3. When Citing information, mention the page number if available.
                                    5. Do not use external knowledge.

                                    Question: {question}

                                    Answer: 
                                    """),("human","{question}")])
    
    def _format_context(self,documents: List[Tuple[Document,float]]) -> str:
        """Format retrieved document into a coherent context string"""
        if not documents:
            return "No relevant information in  the given document"
        
        context_parts = []
        for i,(doc,score) in enumerate(documents):
            page_info = f"(page {doc.metadata.get('page_number','unknown')})" if 'page_number' in doc.metadata else ""
            context_parts.append(
                f"[Excerpt {i}{page_info} - Relevance: {score:.2f}]\n{doc.page_content}\n"
            ) 

        return "\n---\n".join(context_parts)
    async def answer_question(
            self,
            question:str,
            top_k:int = 4,
            include_source:bool = True,
    )->Dict[str,Any]:
        """"Answer a question using the RAG pipeline
            Args:
                question: User's qestion,
                top_k: Number of relevant chunks to retrieve,
                include_source: whether to include source chunks in response

            Returns:
                Dictionay with answer, source, metadata
        """

        start_time = time.time()
        try:
            retrieved = await self.vector_store.similarity_search(
                query = question,
                k = top_k
            )

            sources = None
            if include_source and retrieved:
                sources = []
                for doc, score in retrieved:
                    sources.append({
                        "text":doc.page_content[:800],
                        "page_number":doc.metadat.get('page_number'),
                        "relevance_score": round(score,3),
                        "chunk_index": doc.metadata.get('chunk_index',-1)
                    })
            context = self._format_context(retrieved)

            chain = self.prompt_template | self.llm | StrOutputParser()
            answer = await chain.ainvoke({
                "context":context,
                "question":question
            })
            processing_time = (time.time()- start_time)*1000
            return {
                "answer":answer.strip(),
                "source":sources,
                "processing_time_ms":round(processing_time,2),
                "model_used":settings.LLM_MODEL_NAME
            }
        
        except Exception as e:
            logger.error(f"Question answering failed: {str(e)}")
            return LLMServiceError(f"Failed to answer question: {str(e)}")
        
    async def check_llm_health(self)-> bool:
        """Check if LLM API is accessible"""
        try:
            response = await self.llm.ainvoke("Response with OK")
            return response.content.strip().upper() == "OK"
        except Exception as e:
            logger.error(f"LLM health check failed: {str(e)}")
            return False
        

    async def validate_retrieval(self,question:str, top_k:int = 2)-> Dict:
        """Debug method to test retrieval quality"""
        retrieved = await self.vector_store.similarity_search(question, k=top_k)
        return {
            "question":question,
            "num_retrieved":len(retrieved),
            "top_scores":[(doc.metadata.get('chunk_index'),score) for doc, score in retrieved],
            "has_result":len(retrieved)>0
        }