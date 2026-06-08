"""
Custom exceptions for the RAG pipeline application.

This module defines a hierarchy of exceptions for different error scenarios,
enabling granular error handling and appropriate HTTP responses.
"""

from typing import Optional, Any, Dict
from enum import Enum


class ErrorCode(str, Enum):
    """Standardized error codes for API responses"""
    # Document errors
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    DOCUMENT_PROCESSING_FAILED = "DOCUMENT_PROCESSING_FAILED"
    DOCUMENT_EMPTY = "DOCUMENT_EMPTY"
    DOCUMENT_TOO_LARGE = "DOCUMENT_TOO_LARGE"
    
    # Vector store errors
    VECTOR_STORE_INIT_FAILED = "VECTOR_STORE_INIT_FAILED"
    VECTOR_STORE_QUERY_FAILED = "VECTOR_STORE_QUERY_FAILED"
    VECTOR_STORE_INSERT_FAILED = "VECTOR_STORE_INSERT_FAILED"
    
    # LLM errors
    LLM_API_UNAVAILABLE = "LLM_API_UNAVAILABLE"
    LLM_API_TIMEOUT = "LLM_API_TIMEOUT"
    LLM_API_RATE_LIMIT = "LLM_API_RATE_LIMIT"
    LLM_API_AUTH_FAILED = "LLM_API_AUTH_FAILED"
    LLM_INVALID_RESPONSE = "LLM_INVALID_RESPONSE"
    
    # Configuration errors
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    MISSING_ENV_VAR = "MISSING_ENV_VAR"
    
    # General errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class RAGException(Exception):
    """
    Base exception for the RAG pipeline.
    
    All custom exceptions inherit from this class to ensure consistent
    error handling and response formatting.
    
    Attributes:
        message: Human-readable error message
        status_code: HTTP status code for API responses
        error_code: Standardized error code for programmatic handling
        detail: Additional error details or context
        metadata: Optional dictionary with extra error context
    """
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        detail: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail
        self.metadata = metadata or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON responses"""
        error_dict = {
            "error": self.message,
            "error_code": self.error_code,
            "status_code": self.status_code
        }
        if self.detail:
            error_dict["detail"] = self.detail
        if self.metadata:
            error_dict["metadata"] = self.metadata
        return error_dict


class DocumentProcessingError(RAGException):
    """Raised when PDF document processing fails."""
    
    def __init__(
        self,
        message: str,
        detail: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=422,
            error_code=ErrorCode.DOCUMENT_PROCESSING_FAILED,
            detail=detail,
            metadata=metadata
        )


class DocumentNotFoundError(RAGException):
    """Raised when the PDF file is not found at the configured path."""
    
    def __init__(
        self,
        pdf_path: str,
        detail: Optional[Any] = None
    ):
        message = f"PDF document not found at: {pdf_path}"
        super().__init__(
            message=message,
            status_code=404,
            error_code=ErrorCode.DOCUMENT_NOT_FOUND,
            detail=detail,
            metadata={"pdf_path": pdf_path}
        )


class DocumentEmptyError(RAGException):
    """Raised when the PDF has no extractable content."""
    
    def __init__(
        self,
        message: str = "The PDF document contains no extractable content",
        detail: Optional[Any] = None
    ):
        super().__init__(
            message=message,
            status_code=422,
            error_code=ErrorCode.DOCUMENT_EMPTY,
            detail=detail
        )


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""
    
    def __init__(
        self,
        message: str,
        operation: str = "unknown",
        detail: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        metadata = metadata or {}
        metadata["operation"] = operation
        super().__init__(
            message=message,
            status_code=500,
            error_code=ErrorCode.VECTOR_STORE_QUERY_FAILED,
            detail=detail,
            metadata=metadata
        )


class VectorStoreInitError(VectorStoreError):
    """Raised when vector store initialization fails."""
    
    def __init__(
        self,
        message: str,
        detail: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            operation="initialization",
            detail=detail,
            metadata=metadata
        )
        self.error_code = ErrorCode.VECTOR_STORE_INIT_FAILED


class LLMServiceError(RAGException):
    """Base exception for LLM API related errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 503,
        error_code: ErrorCode = ErrorCode.LLM_API_UNAVAILABLE,
        detail: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            error_code=error_code,
            detail=detail,
            metadata=metadata
        )


class LLMAPIUnavailableError(LLMServiceError):
    """Raised when the LLM API is unreachable."""
    
    def __init__(
        self,
        api_base_url: str,
        detail: Optional[Any] = None
    ):
        message = f"LLM API is unavailable at {api_base_url}"
        super().__init__(
            message=message,
            status_code=503,
            error_code=ErrorCode.LLM_API_UNAVAILABLE,
            detail=detail,
            metadata={"api_base_url": api_base_url}
        )


class LLMAPITimeoutError(LLMServiceError):
    """Raised when LLM API request times out."""
    
    def __init__(
        self,
        timeout_seconds: int,
        detail: Optional[Any] = None
    ):
        message = f"LLM API request timed out after {timeout_seconds} seconds"
        super().__init__(
            message=message,
            status_code=504,
            error_code=ErrorCode.LLM_API_TIMEOUT,
            detail=detail,
            metadata={"timeout_seconds": timeout_seconds}
        )


class LLMAPIRateLimitError(LLMServiceError):
    """Raised when LLM API rate limit is exceeded."""
    
    def __init__(
        self,
        detail: Optional[Any] = None,
        retry_after: Optional[int] = None
    ):
        message = "LLM API rate limit exceeded. Please try again later."
        metadata = {}
        if retry_after:
            metadata["retry_after_seconds"] = retry_after
        
        super().__init__(
            message=message,
            status_code=429,
            error_code=ErrorCode.LLM_API_RATE_LIMIT,
            detail=detail,
            metadata=metadata
        )


class LLMAPIAuthError(LLMServiceError):
    """Raised when LLM API authentication fails."""
    
    def __init__(
        self,
        detail: Optional[Any] = None
    ):
        message = "LLM API authentication failed. Please check your API key."
        super().__init__(
            message=message,
            status_code=401,
            error_code=ErrorCode.LLM_API_AUTH_FAILED,
            detail=detail
        )


class ConfigurationError(RAGException):
    """Raised when application configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        detail: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            detail=detail,
            metadata=metadata
        )


class MissingEnvironmentVariableError(ConfigurationError):
    """Raised when required environment variables are missing."""
    
    def __init__(
        self,
        variable_name: str,
        detail: Optional[Any] = None
    ):
        message = f"Required environment variable is missing: {variable_name}"
        super().__init__(
            message=message,
            detail=detail,
            metadata={"missing_variable": variable_name}
        )
        self.error_code = ErrorCode.MISSING_ENV_VAR


class ValidationError(RAGException):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        detail: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=400,
            error_code=ErrorCode.VALIDATION_ERROR,
            detail=detail,
            metadata=metadata
        )