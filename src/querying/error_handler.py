"""Error handling utilities for query endpoints."""

import logging
from typing import Optional, Tuple
from fastapi import HTTPException, status

from .exceptions import (
    QueryError,
    OpenAIAPIError,
    EmbeddingError,
    RetrievalError,
)

# Set up logging
logger = logging.getLogger(__name__)

# Error mappings: (exception_type, status_code, generic_message)
ERROR_MAPPINGS = {
    OpenAIAPIError: (
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "An error occurred while generating the answer. Please try again later."
    ),
    EmbeddingError: (
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "An error occurred while retrieving context. Please try again later."
    ),
    RetrievalError: (
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "An error occurred while retrieving context. Please try again later."
    ),
    ValueError: (
        status.HTTP_400_BAD_REQUEST,
        "Invalid request. Please check your input and try again."
    ),
}


def handle_query_error(
    error: Exception,
    default_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    default_message: str = "An unexpected error occurred. Please try again later."
) -> HTTPException:
    """
    Handle query-related errors and convert them to HTTP exceptions.
    
    Args:
        error: The exception to handle
        default_status: Default HTTP status code if error type is not mapped
        default_message: Default error message if error type is not mapped
        
    Returns:
        HTTPException with appropriate status code and message
    """
    error_type = type(error)
    
    # Check if error type is in mappings
    if error_type in ERROR_MAPPINGS:
        status_code, message = ERROR_MAPPINGS[error_type]
    else:
        # Check if it's a QueryError base class
        if isinstance(error, QueryError):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            message = "An error occurred while processing your request. Please try again later."
        else:
            status_code = default_status
            message = default_message
    
    # Log the error with essential details only
    logger.error(f"{error_type.__name__}: {str(error)}")
    
    return HTTPException(
        status_code=status_code,
        detail=message
    )


def handle_api_key_error() -> HTTPException:
    """
    Handle missing OpenAI API key error.
    
    Returns:
        HTTPException for missing API key
    """
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="OpenAI API key is not configured. Please set OPENAI_API_KEY environment variable."
    )

