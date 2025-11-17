"""Custom exceptions for querying module."""


class QueryError(Exception):
    """Base exception for query-related errors."""
    pass


class OpenAIAPIError(QueryError):
    """Exception for OpenAI API errors."""
    pass


class EmbeddingError(QueryError):
    """Exception for embedding generation errors."""
    pass


class VectorStoreError(QueryError):
    """Exception for vector store errors."""
    pass


class RetrievalError(QueryError):
    """Exception for retrieval errors."""
    pass

