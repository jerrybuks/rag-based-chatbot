"""Validation utilities for retrieval operations."""

import logging
from typing import Optional, List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .exceptions import EmbeddingError, RetrievalError
from .pricing import (
    get_embedding_model_name,
    get_embedding_max_tokens,
    get_embedding_dimensions,
    get_llm_max_tokens,
)

# Set up logging
logger = logging.getLogger(__name__)


def validate_embedding_tokens(
    question: str,
    embedding_model: Optional[str] = None,
) -> int:
    """
    Validate that the question token count is within the embedding model's limits.
    
    Args:
        question: The question text to validate
        embedding_model: Embedding model name (defaults to default model)
        
    Returns:
        Number of tokens in the question
        
    Raises:
        EmbeddingError: If token count exceeds the maximum allowed
    """
    if embedding_model is None:
        embedding_model = get_embedding_model_name()
    
    # Count tokens using tiktoken if available, otherwise estimate
    embedding_tokens = 0
    try:
        import tiktoken
        # Use cl100k_base encoding (used by text-embedding-ada-002 and text-embedding-3 models)
        encoding = tiktoken.get_encoding("cl100k_base")
        embedding_tokens = len(encoding.encode(question))
    except ImportError:
        # Fallback: estimate tokens (~1 token per 4 characters for English text)
        embedding_tokens = len(question) // 4
    
    # Validate token count
    max_tokens = get_embedding_max_tokens(embedding_model)
    if embedding_tokens > max_tokens:
        raise EmbeddingError(
            f"Question exceeds maximum token limit: {embedding_tokens} tokens "
            f"(max: {max_tokens} for model {embedding_model}). "
            "Please shorten your question."
        )
    
    return embedding_tokens


def validate_embedding_dimensions(
    vectorstore: Chroma,
    question: str,
    embedding_model: Optional[str] = None,
) -> None:
    """
    Validate that the question's embedding dimensions (when converted) match the dimensions
    of embeddings stored in the vectorstore.
    
    This ensures the embedding function will produce embeddings compatible with the vectorstore.
    
    Args:
        vectorstore: Chroma vectorstore instance
        question: The question text to validate (will be converted to embedding)
        embedding_model: Embedding model name (defaults to default model)
        
    Raises:
        RetrievalError: If embedding dimensions don't match between question and vectorstore
    """
    if embedding_model is None:
        embedding_model = get_embedding_model_name()
    
    expected_dimensions = get_embedding_dimensions(embedding_model)
    
    try:
        # Generate embedding for the question to check its dimensions
        embedding_function = vectorstore._embedding_function
        question_embedding = embedding_function.embed_query(question)
        question_dimensions = len(question_embedding) if question_embedding else None
        
        if not question_dimensions:
            raise RetrievalError(
                "Failed to generate embedding for question. Cannot validate dimensions."
            )
        
        # Check dimensions of stored embeddings in the vectorstore
        collection = vectorstore._collection
        vectorstore_dimensions = None
        
        if collection and hasattr(collection, 'count') and collection.count() > 0:
            # Try to peek at stored embeddings to check dimensions
            if hasattr(collection, 'peek'):
                sample = collection.peek(limit=1)
                if sample and 'embeddings' in sample and len(sample['embeddings']) > 0:
                    vectorstore_dimensions = len(sample['embeddings'][0])
        
        # Validate that question embedding dimensions match vectorstore dimensions
        if vectorstore_dimensions:
            if question_dimensions != vectorstore_dimensions:
                logger.warning(
                    f"Embedding dimension mismatch: question embedding has {question_dimensions} dimensions, "
                    f"but vectorstore has {vectorstore_dimensions} dimensions."
                )
                raise RetrievalError(
                    f"Embedding dimension mismatch: question embedding has {question_dimensions} dimensions, "
                    f"but vectorstore has {vectorstore_dimensions} dimensions. "
                    "The embedding function and vectorstore were created with different models. "
                    "Please regenerate the vectorstore with the correct model."
                )
        else:
            # If we can't check vectorstore dimensions, at least validate against expected
            if question_dimensions != expected_dimensions:
                logger.warning(
                    f"Question embedding dimension mismatch: expected {expected_dimensions}, "
                    f"got {question_dimensions} for model {embedding_model}."
                )
                raise RetrievalError(
                    f"Question embedding dimension mismatch: expected {expected_dimensions} dimensions "
                    f"for model {embedding_model}, but got {question_dimensions} dimensions. "
                    "The embedding function may be misconfigured."
                )
    except RetrievalError:
        # Re-raise RetrievalError
        raise
    except Exception as e:
        # If dimension check fails, log but don't fail (might be empty collection or API differences)
        logger.debug(f"Could not validate embedding dimensions: {e}")


def validate_and_truncate_llm_prompt(
    system_prompt: str,
    context_text: str,
    question: str,
    model_name: Optional[str] = None,
) -> Tuple[str, bool]:
    """
    Validate that the total prompt tokens don't exceed the LLM's max token limit.
    If it does, truncate the context to fit within the limit.
    
    Args:
        system_prompt: System prompt text (with {context} placeholder)
        context_text: Context text to potentially truncate
        question: User question
        model_name: LLM model name (defaults to default model)
        
    Returns:
        Tuple of (truncated_context_text, was_truncated)
    """
    if model_name is None:
        from .pricing import get_llm_model_name
        model_name = get_llm_model_name()
    
    max_tokens = get_llm_max_tokens(model_name)
    
    # Count tokens using tiktoken
    try:
        import tiktoken
        # Use cl100k_base encoding (used by GPT-3.5 and GPT-4)
        encoding = tiktoken.get_encoding("cl100k_base")
        
        # Count tokens for system prompt (without context), question, and some overhead
        system_prompt_base = system_prompt.replace("{context}", "")
        system_tokens = len(encoding.encode(system_prompt_base))
        question_tokens = len(encoding.encode(question))
        
        # Reserve tokens for formatting, context placeholder text, and safety margin
        overhead_tokens = 100
        available_tokens = max_tokens - system_tokens - question_tokens - overhead_tokens
        
        if available_tokens <= 0:
            logger.warning(
                f"System prompt and question already exceed max tokens ({max_tokens}). "
                "Cannot include context."
            )
            return "", True
        
        # Count context tokens
        context_tokens = len(encoding.encode(context_text))
        
        # If context fits, return as-is
        if context_tokens <= available_tokens:
            return context_text, False
        
        # Need to truncate context
        logger.warning(
            f"Context exceeds token limit. Truncating from {context_tokens} to ~{available_tokens} tokens."
        )
        
        # Truncate context by removing characters until it fits
        # We'll truncate from the end (oldest context first)
        truncated_context = context_text
        while len(encoding.encode(truncated_context)) > available_tokens:
            # Remove ~10% at a time for efficiency
            truncate_amount = max(1, len(truncated_context) // 10)
            truncated_context = truncated_context[:-truncate_amount]
            
            # Safety check to avoid infinite loop
            if len(truncated_context) == 0:
                break
        
        return truncated_context, True
        
    except ImportError:
        # Fallback: estimate tokens (~1 token per 4 characters)
        system_prompt_base = system_prompt.replace("{context}", "")
        system_tokens = len(system_prompt_base) // 4
        question_tokens = len(question) // 4
        overhead_tokens = 100
        available_tokens = max_tokens - system_tokens - question_tokens - overhead_tokens
        
        if available_tokens <= 0:
            return "", True
        
        context_tokens = len(context_text) // 4
        
        if context_tokens <= available_tokens:
            return context_text, False
        
        # Truncate by character count
        max_chars = available_tokens * 4
        truncated_context = context_text[:max_chars]
        return truncated_context, True

