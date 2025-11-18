"""Retrieval logic for vector search and hybrid search with filters."""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from indexing.embeddings import load_vectorstore
from .exceptions import RetrievalError, EmbeddingError
from .pricing import calculate_embedding_cost, get_embedding_model_name
from .validations import validate_embedding_tokens, validate_embedding_dimensions

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


def retrieve_context(
    question: str,
    vectorstore_path: Path = Path("./chroma_db"),
    k: int = 3,
    filters: Optional[Dict[str, Any]] = None,
    openai_api_key: Optional[str] = None,
    min_similarity: float = 0.8,
) -> tuple[List[Tuple[Document, float]], Dict[str, Any]]:
    """
    Retrieve relevant context using vector search with optional metadata filters.
    
    Performs hybrid search combining:
    - ANN (Approximate Nearest Neighbor) vector similarity search
    - Metadata filtering for precise results
    
    Distance Metric: Chroma is configured to use cosine similarity.
    
    IMPORTANT: Chroma returns cosine DISTANCE (1 - cosine_similarity), not similarity.
    - Lower scores = higher similarity (distance closer to 0 = more similar)
    - To get actual similarity: similarity = 1 - distance
    - Distance range: 0.0 (identical, similarity=1.0) to 2.0 (opposite, similarity=-1.0)
    
    Args:
        question: The user's question
        vectorstore_path: Path to the Chroma vector store
        k: Number of top results to retrieve (default: 3)
        filters: Optional metadata filters (e.g., {"section": "Account & Access"})
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        min_similarity: Minimum similarity threshold (0.0 to 1.0, default: 0.8)
                       Results below this threshold will be filtered out
        
    Returns:
        Tuple of:
        - List of tuples containing (Document, distance_score) pairs
          where score is cosine DISTANCE (lower = more similar, range 0.0 to 2.0)
          Note: To convert to similarity, use similarity = 1 - distance
          Only returns results above min_similarity threshold
        - Dictionary with embedding metadata: {"tokens": int, "cost_usd": float}
    """
    # Load the vector store with error handling
    try:
        vectorstore = load_vectorstore(vectorstore_path, openai_api_key)
    except ValueError as e:
        # Re-raise ValueError from load_vectorstore as RetrievalError
        logger.error(f"Failed to load vector store: {e}")
        raise RetrievalError(str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error loading vector store: {type(e).__name__}: {e}")
        raise RetrievalError(f"Failed to load vector store: {str(e)}") from e
    
    # Validate embedding dimensions (question embedding vs vectorstore embeddings)
    embedding_model = get_embedding_model_name()
    validate_embedding_dimensions(vectorstore, question, embedding_model)
    
    # Get embedding function to track token usage
    embedding_function = vectorstore._embedding_function
    embedding_metadata = {"tokens": 0, "cost_usd": 0.0}
    
    # Prepare filter dict for Chroma
    # Chroma requires $and operator when multiple filters are provided
    where_filter = None
    if filters:
        filter_conditions = []
        for key, value in filters.items():
            # Support exact match or list for $in operator
            if isinstance(value, list):
                filter_conditions.append({key: {"$in": value}})
            else:
                filter_conditions.append({key: value})
        
        # Chroma requires $and when multiple conditions exist
        if len(filter_conditions) > 1:
            where_filter = {"$and": filter_conditions}
        elif len(filter_conditions) == 1:
            where_filter = filter_conditions[0]
    
    # Perform similarity search with filters (hybrid search)
    # Note: Embedding generation happens here (when converting question to vector)
    # Track embedding tokens and cost using actual token counting
    try:
        # Validate and count tokens for embedding
        embedding_tokens = validate_embedding_tokens(question, embedding_model)
        
        # Calculate embedding cost using pricing module
        embedding_cost = calculate_embedding_cost(embedding_tokens, model=embedding_model)
        embedding_metadata = {
            "tokens": embedding_tokens,
            "cost_usd": embedding_cost
        }
        
        if where_filter:
            # Hybrid search: vector similarity + metadata filtering
            results = vectorstore.similarity_search_with_score(
                query=question,
                k=k,
                filter=where_filter,
            )
        else:
            # Pure vector similarity search (ANN)
            results = vectorstore.similarity_search_with_score(
                query=question,
                k=k,
            )
    except Exception as e:
        # If filter causes issues, try to fall back to simple search
        if where_filter:
            logger.warning(f"Filter search failed, attempting fallback to simple search: {e}")
            try:
                results = vectorstore.similarity_search_with_score(
                    query=question,
                    k=k,
                )
            except Exception as fallback_error:
                logger.error(f"Vector search error: {type(fallback_error).__name__} - {str(fallback_error)}")
                raise RetrievalError(
                    "An error occurred during retrieval. Please try again later."
                ) from fallback_error
        else:
            logger.error(f"Vector search error: {type(e).__name__} - {str(e)}")
            raise RetrievalError(
                "An error occurred during retrieval. Please try again later."
            ) from e
    
    # Filter results by minimum similarity threshold
    # Convert distance to similarity: similarity = 1 - distance
    # Filter: keep only results where similarity >= min_similarity
    max_distance = 1.0 - min_similarity
    filtered_results = [
        (doc, distance) for doc, distance in results
        if distance <= max_distance
    ]
    
    # Return filtered documents with their distance scores and embedding metadata
    # Note: Chroma returns cosine DISTANCE (1 - cosine_similarity)
    # To convert to similarity: similarity = 1 - distance
    # Example: distance 0.18 = similarity 0.82 (excellent match!)
    return filtered_results, embedding_metadata


def format_context_for_llm(documents_with_scores: List[Tuple[Document, float]]) -> str:
    """
    Format retrieved documents into context string for LLM.
    
    Args:
        documents_with_scores: List of (Document, score) tuples
        
    Returns:
        Formatted context string
    """
    if not documents_with_scores:
        return ""
    
    context_parts = []
    for idx, (doc, score) in enumerate(documents_with_scores, 1):
        metadata = doc.metadata
        section_info = f"[Source {idx}"
        if metadata.get("section"):
            section_info += f": {metadata.get('section')}"
        # Convert distance to similarity for display
        similarity = 1.0 - score
        section_info += f" (similarity: {similarity:.4f}, distance: {score:.4f})"
        section_info += "]"
        
        context_parts.append(f"{section_info}\n{doc.page_content}\n")
    
    return "\n---\n".join(context_parts)

