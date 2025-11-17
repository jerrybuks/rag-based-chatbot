"""Embedding generation and storage utilities for RAG chunks."""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    vectorstore_path: Optional[Path] = None,
    openai_api_key: Optional[str] = None,
) -> Chroma:
    """
    Generate embeddings for chunks and save to Chroma vector store.
    
    Configured to use cosine similarity for distance metric.
    - Higher scores = higher similarity (closer to 1.0 = more similar)
    - Range: -1.0 (opposite) to 1.0 (identical)
    
    Note: If an existing vectorstore exists with a different distance metric,
    you may need to delete it and regenerate for the new metric to take effect.
    
    Args:
        chunks: List of chunk dictionaries with 'content' field
        vectorstore_path: Path to save Chroma vector store (default: ./chroma_db)
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        
    Returns:
        Chroma vector store instance with embedded chunks (using cosine similarity)
    """
    # Get API key from parameter or environment variable
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
            "or pass openai_api_key parameter."
        )
    
    # Set default vectorstore path if not provided
    if vectorstore_path is None:
        vectorstore_path = Path("./chroma_db")
    vectorstore_path = Path(vectorstore_path)
    
    # Initialize embeddings model
    # Check for custom API base URL (e.g., for OpenRouter)
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, openai_api_base=api_base)
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Prepare documents for Chroma (LangChain Document format)
    from langchain_core.documents import Document
    
    documents = []
    metadatas = []
    ids = []
    
    for chunk in chunks:
        # Create LangChain Document
        doc = Document(
            page_content=chunk.get("content", ""),
            metadata={
                "chunk_id": chunk.get("chunk_id"),
                "section_id": chunk.get("section_id"),
                "section": chunk.get("section"),
                "product_area": chunk.get("product_area"),
                "last_updated": chunk.get("last_updated"),
                "intent_tags": str(chunk.get("intent_tags", [])),
                "word_count": chunk.get("word_count"),
                "chunk_index": chunk.get("chunk_index"),
                "total_chunks": chunk.get("total_chunks"),
            }
        )
        documents.append(doc)
        metadatas.append(doc.metadata)
        ids.append(chunk.get("chunk_id", f"chunk_{len(ids)}"))
    
    # Create Chroma vector store with embeddings
    # Configure to use cosine similarity instead of L2 distance
    print(f"Generating embeddings for {len(documents)} chunks...")
    
    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(vectorstore_path),
            ids=ids,
            collection_metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        print(f"Saved embeddings to vector store at: {vectorstore_path}")
        return vectorstore
    except Exception as e:
        # Log the error with essential details only
        logger.error(f"Embedding generation error: {type(e).__name__} - {str(e)}")
        # Raise generic error message
        raise ValueError(
            "An error occurred while generating embeddings. Please try again later."
        ) from e


def load_vectorstore(
    vectorstore_path: Path,
    openai_api_key: Optional[str] = None,
) -> Chroma:
    """
    Load an existing Chroma vector store.
    
    Args:
        vectorstore_path: Path to the Chroma vector store
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        
    Returns:
        Chroma vector store instance
    """
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
            "or pass openai_api_key parameter."
        )
    
    # Check for custom API base URL (e.g., for OpenRouter)
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, openai_api_base=api_base)
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Load existing vectorstore (will use the distance metric it was created with)
    try:
        vectorstore = Chroma(
            persist_directory=str(vectorstore_path),
            embedding_function=embeddings,
        )
        return vectorstore
    except Exception as e:
        # Log the error with essential details only
        logger.error(f"Vector store load error: {type(e).__name__} - {str(e)}")
        # Raise generic error message
        raise ValueError(
            "An error occurred while loading the vector store. Please ensure the vector store exists and try again."
        ) from e

