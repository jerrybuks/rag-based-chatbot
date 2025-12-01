"""LangChain Tools for RAG operations."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents import Tool
from langchain_core.documents import Document

from .retrieval import retrieve_context, format_context_for_llm


def create_retrieval_tool(
    vectorstore_path: Path = Path("./chroma_db"),
    k: int = 3,
    openai_api_key: Optional[str] = None,
    min_similarity: float = 0.78,
) -> Tool:
    """
    Create a LangChain Tool for retrieval operations.
    
    This tool can be used by agents to retrieve relevant context from the vector store.
    
    Args:
        vectorstore_path: Path to the Chroma vector store
        k: Number of top results to retrieve (default: 3)
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        min_similarity: Minimum similarity threshold (0.0 to 1.0, default: 0.78)
        
    Returns:
        LangChain Tool instance for retrieval
    """
    def retrieval_func(question: str) -> str:
        """
        Retrieve context for a question.
        
        Args:
            question: The user's question
            
        Returns:
            Formatted context string
        """
        try:
            # Retrieve documents with scores
            documents_with_scores, _ = retrieve_context(
                question=question,
                vectorstore_path=vectorstore_path,
                k=k,
                filters=None,  # Can be extended to accept filters
                openai_api_key=openai_api_key,
                min_similarity=min_similarity,
            )
            
            # Format context for LLM
            context = format_context_for_llm(documents_with_scores)
            return context if context else "No relevant context found."
        except Exception as e:
            return f"Error retrieving context: {str(e)}"
    
    return Tool(
        name="retrieval_tool",
        description="""Useful for retrieving relevant context from the knowledge base to answer questions.
        Input should be a question or query string.
        Returns formatted context from the vector store.""",
        func=retrieval_func,
    )


def create_retrieval_tool_with_filters(
    vectorstore_path: Path = Path("./chroma_db"),
    k: int = 3,
    openai_api_key: Optional[str] = None,
    min_similarity: float = 0.78,
) -> Tool:
    """
    Create a LangChain Tool for retrieval operations with filter support.
    
    This is an enhanced version that accepts filters as part of the input.
    Note: This requires parsing the input to extract filters, which is more complex.
    For now, we'll use a simpler approach where filters are passed via tool configuration.
    
    Args:
        vectorstore_path: Path to the Chroma vector store
        k: Number of top results to retrieve (default: 3)
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        min_similarity: Minimum similarity threshold (0.0 to 1.0, default: 0.78)
        
    Returns:
        LangChain Tool instance for retrieval with filter support
    """
    def retrieval_func_with_filters(input_str: str) -> str:
        """
        Retrieve context for a question, potentially with filters.
        
        Args:
            input_str: The user's question (filters can be added later via tool configuration)
            
        Returns:
            Formatted context string
        """
        try:
            # For now, use no filters. Can be extended to parse filters from input_str
            documents_with_scores, _ = retrieve_context(
                question=input_str,
                vectorstore_path=vectorstore_path,
                k=k,
                filters=None,
                openai_api_key=openai_api_key,
                min_similarity=min_similarity,
            )
            
            # Format context for LLM
            context = format_context_for_llm(documents_with_scores)
            return context if context else "No relevant context found."
        except Exception as e:
            return f"Error retrieving context: {str(e)}"
    
    return Tool(
        name="retrieval_tool_with_filters",
        description="""Useful for retrieving relevant context from the knowledge base with optional filters.
        Input should be a question or query string.
        Returns formatted context from the vector store.""",
        func=retrieval_func_with_filters,
    )

