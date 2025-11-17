"""LLM integration for generating answers from context."""

import os
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from .retrieval import format_context_for_llm
from .exceptions import OpenAIAPIError
from .pricing import calculate_llm_cost, DEFAULT_LLM_MODEL
from .validations import validate_and_truncate_llm_prompt

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Maximum length for context content in responses (None = no truncation)
MAX_CONTEXT_CONTENT_LENGTH = None  # Set to None to return full content, or an integer for character limit


def generate_answer(
    question: str,
    context_documents_with_scores: List[tuple],
    openai_api_key: Optional[str] = None,
    model_name: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Generate answer using LLM with retrieved context.
    
    The LLM is instructed to:
    - Only use information from the provided context
    - Avoid hallucinations
    - Say "I don't have enough information" if context doesn't answer the question
    
    Args:
        question: The user's question
        context_documents_with_scores: List of (Document, score) tuples from retrieval
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        model_name: Model to use (defaults to DEFAULT_LLM_MODEL)
        temperature: Model temperature (default: 0.0 for deterministic)
        
    Returns:
        Dictionary with answer, context_used, sources, scores, and no_context_found flag
    """
    # Get API key
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
            "or pass openai_api_key parameter."
        )
    
    # Check for custom API base URL
    api_base = os.getenv("OPENAI_API_BASE")
    
    # If no context found, don't call LLM
    if not context_documents_with_scores:
        return {
            "answer": "I don't have enough information in the knowledge base to answer your question. Please try rephrasing your question or contact support for assistance.",
            "context_used": [],
            "sources": [],
            "no_context_found": True,
        }
    
    # Format context for LLM
    context_text = format_context_for_llm(context_documents_with_scores)
    
    # Prepare system message with strict instructions to avoid hallucinations
    system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided context.

IMPORTANT RULES:
1. ONLY use information from the context provided below. Do not use any outside knowledge.
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question."
3. Do NOT make up or invent information. If you're not sure, say so.
4. Be concise and accurate. Cite the source sections when relevant.

Context:
{context}"""

    # Validate and truncate prompt if needed to fit within token limits
    truncated_context, was_truncated = validate_and_truncate_llm_prompt(
        system_prompt=system_prompt,
        context_text=context_text,
        question=question,
        model_name=model_name,
    )
    
    if was_truncated:
        logger.warning(
            f"Context was truncated to fit within {model_name} token limits. "
            "Some context may be missing."
        )
    
    # Initialize LLM
    if api_base:
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=api_base,
        )
    else:
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
        )
    
    # Create messages with truncated context
    messages = [
        SystemMessage(content=system_prompt.format(context=truncated_context)),
        HumanMessage(content=question),
    ]
    
    # Generate answer with error handling
    try:
        response = llm.invoke(messages)
        answer = response.content
        
        # Validate response
        if not answer or not isinstance(answer, str):
            logger.warning("LLM returned invalid response")
            raise OpenAIAPIError("Invalid response from LLM")
        
        # Extract token usage from response
        # LangChain response may have response_metadata with token usage
        tokens_prompt = 0
        tokens_completion = 0
        total_tokens = 0
        llm_cost_usd = 0.0
        
        if hasattr(response, 'response_metadata') and response.response_metadata:
            usage = response.response_metadata.get('token_usage', {})
            tokens_prompt = usage.get('prompt_tokens', 0)
            tokens_completion = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            
            # Calculate actual cost using pricing module
            llm_cost_usd = calculate_llm_cost(
                tokens_input=tokens_prompt,
                tokens_output=tokens_completion,
                model=model_name
            )
            
    except Exception as e:
        # Log the error with essential details only
        logger.error(f"LLM error: {type(e).__name__} - {str(e)}")
        # Raise generic error message
        raise OpenAIAPIError(
            "An error occurred while generating the answer. Please try again later."
        ) from e
    
    # Extract sources and scores from documents
    sources = []
    context_used = []
    
    for doc, distance_score in context_documents_with_scores:
        if doc.metadata.get("chunk_id"):
            sources.append(doc.metadata.get("chunk_id"))
        # Convert cosine distance to cosine similarity for clarity
        # Chroma returns distance (1 - similarity), so similarity = 1 - distance
        similarity_score = 1.0 - float(distance_score)
        # Format content with optional truncation
        content = doc.page_content
        if MAX_CONTEXT_CONTENT_LENGTH is not None and len(content) > MAX_CONTEXT_CONTENT_LENGTH:
            content = content[:MAX_CONTEXT_CONTENT_LENGTH] + "..."
        
        context_used.append({
            "content": content,
            "section_id": doc.metadata.get("section_id"),
            "section": doc.metadata.get("section"),
            "product_area": doc.metadata.get("product_area"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "similarity_score": similarity_score,  # Actual cosine similarity (0.0 to 1.0)
        })
    
    return {
        "answer": answer,
        "context_used": context_used,
        "sources": sources,
        "no_context_found": False,
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_completion,
        "total_tokens": total_tokens,
        "llm_cost_usd": llm_cost_usd,
    }

