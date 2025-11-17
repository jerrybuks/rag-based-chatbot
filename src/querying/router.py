"""FastAPI router for query endpoints."""

import os
import time
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, status

from .models import QuestionRequest, QuestionResponse
from .retrieval import retrieve_context
from .llm import generate_answer
from .exceptions import OpenAIAPIError, EmbeddingError, RetrievalError
from .metrics import record_request, get_metrics
from .pricing import DEFAULT_LLM_MODEL
from .cache import get_cached_response, save_to_cache, _get_question_hash, get_cached_response_by_id
from .error_handler import handle_query_error, handle_api_key_error
from .evaluator import evaluate_answer
from .evaluation_models import EvaluationResponse

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/query", tags=["query"])

# Default vectorstore path
DEFAULT_VECTORSTORE_PATH = Path("./chroma_db")


@router.post("/", response_model=QuestionResponse)
async def query_question(request: QuestionRequest) -> QuestionResponse:
    """
    Query endpoint that accepts a question and returns an answer.
    
    Process:
    1. Validates question (max 100 words)
    2. Converts question to embeddings
    3. Performs vector search with optional filters (hybrid search)
    4. Calls LLM with question and context
    5. Returns answer with sources
    
    Args:
        request: QuestionRequest with question and optional filters
        
    Returns:
        QuestionResponse with answer, context, and sources
    """
    start_time = time.time()
    success = False
    error_message = None
    tokens_prompt = 0
    tokens_completion = 0
    total_tokens = 0
    model_name = DEFAULT_LLM_MODEL
    cached_response = None
    query_id = None
    
    try:
        # Generate query ID (hash) for this question
        query_id = _get_question_hash(request.question)
        
        # Check cache first
        cached_response = get_cached_response(request.question)
        if cached_response:
            logger.info("Returning cached response")
            # Reorder cached response to have query_id first
            cached_response = {
                "query_id": query_id,
                **{k: v for k, v in cached_response.items() if k != "query_id"}
            }
            # Return cached response immediately (no metrics recording - already recorded on first request)
            return QuestionResponse(**cached_response)
        
        # Get API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise handle_api_key_error()
        
        # Step 1: Validation is handled by Pydantic model
        # Step 2 & 3: Retrieve relevant context (embeddings + vector search with scores)
        embedding_cost_usd = 0.0
        try:
            context_documents_with_scores, embedding_metadata = retrieve_context(
                question=request.question,
                vectorstore_path=DEFAULT_VECTORSTORE_PATH,
                k=3,  # Top 3 results
                filters=request.filters,
                openai_api_key=openai_api_key,
                min_similarity=request.min_similarity,  # Filter by similarity threshold
            )
            
            # Extract embedding cost
            embedding_cost_usd = embedding_metadata.get("cost_usd", 0.0)
        except (EmbeddingError, RetrievalError) as e:
            error_message = str(e)
            raise handle_query_error(e)
        
        # Step 4: Generate answer using LLM (if context found)
        try:
            result = generate_answer(
                question=request.question,
                context_documents_with_scores=context_documents_with_scores,
                openai_api_key=openai_api_key,
            )
            
            # Extract token usage and costs from result
            tokens_prompt = result.get("tokens_prompt", 0)
            tokens_completion = result.get("tokens_completion", 0)
            total_tokens = result.get("total_tokens", 0)
            llm_cost_usd = result.get("llm_cost_usd", 0.0)
            
            # Remove internal fields from result before returning
            result_clean = {
                k: v for k, v in result.items() 
                if not k.startswith("tokens_") and k != "total_tokens" and k != "llm_cost_usd"
            }
            # Create new dict with query_id first
            result_clean = {
                "query_id": query_id,
                **result_clean
            }
            success = True
            
            # Save to cache
            save_to_cache(request.question, result_clean)
            
        except OpenAIAPIError as e:
            error_message = str(e)
            raise handle_query_error(e)
        
        # Step 5: Return response
        return QuestionResponse(**result_clean)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except (ValueError, EmbeddingError, RetrievalError, OpenAIAPIError) as e:
        error_message = str(e)
        raise handle_query_error(e)
    except Exception as e:
        error_message = str(e)
        raise handle_query_error(e)
    finally:
        # Record metrics (only if not from cache)
        if cached_response is None:
            latency_ms = (time.time() - start_time) * 1000
            total_cost_usd = embedding_cost_usd + llm_cost_usd
            
            # Create question snippet (first 100 chars)
            question_snippet = request.question[:100] if len(request.question) > 100 else request.question
            
            # Ensure query_id is set
            if query_id is None:
                query_id = _get_question_hash(request.question)
            
            record_request(
                latency_ms=latency_ms,
                total_tokens=total_tokens,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                embedding_cost_usd=embedding_cost_usd,
                llm_cost_usd=llm_cost_usd,
                total_cost_usd=total_cost_usd,
                success=success,
                error=error_message,
                question_snippet=question_snippet,
                query_id=query_id,
            )


@router.get("/metrics")
async def get_metrics_endpoint():
    """
    Get metrics for query requests.
    
    Returns:
        Dictionary with aggregated metrics including totals, averages, percentiles, and recent requests
    """
    return get_metrics()


@router.get("/evaluate/{query_id}", response_model=EvaluationResponse)
async def evaluate_query(query_id: str) -> EvaluationResponse:
    """
    Evaluate a query response for hallucination and quality.
    
    This endpoint:
    1. Looks up the query by query_id in the cache
    2. Extracts the question, answer, and context used
    3. Makes an LLM call to evaluate the answer for hallucination
    4. Returns evaluation results with verdict, confidence, and hallucination flag
    
    Args:
        query_id: The query ID (hash) to evaluate
        
    Returns:
        EvaluationResponse with evaluation results, question, and answer details
        
    Raises:
        HTTPException: If query_id not found in cache or evaluation fails
    """
    try:
        # Look up query in cache by query_id
        cached_data = get_cached_response_by_id(query_id)
        
        if not cached_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Query ID '{query_id}' not found in cache. Please make a call to the /api/v1/query/ endpoint first to get a query_id."
            )
        
        question = cached_data.get("question", "")
        response = cached_data.get("response", {})
        
        if not question or not response:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Cached data is incomplete. Cannot evaluate."
            )
        
        # Extract answer and context from response
        answer = response.get("answer", "")
        context_used = response.get("context_used", [])
        
        if not answer:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No answer found in cached response. Cannot evaluate."
            )
        
        # Get API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise handle_api_key_error()
        
        # Evaluate the answer
        try:
            evaluation_result = evaluate_answer(
                question=question,
                answer=answer,
                context_used=context_used,
                openai_api_key=openai_api_key,
            )
        except OpenAIAPIError as e:
            raise handle_query_error(e)
        
        # Build evaluation response
        return EvaluationResponse(
            query_id=query_id,
            question=question,
            answer=answer,
            verdict=evaluation_result["verdict"],
            confidence=evaluation_result["confidence"],
            possible_hallucination=evaluation_result["possible_hallucination"],
            reasoning=evaluation_result["reasoning"],
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise handle_query_error(e)

