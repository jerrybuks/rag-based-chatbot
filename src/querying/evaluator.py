"""Evaluation agent for verifying answer quality and detecting hallucinations."""

import os
import logging
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .exceptions import OpenAIAPIError
from .pricing import calculate_llm_cost, DEFAULT_LLM_MODEL

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


def evaluate_answer(
    question: str,
    answer: str,
    context_used: list[dict],
    openai_api_key: Optional[str] = None,
    model_name: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluate an answer for hallucination and quality.
    
    Args:
        question: The original question
        answer: The answer to evaluate
        context_used: List of context chunks that were used to generate the answer
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        model_name: Model to use (defaults to DEFAULT_LLM_MODEL)
        temperature: Model temperature (default: 0.0 for deterministic)
        
    Returns:
        Dictionary with evaluation results including verdict, confidence, and hallucination flag
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
    
    # Format context for evaluation
    context_text = ""
    if context_used:
        context_parts = []
        for idx, ctx in enumerate(context_used, 1):
            content = ctx.get("content", "")
            section = ctx.get("section", "")
            similarity = ctx.get("similarity_score", 0.0)
            context_parts.append(
                f"[Context {idx} - Section: {section}, Similarity: {similarity:.4f}]\n{content}\n"
            )
        context_text = "\n---\n".join(context_parts)
    
    # Prepare system message for evaluation
    system_prompt = """You are an expert evaluator that checks answers for hallucinations and factual accuracy.

Your task is to evaluate whether the given answer:
1. Is supported by the provided context
2. Contains any information not present in the context (hallucination)
3. Accurately represents the information from the context

IMPORTANT: You must respond in the following JSON format:
{
  "verdict": "RELIABLE" or "SUSPECTED_HALLUCINATION",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of your evaluation"
}

Where:
- "verdict": 
  - "RELIABLE": Answer is reliable and well-supported by context
  - "SUSPECTED_HALLUCINATION": Answer has high degree of response not in context and confidence is below 0.75
- "confidence": Your confidence in the evaluation (0.0 = not confident, 1.0 = very confident)
- "reasoning": Brief explanation of your evaluation

"""

    # Prepare evaluation prompt
    evaluation_prompt = f"""Question: {question}

Context Provided:
{context_text if context_text else "No context was provided."}

Answer to Evaluate:
{answer}

Please evaluate this answer and provide your assessment in the JSON format specified."""

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
    
    # Create messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=evaluation_prompt),
    ]
    
    # Generate evaluation with error handling
    try:
        response = llm.invoke(messages)
        evaluation_text = response.content
        
        # Validate response
        if not evaluation_text or not isinstance(evaluation_text, str):
            logger.warning("LLM returned invalid evaluation response")
            raise OpenAIAPIError("Invalid response from LLM")
        
        # Parse JSON response
        import json
        import re
        
        # Try to extract JSON from response (handle cases where LLM adds extra text)
        # Look for JSON object with proper structure
        json_pattern = r'\{[^{}]*(?:"verdict"|"confidence")[^{}]*\}'
        json_match = re.search(json_pattern, evaluation_text, re.DOTALL)
        
        if json_match:
            try:
                evaluation_json = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                # Try a more permissive extraction
                # Find content between first { and last }
                start = evaluation_text.find('{')
                end = evaluation_text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    try:
                        evaluation_json = json.loads(evaluation_text[start:end+1])
                    except json.JSONDecodeError:
                        # Fallback: try parsing the whole response
                        evaluation_json = json.loads(evaluation_text)
                else:
                    evaluation_json = json.loads(evaluation_text)
        else:
            # Try parsing the whole response
            evaluation_json = json.loads(evaluation_text)
        
        # Extract and validate fields
        verdict = evaluation_json.get("verdict", "UNKNOWN").upper()
        confidence = float(evaluation_json.get("confidence", 0.5))
        reasoning = evaluation_json.get("reasoning", "No reasoning provided")
        
        # Clamp confidence to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))
        
        # Validate verdict
        if verdict not in ["RELIABLE", "SUSPECTED_HALLUCINATION"]:
            # Default based on confidence if invalid verdict
            verdict = "SUSPECTED_HALLUCINATION" if confidence < 0.75 else "RELIABLE"
            logger.warning(f"Invalid verdict received: {evaluation_json.get('verdict')}, defaulting to {verdict}")
        
        # Set possible_hallucination based on confidence threshold
        # Only true if confidence is below 0.75
        possible_hallucination = confidence < 0.75
        
        # Extract token usage from response
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
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "possible_hallucination": possible_hallucination,
            "reasoning": reasoning,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "total_tokens": total_tokens,
            "llm_cost_usd": llm_cost_usd,
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse evaluation JSON: {e}")
        raise OpenAIAPIError("Failed to parse evaluation response from LLM")
    except Exception as e:
        # Log the error with essential details only
        logger.error(f"Evaluation error: {type(e).__name__} - {str(e)}")
        # Raise generic error message
        raise OpenAIAPIError(
            "An error occurred while evaluating the answer. Please try again later."
        ) from e

