"""Pricing and cost calculation utilities for OpenAI API calls."""

from typing import Optional


# OpenAI pricing per 1K tokens (as of 2025)
# Embedding models
EMBEDDING_PRICING = {
    "text-embedding-ada-002": 0.0001,  # $0.10 per 1M tokens
    "text-embedding-3-small": 0.00002,  # $0.02 per 1M tokens
    "text-embedding-3-large": 0.00013,  # $0.13 per 1M tokens
}

# Embedding model limits
EMBEDDING_MAX_TOKENS = {
    "text-embedding-ada-002": 8191,
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
}

EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

# Default embedding model pricing (ada-002)
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_EMBEDDING_COST_PER_1K = EMBEDDING_PRICING[DEFAULT_EMBEDDING_MODEL]
DEFAULT_EMBEDDING_MAX_TOKENS = EMBEDDING_MAX_TOKENS[DEFAULT_EMBEDDING_MODEL]
DEFAULT_EMBEDDING_DIMENSIONS = EMBEDDING_DIMENSIONS[DEFAULT_EMBEDDING_MODEL]

# LLM models - pricing per 1K tokens
LLM_PRICING = {
    "gpt-3.5-turbo": {
        "input": 0.0005,   # $0.50 per 1M tokens
        "output": 0.0015,  # $1.50 per 1M tokens
    },
    "gpt-4": {
        "input": 0.03,     # $30 per 1M tokens
        "output": 0.06,    # $60 per 1M tokens
    },
    "gpt-4-turbo": {
        "input": 0.01,     # $10 per 1M tokens
        "output": 0.03,    # $30 per 1M tokens
    },
    "gpt-4o": {
        "input": 0.005,    # $5 per 1M tokens
        "output": 0.015,   # $15 per 1M tokens
    },
    "gpt-4o-mini": {
        "input": 0.00015,  # $0.15 per 1M tokens
        "output": 0.0006,  # $0.60 per 1M tokens
    },
}

# LLM models - max context window tokens
# These are approximate max input tokens (leaving room for output)
LLM_MAX_TOKENS = {
    "gpt-3.5-turbo": 16384,  # ~16K context window, reserve ~2K for output
    "gpt-4": 8192,           # ~8K context window, reserve ~1K for output
    "gpt-4-turbo": 128000,   # ~128K context window, reserve ~4K for output
    "gpt-4o": 128000,        # ~128K context window, reserve ~4K for output
    "gpt-4o-mini": 128000,   # ~128K context window, reserve ~4K for output
}

# Default LLM model pricing (gpt-3.5-turbo)
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
DEFAULT_LLM_PRICING = LLM_PRICING[DEFAULT_LLM_MODEL]
DEFAULT_LLM_MAX_TOKENS = LLM_MAX_TOKENS[DEFAULT_LLM_MODEL]


def calculate_embedding_cost(
    tokens: int,
    model: Optional[str] = None
) -> float:
    """
    Calculate cost for embedding generation.
    
    Args:
        tokens: Number of tokens
        model: Embedding model name (defaults to text-embedding-ada-002)
        
    Returns:
        Cost in USD
    """
    if model is None:
        model = DEFAULT_EMBEDDING_MODEL
    
    cost_per_1k = EMBEDDING_PRICING.get(model, DEFAULT_EMBEDDING_COST_PER_1K)
    return (tokens / 1000.0) * cost_per_1k


def calculate_llm_cost(
    tokens_input: int,
    tokens_output: int,
    model: Optional[str] = None
) -> float:
    """
    Calculate cost for LLM completion.
    
    Args:
        tokens_input: Number of input/prompt tokens
        tokens_output: Number of output/completion tokens
        model: LLM model name (defaults to gpt-3.5-turbo)
        
    Returns:
        Cost in USD
    """
    if model is None:
        model = DEFAULT_LLM_MODEL
    
    pricing = LLM_PRICING.get(model, DEFAULT_LLM_PRICING)
    input_cost = (tokens_input / 1000.0) * pricing["input"]
    output_cost = (tokens_output / 1000.0) * pricing["output"]
    return input_cost + output_cost


def get_embedding_model_name() -> str:
    """
    Get the default embedding model name.
    
    Returns:
        Default embedding model name
    """
    return DEFAULT_EMBEDDING_MODEL


def get_embedding_max_tokens(model: Optional[str] = None) -> int:
    """
    Get the maximum tokens allowed for an embedding model.
    
    Args:
        model: Embedding model name (defaults to DEFAULT_EMBEDDING_MODEL)
        
    Returns:
        Maximum tokens allowed
    """
    if model is None:
        model = DEFAULT_EMBEDDING_MODEL
    return EMBEDDING_MAX_TOKENS.get(model, DEFAULT_EMBEDDING_MAX_TOKENS)


def get_embedding_dimensions(model: Optional[str] = None) -> int:
    """
    Get the expected embedding dimensions for an embedding model.
    
    Args:
        model: Embedding model name (defaults to DEFAULT_EMBEDDING_MODEL)
        
    Returns:
        Expected embedding dimensions
    """
    if model is None:
        model = DEFAULT_EMBEDDING_MODEL
    return EMBEDDING_DIMENSIONS.get(model, DEFAULT_EMBEDDING_DIMENSIONS)


def get_llm_model_name() -> str:
    """
    Get the default LLM model name.
    
    Returns:
        Default LLM model name
    """
    return DEFAULT_LLM_MODEL


def get_llm_max_tokens(model: Optional[str] = None) -> int:
    """
    Get the maximum input tokens allowed for an LLM model.
    
    Args:
        model: LLM model name (defaults to DEFAULT_LLM_MODEL)
        
    Returns:
        Maximum input tokens allowed (reserves room for output)
    """
    if model is None:
        model = DEFAULT_LLM_MODEL
    return LLM_MAX_TOKENS.get(model, DEFAULT_LLM_MAX_TOKENS)

