"""Caching utilities for query responses."""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Cache file path
CACHE_FILE = Path("./cache/cache.json")

# Maximum number of cache entries to keep
MAX_CACHE_ENTRIES = 100


def _get_question_hash(question: str) -> str:
    """
    Generate a hash for the question to use as cache key.
    
    Args:
        question: The user's question
        
    Returns:
        SHA256 hash of the question (normalized)
    """
    # Normalize question: strip whitespace and convert to lowercase
    normalized = question.strip().lower()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def _load_cache() -> Dict[str, Any]:
    """Load cache from JSON file."""
    if not CACHE_FILE.exists():
        return {}
    
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Could not load cache from file: {e}")
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    """Save cache to JSON file."""
    try:
        # Ensure directory exists
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file atomically
        temp_file = CACHE_FILE.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        temp_file.replace(CACHE_FILE)
    except Exception as e:
        logger.warning(f"Could not save cache to file: {e}")


def get_cached_response(question: str) -> Optional[Dict[str, Any]]:
    """
    Get cached response for a question if it exists.
    
    Args:
        question: The user's question
        
    Returns:
        Cached response dictionary if found, None otherwise
    """
    cache = _load_cache()
    question_hash = _get_question_hash(question)
    
    if question_hash in cache:
        logger.debug(f"Cache hit for question: {question[:50]}...")
        return cache[question_hash].get("response")
    
    logger.debug(f"Cache miss for question: {question[:50]}...")
    return None


def get_cached_response_by_id(query_id: str) -> Optional[Dict[str, Any]]:
    """
    Get cached response by query_id (hash).
    
    Args:
        query_id: The query ID (hash) to look up
        
    Returns:
        Dictionary with 'question' and 'response' if found, None otherwise
    """
    cache = _load_cache()
    
    if query_id in cache:
        cache_entry = cache[query_id]
        logger.debug(f"Cache hit for query_id: {query_id[:16]}...")
        return {
            "question": cache_entry.get("question", ""),
            "response": cache_entry.get("response", {})
        }
    
    logger.debug(f"Cache miss for query_id: {query_id[:16]}...")
    return None


def save_to_cache(question: str, response: Dict[str, Any]) -> None:
    """
    Save a response to cache.
    
    Args:
        question: The user's question
        response: The response dictionary to cache
    """
    cache = _load_cache()
    question_hash = _get_question_hash(question)
    
    # Store response with timestamp
    from datetime import datetime
    cache[question_hash] = {
        "question": question,
        "response": response,
        "cached_at": datetime.utcnow().isoformat() + "Z",
    }
    
    # Limit cache size by removing oldest entries
    if len(cache) > MAX_CACHE_ENTRIES:
        # Sort by timestamp and keep only the most recent entries
        sorted_entries = sorted(
            cache.items(),
            key=lambda x: x[1].get("cached_at", ""),
            reverse=True
        )
        cache = dict(sorted_entries[:MAX_CACHE_ENTRIES])
    
    _save_cache(cache)
    logger.debug(f"Cached response for question: {question[:50]}...")


def clear_cache() -> None:
    """Clear all cache entries."""
    empty_cache = {}
    _save_cache(empty_cache)
    logger.info("Cache cleared")

