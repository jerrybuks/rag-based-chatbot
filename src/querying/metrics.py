"""Metrics tracking for query requests."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from utils.time_utils import parse_iso_timestamp

# Metrics file path
METRICS_FILE = Path("./metrics/metrics.json")


def _load_metrics() -> Dict[str, Any]:
    """Load metrics from JSON file."""
    if not METRICS_FILE.exists():
        return {
            "requests": [],
            "successes": 0,
            "failures": 0,
        }
    
    try:
        with open(METRICS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Could not load metrics from file: {e}")
        return {
            "requests": [],
            "successes": 0,
            "failures": 0,
        }


def _save_metrics(data: Dict[str, Any]) -> None:
    """Save metrics to JSON file."""
    try:
        # Ensure directory exists
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file atomically
        temp_file = METRICS_FILE.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        temp_file.replace(METRICS_FILE)
    except Exception as e:
        print(f"Warning: Could not save metrics to file: {e}")


def record_request(
    latency_ms: float,
    total_tokens: int,
    tokens_prompt: int,
    tokens_completion: int,
    embedding_tokens: int,
    embedding_cost_usd: float,
    llm_cost_usd: float,
    total_cost_usd: float,
    success: bool,
    error: Optional[str],
    question_snippet: str,
    query_id: str,
) -> None:
    """
    Record a request metric.
    
    Args:
        latency_ms: Request latency in milliseconds
        total_tokens: Total tokens used (LLM tokens + embedding tokens)
        tokens_prompt: Prompt tokens (LLM)
        tokens_completion: Completion tokens (LLM)
        embedding_tokens: Tokens used for embedding generation
        embedding_cost_usd: Cost for embedding generation in USD
        llm_cost_usd: Cost for LLM call in USD
        total_cost_usd: Total cost (embedding + LLM) in USD
        success: Whether request succeeded
        error: Error message if failed
        question_snippet: Snippet of the question (without XML tags)
        query_id: Unique hash ID for the query
    """
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    request_data = {
        "timestamp": timestamp,
        "latencyMs": latency_ms,
        "total_tokens": total_tokens,
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_completion,
        "embedding_tokens": embedding_tokens,
        "costUsd": total_cost_usd,
        "embeddingCostUsd": embedding_cost_usd,
        "llmCostUsd": llm_cost_usd,
        "success": success,
        "error": error,
        "questionSnippet": question_snippet,
        "queryId": query_id,
    }
    
    # Load current metrics
    data = _load_metrics()
    
    # Add new request (keep only last 50)
    data["requests"].append(request_data)
    data["requests"] = data["requests"][-50:]
    
    # Update counters
    if success:
        data["successes"] = data.get("successes", 0) + 1
    else:
        data["failures"] = data.get("failures", 0) + 1
    
    # Save to file
    _save_metrics(data)


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile from a list of values."""
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    index = (percentile / 100.0) * (len(sorted_values) - 1)
    
    if index.is_integer():
        return sorted_values[int(index)]
    else:
        lower = sorted_values[int(index)]
        upper = sorted_values[int(index) + 1]
        return lower + (upper - lower) * (index - int(index))


def get_metrics() -> Dict[str, Any]:
    """
    Get aggregated metrics.
    
    Returns:
        Dictionary with metrics including totals, averages, percentiles, and recent requests
    """
    # Load metrics from file
    data = _load_metrics()
    requests = data.get("requests", [])
    successes = data.get("successes", 0)
    failures = data.get("failures", 0)
    
    # Extract lists from requests
    latencies = [r.get("latencyMs", 0) for r in requests]
    tokens = [r.get("total_tokens", 0) for r in requests]
    costs = [r.get("costUsd", 0) for r in requests]
    embedding_costs = [r.get("embeddingCostUsd", 0) for r in requests]
    llm_costs = [r.get("llmCostUsd", 0) for r in requests]
    
    total_requests = len(requests)
    total_successes = successes
    total_failures = failures
    
    # Calculate error rate
    error_rate = (total_failures / total_requests) if total_requests > 0 else 0.0
    
    # Calculate latency metrics
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p50_latency = calculate_percentile(latencies, 50) if latencies else 0.0
    p95_latency = calculate_percentile(latencies, 95) if latencies else 0.0
    
    # Calculate token metrics
    total_tokens = sum(tokens) if tokens else 0
    total_prompt = sum(r.get("tokens_prompt", 0) for r in requests)
    total_completion = sum(r.get("tokens_completion", 0) for r in requests)
    total_embedding_tokens = sum(r.get("embedding_tokens", 0) for r in requests)
    
    # Calculate costs (separate and total)
    total_cost = sum(costs) if costs else 0.0
    total_embedding_cost = sum(embedding_costs) if embedding_costs else 0.0
    total_llm_cost = sum(llm_costs) if llm_costs else 0.0
    
    # Calculate throughput (requests per second)
    throughput = 0.0
    if len(requests) >= 2:
        timestamps = [r.get("timestamp") for r in requests if r.get("timestamp")]
        if len(timestamps) >= 2:
            first_time = parse_iso_timestamp(timestamps[0])   # Oldest (first in list)
            last_time = parse_iso_timestamp(timestamps[-1])   # Newest (last in list)
            
            if first_time and last_time:
                time_span = (last_time - first_time).total_seconds()
                if time_span > 0:
                    throughput = len(requests) / time_span
    
    # Generate insights
    insights = []
    if avg_latency > 2000:
        insights.append("Average latency > 2s — consider increasing timeouts or switching models")
    if p95_latency > 5000:
        insights.append("P95 latency > 5s — high latency outliers detected")
    if error_rate > 0.1:
        insights.append(f"Error rate is {error_rate*100:.1f}% — investigate failures")
    if throughput < 0.1 and total_requests > 10:
        insights.append("Low throughput — consider optimizing retrieval pipeline")
    
    # Get recent requests (last 10, most recent first)
    recent = list(reversed(requests[-10:]))
    
    return {
        "totalRequests": total_requests,
        "successes": total_successes,
        "failures": total_failures,
        "errorRate": error_rate,
        "avgLatency": avg_latency,
        "p50Latency": p50_latency,
        "p95Latency": p95_latency,
        "throughput": throughput,
        "totalTokens": total_tokens,
        "totalPrompt": total_prompt,
        "totalCompletion": total_completion,
        "totalEmbeddingTokens": total_embedding_tokens,
        "totalCost": total_cost,
        "totalEmbeddingCost": total_embedding_cost,
        "totalLlmCost": total_llm_cost,
        "insights": insights,
        "recent": recent,
    }


def reset_metrics() -> None:
    """Reset all metrics (useful for testing)."""
    empty_data = {
        "requests": [],
        "successes": 0,
        "failures": 0,
    }
    _save_metrics(empty_data)

