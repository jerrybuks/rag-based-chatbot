"""Time utility functions for parsing and handling timestamps."""

from datetime import datetime
from typing import Optional


def parse_iso_timestamp(timestamp: str) -> Optional[datetime]:
    """
    Parse an ISO format timestamp string to a datetime object.
    Handles 'Z' suffix and timezone information.
    
    Args:
        timestamp: ISO format timestamp string (e.g., "2025-11-08T12:32:06.246Z")
    
    Returns:
        datetime object with timezone removed, or None if parsing fails
    """
    if not timestamp or not isinstance(timestamp, str):
        return None
    
    try:
        # Handle 'Z' suffix (UTC)
        timestamp_clean = timestamp.replace("Z", "+00:00") if timestamp.endswith("Z") else timestamp
        dt = datetime.fromisoformat(timestamp_clean)
        
        # Remove timezone for comparison
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)
        
        return dt
    except (ValueError, TypeError, AttributeError):
        return None

