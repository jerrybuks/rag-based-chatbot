"""General file I/O utilities for saving and loading data."""

import json
from pathlib import Path
from typing import Any, Dict, List


def save_jsonl(data: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save data to JSONL file (one JSON object per line).
    
    Args:
        data: List of dictionaries to save
        output_path: Path to output JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

