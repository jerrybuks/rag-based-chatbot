"""Entry point for building FAQ index from document to JSONL chunks."""

from pathlib import Path

from indexing import build_index_pipeline

if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "faq_document.txt"
    output_file = project_root / "data" / "hrcare_faq_chunks.jsonl"
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Build index
    build_index_pipeline(input_file, output_file)

