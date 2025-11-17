"""Main pipeline for processing FAQ documents and creating RAG-ready chunks."""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from utils.file_io import save_jsonl
from .chunking import create_rag_chunks
from .embeddings import generate_embeddings
from .parsing import load_faq_document

# Load environment variables from .env file
load_dotenv()

def build_index_pipeline(
    input_file: Path,
    output_file: Path,
    chunk_size: int = 100,
    chunk_overlap: int = 20,
    vectorstore_path: Optional[Path] = None,
    openai_api_key: Optional[str] = None,
    generate_embeddings_flag: bool = True,
) -> None:
    """
    Main function to build the FAQ index from document to JSONL chunks with embeddings.
    
    Args:
        input_file: Path to input FAQ document
        output_file: Path to output JSONL file
        chunk_size: Target number of words per chunk (default: 100)
        chunk_overlap: Number of words to overlap between chunks (default: 20)
        vectorstore_path: Path to save Chroma vector store (default: ./chroma_db)
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        generate_embeddings_flag: Whether to generate and save embeddings (default: True)
    """
    print(f"Loading FAQ document from: {input_file}")
    faq_sections = load_faq_document(input_file)
    print(f"Parsed {len(faq_sections)} FAQ sections")
    
    print("Creating RAG-ready chunks...")
    rag_chunks = create_rag_chunks(
        faq_sections,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print(f"Created {len(rag_chunks)} chunks from {len(faq_sections)} sections")
    
    print(f"Saving chunks to: {output_file}")
    save_jsonl(rag_chunks, output_file)
    print(f"Successfully saved {len(rag_chunks)} chunks to {output_file}")
    
    # Generate and save embeddings
    if generate_embeddings_flag:
        print("\nGenerating embeddings and saving to vector store...")
        generate_embeddings(
            chunks=rag_chunks,
            vectorstore_path=vectorstore_path,
            openai_api_key=openai_api_key,
        )
        print("Embedding generation complete!")

