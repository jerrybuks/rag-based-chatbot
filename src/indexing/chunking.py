"""Utilities for building RAG-ready chunks from parsed FAQ sections."""

from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter


def count_words(text: str) -> int:
    """Count the number of words in a text string."""
    return len(text.split())


def split_content_into_chunks(
    content: str,
    chunk_size: int = 100,
    chunk_overlap: int = 20
) -> List[str]:
    """
    Split content into chunks of approximately specified word count with overlap.
    
    Uses langchain's RecursiveCharacterTextSplitter with word-aware splitting.
    The splitter approximates word-based chunking using character-based splitting
    with word boundaries as separators.
    
    Args:
        content: Text content to split
        chunk_size: Target number of words per chunk (~100 words)
        chunk_overlap: Number of words to overlap between chunks (20% overlap = 20 words)
        
    Returns:
        List of text chunks
    """
    # Estimate character count from word count
    # Average English word is ~5 characters plus 1 space = ~6 characters per word
    # Using a conservative estimate to ensure we stay close to word count
    estimated_chars_per_word = 6
    char_chunk_size = chunk_size * estimated_chars_per_word
    char_chunk_overlap = chunk_overlap * estimated_chars_per_word
    
    # Use RecursiveCharacterTextSplitter with word-aware separators
    # This will split at word boundaries while approximating the word count
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_chunk_size,
        chunk_overlap=char_chunk_overlap,
        length_function=len,  # Character-based length
        separators=["\n\n", "\n", ". ", " ", ""],  # Word-aware separators
        keep_separator=True,
    )
    
    chunks = text_splitter.split_text(content)
    
    # Filter and validate chunks
    # Ensure chunks meet minimum size requirements
    filtered_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        word_count = count_words(chunk)
        # Keep chunks with at least 10 words (10% of target) and reasonable length
        if word_count >= 10 and len(chunk) > 20:
            filtered_chunks.append(chunk)
    
    return filtered_chunks


def create_rag_chunks(
    faq_sections: List[Dict[str, Any]],
    chunk_size: int = 100,
    chunk_overlap: int = 20
) -> List[Dict[str, Any]]:
    """
    Create RAG-ready chunks from FAQ sections.
    
    Args:
        faq_sections: List of parsed FAQ sections
        chunk_size: Target number of words per chunk
        chunk_overlap: Number of words to overlap between chunks
        
    Returns:
        List of RAG-ready chunk dictionaries
    """
    rag_chunks = []
    
    for section in faq_sections:
        content = section.get('content', '')
        if not content:
            continue
        
        # Split content into chunks
        content_chunks = split_content_into_chunks(
            content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Create a RAG chunk for each content chunk
        for idx, chunk_text in enumerate(content_chunks):
            rag_chunk = {
                'chunk_id': f"{section['section_id']}_chunk_{idx + 1}",
                'section_id': section.get('section_id'),
                'section': section.get('section'),
                'product_area': section.get('product_area'),
                'last_updated': section.get('last_updated'),
                'intent_tags': section.get('intent_tags', []),
                'content': chunk_text,
                'word_count': count_words(chunk_text),
                'chunk_index': idx + 1,
                'total_chunks': len(content_chunks),
            }
            rag_chunks.append(rag_chunk)
    
    return rag_chunks

