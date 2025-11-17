"""Document parsing utilities for FAQ documents."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import TextLoader


def parse_faq_section(section_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single FAQ section from the document.
    
    Args:
        section_text: Raw text of a single FAQ section
        
    Returns:
        Dictionary with parsed metadata and content, or None if parsing fails
    """
    section_data = {}
    
    # Extract section_id
    section_id_match = re.search(r'section_id:\s*(.+)', section_text)
    if section_id_match:
        section_data['section_id'] = section_id_match.group(1).strip()
    else:
        return None
    
    # Extract section
    section_match = re.search(r'section:\s*(.+)', section_text)
    if section_match:
        section_data['section'] = section_match.group(1).strip()
    
    # Extract product_area
    product_area_match = re.search(r'product_area:\s*(.+)', section_text)
    if product_area_match:
        section_data['product_area'] = product_area_match.group(1).strip()
    
    # Extract intent_tags (JSON array format)
    intent_tags_match = re.search(r'intent_tags:\s*(\[[^\]]+\])', section_text)
    if intent_tags_match:
        try:
            section_data['intent_tags'] = json.loads(intent_tags_match.group(1))
        except json.JSONDecodeError:
            section_data['intent_tags'] = []
    else:
        section_data['intent_tags'] = []
    
    # Extract last_updated
    last_updated_match = re.search(r'last_updated:\s*(.+)', section_text)
    if last_updated_match:
        section_data['last_updated'] = last_updated_match.group(1).strip()
    
    # Extract content (everything after "content:")
    content_match = re.search(r'content:\s*(.+)', section_text, re.DOTALL)
    if content_match:
        section_data['content'] = content_match.group(1).strip()
    else:
        return None
    
    return section_data


def load_faq_document(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load and parse FAQ document into structured sections.
    
    Uses langchain's TextLoader to load the document, then parses it into sections.
    
    Args:
        file_path: Path to the FAQ document
        
    Returns:
        List of parsed FAQ sections
    """
    # Use TextLoader from langchain to load the document
    loader = TextLoader(str(file_path), encoding='utf-8')
    documents = loader.load()
    
    # Extract content from the loaded document
    content = documents[0].page_content if documents else ""
    
    # Split by section separators (---)
    sections = re.split(r'\n---\n', content)
    
    parsed_sections = []
    for section_text in sections:
        section_text = section_text.strip()
        if not section_text:
            continue
        
        parsed_section = parse_faq_section(section_text)
        if parsed_section:
            parsed_sections.append(parsed_section)
    
    return parsed_sections

