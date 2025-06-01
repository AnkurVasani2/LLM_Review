import os
from dataclasses import dataclass
from typing import List, Dict, Any
from .retrive_from_PDF import retrieve_from_pdf, separate_elements
from .summarize_chunks import summarize_chunks, summarize_images

@dataclass
class ProcessedContent:
    text_summaries: List[str]
    texts: List[str]
    table_summaries: List[str]
    tables: List[str]
    image_summaries: List[str]
    images: List[str]

def process_pdf(pdf_path: str) -> ProcessedContent:
    """Process PDF and generate summaries"""
    # Extract content
    element_summary, chunks = retrieve_from_pdf(pdf_path)
    print(f"Extracted elements: {element_summary}")
    
    
    text_elements, image_elements, table_elements = separate_elements(chunks)
    texts = [str(t) for t in text_elements if t is not None]
    tables = [str(t) for t in table_elements if t is not None]
    
    # Generate summaries
    text_summaries, table_summaries = summarize_chunks(chunks)
    
    
    text_summaries = [s for s in text_summaries if s is not None and str(s).strip()]
    table_summaries = [s for s in table_summaries if s is not None and str(s).strip()]
    
    
    all_text_summary = " ".join(text_summaries)
    try:
        image_summaries, images = summarize_images(text_summary=all_text_summary)
        image_summaries = [s for s in image_summaries if s is not None and str(s).strip()]
        images = [img for img in images if img is not None and str(img).strip()]
    except Exception as e:
        print(f"Error summarizing images: {str(e)}")
        image_summaries, images = [], []

    return ProcessedContent(
        text_summaries=text_summaries,
        texts=texts,
        table_summaries=table_summaries,
        tables=tables,
        image_summaries=image_summaries,
        images=images
    )