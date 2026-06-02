import os
import re
from typing import List
from langchain_core.documents import Document
from src.core.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.core.logger import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(dataset: List[Document]) -> List[Document]:
    """
    Splits legal documents into manageable chunks while preserving structural integrity.
    Uses specific legal keywords as separators to keep sections together.
    """
    try:
        if not dataset:
            logger.warning("Received an empty dataset for chunking.")
            return []

        cleaned_dataset = []
        for doc in dataset:
            source_file = os.path.basename(doc.metadata.get("source", "Unknown_Act")).lower()
            text = doc.page_content
            
            # Remove page-top numbers and titles (e.g., "1 \n THE INDIAN PENAL CODE")
            text = re.sub(r"^\d+\s*\n(?:THE INDIAN PENAL CODE|THE BHARATIYA NYAYA SANHITA).*?\n", "", text, flags=re.MULTILINE)
            text = re.sub(r"^\d+\s*SECTIONS.*?\n", "", text, flags=re.MULTILINE)
            
            # Remove standalone page numbers
            text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
            
            doc.page_content = text
            cleaned_dataset.append(doc)
    except Exception as e:
        logger.error(f"Failed basic cleaning: {str(e)}", exc_info=True)
        raise e

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
            is_separator_regex=True,
            separators=[
                r"\nCHAPTER\s+[IVXLCDM]+", # Chapter boundaries
                r"\n\d+[A-Z]?\.\s+[A-Z]",   # Section boundaries (e.g. "300. Murder")
                r"\nSection\s+\d+[A-Z]?",    # Alternative section format
                r"\n\n",                    # Paragraphs
                r"\n",                      # Lines
                r"\.\s+",                   # Sentences
                r" ",                       # Words
                r""
            ]
        )

        chunks = text_splitter.split_documents(cleaned_dataset)

        enriched_chunks = []
        current_chapter = "Unknown"
        current_section = "Unknown"

        for chunk in chunks:
            content = chunk.page_content
            
            chapter_match = re.search(r"CHAPTER\s+([IVXLCDM]+)", content)
            if chapter_match:
                current_chapter = chapter_match.group(1)
            
            section_match = re.search(r"(?:\n|^)(\d+[A-Z]?)\.\s+", content)
            if not section_match:
                section_match = re.search(r"Section\s+(\d+[A-Z]?)", content, re.IGNORECASE)
            
            if section_match:
                current_section = section_match.group(1)

            # Identify statute based on source metadata
            statute = "IPC"
            if "bns" in chunk.metadata.get("source", "").lower(): statute = "BNS"
            elif "bnss" in chunk.metadata.get("source", "").lower(): statute = "BNSS"
            elif "bsa" in chunk.metadata.get("source", "").lower(): statute = "BSA"
            elif "judgment" in chunk.metadata.get("source", "").lower(): statute = "JUDGMENT"

            chunk.metadata.update({
                "chapter": current_chapter,
                "section": current_section,
                "statute": statute
            })
            
            if current_section != "Unknown":
                chunk.page_content = f"[Source: {statute} Ch {current_chapter}, Sec {current_section}]\n{content}"
            
            enriched_chunks.append(chunk)

        logger.info(f"Total context-aware chunks formed: {len(enriched_chunks)}")
        return enriched_chunks
    except Exception as e:
        logger.error(f"Failed text splitting: {str(e)}", exc_info=True)
        raise e