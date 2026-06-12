import re
import hashlib
from typing import List
from langchain_core.documents import Document
from src.core.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.core.logger import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(dataset: List[Document]) -> List[Document]:
    """
    Splits legal documents into manageable chunks while preserving structural integrity.
    Uses specific legal keywords as separators to keep sections together.

    The function performs:
        1. Removal of TOC(Table of contents) and non-content pages.
        2. Page-level cleaning.
        3. Chapter extraction.
        4. Section-aware splitting.
        5. Recursive chunking.
        6. Metadata enrichment.

    Args:
        dataset (List[Document]):
            A list of LangChain Document objects loaded from legal PDF sources.

    Returns:
        List[Document]:
            A list of chunked and metadata-enriched Document objects ready for
            embedding generation and retrieval.
            - statute
            - chapter
            - section
            - section_title
            - page_number
            - anchor_id
    """
    try:
        if not dataset:
            logger.warning("Received an empty dataset for chunking.")
            return []

        # ========== STEP-1: Filter out TOC(Table of Contents) and junk pages ==========
        filtered_docs = []
        toc_patterns = [r"ARRANGEMENT OF SECTIONS", r"LIST OF ABBREVIATIONS", r"INDEX"]
        
        for doc in dataset:
            content = doc.page_content
            if any(re.search(p, content, re.IGNORECASE) for p in toc_patterns):
                continue
            content = re.sub(r"^\d+\s*\n(?:THE INDIAN PENAL CODE|THE BHARATIYA NYAYA SANHITA|THE BHARATIYA).*?\n", "", content, flags=re.MULTILINE) # Removing Main Titles
            content = re.sub(r"^\d+\s*SECTIONS.*?\n", "", content, flags=re.MULTILINE) # Removing Section Titles
            content = re.sub(r"^\s*\d+\s*$", "", content, flags=re.MULTILINE) # Removing Page Numbers

            doc.page_content = content
            filtered_docs.append(doc)
    except Exception as e:
        logger.error(f"Failed basic cleaning: {str(e)}", exc_info=True)
        raise e

    try:
        # ========== STEP-2: Hierarchy Extraction: Split by section before recursive splitting ==========
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
            is_separator_regex=True,
            separators=[r"\n\n", r"\n", r" ", r""] # Removed sentence level separator
        )

        enriched_chunks = []
        current_chapter = "Unknown"
        current_section = "Unknown"
        current_section_title = "Unknown"
        for doc in filtered_docs:
            content = doc.page_content
            source = doc.metadata.get("source", "").lower()
            page_number = doc.metadata.get("page", 0)

            # Mapping: statute -> act name
            statute, act_name = "IPC", "The Indian Penal Code, 1860"
            if "bns" in source: 
                statute, act_name = "BNS", "The Bharatiya Nyaya Sanhita, 2023"
            elif "bnss" in source: 
                statute, act_name = "BNSS", "The Bharatiya Nagarik Suraksha Sanhita, 2023"
            elif "bsa" in source: 
                statute, act_name = "BSA", "The Bharatiya Sakshya Adhiniyam, 2023"

            # Detect Chapter boundary
            chapter_match = re.search(r"CHAPTER\s+([IVXLCDM]+)", content)
            if chapter_match:
                current_chapter = chapter_match.group(1)

            # Split the page by section markers to isolate content, Regex looks for "378. Theft"
            sections_in_page = re.split(r"\n(\d+[A-Z]?)\.\s+([^\n]+)", content)

            # re.split with groups returns [pre-text, num1, title1, content1, num2, title2, content2...]
            if len(sections_in_page) > 1:
                # Handle pre-text: last saved sections (current_chapter, current_section, current_section_title)
                if sections_in_page[0].strip():
                    create_chunks(
                        text=sections_in_page[0], 
                        statute=statute, 
                        act_name=act_name, 
                        chapter=current_chapter, 
                        section=current_section, 
                        section_title=current_section_title, 
                        page=page_number, 
                        splitter=text_splitter, 
                        output_list=enriched_chunks
                    )

                # Handle new sections
                for i in range(1, len(sections_in_page), 3):
                    current_section = sections_in_page[i]
                    current_section_title = sections_in_page[i+1].strip()
                    section_content = sections_in_page[i+2]

                    create_chunks(
                        text=section_content, 
                        statute=statute, 
                        act_name=act_name, 
                        chapter=current_chapter, 
                        section=current_section, 
                        section_title=current_section_title, 
                        page=page_number, 
                        splitter=text_splitter, 
                        output_list=enriched_chunks
                    )
            else:
                # No section header found on this document ("doc")
                create_chunks(
                    text=content, 
                    statute=statute, 
                    act_name=act_name, 
                    chapter=current_chapter, 
                    section=current_section, 
                    section_title=current_section_title, 
                    page=page_number, 
                    splitter=text_splitter, 
                    output_list=enriched_chunks
                )
        logger.info(f"Total context-aware chunks formed: {len(enriched_chunks)}")
        return enriched_chunks
    except Exception as e:
        logger.error(f"Failed text splitting: {str(e)}", exc_info=True)
        raise e


def create_chunks(text: str, statute: str, act_name: str, chapter: str, section: str, section_title: str, page: int, splitter: RecursiveCharacterTextSplitter, output_list: List[Document]):
    """
    Splits a legal section into smaller chunks and enriches each chunk with
    hierarchical legal metadata.

    Args:
        text (str):
            The section content to be chunked.

        statute (str):
            Statute identifier (e.g., IPC, BNS, BNSS, BSA).

        act_name (str):
            Full name of the statute.

        chapter (str):
            Chapter number or identifier containing the section.

        section (str):
            Section number extracted from the statute.

        section_title (str):
            Official title or heading of the section.

        page (int):
            Source PDF page number.

        splitter (RecursiveCharacterTextSplitter):
            Text splitter used to recursively divide content into smaller chunks.

        output_list (List[Document]):
            Mutable list that receives the generated chunk documents.

    Returns:
        None:
            Appends enriched Document objects directly to output_list.
    """
    sub_chunks = splitter.split_text(text)
    for _, chunk_text in enumerate(sub_chunks):
        content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
        anchor_id = f"{statute}_CH_{chapter}_SEC_{section}_{content_hash}"
        
        metadata = {
            "statute": statute,
            "act_name": act_name,
            "chapter": chapter,
            "section": section,
            "parent_section": section,
            "section_title": section_title,
            "page_number": page,
            "anchor_id": anchor_id,
        }
        
        # Truncate title in header to prevent context length overflow in embedding model (Ollama 400 error)
        safe_title = (section_title[:100] + "...") if len(section_title) > 100 else section_title
        header = f"[Source: {statute} Sec {section} ({safe_title})]\n"
        doc = Document(
            page_content=header + chunk_text,
            metadata=metadata
        )
        output_list.append(doc)