from src.core.config import *
from src.core.logger import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(dataset):
    chunks = []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        chunks = text_splitter.split_documents(dataset)
        if not chunks:
            logger.warning("Failed to create chunks")
            return chunks

        logger.info(f"Total chunks formed: {len(chunks)}")
        return chunks
    except Exception as e:
        logger.error("Failed text splitting: {str(e)}", exc_info = True)
    return chunks