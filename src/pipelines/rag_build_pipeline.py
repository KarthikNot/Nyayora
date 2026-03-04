import os
from src.core.logger import logger
from src.core.config import *
from src.components.loader import load_dataset
from src.components.chunker import chunk_documents
from src.components.embeddings import generate_embeddings
from src.components.vector_store import VectorStore


def run_build_pipeline():
    try:
        logger.info("Started loading dataset...")
        dataset = load_dataset()
        if not dataset:
            logger.warning("Failed to load dataset.")
            return None
        logger.info("Dataset loaded successfully.")

        logger.info("Started chunking the document...")
        chunks = chunk_documents(dataset = dataset)
        if not chunks:
            logger.warning("Failed to chunk dataset.")
            return None
        logger.info("Chunking completed.")

        logger.info("Started generating of embeddings")
        embeddings = generate_embeddings(chunks = chunks)
        if not embeddings:
            logger.warning("Failed to generate embedddings")
            return None
        logger.info("Embeddings generated.")

        logger.info("Started to store embeddings in db...")
        MONGO_URI = os.getenv("MONGO_URI")
        if not MONGO_URI:
            logger.warning("MONGO_URI does not exist")
            return None
        vector_store = VectorStore(MONGO_URI=MONGO_URI)
        result = vector_store.store_embeddings(chunks = chunks, embeddings = embeddings)
        if not result:
            logger.warning("Failed to store embeddings.")
            return None
        logger.info("Stored embeddings in db.")


    except Exception as e:
        logger.error(f"Error in Build Pipeline: {str(e)}", exc_info = True)


if __name__ == "__main__":
    run_build_pipeline()