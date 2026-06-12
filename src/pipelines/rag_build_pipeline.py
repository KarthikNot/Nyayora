import asyncio
import os
import pickle
from src.core.logger import logger
from src.core.config import EMBEDDING_BATCH_SIZE, PREPROCESSED_DATASET_PATH
from src.core.utils import check_ollama_availability
from src.components.loader import load_dataset
from src.components.chunker import chunk_documents
from src.components.vector_store import VectorStore


async def run_build_pipeline() -> None:
    try:
        availability, error = check_ollama_availability()
        if not availability:
            logger.error(f"Ollama Error: {error}")
            return None
        logger.info(f"Ollama Connection Successful: {error}")

        logger.info("Started loading dataset...")
        dataset = load_dataset()
        if not dataset:
            logger.warning("Failed to load dataset.")
            return None
        logger.info(f"Dataset loaded successfully. Found {len(dataset)} statutes.")

        vector_store = VectorStore()

        statute_chunks_map = {}
        for statute_name, dataset_docs in dataset.items():
            logger.info(f"--- Processing statute: {statute_name.upper()} ---")

            logger.info(f"Started chunking for {statute_name}...")
            chunks = chunk_documents(dataset=dataset_docs)
            if not chunks:
                logger.warning(f"No chunks generated for {statute_name}. Skipping.")
                continue
            logger.info(f"Storing {len(chunks)} chunks into ChromaDB collection: '{statute_name.lower()}'")

            logger.info(f"Started embedding storage for {statute_name}...")
            result = await vector_store.store_embeddings(chunks=chunks, collection_name=statute_name.lower(), batch_size=EMBEDDING_BATCH_SIZE)
            if result:
                statute_chunks_map[statute_name.lower()] = chunks
                vector_store.initialize_bm25_retriever(chunks, statute_name)
                logger.info(f"Successfully indexed {statute_name}.")
            else:
                logger.error(f"Failed to store embeddings for {statute_name}.")
            logger.info(f"Completed and Stored Vector Embeddings.")

        if statute_chunks_map:
            logger.info("Persisting partitioned chunks to disk for scoped BM25 mapping...")
            os.makedirs(os.path.dirname(PREPROCESSED_DATASET_PATH), exist_ok=True)
            with open(PREPROCESSED_DATASET_PATH, "wb") as f:
                pickle.dump(statute_chunks_map, f)
            
            logger.info("Build pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Error in Build Pipeline: {str(e)}")
        raise e


if __name__ == "__main__":
    asyncio.run(run_build_pipeline())