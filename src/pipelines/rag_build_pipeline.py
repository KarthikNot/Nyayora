import asyncio
import os
import pickle
from src.core.logger import logger
from src.core.config import *
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
        all_chunks = []

        for statute_name, dataset_docs in dataset.items():
            logger.info(f"--- Processing statute: {statute_name.upper()} ---")
            
            chunks = chunk_documents(dataset=dataset_docs)
            if not chunks:
                logger.warning(f"No chunks generated for {statute_name}. Skipping.")
                continue
            
            logger.info(f"Storing {len(chunks)} chunks into ChromaDB collection: '{statute_name.lower()}'")
            result = await vector_store.store_embeddings(chunks=chunks, collection_name=statute_name.lower(), batch_size=EMBEDDING_BATCH_SIZE)
            
            if result:
                all_chunks.extend(chunks)
                logger.info(f"Successfully indexed {statute_name}.")
            else:
                logger.error(f"Failed to store embeddings for {statute_name}.")

        if all_chunks:
            logger.info("Persisting aggregated chunks for BM25 retrieval...")
            os.makedirs(os.path.dirname(PREPROCESSED_DATASET_PATH), exist_ok=True)
            with open(PREPROCESSED_DATASET_PATH, "wb") as f:
                pickle.dump(all_chunks, f)
            
            vector_store.initialize_bm25_retriever(all_chunks)
            logger.info("Build pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Error in Build Pipeline: {str(e)}", exc_info = True)


if __name__ == "__main__":
    asyncio.run(run_build_pipeline())