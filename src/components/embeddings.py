from typing import List
from src.core.logger import logger
from src.core.agents import embedding_model


async def generate_embeddings(chunks: List, batch_size: int = 32) -> List[List[float]]:
    """
    Generates embeddings for document chunks using batch processing and asynchrony.
    Batching prevents memory exhaustion and Ollama service timeouts.
    """
    if not chunks:
        logger.warning("No chunks provided for embedding generation.")
        return []

    try:
        # Filter out empty or whitespace-only chunks to prevent model errors
        texts = [chunk.page_content for chunk in chunks if chunk.page_content and chunk.page_content.strip()]
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        all_embeddings = []
        logger.info(f"Generating embeddings for {len(texts)} chunks in {total_batches} batches.")

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await embedding_model.aembed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Batch Progress: {i // batch_size + 1}/{total_batches} processed.")
            
        return all_embeddings
    except Exception as e:
        logger.error(f"Error while generating embeddings: {str(e)}", exc_info=True)
        raise e


async def user_query_embedding(query: str) -> List[float]:
    """Asynchronously generates an embedding for a single user query."""
    try:
        return await embedding_model.aembed_query(query)
    except Exception as e:
        logger.error(f"Error in user query embedding: {str(e)}", exc_info=True)
        raise e