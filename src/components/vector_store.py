import os
import asyncio
import pickle
from typing import List, Any, Optional, Sequence, Dict
from src.core.config import *
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from src.core.logger import logger
from src.core.agents import embedding_model
from langchain_core.documents import Document

class VectorStore:
    def __init__(self) -> None:
        self.embeddings = embedding_model
        self.base_persist_directory = CHROMA_DB_PATH
        self.vector_dbs: Dict[str, Chroma] = {}
        self.bm25_retriever: Optional[BM25Retriever] = None

        os.makedirs(self.base_persist_directory, exist_ok=True)

        self.load_existing_collections()
        
        self.load_bm25_retriever()

    def load_bm25_retriever(self) -> None:
        """Loads BM25 retriever from persisted chunks if they exist."""
        if os.path.exists(PREPROCESSED_DATASET_PATH):
            try:
                logger.info("Loading persisted chunks for BM25 index...")
                with open(PREPROCESSED_DATASET_PATH, "rb") as f:
                    chunks = pickle.load(f)
                    self.initialize_bm25_retriever(chunks)
                logger.info("BM25 retriever initialized from disk.")
            except Exception as e:
                logger.error(f"Failed to load BM25 retriever: {e}")

    def load_existing_collections(self) -> None:
        """Discovers and loads existing ChromaDB collections from the base persist directory."""
        for item in os.listdir(self.base_persist_directory):
            item_path = os.path.join(self.base_persist_directory, item)
            if os.path.isdir(item_path):
                logger.info(f"Loading existing ChromaDB collection: {item}")
                try:
                    self.vector_dbs[item] = Chroma(
                        persist_directory=item_path,
                        embedding_function=self.embeddings
                    )
                except Exception as e:
                    logger.warning(f"Failed to load ChromaDB collection '{item}': {e}")


    def get_or_create_db(self, collection_name: str) -> Chroma:
        if collection_name not in self.vector_dbs:
            db_path = os.path.join(self.base_persist_directory, collection_name)
            self.vector_dbs[collection_name] = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        return self.vector_dbs[collection_name]


    async def store_embeddings(self, chunks: List[Document], collection_name: str, batch_size: int = 100) -> bool:
        """
        Asynchronously stores document chunks in ChromaDB with batch processing.
        Batching prevents memory exhaustion and ensures persistence for large datasets.
        """
        try:
            if not chunks:
                logger.warning(f"No chunks provided for storage in collection '{collection_name}'.")
                return False

            logger.info(f"Ingesting {len(chunks)} chunks into ChromaDB collection '{collection_name}' in batches of {batch_size}...")
            
            vector_db_instance = self.get_or_create_db(collection_name)
            
            # Process in safe thread-isolated execution blocks
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                await asyncio.to_thread(
                    vector_db_instance.add_documents,
                    documents=batch
                )
                logger.info(f"Storage Progress for '{collection_name}': {min(i + batch_size, len(chunks))}/{len(chunks)} chunks processed.")

            logger.info(f"Successfully stored and persisted all chunks for collection '{collection_name}'.")
            return True
        except Exception as e:
            logger.error(f"Critical error during embedding storage for collection '{collection_name}': {str(e)}", exc_info=True)
            raise e

    def initialize_bm25_retriever(self, all_documents: List[Document]) -> None:
        """Initializes the BM25 retriever with all available documents."""
        if not all_documents:
            logger.warning("No documents provided to initialize BM25 retriever.")
            return
        logger.info(f"Initializing BM25 retriever with {len(all_documents)} documents.")
        self.bm25_retriever = BM25Retriever.from_documents(all_documents)
        logger.info("BM25 retriever initialized.")


    def reciprocal_rank_fusion(self, results_list: List[List[dict]], k: int = 60) -> List[dict]:
        """
        Combines multiple retrieval rankings using Reciprocal Rank Fusion.
        """
        fused_scores = {}
        content_map = {}

        for docs in results_list:
            for rank, doc in enumerate(docs):
                content = doc["content"]
                if content not in fused_scores:
                    fused_scores[content] = 0
                    content_map[content] = doc["metadata"]
                fused_scores[content] += 1 / (rank + k)

        # Sort by score descending
        reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                "content": content,
                "metadata": metadata
            }
            for content, metadata in reranked_results
        ]


    async def retrieve_documents(self, query: str, k: int = 20, use_mmr: bool = True, statute_filter: Optional[Sequence[str]] = None) -> List[dict]:
        """
        Performs Hybrid Search: BM25 (keyword) + Vector (semantic) combined with RRF.
        """
        try:
            all_vector_results = []
            all_bm25_results = []

            # Determine which collections to query
            collections_to_query = []
            if statute_filter:
                # Filter based on provided statutes (which are now collection names)
                collections_to_query = [s.lower() for s in statute_filter if s.lower() in self.vector_dbs] # Ensure case consistency
            else:
                # Query all loaded collections if no filter is provided
                collections_to_query = list(self.vector_dbs.keys())

            if not collections_to_query:
                logger.warning(f"No vector stores found for the given statute filter: {statute_filter}. Returning empty results.")
                return []
            
            logger.info(f"Querying collections: {collections_to_query}")

            # Perform vector search across selected collections
            vector_search_tasks = []
            for collection_name in collections_to_query:
                vector_db_instance = self.get_or_create_db(collection_name)
                if use_mmr:
                    vector_search_tasks.append(
                        asyncio.to_thread(
                            vector_db_instance.max_marginal_relevance_search,
                            query,
                            k,
                            fetch_k=40,
                        )
                    )
                else:
                    vector_search_tasks.append(
                        asyncio.to_thread(vector_db_instance.similarity_search, query, k)
                    )
            
            results_from_dbs = await asyncio.gather(*vector_search_tasks)
            for docs_vector in results_from_dbs:
                all_vector_results.extend([
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in docs_vector
                ])

            # 2. Sparse Search (BM25) - Use the globally initialized BM25 retriever
            # Note: BM25 retriever is initialized once with all documents in the build pipeline.
            if self.bm25_retriever:
                docs_bm25 = self.bm25_retriever.invoke(query)[:k]
                bm25_results = [{"content": d.page_content, "metadata": d.metadata} for d in docs_bm25]
                
                # 3. Hybrid Fusion
                # Combine all vector results and BM25 results for RRF
                return self.reciprocal_rank_fusion([all_vector_results, bm25_results])[:k]

            # If no BM25 retriever, just return the vector results (after RRF if multiple sources)
            return self.reciprocal_rank_fusion([all_vector_results])[:k]
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
            raise e