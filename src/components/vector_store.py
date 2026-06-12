import os
import asyncio
import pickle
from typing import List, Dict
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

        self.bm25_retrievers: Dict[str, BM25Retriever] = {}

        os.makedirs(self.base_persist_directory, exist_ok=True)
        self.load_existing_collections()
        self.load_bm25_retriever()


    def initialize_bm25_retriever(self, chunks: List[Document], statute_name: str) -> None:
        """
        Creates and stores a `BM25 retriever` for a specific statute collection.

        Args:
            chunks (List[Document]):
                List of document chunks belonging to a statute.

            statute_name (str):
                Name of the statute used as the retriever key
                (e.g., "ipc", "bns", "bnss", "bsa").

        Returns:
            None:
                Stores the generated BM25 retriever in the
                bm25_retrievers dictionary for later retrieval operations.
        """
        if chunks:
            self.bm25_retrievers[statute_name.lower()] = BM25Retriever.from_documents(chunks)
            logger.info(f"BM25 index built specifically for collection: {statute_name.lower()}")


    def load_bm25_retriever(self) -> None:
        """
        Restores `BM25 retrievers` from persisted chunk data stored on disk.

        Args:
            None.

        Returns:
            None:
                Recreates statute-specific BM25 retrievers and registers
                them in memory for hybrid retrieval.
        """
        if os.path.exists(PREPROCESSED_DATASET_PATH):
            try:
                with open(PREPROCESSED_DATASET_PATH, "rb") as f:
                    statute_chunks_map = pickle.load(f)
                    
                if isinstance(statute_chunks_map, dict):
                    for statute, chunks in statute_chunks_map.items():
                        self.initialize_bm25_retriever(chunks, statute)
                logger.info("All isolated BM25 retrievers restored successfully.")
            except Exception as e:
                logger.error(f"Failed to load split BM25 retrievers: {str(e)}")


    def load_existing_collections(self) -> None:
        """
        Scans the `ChromaDB` persistence directory and restores all available
        vector database collections.

        Args:
            None.

        Returns:
            None:
                Loads each discovered ChromaDB collection into memory and
                registers it in the vector_dbs dictionary for future searches.
        """
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
        """
        Retrieves an existing `ChromaDB` collection or creates a new one
        if it does not already exist.

        Args:
            collection_name (str):
                Name of the ChromaDB collection to retrieve or create.

        Returns:
            Chroma:
                The requested ChromaDB collection instance, ready for
                document storage and similarity search operations.
        """
        if collection_name not in self.vector_dbs:
            db_path = os.path.join(self.base_persist_directory, collection_name)
            self.vector_dbs[collection_name] = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        return self.vector_dbs[collection_name]


    async def store_embeddings(self, chunks: List[Document], collection_name: str, batch_size: int = 100) -> bool:
        """
        Stores document chunks in a `ChromaDB` collection using batch processing.

        Args:
            chunks (List[Document]):
                List of document chunks to be embedded and stored.

            collection_name (str):
                Name of the ChromaDB collection where the chunks
                will be stored.

            batch_size (int, optional):
                Number of chunks to process in each batch.
                Defaults to 100.

        Returns:
            bool:
                True if all chunks are successfully stored in the
                collection, otherwise False when no chunks are provided.
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
                await asyncio.to_thread(vector_db_instance.add_documents, documents=batch)
                logger.info(f"Storage Progress for '{collection_name}': {min(i + batch_size, len(chunks))}/{len(chunks)} chunks processed.")

            logger.info(f"Successfully stored and persisted all chunks for collection '{collection_name}'.")
            return True
        except Exception as e:
            logger.error(f"Critical error during embedding storage for collection '{collection_name}': {str(e)}")
            raise e


    @staticmethod
    def reciprocal_rank_fusion(results_list: List[List[dict]], k: int = 60) -> List[dict]:
        """
        Merges multiple retrieval result lists into a single ranked list
        using the Reciprocal Rank Fusion (RRF) algorithm.

        Args:
            results_list (List[List[dict]]):
                Collection of ranked retrieval results where each inner list
                contains documents returned by a retrieval method.

            k (int, optional):
                Constant used in the RRF scoring formula to reduce the impact
                of rank position. Defaults to 60.

        Returns:
            List[dict]:
                A reranked list of unique documents ordered by their
                fused RRF scores from highest to lowest relevance.
        """
        try:
            fused_scores = {}
            content_map = {}

            for docs in results_list:
                for rank, doc in enumerate(docs):
                    content = doc["content"]
                    if content not in fused_scores:
                        fused_scores[content] = 0
                        content_map[content] = doc["metadata"]
                    fused_scores[content] += 1 / (rank + k)

            reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            
            return [
                {
                    "content": content,
                    "metadata": content_map[content]
                }
                for content, _ in reranked_results
            ]
        except Exception as e:
            logger.error(f"Failed during Reciprocal Rank Fusion (RRF): {str(e)}", str(e))
            raise e


    async def hybrid_search(self, query: str, query_vector: List[float], target_statutes: List[str], k: int = 10) -> List[dict]:
        """Runs parallel hybrid search locked strictly within the target_statutes."""
        try:
            all_vector_results = []
            bm25_results = []

            # ========== STEP-1: Structured Dense Filtered Search ==========
            vector_search_tasks = []
            for statute in target_statutes:
                vector_db_instance = self.vector_dbs.get(statute.lower())
                if vector_db_instance:
                    vector_search_tasks.append(
                        asyncio.to_thread(vector_db_instance.similarity_search_by_vector, query_vector, k)
                    )
            
            results_from_dbs = await asyncio.gather(*vector_search_tasks)
            for docs_vector in results_from_dbs:
                all_vector_results.extend([
                    {"content": doc.page_content, "metadata": doc.metadata} for doc in docs_vector
                ])

            # ========== STEP-2: Locked Sparse Search (Query ONLY the indices requested by the agent) ==========
            for statute in target_statutes:
                retriever = self.bm25_retrievers.get(statute.lower())
                if retriever:
                    # Invoke BM25 only for the safe, scoped statute pool
                    docs_bm25 = retriever.invoke(query)[:k]
                    bm25_results.extend([
                        {"content": d.page_content, "metadata": d.metadata} for d in docs_bm25
                    ])

            # ========== STEP-3: Hybrid RRF Fusion on pristine, non-polluted candidate sets ==========
            return self.reciprocal_rank_fusion([all_vector_results, bm25_results])[:k]
        except Exception as e:
            logger.error(f"Error executing scoped hybrid search: {str(e)}")
            return []