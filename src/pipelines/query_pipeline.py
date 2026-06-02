import asyncio
import re
import time
from typing import List, Optional
from flashrank import Ranker, RerankRequest
from src.components.vector_store import VectorStore
from src.core.agents import (
    classifier_llm,
    expansion_llm,
    intent_llm,
    reasoning_llm,
    intent_classifier_agent,
    law_reasoning_agent,
    multi_query_expansion,
    query_classifier_agent
)
from src.core.config import DOCS_TO_RETRIEVE, DOCS_TO_RERANK
from src.core.logger import logger
from src.core.schemas import IntentClassification, LegalResponseSchema
from src.core.utils import format_legal_response


class QueryProcessor:
    def __init__(self, vector_store: VectorStore, ranker: Optional[Ranker] = None) -> None:
        self.vector_store = vector_store
        
        self.classifier_llm = classifier_llm
        self.intent_llm = intent_llm
        self.expansion_llm = expansion_llm
        self.reasoning_llm = reasoning_llm

        # Initialize Re-ranker (FlashRank is highly efficient for CPU-based RAG)
        self.ranker = ranker or Ranker()


    def rerank_documents(self, query: str, documents: List[dict], top_n: int) -> List[dict]:
        """
        Refines retrieval results using FlashRank to ensure high precision context.
        This significantly reduces noise from standard vector similarity search.
        """
        if not documents:
            return []

        try:
            # Prepare data for FlashRank ingestion
            passages = [
                {
                    "id": i,
                    "text": doc["content"],
                    "meta": doc["metadata"]
                }
                for i, doc in enumerate(documents)
            ]

            rerank_request = RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(rerank_request)

            # Map back to internal document structure
            reranked_docs = [
                {
                    "content": res["text"],
                    "metadata": res["meta"]
                }
                for res in results[:top_n]
            ]
            return reranked_docs
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}", exc_info=True)
            return documents[:top_n] # Fallback to original order on error


    async def retrieval_agent(self, intent_obj: IntentClassification, query: str) -> List[dict]:
        try:
            logger.info(f"Starting retrieval agent for intent: {intent_obj.intent}")
            
            # Expand query
            all_queries = await multi_query_expansion(query)
            logger.info(f"Expanded query into {len(all_queries)} variations for hybrid search.")
            all_queries.append(f"Law sections about {intent_obj.intent}. Case: {query}")
            
            unique_contents = set()
            final_results = []
            statutes = intent_obj.target_statutes

            # Parallel fetch using native async retrieval
            tasks = [self.vector_store.retrieve_documents(q, k=DOCS_TO_RETRIEVE, statute_filter=statutes) for q in all_queries]
            search_results = await asyncio.gather(*tasks)
            for docs in search_results:
                for doc in docs:
                    if doc["content"] not in unique_contents:
                        unique_contents.add(doc["content"])
                        final_results.append(doc)
            
            logger.info(f"Retrieved {len(final_results)} unique chunks. Proceeding to reranking...")

            # Rerank to get high-quality context
            # We retrieve 30 but rerank to top 8 for the LLM context window
            reranked = self.rerank_documents(query, final_results, top_n=DOCS_TO_RERANK)
            logger.info(f"Successfully reranked to top {len(reranked)} relevant context chunks.")
            return reranked
        except Exception as e:
            logger.error(f"Retrieval agent error: {str(e)}", exc_info=True)
            raise e


    def verify_citations(self, response: LegalResponseSchema, context: List[dict]) -> LegalResponseSchema:
        """Cross-references the LLM generated sections against retrieved context to prevent hallucinations."""
        verified_sections = []
        retrieved_text = " ".join([doc["content"] for doc in context])
        
        logger.info(f"Verifying {len(response.sections)} generated citations against context...")
        
        # Use regex to ensure the section number is a whole word/number match
        # This prevents "42" matching "420"
        for section in response.sections:
            pattern = rf"\bSection\s+{re.escape(section.section_number)}\b"
            # Also check for just the number if 'Section' prefix is missing in some chunks
            if re.search(pattern, retrieved_text, re.IGNORECASE) or f" {section.section_number} " in f" {retrieved_text} ":
                verified_sections.append(section)
            else:
                logger.warning(f"Hallucination Alert: Section {section.section_number} not found in context. Filtering.")
        
        response.sections = verified_sections
        return response


    async def run(self, query: str) -> str:
        try:
            start_time = time.time()
            logger.info(f"--- Pipeline Execution Started for: '{query[:50]}...' ---")

            is_law_related = await query_classifier_agent(query)
            if not is_law_related:
                logger.info("Query classified as non-legal. Short-circuiting pipeline.")
                return "I apologize, but my expertise is strictly limited to Indian criminal law. Your query does not appear to be related to Indian legal matters or involves a non-Indian jurisdiction."
            
            intent_obj = await intent_classifier_agent(query)
            logger.info(f"Intent Classification: {intent_obj.intent} (Confidence: {intent_obj.confidence:.2f})")
            
            context = await self.retrieval_agent(intent_obj, query)

            if not context:
                return "No relevant legal sections were found in the current knowledge base."

            logger.info("Executing law reasoning agent...")
            legal_response = await law_reasoning_agent(query, context)
            if not legal_response:
                return "An error occurred while analyzing the legal context."
            
            # Verification Layer
            verified_response = self.verify_citations(legal_response, context)
            
            result = format_legal_response(verified_response)
            logger.info(f"--- Pipeline Completed in {time.time() - start_time:.2f}s ---")
            return result
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}", exc_info=True)
            raise e