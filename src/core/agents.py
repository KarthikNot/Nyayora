from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.messages import HumanMessage, SystemMessage
from src.core.logger import logger
from typing import Optional, List, cast
from src.core.config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    MINI_AGENT_MODELS,
    VECTOR_EMBEDDINGS_MODEL,
    CLASSIFIER_AGENT_PROMPT,
    INTENT_AGENT_PROMPT,
    PLANNER_AGENT_PROMPT,
    LAW_AGENT_PROMPT,
    MULTI_QUERY_PROMPT,
    REASONING_MODEL_TEMP,
    MINI_MODEL_TEMP
)
from src.core.schemas import QueryClassification, IntentClassification, QueryExpansion, LegalResponseSchema

mini_model = ChatOllama(
    model=MINI_AGENT_MODELS,
    base_url=OLLAMA_BASE_URL,
    format="json",
    temperature=MINI_MODEL_TEMP,
)

reasoning_model = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    format="json",
    temperature=REASONING_MODEL_TEMP,
)

embedding_model = OllamaEmbeddings(
    model=VECTOR_EMBEDDINGS_MODEL,
    base_url=OLLAMA_BASE_URL
)

classifier_llm = mini_model.with_structured_output(QueryClassification)
intent_llm = mini_model.with_structured_output(IntentClassification)
expansion_llm = mini_model.with_structured_output(QueryExpansion)
reasoning_llm = reasoning_model.with_structured_output(LegalResponseSchema)

async def query_classifier_agent(user_query: str) -> bool:
    try:
        prompt = [
            SystemMessage(content=CLASSIFIER_AGENT_PROMPT),
            HumanMessage(content=f"Classify this query: {user_query}")
        ]
        response = await classifier_llm.ainvoke(prompt)
        response = cast(QueryClassification, response)
        return response.is_legal
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise e


async def intent_classifier_agent(query: str) -> IntentClassification:
    try:
        prompt = [
            SystemMessage(content=INTENT_AGENT_PROMPT),
            HumanMessage(content=f"Classify this query: {query}")
        ]
        response = await intent_llm.ainvoke(prompt)
        response = cast(IntentClassification, response)
        return response
    except Exception as e:
        logger.error(f"Planner Agent error: {str(e)}", exc_info=True)
        raise e


async def planner_agent(query: str) -> List[str]:
    """Decomposes a complex query into simpler sub-queries."""
    try:
        prompt = [
            SystemMessage(content=PLANNER_AGENT_PROMPT),
            HumanMessage(content=f"Decompose this query: {query}")
        ]
        response = await expansion_llm.ainvoke(prompt)
        response = cast(QueryExpansion, response)
        logger.info(f"Planner Agent Prompts:\n{response.queries}")
        return response.queries
    except Exception as e:
        logger.error(f"Planner Agent failed: {str(e)}")
        return [query] # Fallback to original query


async def multi_query_expansion(query: str) -> List[str]:
    """Generates multiple versions of the query to improve retrieval."""
    try:
        prompt = MULTI_QUERY_PROMPT.format(question=query)
        response = await expansion_llm.ainvoke(prompt)
        response = cast(QueryExpansion, response)
        logger.info(f"Generated {len(response.queries)} sub-queries.")
        logger.info(f"Generated Queries:\n{response.queries}")
        return response.queries
    except Exception as e:
        logger.error(f"Multi-query expansion failed: {str(e)}")
        raise e


async def law_reasoning_agent(query: str, context: List[dict]) -> Optional[LegalResponseSchema]:
        try:
            context_text = "\n\n".join([doc["content"] for doc in context])

            prompt = [
                SystemMessage(content=LAW_AGENT_PROMPT),
                HumanMessage(content=f"Legal Context:\n{context_text}\n\nUser Query:\n{query}")
            ]
            
            response = await reasoning_llm.ainvoke(prompt)
            response = cast(LegalResponseSchema, response)
            return response
        except Exception as e:
            logger.error(f"Law Interpretation Agent error: {str(e)}", exc_info=True)
            raise e