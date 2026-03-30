from src.core.config import *
from src.core.logger import logger
from langchain_ollama import ChatOllama
from src.components.embeddings import user_query_embedding
from langchain.messages import HumanMessage, SystemMessage

class QueryProcessor:

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatOllama(
            model=MINI_AGENT_MODELS,
            base_url=OLLAMA_BASE_URL
        )

    def query_classifier_agent(self, user_query):
        try:

            prompt = [
                SystemMessage(content=CLASSIFIER_AGENT_PROMPT),
                HumanMessage(content=user_query)
            ]

            response = self.model.invoke(prompt)

            answer = str(response.content).lower().strip()

            logger.info(f"Query Classifier Agent Response: {answer}")

            return answer.startswith("yes")

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return False


    def intent_classifier_agent(self, query):
        try:
            prompt = [
                SystemMessage(content=INTENT_AGENT_PROMPT),
                HumanMessage(content=query)
            ]

            response = self.model.invoke(prompt)

            response = response.content.strip().lower()

            logger.info(f"Intent Classifier Agent Response: {response}")

            return response

        except Exception as e:
            logger.error(f"Planner Agent error: {str(e)}", exc_info=True)
        return "legal_question"


    def retrieval_agent(self, intent : str, query : str, top_k : int = 50):
        try:

            search_query = f"IPC law sections about {intent}. Case: {query}"

            embedded_query = user_query_embedding(search_query)
            
            if not embedded_query:
                logger.warning("Failed to generate embedding for search query")
                return []
            
            docs = self.vector_store.retrieve_documents(embedded_query, k=top_k)

            return [doc['text'] for doc in docs]

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return []


    def law_reasoning_agent(self, query, context):
        try:

            context_text = "\n\n".join(context) if isinstance(context, list) else context

            prompt = [
                SystemMessage(content=LAW_AGENT_PROMPT),
                HumanMessage(content=f"""
                Use the provided legal context to answer the user's query.

                Legal Context:
                {context_text}

                User Query:
                {query}

                Respond in this format:

                Applicable IPC Section(s):
                Explanation:
                Punishment:
                Legal Consequences:
                Related Sections:
                """)
            ]

            for chunk in self.model.stream(prompt):
                yield chunk.content

        except Exception as e:
            logger.error(f"Law Interpretation Agent error: {str(e)}", exc_info=True)
            yield "Error generating response."


    def run(self, query):
        try:

            is_law_related = self.query_classifier_agent(query)

            if not is_law_related:
                yield "Query is not related to LAW."
                return None
            
            logger.info("User's query is LAW RELATED.")

            intent = self.intent_classifier_agent(query)

            context = self.retrieval_agent(intent, query)

            if not context:
                yield "No relevant IPC sections were found in the knowledge base."
                return

            for token in self.law_reasoning_agent(query, context):
                yield token

        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}", exc_info=True)
            yield "An error occurred while processing the legal query."