import os
import asyncio
import time
import streamlit as st
from typing import Optional
from dotenv import load_dotenv
from src.core.config import *
from src.core.logger import logger
from src.core.utils import check_ollama_availability
from src.components.vector_store import VectorStore
from src.pipelines.query_pipeline import QueryProcessor


@st.cache_resource
def initialize_vector_store() -> VectorStore:
    """Initializes and caches the ChromaDB container safely."""
    try:
        vc = VectorStore()
        logger.info("Vector store resource loaded.")
        return vc
    except Exception as e:
        logger.error(f"VectorStore initialization failure: {str(e)}", exc_info=True)
        st.error(f"❌ **Database Offline:** Completely unable to access Chroma DB storage.")
        st.stop()


@st.cache_resource
def initialize_query_processor(_vector_store: VectorStore) -> QueryProcessor:
    """Ensures agent graphs and pipeline components instantiate precisely once."""
    try:
        processor = QueryProcessor(_vector_store)
        logger.info("Query pipeline engines operational.")
        return processor
    except Exception as e:
        logger.error(f"QueryProcessor compilation failure: {str(e)}", exc_info=True)
        st.error(f"❌ **Pipeline Failure:** Could not assemble multi-agent routing steps.")
        raise e


def process_query(query_processor: QueryProcessor, user_query: str):
    """Simple wrapper to run the RAG pipeline."""
    try:
        logger.info(f"Incoming Request: {user_query}")
        start_time = time.time()
        with st.status("⚖️ Nyayora is analyzing your query...", expanded=True) as status:
            status.write("🧠 Activating Legal Classifier Agents...")
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            status.write("📚 Searching Multi-Statute Knowledge Base...")
            result = loop.run_until_complete(query_processor.run(user_query))
            
            status.update(label="✅ Analysis Complete", state="complete", expanded=False)
            logger.info(f"Request fulfilled in {time.time() - start_time:.2f} seconds.")
            return result, None
    except Exception as e:
        logger.error(f"Execution sequence halted unexpectedly: {str(e)}", exc_info=True)
        return None, str(e)


# ================================ MAIN VIEW APPLICATION ENTRYPOINT ================================
def main() -> None:
    st.set_page_config(
        page_title="Nyayora | Legal AI",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_dotenv()
    
    # Initialize resources early so the sidebar can access database status
    vc = initialize_vector_store()
    query_processor = initialize_query_processor(vc)
    
    # --- Sidebar Configuration ---
    with st.sidebar:
        st.title("⚖️ Nyayora Dashboard")
        st.markdown("---")
        
        is_ok, baseline_message = check_ollama_availability()
        if is_ok:
            st.success(f"**Ollama Status:** Online\n\n**Primary Model:** `{LLM_MODEL}`")
        else:
            st.error(f"**Ollama Status:** Offline\n\n{baseline_message}")
            st.stop()
        
        # Dynamically list loaded collections from the multi-DB VectorStore
        active_statutes = list(vc.vector_dbs.keys())
        if active_statutes:
            st.info(f"📚 **Knowledge Base ({len(active_statutes)}):**\n" + 
                    "\n".join([f"- {s.upper()}" for s in active_statutes]))
        else:
            st.warning("⚠️ No local databases detected. Please run the build pipeline.")

        # Display Workflow Status (Hybrid Search + Reranking)
        st.markdown("---")
        st.caption(f"🔍 **Retrieval:** Hybrid (Semantic + {'BM25' if vc.bm25_retriever else 'Vector Only'})")
        st.caption("🎯 **Ranking:** RRF + FlashRank Reranker (Top-8)")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    st.title("⚖️ Nyayora")
    st.markdown("##### *Intelligent Legal Assistant for the New Indian Criminal Law Era*")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about Indian Law..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response, error = process_query(query_processor, prompt)
            if error or response is None:
                st.error("Something went wrong. Please try again.")
            else:
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()