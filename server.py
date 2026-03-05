import os
import streamlit as st
from src.core.config import *
from dotenv import load_dotenv
from src.core.logger import logger
from src.components.vector_store import VectorStore
from src.pipelines.query_pipeline import QueryProcessor


@st.cache_resource
def initialize_vector_store(mongo_uri):
    """Initialize and cache the vector store to avoid multiple connections."""
    if not mongo_uri:
        logger.error("MONGO_URI environment variable is not set")
        st.error("❌ Database configuration missing. Please set MONGO_URI environment variable.")
        st.stop()
    
    try:
        vc = VectorStore(MONGO_URI=mongo_uri)
        logger.info("Vector store initialized successfully")
        return vc
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}", exc_info=True)
        st.error(f"❌ Failed to connect to database: {str(e)}")
        st.stop()


def initialize_query_processor(vector_store):
    """Initialize and cache the query processor to avoid repeated instantiations."""
    try:
        processor = QueryProcessor(vector_store)
        logger.info("Query processor initialized successfully")
        return processor
    except Exception as e:
        logger.error(f"Failed to initialize query processor: {str(e)}", exc_info=True)
        st.error(f"❌ Failed to initialize query processor: {str(e)}")
        st.stop()


def validate_query(query):
    """Validate user input query."""
    if not query:
        return False, "Please enter a query to search."
    
    if len(query.strip()) < 3:
        return False, "Query must be at least 3 characters long."
    
    return True, ""


def process_query(query_processor, user_query):
    """Process the user query through the QueryProcessor pipeline."""
    try:
        is_valid, error_msg = validate_query(user_query)
        if not is_valid:
            return None, error_msg
        
        logger.info(f"Processing query: {user_query[:50]}...")
        with st.spinner("🔍 Analyzing query..."):
            response = query_processor.run(user_query)
        
        return response, None
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return None, error_msg


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Nyayora - Legal AI Assistant",
        page_icon="⚖️",
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    st.title("⚖️ Nyayora - Legal AI Assistant")
    st.markdown("""
    Get instant legal guidance based on Indian Penal Code and Criminal Procedure Code.
    This AI assistant specializes in Indian criminal law and provides information-only responses.
    """)
    
    # Initialize resources
    mongo_uri = os.getenv("MONGO_URI")
    vc = initialize_vector_store(mongo_uri)
    query_processor = initialize_query_processor(vc)
    
    # Create columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_query = st.text_input(
            label="Enter your legal query:",
            placeholder="e.g., What is the punishment for theft under IPC?",
            help="Ask any question related to Indian criminal law"
        )
    
    with col2:
        submit_button = st.button("🔍 Search", use_container_width=True)
    
    if submit_button:
        if user_query:
            response_stream, error = process_query(query_processor, user_query)
            
            if error:
                st.warning(f"⚠️ {error}")
                logger.warning(f"Query processing issue: {error}")
            else:
                with st.spinner("⚖️ Generating legal analysis..."):
                    full_response = st.write_stream(response_stream)
                
                # Add disclaimer
                with st.expander("📋 Disclaimer"):
                    st.info(
                        "This is for informational purposes only and not legal advice. "
                        "Please consult with a qualified legal professional for specific legal guidance."
                    )
        else:
            st.info("ℹ️ Please enter a query to begin.")
    
    # Sidebar with additional information
    with st.sidebar:
        st.header("ℹ️ About Nyayora")
        st.markdown("""
        **Nyayora** is an AI-powered legal assistant that provides information on:
        - Indian Penal Code (IPC)
        - Criminal Procedure Code (CrPC)
        - Related Indian criminal laws
        
        **Features:**
        - Fast document retrieval
        - Accurate legal information
        - Evidence-based responses
        - Professional legal formatting
        """)
        
        st.divider()
        
        st.subheader("📝 Tips for Better Results:")
        st.markdown("""
        1. Be specific with your query
        2. Mention the subject matter (theft, assault, etc.)
        3. Ask about specific sections or punishments
        4. Use clear, legal terminology when possible
        """)
        
        st.divider()
        
        st.subheader("⚠️ Important Notice:")
        st.warning(
            "This tool provides general legal information only. "
            "It is not a substitute for professional legal advice. "
            "Always consult with a qualified lawyer."
        )


if __name__ == "__main__":
    load_dotenv()
    main()