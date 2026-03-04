import os
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from src.components.vector_store import VectorStore
from src.core.config import *
from src.core.logger import logger
from langchain.messages import SystemMessage, HumanMessage
from src.components.embeddings import user_query_embedding


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


@st.cache_resource
def initialize_chat_model():
    """Initialize and cache the chat model to avoid repeated instantiations."""
    try:
        model = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL
            temperature=0.0,
        )
        logger.info("Chat model initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize chat model: {str(e)}", exc_info=True)
        st.error(f"❌ Failed to initialize chat model: {str(e)}")
        st.stop()


def validate_query(query):
    """Validate user input query."""
    if not query:
        return False, "Please enter a query to search."
    
    if len(query.strip()) < 3:
        return False, "Query must be at least 3 characters long."
    
    return True, ""


def process_query(chat_model, vc, user_query):
    """Process the user query through the RAG pipeline."""
    try:
        # Validate input
        is_valid, error_msg = validate_query(user_query)
        if not is_valid:
            return None, error_msg
        
        # Generate embedding for the query
        logger.info(f"Processing query: {user_query[:50]}...")
        with st.spinner("🔍 Embedding query..."):
            embedded_query = user_query_embedding(user_query)
        
        if not embedded_query:
            error_msg = "Failed to generate query embedding. Please try again."
            logger.error(error_msg)
            return None, error_msg
        
        # Retrieve relevant documents
        with st.spinner("📚 Retrieving relevant documents..."):
            retrieved_docs = vc.retrieve_documents(embedded_query, k=50)
        
        if not retrieved_docs:
            logger.warning("No documents retrieved for query")
            return None, "No relevant documents found in the knowledge base."
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        
        if not context.strip():
            logger.warning("Empty context generated from retrieved documents")
            return None, "Retrieved documents are empty. Please try a different query."
        
        # Generate response using the chat model
        if SYSTEM_PROMPT:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"""
Retrieved Context:
{context}

User Question:
{user_query}
""")
            ]

            logger.info("Streaming response from chat model...")
            response_generator = chat_model.stream(messages)
            
            if response_generator:
                logger.info("Response stream initiated successfully")
                return response_generator, None
            else:
                error_msg = "Failed to initiate response stream."
                logger.error(error_msg)
                return None, error_msg
        else:
            error_msg = "System prompt is not configured."
            logger.error(error_msg)
            return None, error_msg
    
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg, exc_info=True)
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
    chat_model = initialize_chat_model()
    
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
    
    # Process query when submit button is clicked or enter is pressed
    if submit_button or user_query:
        if user_query:
            response_stream, error = process_query(chat_model, vc, user_query)
            
            if error:
                st.warning(f"⚠️ {error}")
                logger.warning(f"Query processing issue: {error}")
            else:
                st.subheader("Answer")
                
                # Stream the response in real-time
                with st.spinner("💭 Generating response..."):
                    response_text = ""
                    response_container = st.empty()
                    
                    for chunk in response_stream:
                        if chunk.content:
                            response_text += chunk.content
                            response_container.markdown(response_text)
                
                logger.info("Response streamed successfully")
                
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