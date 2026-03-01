from src.core.config import *
from src.core.logger import logger
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings


def generate_embeddings(chunks):
    embeddings = []
    try:

        embedding_model = OllamaEmbeddings(
            model=VECTOR_EMBEDDINGS_MODEL
        )

        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_model.embed_documents(texts)

        return embeddings
    except Exception as e:
        logger.error(f"Error while generating embeddings: {str(e)}", exc_info = True)
    return embeddings