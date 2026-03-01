from src.core.config import *
from dotenv import load_dotenv
from pymongo import MongoClient
from src.core.logger import logger

load_dotenv()

class VectorStore:
    def __init__(self, MONGO_URI):
        client = MongoClient(MONGO_URI)
        self.collection = client[MONGO_DATABASE][MONGO_COLLECTION]


    def index_documents(self, chunks):
        try:
            pass
        except Exception as e:
            logger.error(f"Error in indexing documents: {str(e)}", exc_info = True)


    def store_embeddings(self, chunks, embeddings):
        try:
            documents = []

            for chunk, embedding in zip(chunks, embeddings):
                documents.append(
                    {
                        "text": chunk.page_content,
                        "embedding": embedding
                    }
                )

            self.collection.insert_many(documents)
            return True
        except Exception as e:
            logger.error(f"Error while storing embeddings: {str(e)}", exc_info = True)
        return False