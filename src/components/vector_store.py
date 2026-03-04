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

            self.collection.delete_many({})

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


    def retrieve_documents(self, user_query_embedding, k = 20):
        try:
            

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": user_query_embedding,
                        "numCandidates": k * 10,
                        "limit": k
                    }
                },
                {
                    "$project": {
                        "text": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            result = list(self.collection.aggregate(pipeline=pipeline))
            return result
        except Exception as e:
            logger.error(f"Error while retrieving documents: {str(e)}", exc_info = True)
        return []