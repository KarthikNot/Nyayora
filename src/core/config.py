import os

DATASET_DOWNLOAD_LINK = "https://www.iitk.ac.in/wc/data/IPC_186045.pdf"

DATASET_PATH = os.path.join(os.getcwd(), "data", "indian_penal_code.pdf") 

PREPROCESSED_DATASET_PATH = os.path.join(os.getcwd(), "artifacts", "chunked_dataset.pkl")

VECTOR_EMBEDDINGS_PATH = os.path.join(os.getcwd(), "artifacts", "vectore_store.pkl")


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

VECTOR_EMBEDDINGS_MODEL = "nomic-embed-text"
MONGO_DATABASE = "rag_data"
MONGO_COLLECTION = "rag_embeddings"