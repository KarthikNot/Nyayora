import urllib.request
from src.core.config import *
from src.core.logger import logger
from langchain_community.document_loaders import PyPDFLoader


def load_dataset():
    try:

        data_dir = os.path.dirname(DATASET_PATH)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        urllib.request.urlretrieve(DATASET_DOWNLOAD_LINK, DATASET_PATH)
        if not os.path.exists(DATASET_PATH):
            logger.warning("Failed to download dataset")
            return None

        loader = PyPDFLoader(DATASET_PATH)
        ipc_pdf = loader.load()
        if not ipc_pdf:
            logger.warning("Failed to load document")
            return None
        
        return ipc_pdf
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}", exc_info=True)
    return []


