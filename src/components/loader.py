import os
import urllib.request
from src.core.config import IPC_DATASET_DOWNLOAD_LINK, BNS_DATASET_2023_LINK, BNSS_DATASET_2023_LINK, BSA_DATASET_2023_LINK, BASE_PATH
from src.core.logger import logger
from langchain_community.document_loaders import PyPDFLoader


def load_dataset() -> dict:
    try:
        data_dir = os.path.join(BASE_PATH, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        datasets = {
            "ipc.pdf": IPC_DATASET_DOWNLOAD_LINK,
            "bns.pdf": BNS_DATASET_2023_LINK,
            "bnss.pdf": BNSS_DATASET_2023_LINK,
            "bsa.pdf": BSA_DATASET_2023_LINK
        }

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/pdf",
        }
        
        all_datasets_docs = {}
        for filename, link in datasets.items():
            dest_path = os.path.join(data_dir, filename)
            if not os.path.exists(dest_path):
                logger.info(f"Downloading {filename}...")
                request = urllib.request.Request(link, headers=headers)
                with urllib.request.urlopen(request) as response:
                    with open(dest_path, "wb") as f:
                        f.write(response.read())
            
            if os.path.exists(dest_path):
                loader = PyPDFLoader(dest_path)
                # Store documents per dataset, using filename without extension as key
                all_datasets_docs[os.path.splitext(filename)[0]] = loader.load()

        if not all_datasets_docs:
            logger.warning("No documents loaded from any source.")
            return {}
        
        return all_datasets_docs
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}", exc_info=True)
        raise e
