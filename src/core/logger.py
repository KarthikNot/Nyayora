import logging

logging.basicConfig(
    level=logging.INFO,
    format=" ---> %(name)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)