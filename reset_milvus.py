import logging
from app.services.milvus_service import get_milvus_service
from pymilvus import utility, connections

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_collection():
    service = get_milvus_service()
    settings = service.settings
    
    logger.info(f"Connecting to Milvus at {settings.milvus_host}:{settings.milvus_port}")
    connections.connect(
        alias="reset",
        host=settings.milvus_host,
        port=settings.milvus_port,
    )
    
    collection_name = settings.milvus_collection
    if utility.has_collection(collection_name, using="reset"):
        logger.info(f"Dropping collection: {collection_name}")
        utility.drop_collection(collection_name, using="reset")
        logger.info("Collection dropped successfully.")
    else:
        logger.info(f"Collection {collection_name} does not exist.")

if __name__ == "__main__":
    reset_collection()
