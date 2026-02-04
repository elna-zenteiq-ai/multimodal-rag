"""Milvus service for vector storage and retrieval."""

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections, utility

from app.config import get_settings


class MilvusService:
    """Service for interacting with Milvus vector database."""

    def __init__(self):
        """Initialize Milvus connection and embeddings."""
        self.settings = get_settings()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embed_model_id
        )
        self._vectorstore: Milvus | None = None

    def _get_connection_args(self) -> dict:
        """Get Milvus connection arguments."""
        return {
            "host": self.settings.milvus_host,
            "port": self.settings.milvus_port,
        }

    def _collection_exists(self) -> bool:
        """Check if the collection already exists."""
        try:
            connections.connect(
                alias="collection_check",
                host=self.settings.milvus_host,
                port=self.settings.milvus_port,
            )
            exists = utility.has_collection(
                self.settings.milvus_collection, using="collection_check"
            )
            connections.disconnect("collection_check")
            return exists
        except Exception:
            return False

    @property
    def vectorstore(self) -> Milvus:
        """Get or initialize vectorstore."""
        if self._vectorstore is None:
            # If collection exists, load it
            if self._collection_exists():
                self._vectorstore = Milvus(
                    embedding_function=self.embeddings,
                    collection_name=self.settings.milvus_collection,
                    connection_args=self._get_connection_args(),
                    auto_id=True,
                )
            else:
                # If it doesn't exist, we can't search yet, but we shouldn't return None for a property 
                # that claims to return Milvus. However, for search operations effectively there is no store.
                # We will handle the None case in search() instead.
                pass
        return self._vectorstore

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        # If collection doesn't exist or vectorstore not init, create it
        if self._vectorstore is None:
            self._vectorstore = Milvus.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.settings.milvus_collection,
                connection_args=self._get_connection_args(),
                auto_id=True,
                drop_old=False,
            )
            return [f"doc_{i}" for i in range(len(documents))]

        # Collection exists, use add_documents
        ids = self.vectorstore.add_documents(documents)
        return ids

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching Document objects with metadata
        """
        if self.vectorstore is None:
            return []

        results = self.vectorstore.similarity_search(
            query=query,
            k=top_k,
        )
        return results

    def search_with_scores(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        """Search for similar documents with relevance scores.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (Document, score) tuples
        """
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k,
        )
        return results

    def check_health(self) -> bool:
        """Check if Milvus is accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            connections.connect(
                alias="health_check",
                host=self.settings.milvus_host,
                port=self.settings.milvus_port,
            )
            utility.list_collections(using="health_check")
            connections.disconnect("health_check")
            return True
        except Exception:
            return False


# Singleton instance
_milvus_service: MilvusService | None = None


def get_milvus_service() -> MilvusService:
    """Get or create Milvus service instance."""
    global _milvus_service
    if _milvus_service is None:
        _milvus_service = MilvusService()
    return _milvus_service
