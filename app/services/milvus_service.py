"""Milvus service for vector storage and retrieval — uses pymilvus directly."""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

# 384-dim for sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM = 384

# Metadata fields stored alongside every chunk.
# Each one becomes a VARCHAR column in Milvus.
_META_FIELDS = [
    "filename",
    "minio_url",
    "source",
    "page_numbers",
    "heading",
    "headings",
    "file_id",
    "chunk_type",
    "image_minio_url",
]


class MilvusService:
    """Direct pymilvus service for vector storage and retrieval."""

    _ALIAS = "default"  # single persistent connection alias

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embed_model_id
        )
        self._collection: Optional[Collection] = None
        self._connect()

    # ------------------------------------------------------------------
    # Connection & schema helpers
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        """Establish a persistent Milvus connection."""
        try:
            connections.connect(
                alias=self._ALIAS,
                host=self.settings.milvus_host,
                port=self.settings.milvus_port,
            )
            logger.info(
                f"Connected to Milvus at "
                f"{self.settings.milvus_host}:{self.settings.milvus_port}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")

    def _ensure_collection(self) -> Collection:
        """Return the collection, creating it if it doesn't exist."""
        if self._collection is not None:
            return self._collection

        name = self.settings.milvus_collection

        if utility.has_collection(name, using=self._ALIAS):
            self._collection = Collection(name=name, using=self._ALIAS)
            self._collection.load()
            logger.info(f"Loaded existing collection '{name}'")
            return self._collection

        # Build schema — INT64 auto-id PK, vector, text, plus metadata VARCHARs
        fields = [
            FieldSchema(
                name="pk",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=EMBEDDING_DIM,
            ),
        ]
        for meta_name in _META_FIELDS:
            fields.append(
                FieldSchema(
                    name=meta_name,
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                    default_value="",
                )
            )

        schema = CollectionSchema(
            fields=fields,
            description="RAG document chunks with embeddings",
            auto_id=True,
        )

        self._collection = Collection(
            name=name, schema=schema, using=self._ALIAS
        )

        # Create IVF_FLAT index on the vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self._collection.create_index(
            field_name="vector", index_params=index_params
        )
        self._collection.load()
        logger.info(f"Created and loaded new collection '{name}'")
        return self._collection

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        return self.embeddings.embed_documents(texts)

    def _embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embeddings.embed_query(text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_documents(
        self, documents: list[Document], file_id: str | None = None
    ) -> list[str]:
        """Insert document chunks into Milvus.

        Args:
            documents: LangChain Document objects (page_content + metadata).
            file_id: Optional file ID injected into every chunk's metadata.

        Returns:
            List of stringified Milvus primary-key IDs.
        """
        if not documents:
            return []

        col = self._ensure_collection()

        # Inject file_id
        if file_id:
            for doc in documents:
                doc.metadata["file_id"] = file_id

        texts = [doc.page_content for doc in documents]
        vectors = self._embed(texts)

        # Build per-row insert data
        insert_data: list[dict] = []
        for doc, vec in zip(documents, vectors):
            row: dict = {
                "text": doc.page_content,
                "vector": vec,
            }
            for meta_name in _META_FIELDS:
                row[meta_name] = str(doc.metadata.get(meta_name, ""))
            insert_data.append(row)

        result = col.insert(insert_data)
        col.flush()

        # result.primary_keys contains the auto-generated INT64 PKs
        pks = [str(pk) for pk in result.primary_keys]
        logger.info(
            f"Inserted {len(pks)} chunks into Milvus "
            f"(sample PKs: {pks[:3]})"
        )
        return pks

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """Similarity search across all chunks.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            List of Document objects with metadata.
        """
        col = self._ensure_collection()
        query_vec = self._embed_query(query)

        output_fields = ["text"] + _META_FIELDS
        results = col.search(
            data=[query_vec],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=output_fields,
        )

        return self._hits_to_documents(results)

    def search_with_scores(
        self, query: str, top_k: int = 5
    ) -> list[tuple[Document, float]]:
        """Similarity search with distance scores.

        Args:
            query: Search query.
            top_k: Number of results.

        Returns:
            List of (Document, distance) tuples.
        """
        col = self._ensure_collection()
        query_vec = self._embed_query(query)

        output_fields = ["text"] + _META_FIELDS
        results = col.search(
            data=[query_vec],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=output_fields,
        )

        docs_with_scores: list[tuple[Document, float]] = []
        if results:
            for hit in results[0]:
                entity = hit.entity
                doc = Document(
                    page_content=entity.get("text", ""),
                    metadata={k: entity.get(k, "") for k in _META_FIELDS},
                )
                docs_with_scores.append((doc, hit.distance))
        return docs_with_scores

    def search_by_chunk_ids(
        self, query: str, chunk_ids: list[str], top_k: int = 5
    ) -> list[Document]:
        """Similarity search scoped to specific primary-key chunk IDs.

        Args:
            query: Search query.
            chunk_ids: Stringified INT64 Milvus PKs.
            top_k: Number of results.

        Returns:
            List of matching Document objects.
        """
        if not chunk_ids:
            return []

        col = self._ensure_collection()

        # Convert to ints (skip any legacy non-integer IDs like "init_*")
        int_ids = []
        for cid in chunk_ids:
            try:
                int_ids.append(int(cid))
            except (ValueError, TypeError):
                logger.warning(f"Skipping non-integer chunk_id: {cid}")
        if not int_ids:
            logger.warning("No valid integer chunk IDs to search")
            return []

        query_vec = self._embed_query(query)
        ids_csv = ", ".join(str(i) for i in int_ids)
        expr = f"pk in [{ids_csv}]"

        output_fields = ["text"] + _META_FIELDS
        results = col.search(
            data=[query_vec],
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 16}},
            limit=min(top_k, len(int_ids)),
            expr=expr,
            output_fields=output_fields,
        )

        docs = self._hits_to_documents(results)
        logger.info(
            f"search_by_chunk_ids: query='{query[:50]}', "
            f"chunk_ids_count={len(int_ids)}, results={len(docs)}"
        )
        return docs

    def check_health(self) -> bool:
        """Check if Milvus is reachable."""
        try:
            utility.list_collections(using=self._ALIAS)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _hits_to_documents(
        results, *, max_l2_distance: float = 1.0
    ) -> list[Document]:
        """Convert pymilvus search results to LangChain Documents.

        Args:
            results: Raw pymilvus search results.
            max_l2_distance: Drop hits whose L2 distance exceeds this
                threshold.  Lower = stricter.  1.0 works well for
                384-dim MiniLM embeddings — keeps genuinely relevant
                passages and drops cross-file noise.
        """
        docs: list[Document] = []
        if not results:
            return docs
        for hit in results[0]:
            if hit.distance > max_l2_distance:
                logger.debug(
                    f"Skipping hit pk={hit.id} with L2 distance "
                    f"{hit.distance:.3f} > {max_l2_distance}"
                )
                continue
            entity = hit.entity
            metadata = {k: entity.get(k, "") for k in _META_FIELDS}
            metadata["_distance"] = hit.distance
            docs.append(
                Document(
                    page_content=entity.get("text", ""),
                    metadata=metadata,
                )
            )
        return docs


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------
_milvus_service: MilvusService | None = None


def get_milvus_service() -> MilvusService:
    """Get or create the MilvusService singleton."""
    global _milvus_service
    if _milvus_service is None:
        _milvus_service = MilvusService()
    return _milvus_service
