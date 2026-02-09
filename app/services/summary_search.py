"""Service for searching file summaries using similarity search — uses pymilvus directly."""

import logging
from typing import Optional

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
from app.database import File, get_db_service

logger = logging.getLogger(__name__)

# Same embedding dim as the main collection
_EMBEDDING_DIM = 384

# Metadata fields for the summary collection
_SUMMARY_META = ["file_id", "conversation_id", "type"]


class FileSummarySearchService:
    """Service for searching and retrieving file summaries from Milvus."""

    SUMMARY_COLLECTION = "file_summaries"
    MAX_SUMMARY_CONTEXT = 10
    TOP_K_RELEVANT = 5

    _ALIAS = "default"  # reuse the shared Milvus connection

    def __init__(self):
        self.settings = get_settings()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embed_model_id
        )
        self.db_service = get_db_service()
        self._collection: Optional[Collection] = None

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------
    def _ensure_collection(self) -> Collection:
        """Return the summary collection, creating it if needed."""
        if self._collection is not None:
            return self._collection

        # Make sure connection exists (MilvusService opens it on init,
        # but be defensive in case summary_search is used standalone).
        if not connections.has_connection(self._ALIAS):
            connections.connect(
                alias=self._ALIAS,
                host=self.settings.milvus_host,
                port=self.settings.milvus_port,
            )

        name = self.SUMMARY_COLLECTION

        if utility.has_collection(name, using=self._ALIAS):
            self._collection = Collection(name=name, using=self._ALIAS)
            self._collection.load()
            return self._collection

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=_EMBEDDING_DIM),
        ]
        for meta_name in _SUMMARY_META:
            fields.append(
                FieldSchema(name=meta_name, dtype=DataType.VARCHAR, max_length=65535, default_value="")
            )

        schema = CollectionSchema(fields=fields, description="File summaries", auto_id=True)
        self._collection = Collection(name=name, schema=schema, using=self._ALIAS)
        self._collection.create_index(
            field_name="vector",
            index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 64}},
        )
        self._collection.load()
        logger.info(f"Created summary collection '{name}'")
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_summary(self, file_id: str, conversation_id: str, summary: str) -> bool:
        """Add a file summary to the search index."""
        if not summary or not summary.strip():
            logger.warning(f"Empty summary for file {file_id}")
            return False

        try:
            col = self._ensure_collection()
            vec = self.embeddings.embed_documents([summary])[0]

            col.insert([{
                "text": summary,
                "vector": vec,
                "file_id": file_id,
                "conversation_id": conversation_id,
                "type": "summary",
            }])
            col.flush()
            logger.info(f"Added summary for file {file_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding summary to search index: {e}")
            return False

    def search_summaries(
        self, query: str, conversation_id: str, top_k: Optional[int] = None
    ) -> list[tuple[str, str, float]]:
        """Search for relevant file summaries in a conversation.

        Returns:
            List of tuples: (file_id, summary, distance)
        """
        if top_k is None:
            top_k = self.TOP_K_RELEVANT

        try:
            col = self._ensure_collection()
            query_vec = self.embeddings.embed_query(query)

            results = col.search(
                data=[query_vec],
                anns_field="vector",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k * 2,
                output_fields=["text"] + _SUMMARY_META,
            )

            conversation_results: list[tuple[str, str, float]] = []
            if results:
                for hit in results[0]:
                    entity = hit.entity
                    if entity.get("conversation_id") == conversation_id:
                        conversation_results.append(
                            (entity.get("file_id", ""), entity.get("text", ""), hit.distance)
                        )
                    if len(conversation_results) >= top_k:
                        break

            return conversation_results

        except Exception as e:
            logger.error(f"Error searching summaries: {e}")
            return []

    def get_conversation_summaries(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> list[tuple[str, str]]:
        """Get all summaries for a conversation.

        Args:
            conversation_id: Conversation ID
            limit: Maximum number of summaries (defaults to MAX_SUMMARY_CONTEXT)

        Returns:
            List of tuples: (file_id, summary)
        """
        if limit is None:
            limit = self.MAX_SUMMARY_CONTEXT

        try:
            # Get ready files from database
            files = self.db_service.get_conversation_files(conversation_id)

            # Filter to ready files with summaries
            summaries = []
            for file in files:
                if file.summary and file.summary.strip():
                    summaries.append((file.file_id, file.summary))

                if len(summaries) >= limit:
                    break

            logger.info(f"Retrieved {len(summaries)} summaries for conversation {conversation_id}")
            return summaries

        except Exception as e:
            logger.error(f"Error getting conversation summaries: {e}")
            return []

    async def analyze_files_for_suggestions(
        self, conversation_id: str, file_ids: list[str]
    ) -> str:
        """Generate suggestions for what can be done with uploaded files.

        Args:
            conversation_id: Conversation ID
            file_ids: List of uploaded file IDs

        Returns:
            Suggestions text
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            from app.config import get_settings

            # Get summaries for the files
            summaries = []
            db_files = self.db_service.get_conversation_files(conversation_id)
            for file_id in file_ids:
                for f in db_files:
                    if f.file_id == file_id and f.summary:
                        summaries.append(f"- {f.filename}: {f.summary}")
                        break

            if not summaries:
                return "Files are being processed. Please wait before making requests."

            summary_text = "\n".join(summaries)

            # Use LLM to generate suggestions
            settings = get_settings()
            llm = ChatNVIDIA(
                model=settings.nvidia_model,
                api_key=settings.nvidia_api_key,
                temperature=0.5,
                max_tokens=300,
            )

            prompt = ChatPromptTemplate.from_template(
                """Based on these document summaries, what analyses or queries would be most useful?

Summaries:
{summaries}

Provide 3-4 concrete suggestions for what the user could ask about or analyze:"""
            )

            chain = prompt | llm
            response = await chain.ainvoke({"summaries": summary_text})
            suggestions = response.content if hasattr(response, "content") else str(response)

            return suggestions

        except Exception as e:
            logger.error(f"Error generating file suggestions: {e}")
            return "Unable to generate suggestions at this time."


# Singleton instance
_summary_search_service: Optional[FileSummarySearchService] = None


def get_summary_search_service() -> FileSummarySearchService:
    """Get or create file summary search service instance."""
    global _summary_search_service
    if _summary_search_service is None:
        _summary_search_service = FileSummarySearchService()
    return _summary_search_service
