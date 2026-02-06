"""Service for searching file summaries using similarity search."""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from app.config import get_settings
from app.database import File, get_db_service

logger = logging.getLogger(__name__)


class FileSummarySearchService:
    """Service for searching and retrieving file summaries from Milvus."""

    SUMMARY_COLLECTION = "file_summaries"  # Separate collection for summaries
    MAX_SUMMARY_CONTEXT = 10  # Max summaries to pass as context
    TOP_K_RELEVANT = 5  # Top K relevant files when more than MAX

    def __init__(self):
        """Initialize file summary search service."""
        self.settings = get_settings()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embed_model_id
        )
        self.db_service = get_db_service()
        self._vectorstore: Optional[Milvus] = None

    def _get_connection_args(self) -> dict:
        """Get Milvus connection arguments."""
        return {
            "host": self.settings.milvus_host,
            "port": self.settings.milvus_port,
        }

    def _init_vectorstore(self) -> Optional[Milvus]:
        """Initialize or get vectorstore for summaries."""
        if self._vectorstore is None:
            try:
                from pymilvus import connections, utility

                # Check if collection exists
                connections.connect(
                    alias="summary_check",
                    host=self.settings.milvus_host,
                    port=self.settings.milvus_port,
                )
                exists = utility.has_collection(self.SUMMARY_COLLECTION, using="summary_check")
                connections.disconnect("summary_check")

                if exists:
                    self._vectorstore = Milvus(
                        embedding_function=self.embeddings,
                        collection_name=self.SUMMARY_COLLECTION,
                        connection_args=self._get_connection_args(),
                        auto_id=True,
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize summary vectorstore: {e}")
        return self._vectorstore

    def add_summary(self, file_id: str, conversation_id: str, summary: str) -> bool:
        """Add a file summary to the search index.

        Args:
            file_id: File ID
            conversation_id: Conversation ID
            summary: Summary text

        Returns:
            True if successful, False otherwise
        """
        if not summary or not summary.strip():
            logger.warning(f"Empty summary for file {file_id}")
            return False

        try:
            # Create document with metadata
            doc = Document(
                page_content=summary,
                metadata={
                    "file_id": file_id,
                    "conversation_id": conversation_id,
                    "type": "summary",
                },
            )

            # Initialize vectorstore if needed
            if self._vectorstore is None:
                self._vectorstore = Milvus.from_documents(
                    documents=[doc],
                    embedding=self.embeddings,
                    collection_name=self.SUMMARY_COLLECTION,
                    connection_args=self._get_connection_args(),
                    auto_id=True,
                    drop_old=False,
                )
            else:
                self._vectorstore.add_documents([doc])

            logger.info(f"Added summary for file {file_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding summary to search index: {e}")
            return False

    def search_summaries(
        self, query: str, conversation_id: str, top_k: Optional[int] = None
    ) -> list[tuple[str, str, float]]:
        """Search for relevant file summaries in a conversation.

        Args:
            query: Search query
            conversation_id: Conversation ID to scope search
            top_k: Number of results to return (defaults to TOP_K_RELEVANT)

        Returns:
            List of tuples: (file_id, summary, score)
        """
        if top_k is None:
            top_k = self.TOP_K_RELEVANT

        try:
            vectorstore = self._init_vectorstore()
            if vectorstore is None:
                logger.warning("Summary vectorstore not available")
                return []

            # Search for similar summaries
            results = vectorstore.similarity_search_with_score(
                query=query,
                k=top_k * 2,  # Get extra to filter by conversation
            )

            # Filter by conversation_id and return top_k
            conversation_results = []
            for doc, score in results:
                if doc.metadata.get("conversation_id") == conversation_id:
                    file_id = doc.metadata.get("file_id", "")
                    summary = doc.page_content
                    conversation_results.append((file_id, summary, score))

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
