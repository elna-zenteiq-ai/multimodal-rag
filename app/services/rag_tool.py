"""RAG tool for LangChain agent - acts as a retrieval tool."""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.tools import tool

from app.services.milvus_service import get_milvus_service

logger = logging.getLogger(__name__)


@tool
def rag_search(
    query: str, file_ids: list[str], top_k: int = 5
) -> str:
    """Search through documents for relevant information.

    This tool searches the vector database for content related to the query,
    scoped to specific files if provided.

    Args:
        query: The search query/question
        file_ids: List of file IDs to search within (from tool runtime context)
        top_k: Number of results to return

    Returns:
        Formatted search results as a string
    """
    try:
        milvus = get_milvus_service()

        # Search in Milvus
        results = milvus.search(query=query, top_k=top_k)

        if not results:
            return "No relevant information found in the documents."

        # Format results with source information
        formatted_results = _format_search_results(results, file_ids)
        return formatted_results

    except Exception as e:
        logger.error(f"Error in RAG search: {e}")
        return f"Error searching documents: {str(e)}"


def _format_search_results(docs: list[Document], scoped_file_ids: Optional[list[str]] = None) -> str:
    """Format search results for presentation to LLM.

    Args:
        docs: List of retrieved documents
        scoped_file_ids: Optional list of file IDs to prioritize

    Returns:
        Formatted string with results
    """
    if not docs:
        return "No results found."

    # Filter by file_ids if provided
    if scoped_file_ids:
        docs = [d for d in docs if d.metadata.get("file_id") in scoped_file_ids]

    results_lines = ["## Search Results\n"]

    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        filename = metadata.get("filename", "Unknown")
        page_numbers = metadata.get("page_numbers", [])
        heading = metadata.get("heading", "")

        # Build source citation
        source_parts = [f"**Source {i}: {filename}**"]
        if page_numbers:
            pages_str = ", ".join(str(p) for p in page_numbers)
            source_parts.append(f"Pages: {pages_str}")
        if heading:
            source_parts.append(f"Section: {heading}")

        source_info = " | ".join(source_parts)
        results_lines.append(f"\n{source_info}\n")
        results_lines.append(f"{doc.page_content}\n")
        results_lines.append("---")

    return "\n".join(results_lines)


class RAGToolRuntime:
    """Runtime context manager for RAG tool with scoped file/chunk IDs."""

    def __init__(self, file_ids: list[str], chunk_ids: list[str] | None = None):
        """Initialize with file IDs and optional chunk IDs to scope searches.

        Args:
            file_ids: List of file IDs available in the context
            chunk_ids: List of Milvus chunk IDs for targeted search
        """
        self.file_ids = file_ids
        self.chunk_ids = chunk_ids

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """Run a scoped search and return raw Document objects.

        If chunk_ids are available, performs Milvus search scoped to those
        specific chunks.  Otherwise falls back to metadata-based filtering.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of Document objects (may be empty)
        """
        milvus = get_milvus_service()

        if self.chunk_ids:
            results = milvus.search_by_chunk_ids(
                query=query, chunk_ids=self.chunk_ids, top_k=top_k
            )
        else:
            results = milvus.search(query=query, top_k=top_k)
            if self.file_ids:
                results = [d for d in results if d.metadata.get("file_id") in self.file_ids]

        return results

    def execute(self, query: str, top_k: int = 5) -> str:
        """Execute RAG search with scoped file/chunk IDs.

        Convenience wrapper that calls :meth:`search` and formats the output.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Search results as formatted string
        """
        results = self.search(query=query, top_k=top_k)

        if not results:
            return "No relevant information found in the documents."

        return _format_search_results(results, self.file_ids)
