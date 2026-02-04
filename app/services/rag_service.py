"""RAG service for retrieval and generation with NVIDIA GPT-OSS-120B."""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.config import get_settings
from app.models import QueryResponse, SourceDocument
from app.services.milvus_service import get_milvus_service


class RAGService:
    """Service for RAG pipeline with NVIDIA GPT-OSS-120B."""

    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer questions. If you're unsure or the context
doesn't contain the relevant information, say so.

When referencing information, mention which source it comes from when relevant.

Context:
{context}
"""

    def __init__(self):
        """Initialize RAG service with NVIDIA LLM."""
        self.settings = get_settings()
        self.milvus = get_milvus_service()

        # Initialize NVIDIA ChatNVIDIA
        self.llm = ChatNVIDIA(
            model=self.settings.nvidia_model,
            api_key=self.settings.nvidia_api_key,
            temperature=self.settings.nvidia_temperature,
            top_p=self.settings.nvidia_top_p,
            max_tokens=self.settings.nvidia_max_tokens,
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

    def query(self, question: str, top_k: int = 5) -> QueryResponse:
        """Execute RAG query and return answer with sources.

        Args:
            question: User's question
            top_k: Number of sources to retrieve

        Returns:
            QueryResponse with answer and source documents
        """
        # Retrieve relevant documents
        docs = self.milvus.search(
            query=question,
            file_id=committed_file_id,
            top_k=top_k
        )

        # Format context with source info
        context = self._format_context(docs)

        # Generate answer
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question,
        })

        # Extract answer text
        answer = response.content if hasattr(response, "content") else str(response)

        # Format source documents
        sources = self._format_sources(docs)

        return QueryResponse(
            answer=answer,
            sources=sources,
        )

    async def aquery(self, question: str, top_k: int = 5) -> QueryResponse:
        """Async version of query.

        Args:
            question: User's question
            top_k: Number of sources to retrieve

        Returns:
            QueryResponse with answer and source documents
        """
        # Retrieve relevant documents
        docs = self.milvus.search(query=question, top_k=top_k)

        # Format context with source info
        context = self._format_context(docs)

        # Generate answer asynchronously
        chain = self.prompt | self.llm
        response = await chain.ainvoke({
            "context": context,
            "question": question,
        })

        # Extract answer text
        answer = response.content if hasattr(response, "content") else str(response)

        # Format source documents
        sources = self._format_sources(docs)

        return QueryResponse(
            answer=answer,
            sources=sources,
        )

    def _format_context(self, docs: list[Document]) -> str:
        """Format retrieved documents into context string with source info.

        Args:
            docs: List of retrieved documents

        Returns:
            Formatted context string
        """
        contexts = []

        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata

            # Build source citation
            source_parts = []
            filename = metadata.get("filename", "Unknown")
            source_parts.append(f"Source: {filename}")

            page_numbers_raw = metadata.get("page_numbers", [])
            if isinstance(page_numbers_raw, str):
                page_numbers = [int(p) for p in page_numbers_raw.split(",") if p.strip()]
            else:
                page_numbers = page_numbers_raw

            if page_numbers:
                pages_str = ", ".join(str(p) for p in page_numbers)
                source_parts.append(f"Page(s): {pages_str}")

            heading = metadata.get("heading")
            if heading:
                source_parts.append(f"Section: {heading}")

            # Combine text with source info
            source_info = " | ".join(source_parts)
            contexts.append(f"[{i}] {doc.page_content}\n--- {source_info} ---")

        return "\n\n".join(contexts)

    def _format_sources(self, docs: list[Document]) -> list[SourceDocument]:
        """Format retrieved documents into SourceDocument objects.

        Args:
            docs: List of retrieved documents

        Returns:
            List of SourceDocument objects
        """
        sources = []

        for doc in docs:
            metadata = doc.metadata

            # Parse page numbers from string if needed
            page_numbers_raw = metadata.get("page_numbers", [])
            if isinstance(page_numbers_raw, str):
                page_numbers = [int(p) for p in page_numbers_raw.split(",") if p.strip()]
            else:
                page_numbers = page_numbers_raw

            source = SourceDocument(
                text=doc.page_content,
                filename=metadata.get("filename", "Unknown"),
                page_numbers=page_numbers,
                heading=metadata.get("heading"),
                minio_url=metadata.get("minio_url"),
            )
            sources.append(source)

        return sources


# Singleton instance
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Get or create RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
