"""Docling service for document processing."""

import os
import tempfile
from pathlib import Path

from docling.chunking import HybridChunker
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from app.config import get_settings


class DoclingService:
    """Service for processing documents using Docling."""

    def __init__(self):
        """Initialize Docling service."""
        self.settings = get_settings()
        # Suppress tokenizer parallelism warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def process_document(
        self,
        file_content: bytes,
        filename: str,
        file_id: str,
        minio_url: str | None = None,
    ) -> list[Document]:
        """Process a document and return chunks with metadata.

        Args:
            file_content: Raw file bytes
            filename: Original filename
            minio_url: Optional MinIO URL to include in metadata

        Returns:
            List of LangChain Document objects with rich metadata
        """
        # Write content to temp file for Docling to process
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(filename).suffix
        ) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name

        try:
            # Create Docling loader with HybridChunker
            loader = DoclingLoader(
                file_path=tmp_path,
                export_type=ExportType.DOC_CHUNKS,
                chunker=HybridChunker(tokenizer=self.settings.embed_model_id),
            )

            # Load and process document
            docs = loader.load()

            # Enrich metadata
            enriched_docs = []
            for doc in docs:
                # Build clean metadata for Milvus (only primitive types)
                metadata = {
                    "file_id": file_id,
                    "filename": filename,
                    "minio_url": minio_url or "",
                    "source": doc.metadata.get("source", ""),
                }

                # Extract page numbers from dl_meta if available
                page_numbers = self._extract_page_numbers(doc.metadata)
                if page_numbers:
                    metadata["page_numbers"] = ",".join(str(p) for p in page_numbers)

                # Extract headings
                headings = self._extract_headings(doc.metadata)
                metadata["heading"] = headings[0] if headings else ""
                metadata["headings"] = "|".join(headings) if headings else ""

                enriched_docs.append(
                    Document(
                        page_content=doc.page_content,
                        metadata=metadata,
                    )
                )

            return enriched_docs

        finally:
            # Cleanup temp file
            Path(tmp_path).unlink(missing_ok=True)

    def process_from_url(self, url: str, filename: str | None = None) -> list[Document]:
        """Process a document from URL.

        Args:
            url: URL to the document (can be MinIO presigned URL)
            filename: Optional filename to use in metadata

        Returns:
            List of LangChain Document objects
        """
        # Create Docling loader with URL
        loader = DoclingLoader(
            file_path=url,
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(tokenizer=self.settings.embed_model_id),
        )

        docs = loader.load()

        # Enrich metadata
        enriched_docs = []
        for doc in docs:
            # Build clean metadata for Milvus (only primitive types)
            metadata = {
                "filename": filename or "",
                "source_url": url,
                "source": doc.metadata.get("source", ""),
            }

            # Extract page numbers
            page_numbers = self._extract_page_numbers(doc.metadata)
            if page_numbers:
                metadata["page_numbers"] = ",".join(str(p) for p in page_numbers)

            # Extract headings
            headings = self._extract_headings(doc.metadata)
            metadata["heading"] = headings[0] if headings else ""
            metadata["headings"] = "|".join(headings) if headings else ""

            enriched_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )

        return enriched_docs

    @staticmethod
    def _extract_page_numbers(metadata: dict) -> list[int]:
        """Extract page numbers from Docling metadata."""
        page_numbers = []

        # Try to get from dl_meta structure
        dl_meta = metadata.get("dl_meta", {})
        if isinstance(dl_meta, dict):
            doc_items = dl_meta.get("doc_items", [])
            for item in doc_items:
                prov = item.get("prov", [])
                for p in prov:
                    page_no = p.get("page_no")
                    if page_no and page_no not in page_numbers:
                        page_numbers.append(page_no)

        return sorted(page_numbers)

    @staticmethod
    def _extract_headings(metadata: dict) -> list[str]:
        """Extract headings from Docling metadata."""
        # Try to get from dl_meta structure
        dl_meta = metadata.get("dl_meta", {})
        if isinstance(dl_meta, dict):
            return dl_meta.get("headings", [])

        return []


# Singleton instance
_docling_service: DoclingService | None = None


def get_docling_service() -> DoclingService:
    """Get or create Docling service instance."""
    global _docling_service
    if _docling_service is None:
        _docling_service = DoclingService()
    return _docling_service
