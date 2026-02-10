"""Docling service for document processing with image extraction."""

import io
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem
from langchain_core.documents import Document

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """An image extracted from a document by Docling."""

    image_bytes: bytes
    page_number: int
    image_index: int  # index within the page
    caption: str = ""  # text caption from the document, if any


@dataclass
class ProcessingResult:
    """Result of processing a document: text chunks + extracted images."""

    text_chunks: list[Document] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)


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
        minio_url: str | None = None,
    ) -> ProcessingResult:
        """Process a document and return text chunks + extracted images.

        Args:
            file_content: Raw file bytes
            filename: Original filename
            minio_url: Optional MinIO URL to include in metadata

        Returns:
            ProcessingResult with text_chunks and images
        """
        logger.info("Starting Docling processing for '%s'", filename)
        # Write content to temp file for Docling to process
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(filename).suffix
        ) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name

        try:
            # Configure PDF pipeline with image extraction enabled
            pdf_pipeline_options = PdfPipelineOptions(
                generate_picture_images=True,
                images_scale=2.0,
            )
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pdf_pipeline_options,
                    )
                }
            )

            # Convert document
            conv_result = converter.convert(tmp_path)
            doc_obj = conv_result.document

            # --- Text chunks via HybridChunker ---
            chunker = HybridChunker(tokenizer=self.settings.embed_model_id)
            chunks = list(chunker.chunk(doc_obj))

            enriched_docs = []
            for chunk in chunks:
                # Build metadata
                meta = chunk.meta
                page_numbers = []
                if hasattr(meta, "doc_items"):
                    for item in meta.doc_items:
                        if hasattr(item, "prov"):
                            for prov in item.prov:
                                pn = prov.page_no if hasattr(prov, "page_no") else None
                                if pn and pn not in page_numbers:
                                    page_numbers.append(pn)

                headings = meta.headings if hasattr(meta, "headings") else []

                metadata = {
                    "filename": filename,
                    "minio_url": minio_url or "",
                    "source": str(tmp_path),
                    "page_numbers": ",".join(str(p) for p in sorted(page_numbers)) if page_numbers else "",
                    "heading": headings[0] if headings else "",
                    "headings": "|".join(headings) if headings else "",
                    "chunk_type": "text",
                    "image_minio_url": "",
                }

                enriched_docs.append(
                    Document(
                        page_content=chunk.text,
                        metadata=metadata,
                    )
                )

            # --- Extract images from PictureItems ---
            extracted_images: list[ExtractedImage] = []
            page_img_counter: dict[int, int] = {}
            for element, _level in doc_obj.iterate_items():
                if isinstance(element, PictureItem):
                    pil_image = element.get_image(doc_obj)
                    if pil_image is None:
                        continue

                    # Determine page number
                    page_no = 0
                    if hasattr(element, "prov") and element.prov:
                        page_no = element.prov[0].page_no if hasattr(element.prov[0], "page_no") else 0

                    # Track per-page index
                    page_img_counter[page_no] = page_img_counter.get(page_no, 0) + 1
                    img_idx = page_img_counter[page_no]

                    # Extract caption text if available
                    caption = ""
                    if hasattr(element, "caption") and element.caption:
                        caption = str(element.caption)

                    # Convert PIL image to PNG bytes
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
                    img_bytes = buf.getvalue()

                    extracted_images.append(
                        ExtractedImage(
                            image_bytes=img_bytes,
                            page_number=page_no,
                            image_index=img_idx,
                            caption=caption,
                        )
                    )

            logger.info(
                "Processed '%s': %d text chunks, %d images extracted",
                filename,
                len(enriched_docs),
                len(extracted_images),
            )

            return ProcessingResult(
                text_chunks=enriched_docs,
                images=extracted_images,
            )

        finally:
            # Cleanup temp file
            Path(tmp_path).unlink(missing_ok=True)

    def process_standalone_image(
        self, file_content: bytes, filename: str, minio_url: str = ""
    ) -> ProcessingResult:
        """Process a standalone image upload (not embedded in a PDF).

        Returns a ProcessingResult with no text chunks and a single
        ExtractedImage.
        """
        logger.info("Processing standalone image '%s'", filename)
        return ProcessingResult(
            text_chunks=[],
            images=[
                ExtractedImage(
                    image_bytes=file_content,
                    page_number=1,
                    image_index=1,
                    caption="",
                )
            ],
        )

    @staticmethod
    def is_image_file(filename: str) -> bool:
        """Return True if the filename looks like a standalone image."""
        ext = Path(filename).suffix.lower()
        return ext in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}


# Singleton instance
_docling_service: DoclingService | None = None


def get_docling_service() -> DoclingService:
    """Get or create Docling service instance."""
    global _docling_service
    if _docling_service is None:
        _docling_service = DoclingService()
    return _docling_service
