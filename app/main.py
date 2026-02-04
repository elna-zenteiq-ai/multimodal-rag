"""FastAPI application with multimodal RAG endpoints."""

import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    DocumentInfo,
    DocumentUploadResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.docling_service import get_docling_service
from app.services.milvus_service import get_milvus_service
from app.services.minio_service import get_minio_service
from app.services.rag_service import get_rag_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multimodal RAG API",
    description="RAG system with Docling, Milvus, MinIO and NVIDIA GPT-OSS-120B",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check health status of all services."""
    try:
        minio_service = get_minio_service()
        minio_status = "connected" if minio_service.check_health() else "disconnected"
    except Exception as e:
        logger.warning(f"MinIO health check failed: {e}")
        minio_status = "error"

    try:
        milvus_service = get_milvus_service()
        milvus_status = "connected" if milvus_service.check_health() else "disconnected"
    except Exception as e:
        logger.warning(f"Milvus health check failed: {e}")
        milvus_status = "error"

    overall_status = "healthy" if minio_status == "connected" and milvus_status == "connected" else "degraded"

    return HealthResponse(
        status=overall_status,
        milvus=milvus_status,
        minio=minio_status,
    )


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    tags=["Documents"],
)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Upload a document, store in MinIO, process with Docling, and index in Milvus.

    Supported formats: PDF, DOCX, PPTX, XLSX, HTML, Markdown, Images
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    logger.info(f"Uploading document: {file.filename}")

    try:
        # Read file content
        file_content = await file.read()

        # Upload to MinIO
        minio_service = get_minio_service()
        doc_id, minio_url = minio_service.upload_document(file_content, file.filename)
        logger.info(f"Document uploaded to MinIO: {doc_id}")

        # Process with Docling
        docling_service = get_docling_service()
        documents = docling_service.process_document(
            file_content=file_content,
            filename=file.filename,
            minio_url=minio_url,
        )
        logger.info(f"Document processed into {len(documents)} chunks")

        # Index in Milvus
        milvus_service = get_milvus_service()
        milvus_service.add_documents(documents)
        logger.info(f"Document indexed in Milvus")

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=file.filename,
            minio_url=minio_url,
            chunks_count=len(documents),
        )

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
async def list_documents() -> list[DocumentInfo]:
    """List all uploaded documents."""
    try:
        minio_service = get_minio_service()
        docs = minio_service.list_documents()
        return [DocumentInfo(**doc) for doc in docs]
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest) -> QueryResponse:
    """Query the RAG system and get an answer with source citations.

    The response includes:
    - answer: The generated answer from NVIDIA GPT-OSS-120B
    - sources: List of source documents with text, filename, page numbers, and headings
    """
    logger.info(f"RAG query: {request.query[:100]}...")

    try:
        rag_service = get_rag_service()
        response = await rag_service.aquery(
            question=request.query,
            top_k=request.top_k,
        )
        return response

    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Multimodal RAG API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
