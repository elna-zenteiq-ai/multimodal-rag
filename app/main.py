"""FastAPI application with multimodal RAG endpoints."""

import logging
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, Header
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


# ----------------------------
# Conversation Header Dependency
# ----------------------------
def get_conversation_id(
    x_conversation_id: str = Header(...)
) -> str:
    return x_conversation_id


# ----------------------------
# Configure logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------
# Create FastAPI app
# ----------------------------
app = FastAPI(
    title="Multimodal RAG API",
    description="RAG system with Docling, Milvus, MinIO and NVIDIA GPT-OSS-120B",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Health
# ----------------------------
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
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

    overall_status = (
        "healthy"
        if minio_status == "connected" and milvus_status == "connected"
        else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        milvus=milvus_status,
        minio=minio_status,
    )


# ----------------------------
# Upload Document
# ----------------------------
@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    tags=["Documents"],
)
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str = Depends(get_conversation_id),
) -> DocumentUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    logger.info(
        f"Uploading document for conversation_id={conversation_id}, filename={file.filename}"
    )

    try:
        # Read file content
        file_content = await file.read()

        # Upload to MinIO
        minio_service = get_minio_service()
        doc_id, minio_url = minio_service.upload_document(
            file_content, file.filename
        )

        logger.info(f"Document uploaded to MinIO: {doc_id}")

        # NOTE: DB persistence (files ↔ conversation) intentionally deferred (Phase-1)

        # Process with Docling
        docling_service = get_docling_service()
        documents = docling_service.process_document(
            file_content=file_content,
            filename=file.filename,
            file_id=doc_id,
            minio_url=minio_url,
        )

        logger.info(f"Document processed into {len(documents)} chunks")

        # Index in Milvus
        milvus_service = get_milvus_service()
        milvus_service.add_documents(documents)

        logger.info("Document indexed in Milvus")

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=file.filename,
            minio_url=minio_url,
            chunks_count=len(documents),
        )

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# List Documents
# ----------------------------
@app.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
async def list_documents() -> list[DocumentInfo]:
    try:
        minio_service = get_minio_service()
        docs = minio_service.list_documents()
        return [DocumentInfo(**doc) for doc in docs]
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Query RAG
# ----------------------------
@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(
    request: QueryRequest,
    conversation_id: str = Depends(get_conversation_id),
) -> QueryResponse:
    logger.info(
        f"RAG query for conversation_id={conversation_id}: {request.query[:100]}..."
    )

    try:
        # Step 8: message lifecycle (ID only)
        user_message_id = str(uuid.uuid4())
        logger.info(
            f"user_message_id={user_message_id} conversation_id={conversation_id}"
        )

        rag_service = get_rag_service()
        response = await rag_service.aquery(
            question=request.query,
            top_k=request.top_k,
        )

        assistant_message_id = str(uuid.uuid4())
        logger.info(
            f"assistant_message_id={assistant_message_id} conversation_id={conversation_id}"
        )

        return response

    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Root
# ----------------------------
@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "Multimodal RAG API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
