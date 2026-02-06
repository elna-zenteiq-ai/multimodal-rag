"""FastAPI application with agent-based multimodal RAG endpoints."""

import logging
import uuid
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.database import FileStatus, MessageIntentType, get_db_service
from app.models import (
    ConversationFilesResponse,
    ConversationInitResponse,
    DocumentInfo,
    DocumentUploadResponse,
    FileUploadInfo,
    HealthResponse,
    MessageResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.agent_rag_service import get_agent_rag_service
from app.services.docling_service import get_docling_service
from app.services.milvus_service import get_milvus_service
from app.services.minio_service import get_minio_service
from app.services.rag_service import get_rag_service
from app.services.summarization_service import get_summarization_service
from app.services.summary_search import get_summary_search_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multimodal RAG API",
    description="Agent-based RAG system with Docling, Milvus, MinIO and NVIDIA GPT-OSS-120B",
    version="0.2.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== BACKGROUND TASKS =====
async def process_file_async(
    file_id: str,
    conversation_id: str,
    file_content: bytes,
    filename: str,
) -> None:
    """Process uploaded file asynchronously (summarize and index).

    Args:
        file_id: File identifier
        conversation_id: Conversation identifier
        file_content: Raw file bytes
        filename: Original filename
    """
    db_service = get_db_service()
    try:
        # Process with Docling
        docling_service = get_docling_service()
        documents = docling_service.process_document(
            file_content=file_content,
            filename=filename,
            minio_url="",  # Will be set later
        )
        logger.info(f"Processed file into {len(documents)} chunks")

        # Index in Milvus
        milvus_service = get_milvus_service()
        milvus_service.add_documents(documents)

        # Generate summary
        summarization_service = get_summarization_service()
        # Combine first few chunks for summary
        combined_content = " ".join([doc.page_content for doc in documents[:3]])
        summary = await summarization_service.asummarize(combined_content)

        # Update file status to READY with summary
        if summary:
            db_service.update_file_status(file_id, FileStatus.READY, summary=summary)
            # Add summary to search index
            summary_search = get_summary_search_service()
            summary_search.add_summary(file_id, conversation_id, summary)
        else:
            db_service.update_file_status(file_id, FileStatus.READY)

        logger.info(f"File {file_id} processing complete")

    except Exception as e:
        logger.error(f"Error processing file {file_id}: {e}")
        db_service.update_file_status(file_id, FileStatus.FAILED, error_message=str(e))


# ===== HEALTH CHECK =====
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


# ===== CONVERSATION MANAGEMENT =====
@app.post("/conversations", response_model=ConversationInitResponse, tags=["Conversations"])
async def create_conversation() -> ConversationInitResponse:
    """Create a new conversation."""
    try:
        conversation_id = str(uuid.uuid4())
        db_service = get_db_service()
        conversation_data = db_service.create_conversation(conversation_id)

        return ConversationInitResponse(
            conversation_id=conversation_data["conversation_id"],
            created_at=conversation_data["created_at"],
        )

    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")


# ===== UNIFIED MESSAGE/FILE ENDPOINT =====
@app.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageResponse,
    tags=["Messages"],
)
async def send_message_with_files(
    conversation_id: str,
    message: Optional[str] = Form(None, description="Message text content"),
    files: Optional[list[UploadFile]] = File(None),
    background_tasks: BackgroundTasks = None,
) -> MessageResponse:
    """
    Unified endpoint for sending messages and/or uploading files.

    At least one of message or files must be provided.
    Max 10 files per request.

    Args:
        conversation_id: Conversation identifier (path parameter)
        message: Optional message text (form field)
        files: Optional list of files (multipart files)
        background_tasks: Background task manager

    Returns:
        MessageResponse with AI answer and processing info
    """
    # Validate input
    if not message and not files:
        raise HTTPException(
            status_code=400,
            detail="At least one of message or files must be provided",
        )

    if files and len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files per upload",
        )

    db_service = get_db_service()
    minio_service = get_minio_service()
    agent_rag = get_agent_rag_service()

    try:
        # Ensure conversation exists
        conversation = db_service.get_conversation(conversation_id)
        if not conversation:
            conversation = db_service.create_conversation(conversation_id)

        # Create message record if message provided
        message_id = str(uuid.uuid4())
        file_ids = []

        # Process files if provided
        if files:
            for uploaded_file in files:
                if not uploaded_file.filename:
                    continue

                # Generate file ID
                file_id = str(uuid.uuid4())
                file_content = await uploaded_file.read()

                # Upload to MinIO
                doc_id, minio_url = minio_service.upload_document(file_content, uploaded_file.filename)

                # Create file record in DB
                db_service.create_file(
                    file_id=file_id,
                    conversation_id=conversation_id,
                    minio_url=minio_url,
                    filename=uploaded_file.filename,
                    file_size=len(file_content),
                )

                file_ids.append(file_id)

                # Queue async processing
                if background_tasks:
                    background_tasks.add_task(
                        process_file_async,
                        file_id,
                        conversation_id,
                        file_content,
                        uploaded_file.filename,
                    )

                logger.info(f"Queued file {file_id} for processing")

        # If only files (no message), return file upload info
        if not message:
            # Get file info for response
            all_files = db_service.get_conversation_files(conversation_id)
            files_info = [
                FileUploadInfo(
                    file_id=f.file_id,
                    filename=f.filename,
                    status=f.status.value,
                )
                for f in all_files if f.file_id in file_ids
            ]

            # Generate suggestions for uploaded files
            suggestions = await agent_rag.analyze_and_suggest(conversation_id)

            return MessageResponse(
                message_id=message_id,
                conversation_id=conversation_id,
                answer=suggestions or "Files uploaded successfully. They are being processed.",
                intent=MessageIntentType.HIGH_LEVEL.value,
                used_rag=False,
                is_ambiguous=False,
                clarification_question=None,
                files_linked=file_ids,
            )

        # If message provided, process with agent
        # First, create message record
        from app.services.intent_classifier import get_intent_classifier
        intent_classifier = get_intent_classifier()
        classification = await intent_classifier.aclassify(message)

        message_obj = db_service.create_message(
            message_id=message_id,
            conversation_id=conversation_id,
            content=message,
            intent=classification["intent"],
            is_ambiguous=classification["is_ambiguous"],
            clarification_question=classification["clarification_question"],
        )

        # Link files to message if any
        if file_ids:
            db_service.link_message_files(message_id, file_ids, conversation_id)

        # Get all files in conversation
        all_files = db_service.get_conversation_files(conversation_id)
        ready_file_ids = [f.file_id for f in all_files if f.status == FileStatus.READY]

        # Check if any file is still PROCESSING
        processing_files = [f for f in all_files if f.status == FileStatus.PROCESSING]
        if processing_files and not ready_file_ids:
            return MessageResponse(
                message_id=message_id,
                conversation_id=conversation_id,
                answer="Files are still being processed. Please wait a moment and try again.",
                intent=MessageIntentType.HIGH_LEVEL.value,
                used_rag=False,
                is_ambiguous=False,
                clarification_question=None,
                files_linked=file_ids,
            )

        # Use agent to process query
        agent_response = await agent_rag.query_with_files(
            query=message,
            conversation_id=conversation_id,
            file_ids=ready_file_ids if ready_file_ids else None,
            chat_history=[],
        )

        return MessageResponse(
            message_id=message_id,
            conversation_id=conversation_id,
            answer=agent_response["answer"],
            intent=agent_response["intent"].value,
            used_rag=agent_response["used_rag"],
            is_ambiguous=classification["is_ambiguous"],
            clarification_question=classification["clarification_question"],
            files_linked=file_ids,
        )

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


@app.get(
    "/conversations/{conversation_id}/files",
    response_model=ConversationFilesResponse,
    tags=["Conversations"],
)
async def get_conversation_files(conversation_id: str) -> ConversationFilesResponse:
    """Get all files in a conversation with suggestions."""
    db_service = get_db_service()
    agent_rag = get_agent_rag_service()

    try:
        files = db_service.get_conversation_files(conversation_id)
        files_info = [
            FileUploadInfo(
                file_id=f.file_id,
                filename=f.filename,
                status=f.status.value,
            )
            for f in files
        ]

        ready_count = len([f for f in files if f.status == FileStatus.READY])

        # Generate suggestions if files are ready
        suggestions = None
        if ready_count > 0:
            suggestions = await agent_rag.analyze_and_suggest(conversation_id)

        return ConversationFilesResponse(
            conversation_id=conversation_id,
            files=files_info,
            ready_files_count=ready_count,
            suggestions=suggestions,
        )

    except Exception as e:
        logger.error(f"Error getting conversation files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get files: {str(e)}")


# ===== LEGACY ENDPOINTS (for backward compatibility) =====
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
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "conversations": {
                "create": "POST /conversations",
                "send_message": "POST /conversations/{conversation_id}/messages",
                "get_files": "GET /conversations/{conversation_id}/files",
            },
            "legacy": {
                "upload": "POST /documents/upload",
                "list": "GET /documents",
                "query": "POST /query",
            },
        },
    }

