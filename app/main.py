"""FastAPI application with multimodal RAG endpoints."""

import logging
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi import Form, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

# Configure logging without interfering with uvicorn
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from app.models import (
    HealthResponse,
    MessageResponse,
    QueryRequest,
    QueryResponse,
    ConversationStatusResponse,
    FileStatusInfo,
)
from app.services.rag_service import get_rag_service
from app.services.docling_service import get_docling_service
from app.services.milvus_service import get_milvus_service
from app.services.minio_service import get_minio_service
from langchain_core.documents import Document
from app.services.db_service import (
    get_db_service,
    create_conversation,
    get_conversation,
    add_file_record,
    update_file_status,
    add_message_record,
    add_message_file,
    any_processing_files,
    get_files_by_conversation,
    FileStatus,
)

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

    try:
        db = get_db_service()
        session = db.get_session()
        session.execute(text("SELECT 1"))
        session.close()
        postgres_status = "connected"
    except Exception as e:
        logger.warning(f"PostgreSQL health check failed: {e}")
        postgres_status = "error"

    overall_status = "healthy" if minio_status == "connected" and milvus_status == "connected" and postgres_status == "connected" else "degraded"

    return HealthResponse(
        status=overall_status,
        milvus=milvus_status,
        minio=minio_status,
        postgres=postgres_status,
    )


@app.post("/upload", response_model=MessageResponse, tags=["Files"])
async def upload_files_only(
    files: list[UploadFile] = File(...),
    conversation_id: str = Header(...),
) -> MessageResponse:
    """Upload files to a conversation without sending a message.

    Headers:
    - conversation_id: required - the conversation to add files to
    
    Form fields:
    - files: required - list of files (max 10, max 10 MiB each)
    
    Behavior:
    - Maximum 10 files per request
    - Maximum 10 MiB per file
    - If any file in the conversation is still PROCESSING, blocks execution
    - Files are uploaded to MinIO, processed with Docling, indexed in Milvus
    """
    # Validate: max 10 files
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per request")

    # Get or create conversation
    db_service = get_db_service()
    session = db_service.get_session()
    conv = get_conversation(session, conversation_id)
    if not conv:
        # Create new conversation with provided ID
        conv = create_conversation(session, conversation_id=conversation_id)
        logger.info(f"Created new conversation: {conversation_id}")

    # Check if any file in this conversation is still PROCESSING
    if any_processing_files(session, conversation_id):
        raise HTTPException(
            status_code=409,
            detail="Cannot process files while other files in conversation are still PROCESSING",
        )

    # Process files
    uploaded_file_ids = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        # Check file size (max 10 MiB per file)
        file_content = await file.read()
        if len(file_content) > 10 * 1024 * 1024:  # 10 MiB
            raise HTTPException(
                status_code=413,
                detail=f"File {file.filename} exceeds 10 MiB limit",
            )

        logger.info(
            f"Processing file: {file.filename} for conversation: {conversation_id}"
        )

        # Upload to MinIO
        minio_service = get_minio_service()
        file_id, minio_url = minio_service.upload_document(
            file_content, file.filename
        )
        logger.info(f"File uploaded to MinIO: {file_id}")

        # Add file record to DB with PROCESSING status
        file_record = add_file_record(
            session,
            conversation_id=conversation_id,
            filename=file.filename,
            minio_url=minio_url,
        )
        db_file_id = file_record.id

        # Process with Docling (add file_id and conversation_id to metadata)
        docling_service = get_docling_service()
        documents = docling_service.process_document(
            file_content=file_content,
            filename=file.filename,
            minio_url=minio_url,
            file_id=db_file_id,
            conversation_id=conversation_id,
        )
        logger.info(f"File processed into {len(documents)} chunks")

        # Generate summary (first 1000 chars of concatenated chunks)
        summary = " ".join([doc.page_content for doc in documents])[:1000]

        # Add type field to all chunk documents
        for doc in documents:
            if "type" not in doc.metadata:
                doc.metadata["type"] = "chunk"

        # Store summary in Milvus
        milvus_service = get_milvus_service()
        summary_docs = [
            Document(
                page_content=summary,
                metadata={
                    "file_id": db_file_id,
                    "conversation_id": conversation_id,
                    "filename": file.filename,
                    "type": "summary",
                },
            )
        ]
        milvus_service.add_documents(summary_docs)
        logger.info(f"Summary indexed in Milvus for file: {db_file_id}")

        # Index chunks in Milvus
        milvus_service.add_documents(documents)
        logger.info(f"File chunks indexed in Milvus")

        # Update file record status to READY
        update_file_status(session, db_file_id, FileStatus.READY)
        logger.info(f"File status updated to READY: {db_file_id}")

        uploaded_file_ids.append(db_file_id)

    return MessageResponse(
        message_id=None,
        file_ids=uploaded_file_ids,
        conversation_id=conversation_id,
    )


@app.post("/send", response_model=MessageResponse, tags=["Messages"])
async def send_message_or_file(
    message: str | None = Form(None),
    files: list[UploadFile] | None = File(None),
    conversation_id: str = Header(...),
) -> MessageResponse:
    """Send a message and/or upload files to a conversation.

    At least one of `message` or `files` is required.
    
    Headers:
    - conversation_id: required - the conversation to add message/files to
    
    Form fields:
    - message: optional - text message content
    - files: optional - list of files (max 10, max 10 MiB each)
    
    Behavior:
    - Maximum 10 files per request
    - Maximum 10 MiB per file
    - If any file in the conversation is still PROCESSING, blocks execution
    - Files are uploaded to MinIO, processed with Docling, indexed in Milvus
    - File IDs are linked to the message via message_files table
    """
    # Validate: at least one of message or files is required
    if not message and not files:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'message' or 'files' is required",
        )

    # Validate: max 10 files
    if files and len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per request")

    # Get or create conversation
    db_service = get_db_service()
    session = db_service.get_session()
    conv = get_conversation(session, conversation_id)
    if not conv:
        # Create new conversation with provided ID
        conv = create_conversation(session, conversation_id=conversation_id)
        logger.info(f"Created new conversation: {conversation_id}")

    # Check if any file in this conversation is still PROCESSING
    if any_processing_files(session, conversation_id):
        raise HTTPException(
            status_code=409,
            detail="Cannot process message while files in conversation are still PROCESSING",
        )

    # Process files if provided
    uploaded_file_ids = []
    if files:
        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="Filename is required")

            # Check file size (max 10 MiB per file)
            file_content = await file.read()
            if len(file_content) > 10 * 1024 * 1024:  # 10 MiB
                raise HTTPException(
                    status_code=413,
                    detail=f"File {file.filename} exceeds 10 MiB limit",
                )

            logger.info(
                f"Processing file: {file.filename} for conversation: {conversation_id}"
            )

            # Upload to MinIO
            minio_service = get_minio_service()
            file_id, minio_url = minio_service.upload_document(
                file_content, file.filename
            )
            logger.info(f"File uploaded to MinIO: {file_id}")

            # Add file record to DB with PROCESSING status
            file_record = add_file_record(
                session,
                conversation_id=conversation_id,
                filename=file.filename,
                minio_url=minio_url,
            )
            db_file_id = file_record.id

            # Process with Docling (add file_id and conversation_id to metadata)
            docling_service = get_docling_service()
            documents = docling_service.process_document(
                file_content=file_content,
                filename=file.filename,
                minio_url=minio_url,
                file_id=db_file_id,
                conversation_id=conversation_id,
            )
            logger.info(f"File processed into {len(documents)} chunks")

            # Generate summary (first 1000 chars of concatenated chunks)
            summary = " ".join([doc.page_content for doc in documents])[:1000]

            # Add type field to all chunk documents
            for doc in documents:
                if "type" not in doc.metadata:
                    doc.metadata["type"] = "chunk"

            # Store summary in Milvus
            milvus_service = get_milvus_service()
            summary_docs = [
                Document(
                    page_content=summary,
                    metadata={
                        "file_id": db_file_id,
                        "conversation_id": conversation_id,
                        "filename": file.filename,
                        "type": "summary",
                    },
                )
            ]
            milvus_service.add_documents(summary_docs)
            logger.info(f"Summary indexed in Milvus for file: {db_file_id}")

            # Index chunks in Milvus
            milvus_service.add_documents(documents)
            logger.info(f"File chunks indexed in Milvus")

            # Update file record status to READY
            update_file_status(session, db_file_id, FileStatus.READY)
            logger.info(f"File status updated to READY: {db_file_id}")

            uploaded_file_ids.append(db_file_id)

    # Create message record and optionally run RAG query
    message_id = None
    answer = None
    sources = []
    suggestion = None
    
    if message:
        msg_record = add_message_record(
            session,
            conversation_id=conversation_id,
            content=message,
        )
        message_id = msg_record.id
        logger.info(f"Message created: {message_id}")

        # Link files to message via message_files table
        for file_id in uploaded_file_ids:
            add_message_file(session, message_id, file_id)
            logger.info(f"Linked message {message_id} to file {file_id}")
        
        # If files exist (either newly uploaded or existing in conversation), run RAG query
        conversation_files = get_files_by_conversation(session, conversation_id)
        if conversation_files or uploaded_file_ids:
            try:
                logger.info(f"Running RAG query for message: {message}")
                rag_service = get_rag_service()
                rag_response = await rag_service.aquery(
                    question=message,
                    top_k=5,
                    conversation_id=conversation_id,
                )
                answer = rag_response.answer
                sources = [s.model_dump() if hasattr(s, 'model_dump') else s for s in rag_response.sources]
                logger.info(f"RAG query completed, got answer and {len(sources)} sources")
            except Exception as e:
                logger.error(f"RAG query failed: {e}")
                answer = f"Error: {str(e)}"
        else:
            answer = "No files in conversation to search. Please upload files first."
    else:
        # Only files uploaded, suggest what to do
        if uploaded_file_ids:
            suggestion = f"Files uploaded successfully. You can now ask me questions about them!"
    
    session.close()

    return MessageResponse(
        message_id=message_id,
        file_ids=uploaded_file_ids,
        conversation_id=conversation_id,
        answer=answer,
        sources=sources,
        suggestion=suggestion,
    )


@app.get("/status", response_model=ConversationStatusResponse, tags=["Messages"])
async def get_conversation_status(
    conversation_id: str = Header(...),
) -> ConversationStatusResponse:
    """Get file statuses for a conversation.

    Headers:
    - conversation_id: required - the conversation to check

    Response:
    - conversation_id: the conversation ID
    - files: list of files with their status (PROCESSING, READY, FAILED)
    """
    db_service = get_db_service()
    session = db_service.get_session()
    
    files = get_files_by_conversation(session, conversation_id)
    session.close()
    
    file_statuses = [
        FileStatusInfo(
            file_id=f.id,
            filename=f.filename,
            status=f.status.value,
        )
        for f in files
    ]
    
    return ConversationStatusResponse(
        conversation_id=conversation_id,
        files=file_statuses,
    )
async def query_rag(
    request: QueryRequest,
    conversation_id: str = Header(...),
) -> QueryResponse:
    """Query the RAG system and get an answer with source citations.

    Headers:
    - conversation_id: required - the conversation to query

    Body:
    - query: the question to ask
    - top_k: optional - number of top documents to retrieve (default 5)
    
    Response:
    - answer: The generated answer from NVIDIA GPT-OSS-120B
    - sources: List of source documents with text, filename, page numbers, and headings
    """
    logger.info(f"RAG query: {request.query[:100]}...")

    try:
        rag_service = get_rag_service()
        response = await rag_service.aquery(
            question=request.query,
            top_k=request.top_k,
            conversation_id=conversation_id,
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
