"""FastAPI application with agent-based multimodal RAG endpoints."""

import logging
import uuid

from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document as LCDocument

from app.database import FileStatus, get_db_service
from app.models import (
    ChatRequest,
    ChatResponse,
    ConversationInitResponse,
    FileStatusResponse,
    FileUploadResponse,
    HealthResponse,
    SourceDocument,
)
from app.services.agent_rag_service import get_agent_rag_service
from app.services.docling_service import DoclingService, get_docling_service
from app.services.milvus_service import get_milvus_service
from app.services.minio_service import get_minio_service
from app.services.summarization_service import get_summarization_service
from app.services.summary_search import get_summary_search_service
from app.services.vision_service import get_vision_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multimodal RAG API",
    description="Agent-based RAG system with Docling, Milvus, MinIO and NVIDIA GPT-OSS-120B",
    version="0.3.0",
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
    minio_url: str = "",
) -> None:
    """Process uploaded file asynchronously (summarize and index).

    Steps:
        1. Process document with Docling into text chunks + extracted images
        2. For each extracted image: upload to MinIO, generate caption,
           create a Document chunk with chunk_type='image'
        3. Merge text + image chunks and index in Milvus
        4. Generate summary via LLM
        5. Update SQL table with status=READY, summary, chunk_ids

    Args:
        file_id: File identifier
        conversation_id: Conversation identifier
        file_content: Raw file bytes
        filename: Original filename
        minio_url: MinIO URL for the uploaded file
    """
    db_service = get_db_service()
    try:
        docling_service = get_docling_service()
        milvus_service = get_milvus_service()
        minio_service = get_minio_service()

        # 1. Process document (or standalone image)
        if DoclingService.is_image_file(filename):
            result = docling_service.process_standalone_image(
                file_content, filename, minio_url
            )
        else:
            result = docling_service.process_document(
                file_content=file_content,
                filename=filename,
                minio_url=minio_url,
            )

        documents = list(result.text_chunks)  # mutable copy
        logger.info(
            f"Processed file into {len(documents)} text chunks, "
            f"{len(result.images)} images"
        )

        # 2. Handle extracted images — caption & create image chunks
        if result.images:
            vision_service = get_vision_service()
            for img in result.images:
                # Upload image to MinIO
                img_filename = f"figure_p{img.page_number}_{img.image_index}.png"
                image_minio_url = minio_service.upload_image(
                    img.image_bytes, img_filename, file_id
                )

                # Generate caption via VLM
                caption = await vision_service.caption_image(
                    img.image_bytes, context=img.caption
                )

                # Create a Document chunk for the image caption
                documents.append(
                    LCDocument(
                        page_content=caption,
                        metadata={
                            "filename": filename,
                            "minio_url": minio_url,
                            "source": "",
                            "page_numbers": str(img.page_number) if img.page_number else "",
                            "heading": img.caption or f"Figure (page {img.page_number})",
                            "headings": "",
                            "chunk_type": "image",
                            "image_minio_url": image_minio_url,
                        },
                    )
                )
            logger.info(f"Created {len(result.images)} image chunks for file {file_id}")

        # 3. Index all chunks in Milvus
        chunk_ids = milvus_service.add_documents(documents, file_id=file_id)
        logger.info(f"Indexed {len(chunk_ids)} chunks in Milvus for file {file_id}")

        # 4. Generate summary (use text chunks only)
        summarization_service = get_summarization_service()
        text_only = [d for d in documents if d.metadata.get("chunk_type") != "image"]
        combined_content = " ".join([doc.page_content for doc in text_only[:3]])
        if not combined_content and documents:
            combined_content = documents[0].page_content
        summary = await summarization_service.asummarize(combined_content)

        # 5. Update SQL table: status, summary, and chunk_ids
        if summary:
            db_service.update_file_status(
                file_id, FileStatus.READY, summary=summary, chunk_ids=chunk_ids
            )
            # Add summary to search index
            summary_search = get_summary_search_service()
            summary_search.add_summary(file_id, conversation_id, summary)
        else:
            db_service.update_file_status(
                file_id, FileStatus.READY, chunk_ids=chunk_ids
            )

        logger.info(f"File {file_id} processing complete")

    except Exception as e:
        logger.error(f"Error processing file {file_id}: {e}", exc_info=True)
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


# # ===== UNIFIED MESSAGE/FILE ENDPOINT (commented out — using separate /upload + /chat) =====
# @app.post(
#     "/conversations/{conversation_id}/messages",
#     response_model=MessageResponse,
#     tags=["Messages"],
# )
# async def send_message_with_files(
#     conversation_id: str,
#     message: Optional[str] = Form(None, description="Message text content"),
#     files: Optional[list[UploadFile]] = File(None),
#     background_tasks: BackgroundTasks = None,
# ) -> MessageResponse:
#     """
#     Unified endpoint for sending messages and/or uploading files.
#
#     At least one of message or files must be provided.
#     Max 10 files per request.
#     """
#     # Validate input
#     if not message and not files:
#         raise HTTPException(
#             status_code=400,
#             detail="At least one of message or files must be provided",
#         )
#
#     if files and len(files) > 10:
#         raise HTTPException(
#             status_code=400,
#             detail="Maximum 10 files per upload",
#         )
#
#     db_service = get_db_service()
#     minio_service = get_minio_service()
#     agent_rag = get_agent_rag_service()
#
#     try:
#         conversation = db_service.get_conversation(conversation_id)
#         if not conversation:
#             conversation = db_service.create_conversation(conversation_id)
#
#         message_id = str(uuid.uuid4())
#         file_ids = []
#
#         if files:
#             for uploaded_file in files:
#                 if not uploaded_file.filename:
#                     continue
#                 file_id = str(uuid.uuid4())
#                 file_content = await uploaded_file.read()
#                 doc_id, minio_url = minio_service.upload_document(file_content, uploaded_file.filename)
#                 db_service.create_file(
#                     file_id=file_id,
#                     conversation_id=conversation_id,
#                     minio_url=minio_url,
#                     filename=uploaded_file.filename,
#                     file_size=len(file_content),
#                 )
#                 file_ids.append(file_id)
#                 if background_tasks:
#                     background_tasks.add_task(
#                         process_file_async, file_id, conversation_id,
#                         file_content, uploaded_file.filename,
#                     )
#                 logger.info(f"Queued file {file_id} for processing")
#
#         if not message:
#             all_files = db_service.get_conversation_files(conversation_id)
#             files_info = [
#                 FileUploadInfo(
#                     file_id=f.file_id, filename=f.filename, status=f.status.value,
#                 )
#                 for f in all_files if f.file_id in file_ids
#             ]
#             suggestions = await agent_rag.analyze_and_suggest(conversation_id)
#             return MessageResponse(
#                 message_id=message_id, conversation_id=conversation_id,
#                 answer=suggestions or "Files uploaded successfully. They are being processed.",
#                 intent=MessageIntentType.HIGH_LEVEL.value, used_rag=False,
#                 is_ambiguous=False, clarification_question=None, files_linked=file_ids,
#             )
#
#         from app.services.intent_classifier import get_intent_classifier
#         intent_classifier = get_intent_classifier()
#         classification = await intent_classifier.aclassify(message)
#         message_obj = db_service.create_message(
#             message_id=message_id, conversation_id=conversation_id,
#             content=message, intent=classification["intent"],
#             is_ambiguous=classification["is_ambiguous"],
#             clarification_question=classification["clarification_question"],
#         )
#         if file_ids:
#             db_service.link_message_files(message_id, file_ids, conversation_id)
#
#         all_files = db_service.get_conversation_files(conversation_id)
#         ready_file_ids = [f.file_id for f in all_files if f.status == FileStatus.READY]
#         processing_files = [f for f in all_files if f.status == FileStatus.PROCESSING]
#         if processing_files and not ready_file_ids:
#             return MessageResponse(
#                 message_id=message_id, conversation_id=conversation_id,
#                 answer="Files are still being processed. Please wait a moment and try again.",
#                 intent=MessageIntentType.HIGH_LEVEL.value, used_rag=False,
#                 is_ambiguous=False, clarification_question=None, files_linked=file_ids,
#             )
#
#         agent_response = await agent_rag.query_with_files(
#             query=message, conversation_id=conversation_id,
#             file_ids=ready_file_ids if ready_file_ids else None, chat_history=[],
#         )
#         return MessageResponse(
#             message_id=message_id, conversation_id=conversation_id,
#             answer=agent_response["answer"], intent=agent_response["intent"].value,
#             used_rag=agent_response["used_rag"],
#             is_ambiguous=classification["is_ambiguous"],
#             clarification_question=classification["clarification_question"],
#             files_linked=file_ids,
#         )
#
#     except Exception as e:
#         logger.error(f"Error processing message: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


# @app.get(
#     "/conversations/{conversation_id}/files",
#     response_model=ConversationFilesResponse,
#     tags=["Conversations"],
# )
# async def get_conversation_files(conversation_id: str) -> ConversationFilesResponse:
#     """Get all files in a conversation with suggestions."""
#     db_service = get_db_service()
#     agent_rag = get_agent_rag_service()
#
#     try:
#         files = db_service.get_conversation_files(conversation_id)
#         files_info = [
#             FileUploadInfo(
#                 file_id=f.file_id, filename=f.filename, status=f.status.value,
#             )
#             for f in files
#         ]
#         ready_count = len([f for f in files if f.status == FileStatus.READY])
#         suggestions = None
#         if ready_count > 0:
#             suggestions = await agent_rag.analyze_and_suggest(conversation_id)
#
#         return ConversationFilesResponse(
#             conversation_id=conversation_id, files=files_info,
#             ready_files_count=ready_count, suggestions=suggestions,
#         )
#
#     except Exception as e:
#         logger.error(f"Error getting conversation files: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to get files: {str(e)}")


# ===== DEDICATED UPLOAD / STATUS / CHAT ENDPOINTS =====
# Inspired by Open WebUI's pattern: files are uploaded independently,
# status is polled, and chat is a separate flow that references ready files.


@app.post("/upload", response_model=FileUploadResponse, tags=["Files"])
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    actor_id: str = Header(..., alias="x-actor-id", description="Actor/user ID from frontend"),
    conversation_id: str = Header(..., alias="x-conversation-id", description="Conversation ID from frontend"),
) -> FileUploadResponse:
    """Upload a single file.

    Accepts **actor_id** and **conversation_id** from request headers.
    Immediately stores the file in MinIO, creates a DB record, and returns
    the ``file_id``.  Processing (Docling + Milvus indexing + summarisation)
    runs as a background task.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    db_service = get_db_service()
    minio_service = get_minio_service()

    try:
        # Ensure conversation exists
        conversation = db_service.get_conversation(conversation_id)
        if not conversation:
            db_service.create_conversation(conversation_id)

        file_id = str(uuid.uuid4())
        file_content = await file.read()

        # 1. Upload to MinIO (first priority — fast, synchronous)
        _doc_id, minio_url = minio_service.upload_document(file_content, file.filename)

        # 2. Create file record in SQL
        db_service.create_file(
            file_id=file_id,
            conversation_id=conversation_id,
            minio_url=minio_url,
            filename=file.filename,
            file_size=len(file_content),
            actor_id=actor_id,
        )

        # 3. Queue async processing (Docling → Milvus → Summary)
        background_tasks.add_task(
            process_file_async,
            file_id,
            conversation_id,
            file_content,
            file.filename,
            minio_url,
        )
        logger.info(f"File {file_id} uploaded by actor {actor_id}, processing queued")

        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            status=FileStatus.PROCESSING.value,
            message="File uploaded. Processing started.",
        )

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@app.get("/files/{file_id}/status", response_model=FileStatusResponse, tags=["Files"])
async def get_file_status(file_id: str) -> FileStatusResponse:
    """Get the processing status of a file.

    Reads the SQL table row for the given ``file_id`` and returns:
    - **status** — PROCESSING / READY / FAILED
    - **summary** — available once processing is complete
    - **chunk_count** — number of Milvus chunk IDs stored
    - **error_message** — details if processing failed
    """
    db_service = get_db_service()
    file_record = db_service.get_file(file_id)

    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    # Parse chunk_ids to get count
    chunk_count = 0
    if file_record.chunk_ids:
        import json as _json
        try:
            ids = _json.loads(file_record.chunk_ids)
            chunk_count = len(ids) if isinstance(ids, list) else 0
        except Exception:
            pass

    return FileStatusResponse(
        file_id=file_record.file_id,
        filename=file_record.filename,
        status=file_record.status.value,
        summary=file_record.summary,
        chunk_count=chunk_count,
        error_message=file_record.error_message,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    actor_id: str = Header(..., alias="x-actor-id", description="Actor/user ID from frontend"),
    conversation_id: str = Header(..., alias="x-conversation-id", description="Conversation ID from frontend"),
) -> ChatResponse:
    """Chat endpoint — activated once files reach READY status.

    The agent uses short-term memory (PostgresSaver checkpointer keyed by
    ``conversation_id``) so it remembers all previous turns automatically.
    File summaries are injected once on the first query; the agent decides
    autonomously whether to answer from memory or call the rag_search tool.
    """
    db_service = get_db_service()
    agent_rag = get_agent_rag_service()

    try:
        # Verify conversation
        conversation = db_service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get all files for this conversation
        all_files = db_service.get_conversation_files(conversation_id)
        ready_files = [f for f in all_files if f.status == FileStatus.READY]
        processing_files = [f for f in all_files if f.status == FileStatus.PROCESSING]

        if not ready_files and processing_files:
            return ChatResponse(
                message_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                answer="Files are still being processed. Please wait a moment and try again.",
                used_rag=False,
                file_ids=[],
            )

        if not ready_files and not processing_files:
            return ChatResponse(
                message_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                answer="No files found for this conversation. Please upload files first.",
                used_rag=False,
                file_ids=[],
            )

        ready_file_ids = [f.file_id for f in ready_files]
        message_id = str(uuid.uuid4())

        # Resolve chunk IDs from SQL for the ready files
        chunk_ids = db_service.get_chunk_ids_for_files(ready_file_ids)

        # Run query through the agent (short-term memory handles history)
        agent_response = await agent_rag.query(
            query=request.query,
            conversation_id=conversation_id,
            file_ids=ready_file_ids,
            chunk_ids=chunk_ids,
        )

        # Persist message in SQL
        db_service.create_message(
            message_id=message_id,
            conversation_id=conversation_id,
            content=request.query,
        )
        db_service.link_message_files(message_id, ready_file_ids, conversation_id)

        return ChatResponse(
            message_id=message_id,
            conversation_id=conversation_id,
            answer=agent_response["answer"],
            used_rag=agent_response["used_rag"],
            file_ids=ready_file_ids,
            sources=[SourceDocument(**s) for s in agent_response.get("sources", [])],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")


# # ===== LEGACY ENDPOINTS (commented out — replaced by /upload, /chat) =====
# @app.post("/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
# async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
#     """Upload a document, store in MinIO, process with Docling, and index in Milvus."""
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="Filename is required")
#     try:
#         file_content = await file.read()
#         minio_service = get_minio_service()
#         doc_id, minio_url = minio_service.upload_document(file_content, file.filename)
#         docling_service = get_docling_service()
#         documents = docling_service.process_document(
#             file_content=file_content, filename=file.filename, minio_url=minio_url,
#         )
#         milvus_service = get_milvus_service()
#         milvus_service.add_documents(documents)
#         return DocumentUploadResponse(
#             document_id=doc_id, filename=file.filename,
#             minio_url=minio_url, chunks_count=len(documents),
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
#
# @app.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
# async def list_documents() -> list[DocumentInfo]:
#     try:
#         minio_service = get_minio_service()
#         docs = minio_service.list_documents()
#         return [DocumentInfo(**doc) for doc in docs]
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")
#
# @app.post("/query", response_model=QueryResponse, tags=["RAG"])
# async def query_rag(request: QueryRequest) -> QueryResponse:
#     try:
#         rag_service = get_rag_service()
#         response = await rag_service.aquery(question=request.query, top_k=request.top_k)
#         return response
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Multimodal RAG API",
        "version": "0.3.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "files": {
                "upload": "POST /upload  (headers: x-actor-id, x-conversation-id)",
                "status": "GET /files/{file_id}/status",
            },
            "chat": {
                "send": "POST /chat  (headers: x-actor-id, x-conversation-id)",
            },
            "conversations": {
                "create": "POST /conversations",
            },
        },
    }

