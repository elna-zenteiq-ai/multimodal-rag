"""Pydantic models for API request/response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


# ===== LEGACY MODELS (kept for backward compatibility) =====
# class DocumentUploadResponse(BaseModel):
#     """Response schema for document upload."""

#     document_id: str = Field(..., description="Unique identifier for the document")
#     filename: str = Field(..., description="Original filename")
#     minio_url: str = Field(..., description="Presigned URL to access the raw file")
#     chunks_count: int = Field(..., description="Number of chunks created")


# class QueryRequest(BaseModel):
#     """Request schema for RAG query."""

#     query: str = Field(..., description="The question to ask")
#     top_k: int = Field(default=5, ge=1, le=20, description="Number of sources to retrieve")


class SourceDocument(BaseModel):
    """Schema for a source document citation."""

    text: str = Field(..., description="Chunk text content")
    filename: str = Field(..., description="Source filename")
    page_numbers: list[int] = Field(default_factory=list, description="Page numbers")
    heading: str | None = Field(default=None, description="Section heading")
    minio_url: str | None = Field(default=None, description="URL to raw file in MinIO")
    image_url: str | None = Field(default=None, description="URL to extracted image in MinIO")


# class QueryResponse(BaseModel):
#     """Response schema for RAG query with sources."""

#     answer: str = Field(..., description="Generated answer")
#     sources: list[SourceDocument] = Field(default_factory=list, description="Source documents")


# class DocumentInfo(BaseModel):
#     """Schema for document info in list."""

#     document_id: str
#     filename: str
#     minio_url: str
#     uploaded_at: str | None = None


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    milvus: str
    minio: str
    postgres: str


# ===== NEW AGENT-BASED RAG MODELS =====
# class MessageResponse(BaseModel):
#     """Response schema for message endpoint."""

#     message_id: str = Field(..., description="Unique message identifier")
#     conversation_id: str = Field(..., description="Conversation identifier")
#     answer: str = Field(..., description="Agent's response")
#     intent: str = Field(
#         ...,
#         description="Classified intent: HIGH_LEVEL, FACTUAL, or AMBIGUOUS",
#     )
#     used_rag: bool = Field(
#         ...,
#         description="Whether RAG search tool was used",
#     )
#     is_ambiguous: bool = Field(
#         ...,
#         description="Whether message is ambiguous",
#     )
#     clarification_question: Optional[str] = Field(
#         None,
#         description="Clarification question if message is ambiguous",
#     )
#     files_linked: list[str] = Field(
#         default_factory=list,
#         description="File IDs linked with this message",
#     )


# class FileUploadInfo(BaseModel):
#     """Info about uploaded files in response."""

#     file_id: str = Field(..., description="Unique file identifier")
#     filename: str = Field(..., description="Original filename")
#     status: str = Field(
#         ...,
#         description="Processing status: PROCESSING, READY, or FAILED",
#     )


class ConversationInitResponse(BaseModel):
    """Response for conversation initialization."""

    conversation_id: str = Field(..., description="New conversation ID")
    created_at: str = Field(..., description="Creation timestamp")


# class ConversationFilesResponse(BaseModel):
#     """Response with conversation files and suggestions."""

#     conversation_id: str = Field(..., description="Conversation ID")
#     files: list[FileUploadInfo] = Field(
#         ...,
#         description="Files in conversation",
#     )
#     ready_files_count: int = Field(
#         ...,
#         description="Number of files ready for querying",
#     )
#     suggestions: Optional[str] = Field(
#         None,
#         description="AI suggestions for what can be done with files",
#     )


# class AgentQueryResponse(BaseModel):
#     """Response from agent query."""

#     answer: str = Field(..., description="Agent's answer")
#     intent: str = Field(..., description="Classified intent")
#     used_rag: bool = Field(
#         ...,
#         description="Whether RAG retrieval was performed",
#     )
#     sources: list[SourceDocument] = Field(
#         default_factory=list,
#         description="Source documents if available",
#     )


class FileUploadResponse(BaseModel):
    """Response schema for the dedicated file upload endpoint."""

    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    status: str = Field(
        default="PROCESSING",
        description="Processing status: PROCESSING, READY, or FAILED",
    )
    message: str = Field(
        default="File uploaded. Processing started.",
        description="Human-readable status message",
    )


# class FileStatusResponse(BaseModel):
#     """Response schema for the file status endpoint."""

#     file_id: str = Field(..., description="File identifier")
#     filename: str = Field(..., description="Original filename")
#     status: str = Field(
#         ...,
#         description="Processing status: PROCESSING, READY, or FAILED",
#     )
#     summary: Optional[str] = Field(None, description="File summary (available when READY)")
#     chunk_count: int = Field(
#         default=0,
#         description="Number of indexed chunks in Milvus",
#     )
#     error_message: Optional[str] = Field(None, description="Error details if FAILED")


class ChatRequest(BaseModel):
    """Request schema for the chat endpoint."""

    query: str = Field(..., description="User's question / message")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class ChatResponse(BaseModel):
    """Response schema for the chat endpoint."""

    message_id: str = Field(..., description="Unique message identifier")
    conversation_id: str = Field(..., description="Conversation identifier")
    answer: str = Field(..., description="Agent's response")
    used_rag: bool = Field(..., description="Whether RAG search tool was used")
    file_ids: list[str] = Field(
        default_factory=list,
        description="File IDs used for this query",
    )
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="Source documents if available",
    )
