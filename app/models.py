"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response schema for document upload."""

    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    minio_url: str = Field(..., description="Presigned URL to access the raw file")
    chunks_count: int = Field(..., description="Number of chunks created")


class QueryRequest(BaseModel):
    """Request schema for RAG query."""

    query: str = Field(..., description="The question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources to retrieve")


class SourceDocument(BaseModel):
    """Schema for a source document citation."""

    text: str = Field(..., description="Chunk text content")
    filename: str = Field(..., description="Source filename")
    page_numbers: list[int] = Field(default_factory=list, description="Page numbers")
    heading: str | None = Field(default=None, description="Section heading")


class QueryResponse(BaseModel):
    """Response schema for RAG query with sources."""

    answer: str = Field(..., description="Generated answer")
    sources: list[SourceDocument] = Field(default_factory=list, description="Source documents")


class DocumentInfo(BaseModel):
    """Schema for document info in list."""

    document_id: str
    filename: str
    minio_url: str
    uploaded_at: str | None = None


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    milvus: str
    minio: str
    postgres: str


class MessageResponse(BaseModel):
    """Response schema for sending a message with optional files."""

    message_id: str | None = Field(default=None, description="Created message id if message provided")
    file_ids: list[str] = Field(default_factory=list, description="IDs of uploaded files")
    conversation_id: str | None = Field(default=None, description="Conversation id created or used for this request")

class FileStatusInfo(BaseModel):
    """File status information."""

    file_id: str = Field(..., description="File ID")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="File status (PROCESSING, READY, FAILED)")


class ConversationStatusResponse(BaseModel):
    """Response schema for conversation status."""

    conversation_id: str = Field(..., description="Conversation ID")
    files: list[FileStatusInfo] = Field(default_factory=list, description="Files in conversation")