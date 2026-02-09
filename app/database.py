"""Database models and initialization using SQLAlchemy."""

from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
import json as _json
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class FileStatus(str, PyEnum):
    """File processing status."""

    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class MessageIntentType(str, PyEnum):
    """Message intent classification."""

    HIGH_LEVEL = "HIGH_LEVEL"  # Overview, compare, summarize all
    FACTUAL = "FACTUAL"  # Requires retrieval
    AMBIGUOUS = "AMBIGUOUS"  # Needs clarification


class Conversation(Base):
    """Stores conversation and lifecycle metadata."""

    __tablename__ = "conversations"

    conversation_id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    files = relationship("File", back_populates="conversation", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class File(Base):
    """Stores file metadata, ingestion status, summary and embeddings reference."""

    __tablename__ = "files"

    file_id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), ForeignKey("conversations.conversation_id"), nullable=False)
    actor_id = Column(String(36), nullable=True)  # Actor/user who uploaded
    minio_url = Column(String(500), nullable=False)
    filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=True)  # in bytes
    status = Column(Enum(FileStatus), default=FileStatus.PROCESSING, nullable=False)
    summary = Column(Text, nullable=True)  # Generated summary text
    summary_embedding_id = Column(String(100), nullable=True)  # Reference to Milvus embedding
    chunk_ids = Column(Text, nullable=True)  # JSON list of Milvus chunk IDs
    error_message = Column(Text, nullable=True)  # Error details if FAILED
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    conversation = relationship("Conversation", back_populates="files")
    message_files = relationship("MessageFile", back_populates="file", cascade="all, delete-orphan")


class Message(Base):
    """Stores message content, intent label and timestamps."""

    __tablename__ = "messages"

    message_id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), ForeignKey("conversations.conversation_id"), nullable=False)
    content = Column(Text, nullable=False)
    intent = Column(Enum(MessageIntentType), nullable=True)  # Intent classification result
    is_ambiguous = Column(Integer, default=0, nullable=False)  # Boolean: 0 or 1
    clarification_question = Column(Text, nullable=True)  # If ambiguous, ask this
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    message_files = relationship("MessageFile", back_populates="message", cascade="all, delete-orphan")


class MessageFile(Base):
    """Grounding table: explicit (message_id, file_id) mappings (append-only)."""

    __tablename__ = "message_files"

    message_file_id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String(36), ForeignKey("messages.message_id"), nullable=True)
    file_id = Column(String(36), ForeignKey("files.file_id"), nullable=False)
    conversation_id = Column(String(36), ForeignKey("conversations.conversation_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    message = relationship("Message", back_populates="message_files")
    file = relationship("File", back_populates="message_files")


class DatabaseService:
    """Service for database operations."""

    def __init__(self, database_url: str | None = None):
        """Initialize database connection.

        Args:
            database_url: SQLAlchemy database URL. Defaults to settings.database_url (PostgreSQL).
        """
        if database_url is None:
            from app.config import get_settings
            database_url = get_settings().database_url

        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()

    def create_conversation(self, conversation_id: str) -> dict:
        """Create a new conversation.

        Args:
            conversation_id: Unique conversation ID

        Returns:
            Dict with conversation data
        """
        session = self.get_session()
        try:
            conversation = Conversation(conversation_id=conversation_id)
            session.add(conversation)
            session.commit()
            # Access all attributes before session closes to load them
            conv_data = {
                "conversation_id": conversation.conversation_id,
                "created_at": conversation.created_at.isoformat(),
            }
            return conv_data
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation object or None
        """
        session = self.get_session()
        try:
            return session.query(Conversation).filter_by(conversation_id=conversation_id).first()
        finally:
            session.close()

    def create_file(
        self,
        file_id: str,
        conversation_id: str,
        minio_url: str,
        filename: str,
        file_size: int | None = None,
        actor_id: str | None = None,
    ) -> File:
        """Create a new file record.

        Args:
            file_id: Unique file ID
            conversation_id: Associated conversation ID
            minio_url: URL to file in MinIO
            filename: Original filename
            file_size: File size in bytes
            actor_id: Actor/user who uploaded the file

        Returns:
            File object
        """
        session = self.get_session()
        try:
            file = File(
                file_id=file_id,
                conversation_id=conversation_id,
                actor_id=actor_id,
                minio_url=minio_url,
                filename=filename,
                file_size=file_size,
                status=FileStatus.PROCESSING,
            )
            session.add(file)
            session.commit()
            return file
        finally:
            session.close()

    def update_file_status(
        self,
        file_id: str,
        status: FileStatus,
        summary: str | None = None,
        chunk_ids: list[str] | None = None,
        error_message: str | None = None,
    ) -> Optional[File]:
        """Update file processing status.

        Args:
            file_id: File ID
            status: New status
            summary: Summary text if ready
            chunk_ids: List of Milvus chunk IDs
            error_message: Error message if failed

        Returns:
            Updated File object
        """
        session = self.get_session()
        try:
            file = session.query(File).filter_by(file_id=file_id).first()
            if file:
                file.status = status
                if summary:
                    file.summary = summary
                if chunk_ids is not None:
                    file.chunk_ids = _json.dumps(chunk_ids)
                if error_message:
                    file.error_message = error_message
                session.commit()
            return file
        finally:
            session.close()

    def get_file(self, file_id: str) -> Optional[File]:
        """Get a single file by ID.

        Args:
            file_id: File ID

        Returns:
            File object or None
        """
        session = self.get_session()
        try:
            return session.query(File).filter_by(file_id=file_id).first()
        finally:
            session.close()

    def get_chunk_ids_for_files(self, file_ids: list[str]) -> list[str]:
        """Get all Milvus chunk IDs for a list of file IDs.

        Goes to the SQL table, reads the chunk_ids JSON column for each file,
        and returns a flat list of all chunk IDs.

        Args:
            file_ids: List of file IDs

        Returns:
            Flat list of Milvus chunk IDs
        """
        session = self.get_session()
        try:
            all_chunk_ids: list[str] = []
            files = session.query(File).filter(File.file_id.in_(file_ids)).all()
            for f in files:
                if f.chunk_ids:
                    try:
                        ids = _json.loads(f.chunk_ids)
                        if isinstance(ids, list):
                            all_chunk_ids.extend(ids)
                    except (_json.JSONDecodeError, TypeError):
                        pass
            return all_chunk_ids
        finally:
            session.close()

    def create_message(
        self,
        message_id: str,
        conversation_id: str,
        content: str,
        intent: MessageIntentType | None = None,
        is_ambiguous: bool = False,
        clarification_question: str | None = None,
    ) -> Message:
        """Create a new message.

        Args:
            message_id: Unique message ID
            conversation_id: Associated conversation ID
            content: Message content
            intent: Intent classification
            is_ambiguous: Whether message is ambiguous
            clarification_question: Clarification if ambiguous

        Returns:
            Message object
        """
        session = self.get_session()
        try:
            message = Message(
                message_id=message_id,
                conversation_id=conversation_id,
                content=content,
                intent=intent,
                is_ambiguous=1 if is_ambiguous else 0,
                clarification_question=clarification_question,
            )
            session.add(message)
            session.commit()
            return message
        finally:
            session.close()

    def get_message(self, message_id: str) -> Optional[Message]:
        """Get message by ID.

        Args:
            message_id: Message ID

        Returns:
            Message object or None
        """
        session = self.get_session()
        try:
            return session.query(Message).filter_by(message_id=message_id).first()
        finally:
            session.close()

    def get_conversation_messages(self, conversation_id: str) -> list[Message]:
        """Get all messages in a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            List of Message objects
        """
        session = self.get_session()
        try:
            return session.query(Message).filter_by(conversation_id=conversation_id).order_by(Message.created_at).all()
        finally:
            session.close()

    def get_conversation_files(self, conversation_id: str, status: FileStatus | None = None) -> list[File]:
        """Get all files in a conversation.

        Args:
            conversation_id: Conversation ID
            status: Optional status filter

        Returns:
            List of File objects
        """
        session = self.get_session()
        try:
            query = session.query(File).filter_by(conversation_id=conversation_id)
            if status:
                query = query.filter_by(status=status)
            return query.all()
        finally:
            session.close()

    def link_message_files(self, message_id: str, file_ids: list[str], conversation_id: str) -> list[MessageFile]:
        """Link files to a message.

        Args:
            message_id: Message ID
            file_ids: List of file IDs to link
            conversation_id: Associated conversation ID

        Returns:
            List of MessageFile mapping objects
        """
        session = self.get_session()
        try:
            mappings = []
            for file_id in file_ids:
                mapping = MessageFile(
                    message_id=message_id,
                    file_id=file_id,
                    conversation_id=conversation_id,
                )
                session.add(mapping)
                mappings.append(mapping)
            session.commit()
            return mappings
        finally:
            session.close()

    def link_file_without_message(self, file_id: str, conversation_id: str) -> MessageFile:
        """Link a file to conversation without message.

        Args:
            file_id: File ID
            conversation_id: Conversation ID

        Returns:
            MessageFile mapping object
        """
        session = self.get_session()
        try:
            mapping = MessageFile(
                message_id=None,
                file_id=file_id,
                conversation_id=conversation_id,
            )
            session.add(mapping)
            session.commit()
            return mapping
        finally:
            session.close()

    def get_message_files(self, message_id: str) -> list[File]:
        """Get all files linked to a message.

        Args:
            message_id: Message ID

        Returns:
            List of File objects
        """
        session = self.get_session()
        try:
            files = (
                session.query(File)
                .join(MessageFile)
                .filter(MessageFile.message_id == message_id)
                .all()
            )
            return files
        finally:
            session.close()


# Singleton instance
_db_service: DatabaseService | None = None


def get_db_service(database_url: str | None = None) -> DatabaseService:
    """Get or create database service instance."""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService(database_url)
    return _db_service
