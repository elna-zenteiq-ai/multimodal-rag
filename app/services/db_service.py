"""DB service using SQLAlchemy for conversations, files, messages, and mappings.

This is a lightweight implementation to persist records described in the design:
- conversations
- files (status, summary, minio_url)
- messages
- message_files (mapping table)

The module exposes helper functions to create and update these records.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Integer,
    String,
    Text,
    create_engine,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from app.config import get_settings


Base = declarative_base()


class FileStatus(str, enum.Enum):
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)


class File(Base):
    __tablename__ = "files"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    filename = Column(String, nullable=False)
    status = Column(Enum(FileStatus), default=FileStatus.PROCESSING)
    summary = Column(Text, nullable=True)
    minio_url = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    content = Column(Text, nullable=True)
    intent = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class MessageFile(Base):
    __tablename__ = "message_files"
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String, ForeignKey("messages.id"), nullable=True)
    file_id = Column(String, ForeignKey("files.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DBService:
    def __init__(self):
        settings = get_settings()
        # create_engine will accept postgres, sqlite, etc. For sqlite we disable thread check
        connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite:") else {}
        self.engine = create_engine(settings.database_url, connect_args=connect_args)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()


# singleton
_db: DBService | None = None


def get_db_service() -> DBService:
    global _db
    if _db is None:
        _db = DBService()
    return _db


def create_conversation(session: Session, conversation_id: str | None = None) -> Conversation:
    convo = Conversation(id=conversation_id or str(uuid.uuid4()))
    session.add(convo)
    session.commit()
    session.refresh(convo)
    return convo


def get_conversation(session: Session, conversation_id: str) -> Conversation | None:
    return session.query(Conversation).filter_by(id=conversation_id).first()


def add_file_record(session: Session, conversation_id: str, filename: str, minio_url: str | None = None) -> File:
    f = File(conversation_id=conversation_id, filename=filename, status=FileStatus.PROCESSING, minio_url=minio_url)
    session.add(f)
    session.commit()
    session.refresh(f)
    return f


def update_file_status(session: Session, file_id: str, status: FileStatus, summary: str | None = None, minio_url: str | None = None) -> None:
    f = session.query(File).filter_by(id=file_id).first()
    if not f:
        return
    f.status = status
    if summary is not None:
        f.summary = summary
    if minio_url is not None:
        f.minio_url = minio_url
    session.add(f)
    session.commit()


def add_message_record(session: Session, conversation_id: str, content: str | None = None, intent: str | None = None) -> Message:
    m = Message(conversation_id=conversation_id, content=content, intent=intent)
    session.add(m)
    session.commit()
    session.refresh(m)
    return m


def add_message_file(session: Session, message_id: str | None, file_id: str) -> MessageFile:
    mf = MessageFile(message_id=message_id, file_id=file_id)
    session.add(mf)
    session.commit()
    session.refresh(mf)
    return mf


def any_processing_files(session: Session, conversation_id: str) -> bool:
    return session.query(File).filter_by(conversation_id=conversation_id, status=FileStatus.PROCESSING).count() > 0


def get_files_by_conversation(session: Session, conversation_id: str) -> list[File]:
    return session.query(File).filter_by(conversation_id=conversation_id).all()
