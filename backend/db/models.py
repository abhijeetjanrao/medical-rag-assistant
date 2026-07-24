"""
SQLAlchemy models + session management.

Stores conversation turns, retrieved-source metadata, and guardrail decisions
so the whole system is auditable — important for anything medical-adjacent.
"""
import os
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, DateTime, Text, Float, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

load_dotenv()

DB_URL = os.getenv("DB_URL", "sqlite:///./medical_rag.db")

connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
engine = create_engine(DB_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def gen_id() -> str:
    return str(uuid.uuid4())


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=gen_id)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=gen_id)
    conversation_id = Column(String, ForeignKey("conversations.id"))
    role = Column(String)  # "user" | "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Populated for assistant messages only
    guardrail_flagged = Column(Boolean, default=False)
    guardrail_reason = Column(String, nullable=True)
    latency_ms = Column(Float, nullable=True)

    conversation = relationship("Conversation", back_populates="messages")
    sources = relationship("SourceCitation", back_populates="message", cascade="all, delete-orphan")


class SourceCitation(Base):
    """One row per retrieved chunk that was used to ground an answer."""
    __tablename__ = "source_citations"

    id = Column(String, primary_key=True, default=gen_id)
    message_id = Column(String, ForeignKey("messages.id"))
    document_name = Column(String)
    page_number = Column(String, nullable=True)
    chunk_text = Column(Text)
    relevance_score = Column(Float)

    message = relationship("Message", back_populates="sources")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
