from datetime import datetime
from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base

# ORM机制：Python类 = 数据库的一张表，类里的属性 = 表的字段（列）
class User(Base):
    __tablename__ = "users"   # 指定数据库表名

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # 关系：1个用户 → 多个聊天会话
    sessions = relationship("ChatSession", back_populates = "user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"   # 指定数据库表名

    # 复合唯一约束：同一个用户不能有重复的session_id
    __table_args__ = (UniqueConstraint("user_id", "session_id", name="uq_user_session"),) 

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # 关系：1个聊天会话 → 多条消息  
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

# 存储聊天窗口里的每一条信息
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_ref_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    message_type: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    rag_trace: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    session = relationship("ChatSession", back_populates="messages")

# 存储文档分块信息（包括原始文件路径、分块文本内容、层级关系等）
class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    chunk_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    file_type: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    parent_chunk_id: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    root_chunk_id: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    chunk_level: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_idx: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
