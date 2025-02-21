from sqlalchemy import Column, Integer, String, Text, ForeignKey, Enum, DateTime
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
import enum

Base = declarative_base()

class SummaryType(enum.Enum):
    CHAPTER = "chapter"
    BOOK = "book"

class SummaryLevel(enum.Enum):
    DETAILED = 0
    CONCISE = 1
    BRIEF = 2

class Book(Base):
    __tablename__ = 'books'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    author = Column(String(255))
    content_hash = Column(String(64), unique=True)
    book_metadata = Column(Text)
    vector_id = Column(String(64))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    summaries = relationship("Summary", back_populates="book", cascade="all, delete-orphan")

class Summary(Base):
    __tablename__ = 'summaries'
    
    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey('books.id'))
    level = Column(Enum(SummaryLevel))
    content = Column(Text)
    vector_id = Column(String(64))
    summary_type = Column(Enum(SummaryType))
    chapter_index = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    book = relationship("Book", back_populates="summaries")
