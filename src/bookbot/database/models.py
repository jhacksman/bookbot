from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Book(Base):
    __tablename__ = 'books'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    author = Column(String(255))
    content_hash = Column(String(64), unique=True)
    book_metadata = Column(Text)  # Renamed from metadata
    vector_id = Column(String(64))
    summaries = relationship("Summary", back_populates="book")

class Summary(Base):
    __tablename__ = 'summaries'
    
    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey('books.id'))
    level = Column(Integer)
    content = Column(Text)
    vector_id = Column(String(64))
    book = relationship("Book", back_populates="summaries")
