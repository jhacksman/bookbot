from typing import Any, Dict, List, Optional
import json
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.future import select
from ..base import Agent
from ...database.models import Base, Book, Summary
from ...utils.venice_client import VeniceClient, VeniceConfig
from ...utils.vector_store import VectorStore
from ...utils.epub_processor import EPUBProcessor
from ...utils.calibre_connector import CalibreConnector

class LibrarianAgent(Agent):
    def __init__(self, venice_config: VeniceConfig, 
             db_url: str = "sqlite+aiosqlite:///:memory:",
             calibre_path: Optional[Path] = None,
             vram_limit: float = 16.0):
        super().__init__(vram_limit)
        self.venice = VeniceClient(venice_config)
        self.vector_store = VectorStore("librarian_agent")
        self.epub_processor = EPUBProcessor()
        self.engine = create_async_engine(db_url, echo=True)
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        self.calibre = CalibreConnector(calibre_path) if calibre_path else None
    
    async def initialize(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        if hasattr(self, 'calibre'):
            await self.calibre.initialize()
        self.is_active = True
    
    async def cleanup(self) -> None:
        await self.engine.dispose()
        if hasattr(self, 'calibre'):
            await self.calibre.cleanup()
        self.is_active = False
    
    async def add_book(self, book_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with self.async_session() as session:
                metadata = json.dumps(book_data.get("metadata")) if book_data.get("metadata") else None
                
                book = Book(
                    title=book_data["title"],
                    author=book_data.get("author"),
                    content_hash=book_data.get("content_hash"),
                    book_metadata=metadata,
                    vector_id=book_data.get("vector_id")
                )
                session.add(book)
                await session.commit()
                await session.refresh(book)
                return {
                    "book_id": book.id,
                    "status": "success"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def add_summary(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        async with self.async_session() as session:
            summary = Summary(
                book_id=summary_data["book_id"],
                level=summary_data["level"],
                content=summary_data["content"],
                vector_id=summary_data["vector_id"]
            )
            session.add(summary)
            await session.commit()
            return {"summary_id": summary.id}
    
    async def get_book(self, book_id: int) -> Optional[Dict[str, Any]]:
        async with self.async_session() as session:
            result = await session.execute(
                select(Book).where(Book.id == book_id)
            )
            book = result.scalar_one_or_none()
            if book:
                return {
                    "id": book.id,
                    "title": book.title,
                    "author": book.author,
                    "content_hash": book.content_hash,
                    "metadata": book.book_metadata,
                    "vector_id": book.vector_id
                }
            return None
    
    async def process_epub(self, file_path: str) -> Dict[str, Any]:
        try:
            # Process EPUB file
            epub_data = await self.epub_processor.process_file(file_path)
            
            # Add content chunks to vector store
            chunk_ids = await self.vector_store.add_texts(
                texts=epub_data["chunks"],
                metadata=[{"content_hash": epub_data["content_hash"]} for _ in epub_data["chunks"]]
            )
            
            # Add book to database
            book_data = {
                "title": epub_data["metadata"]["title"],
                "author": epub_data["metadata"]["author"],
                "content_hash": epub_data["content_hash"],
                "metadata": epub_data["metadata"],
                "vector_id": chunk_ids[0]  # Store first chunk ID
            }
            
            book_result = await self.add_book(book_data)
            if "book_id" not in book_result:
                raise RuntimeError(f"Failed to add book: {book_result.get('message', 'Unknown error')}")
                
            return {
                "status": "success",
                "book_id": book_result["book_id"],
                "vector_ids": chunk_ids
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def sync_calibre_library(self) -> Dict[str, Any]:
        if not hasattr(self, 'calibre') or not self.calibre:
            return {"status": "error", "message": "Calibre integration not configured"}
        
        try:
            books = await self.calibre.get_books()
            synced_count = 0
            
            for book in books:
                async with self.async_session() as session:
                    result = await session.execute(
                        select(Book).where(Book.title == book["title"])
                    )
                    existing = result.scalar_one_or_none()
                    
                    calibre_metadata = {
                        "calibre_id": book["id"],
                        "format": book["format"],
                        "identifiers": book["identifiers"],
                        "tags": book["tags"],
                        "series": book.get("series"),
                        "series_index": book.get("series_index"),
                        "last_modified": book["last_modified"].isoformat() if book.get("last_modified") else None
                    }
                    
                    if existing:
                        # Update metadata while preserving existing BookBot data
                        current_metadata = json.loads(existing.book_metadata) if existing.book_metadata else {}
                        current_metadata.update(calibre_metadata)
                        existing.book_metadata = json.dumps(current_metadata)
                        await session.commit()
                    else:
                        # Add new book
                        new_book = Book(
                            title=book["title"],
                            author=book["author"],
                            book_metadata=json.dumps(calibre_metadata)
                        )
                        session.add(new_book)
                        await session.commit()
                    
                    synced_count += 1
            
            return {
                "status": "success",
                "books_synced": synced_count,
                "total_books": len(books)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        action = input_data.get("action")
        if not action:
            return {
                "status": "error",
                "message": "No action specified"
            }
        
        try:
            if action == "add_book":
                result = await self.add_book(input_data["book"])
                return {
                    "status": "success",
                    "book_id": result["book_id"]
                }
            elif action == "add_summary":
                result = await self.add_summary(input_data["summary"])
                return {
                    "status": "success",
                    "summary_id": result["summary_id"]
                }
            elif action == "get_book":
                book = await self.get_book(input_data["book_id"])
                return {
                    "status": "success",
                    "book": book
                }
            elif action == "process_epub":
                return await self.process_epub(input_data["file_path"])
            elif action == "sync_calibre":
                return await self.sync_calibre_library()
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {action}"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
