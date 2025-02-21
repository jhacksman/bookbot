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
             session: Optional[AsyncSession] = None,
             db_url: str = "sqlite+aiosqlite:///:memory:",
             calibre_path: Optional[Path] = None,
             vram_limit: float = 16.0):
        super().__init__(vram_limit)
        self.venice = VeniceClient(venice_config)
        self.vector_store = VectorStore("librarian_agent")
        self.epub_processor = EPUBProcessor()
        
        if session:
            self.session = session
            self.engine = None
        else:
            self.engine = create_async_engine(db_url, echo=True)
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            self.session = None
        
        self.calibre = CalibreConnector(calibre_path) if calibre_path else None
    
    async def initialize(self) -> None:
        if not self.session and hasattr(self, 'engine'):
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.session = self.async_session()
            
        if self.calibre:
            await self.calibre.initialize()
        self.is_active = True
    
    async def cleanup(self) -> None:
        try:
            if hasattr(self, 'engine') and self.engine is not None:
                await self.engine.dispose()
            if hasattr(self, 'calibre') and self.calibre:
                await self.calibre.cleanup()
        except Exception as e:
            print(f"Warning during cleanup: {e}")
        finally:
            self.is_active = False
    
    async def add_book(self, book_data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(book_data, dict):
            return {
                "status": "error",
                "message": "Book data must be a dictionary"
            }
            
        try:
            title = str(book_data.get("title", "Unknown Title"))
            author = str(book_data.get("author", "Unknown Author"))
            content_hash = str(book_data.get("content_hash", ""))
            vector_id = str(book_data.get("vector_id", ""))
            
            try:
                metadata = json.dumps(book_data.get("metadata", {}))
            except (TypeError, ValueError):
                metadata = "{}"
            
            book = Book(
                title=title,
                author=author,
                content_hash=content_hash,
                book_metadata=metadata,
                vector_id=vector_id
            )
            
            if not self.session:
                return {
                    "status": "error",
                    "message": "Session not initialized. Call initialize() first."
                }
            
            self.session.add(book)
            await self.session.commit()
            await self.session.refresh(book)
            
            if self.calibre:
                try:
                    await self.calibre.add_book({
                        "title": title,
                        "author": author,
                        "path": book_data.get("path", ""),
                        "format": book_data.get("format", "unknown"),
                        "identifiers": book_data.get("identifiers", {}),
                        "tags": book_data.get("tags", []),
                        "series": book_data.get("series"),
                        "series_index": book_data.get("series_index", 1.0)
                    })
                except Exception as e:
                    print(f"Warning: Failed to add book to Calibre: {e}")
            
            return {
                "status": "success",
                "book_id": book.id
            }
        except Exception as e:
            print(f"Error adding book: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def add_summary(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            summary = Summary(
                book_id=summary_data["book_id"],
                level=summary_data["level"],
                content=summary_data["content"],
                vector_id=summary_data["vector_id"]
            )
            self.session.add(summary)
            await self.session.commit()
            await self.session.refresh(summary)
            return {
                "status": "success",
                "summary_id": summary.id
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_book(self, book_id: int) -> Optional[Dict[str, Any]]:
        result = await self.session.execute(
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
            if not epub_data or not epub_data.get("chunks"):
                return {
                    "status": "error",
                    "message": "Failed to process EPUB file or no content found"
                }
            
            # Add content chunks to vector store
            chunk_ids = await self.vector_store.add_texts(
                texts=epub_data["chunks"],
                metadata=[{"content_hash": epub_data["content_hash"]} for _ in epub_data["chunks"]]
            )
            
            # Add book to database
            book_data = {
                "title": epub_data["metadata"].get("title", "Unknown Title"),
                "author": epub_data["metadata"].get("author", "Unknown Author"),
                "content_hash": epub_data["content_hash"],
                "metadata": epub_data["metadata"],
                "vector_id": chunk_ids[0] if chunk_ids else ""  # Store first chunk ID
            }
            
            book_result = await self.add_book(book_data)
            if book_result["status"] != "success":
                return book_result
                
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
