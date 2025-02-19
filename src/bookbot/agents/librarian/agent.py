from typing import Any, Dict, List, Optional
import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from ..base import Agent
from ...database.models import Base, Book, Summary
from ...utils.venice_client import VeniceClient, VeniceConfig
from ...utils.vector_store import VectorStore
from ...utils.epub_processor import EPUBProcessor

class LibrarianAgent(Agent):
    def __init__(self, venice_config: VeniceConfig, db_url: str = "sqlite+aiosqlite:///:memory:", vram_limit: float = 16.0):
        super().__init__(vram_limit)
        self.venice = VeniceClient(venice_config)
        self.vector_store = VectorStore("librarian_agent")
        self.epub_processor = EPUBProcessor()
        self.engine = create_async_engine(db_url, echo=True)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def initialize(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self.is_active = True
    
    async def cleanup(self) -> None:
        await self.engine.dispose()
        self.is_active = False
    
    async def add_book(self, book_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with self.async_session() as session:
                async with session.begin():
                    # Convert metadata to JSON string if it exists
                    metadata = json.dumps(book_data.get("metadata")) if book_data.get("metadata") else None
                    
                    book = Book(
                        title=book_data["title"],
                        author=book_data.get("author"),
                        content_hash=book_data.get("content_hash"),
                        book_metadata=metadata,
                        vector_id=book_data.get("vector_id")
                    )
                    session.add(book)
                    await session.flush()
                    book_id = book.id
                    return {
                        "status": "success",
                        "book_id": book_id
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
                metadata={"content_hash": epub_data["content_hash"]}
            )
            
            # Add book to database
            book_result = await self.add_book({
                "title": epub_data["metadata"]["title"],
                "author": epub_data["metadata"]["author"],
                "content_hash": epub_data["content_hash"],
                "metadata": epub_data["metadata"],
                "vector_id": chunk_ids[0]  # Store first chunk ID
            })
            
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
