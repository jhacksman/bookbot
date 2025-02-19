from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from ..base import Agent
from ...database.models import Base, Book, Summary
from ...utils.venice_client import VeniceClient, VeniceConfig
from ...utils.vector_store import VectorStore

class LibrarianAgent(Agent):
    def __init__(self, venice_config: VeniceConfig, db_url: str = "sqlite+aiosqlite:///:memory:", vram_limit: float = 16.0):
        super().__init__(vram_limit)
        self.venice = VeniceClient(venice_config)
        self.vector_store = VectorStore("librarian_agent")
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
        async with self.async_session() as session:
            book = Book(
                title=book_data["title"],
                author=book_data.get("author"),
                content_hash=book_data.get("content_hash"),
                book_metadata=book_data.get("metadata"),
                vector_id=book_data.get("vector_id")
            )
            session.add(book)
            await session.commit()
            return {"book_id": book.id}
    
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
