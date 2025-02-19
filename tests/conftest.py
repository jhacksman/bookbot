import pytest
from unittest.mock import AsyncMock
from ebooklib import epub
from bookbot.utils.resource_manager import VRAMManager
from bookbot.utils.venice_client import VeniceClient, VeniceConfig
from bookbot.database.models import Base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from typing import Dict, Any, AsyncGenerator
import aiosqlite
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession, AsyncEngine

@pytest.fixture
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    """Fixture that provides an async SQLAlchemy session."""
    engine: AsyncEngine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        session_maker = async_sessionmaker(
            engine,
            expire_on_commit=False,
            class_=AsyncSession
        )
        
        async with session_maker() as session:
            yield session
            await session.rollback()
    finally:
        await engine.dispose()

@pytest.fixture(autouse=True)
def mock_venice_client(monkeypatch):
    class MockVeniceClient:
        def __init__(self, *args, **kwargs):
            self.config = kwargs.get("config")
            if not self.config and args:
                self.config = args[0]
            if not self.config:
                self.config = VeniceConfig(api_key="test_key")

        async def generate(self, prompt: str, *args, **kwargs):
            try:
                if "hierarchical" in prompt.lower():
                    level = 0
                    if "concise" in prompt.lower():
                        level = 1
                    elif "brief" in prompt.lower():
                        level = 2
                    return {
                        "choices": [{
                            "text": f"Summary level {level}: This is a test summary of the content. It covers key concepts and technical details at varying levels of detail depending on the summary level."
                        }]
                    }
                elif "evaluate" in prompt.lower():
                    return {
                        "choices": [{
                            "text": '{"score": 95, "reasoning": "This book is highly relevant for AI research", "key_topics": ["deep learning", "neural networks", "machine learning"]}'
                        }]
                    }
                else:
                    # For queries with no relevant content, return empty citations
                    if "meaning of life" in prompt.lower():
                        return {
                            "choices": [{
                                "text": '{"answer": "No relevant information found.", "citations": [], "confidence": 0.0}'
                            }]
                        }
                    # For queries with content, return citations
                    return {
                        "choices": [{
                            "text": '{"answer": "This book discusses artificial intelligence and machine learning concepts.", "citations": [{"book_id": 1, "title": "Test Book", "author": "Test Author", "quoted_text": "This is a test summary about AI."}], "confidence": 0.9}'
                        }]
                    }
            except Exception as e:
                return {
                    "choices": [{
                        "text": str(e)
                    }]
                }

        async def embed(self, input: str, *args, **kwargs) -> dict:
            if isinstance(input, list):
                return {"data": [{"embedding": [0.1, 0.2, 0.3] * 128} for _ in range(len(input))]}
            return {"data": [{"embedding": [0.1, 0.2, 0.3] * 128}]}

    # Patch VeniceClient in all modules
    for module in [
        "bookbot.utils.venice_client",
        "bookbot.agents.selection.agent",
        "bookbot.agents.summarization.agent",
        "bookbot.agents.librarian.agent",
        "bookbot.agents.query.agent"
    ]:
        monkeypatch.setattr(f"{module}.VeniceClient", MockVeniceClient)
    
    return MockVeniceClient()

@pytest.fixture
def venice_config(mock_venice_client):
    return VeniceConfig(api_key="test_key")

@pytest.fixture
def vram_manager():
    return VRAMManager(total_vram=64.0)

@pytest.fixture
def test_book_data() -> Dict[str, Any]:
    return {
        "title": "Test Book",
        "author": "Test Author",
        "content": "Test content for verification",
        "metadata": {
            "format": "pdf",
            "pages": 100,
            "language": "en"
        }
    }

@pytest.fixture
def test_summary_data() -> Dict[str, Any]:
    return {
        "book_id": 1,
        "level": 0,
        "content": "Test summary content",
        "vector_id": "test_vector_123"
    }

@pytest.fixture
def test_epub_path(tmp_path):
    from ebooklib import epub
    
    book = epub.EpubBook()
    book.set_identifier('test123')
    book.set_title('Test Book')
    book.set_language('en')
    book.add_author('Test Author')
    
    # Add chapter
    c1 = epub.EpubHtml(title='Chapter 1', file_name='chap_01.xhtml', lang='en')
    c1.content = '<h1>Chapter 1</h1><p>This is a test chapter.</p>'
    book.add_item(c1)
    
    # Add navigation files
    nav = epub.EpubNav()
    book.add_item(nav)
    
    # Create table of contents and spine
    book.toc = [c1]
    book.spine = [nav, c1]
    
    # Write EPUB with NCX disabled
    epub_path = tmp_path / "test.epub"
    epub.write_epub(str(epub_path), book, options={'ignore_ncx': True})
    return str(epub_path)

@pytest.fixture
async def initialized_vram_manager(vram_manager):
    yield vram_manager
    # Cleanup any remaining allocations
    vram_manager.allocated_vram = 0.0
    vram_manager.allocations.clear()
