import pytest
import asyncio
import sys
import os
from io import StringIO
from unittest.mock import AsyncMock, patch

# Disable all telemetry and logging
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_LOGGING_ENABLE"] = "False"
os.environ["POSTHOG_DISABLED"] = "True"
os.environ["DISABLE_TELEMETRY"] = "True"
os.environ["TELEMETRY_DISABLED"] = "True"
os.environ["DISABLE_ANALYTICS"] = "True"

if sys.platform.startswith('linux'):
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Mock ChromaDB to prevent background threads
@pytest.fixture(autouse=True)
def mock_chromadb(monkeypatch):
    class MockChromaClient:
        def __init__(self, *args, **kwargs):
            self.collections = {}
        
        async def heartbeat(self):
            return True
        
        def reset(self):
            self.collections.clear()
        
        def close(self):
            pass
            
        def get_or_create_collection(self, name, **kwargs):
            if name not in self.collections:
                self.collections[name] = MockCollection(name)
            return self.collections[name]
            
        def get_collection(self, name):
            if name not in self.collections:
                raise ValueError(f"Collection {name} does not exist")
            return self.collections[name]
            
        def list_collections(self):
            return list(self.collections.values())
            
    class MockCollection:
        def __init__(self, name):
            self.name = name
            self.texts = []
            self.metadatas = []
            self.ids = []
            
        def add(self, documents=None, metadatas=None, ids=None):
            if not documents:
                return []
            start_idx = len(self.texts)
            self.texts.extend(documents)
            self.metadatas.extend(metadatas or [None] * len(documents))
            self.ids.extend(ids or [str(i + start_idx) for i in range(len(documents))])
            return self.ids[-len(documents):]
            
        def query(self, query_texts, n_results=1, where=None, **kwargs):
            if not self.texts:
                return {
                    "ids": [[]],
                    "distances": [[]],
                    "metadatas": [[]],
                    "documents": [[]]
                }
            results = min(n_results, len(self.texts))
            return {
                "ids": [self.ids[:results] if self.ids else []],
                "distances": [[0.0] * results] if results > 0 else [[]],
                "metadatas": [self.metadatas[:results] if self.metadatas else []],
                "documents": [self.texts[:results] if self.texts else []]
            }
    
    monkeypatch.setattr("chromadb.Client", MockChromaClient)
    monkeypatch.setattr("chromadb.PersistentClient", MockChromaClient)
from ebooklib import epub
from bookbot.utils.resource_manager import VRAMManager
from bookbot.utils.venice_client import VeniceClient, VeniceConfig
from bookbot.database.models import Base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from typing import Dict, Any, AsyncGenerator
import aiosqlite
from sqlalchemy.ext.asyncio import async_sessionmaker

@pytest.fixture(scope="function", autouse=True)
def event_loop():
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    try:
        # Cancel all running tasks
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        # Close any remaining aiohttp sessions
        for task in pending:
            if 'aiohttp' in str(task):
                task.cancel()
                try:
                    loop.run_until_complete(task)
                except (asyncio.CancelledError, Exception):
                    pass
        # Force cleanup of any remaining threads
        import threading
        for thread in threading.enumerate():
            if thread is not threading.current_thread():
                thread.join(timeout=1.0)
    finally:
        loop.close()
        asyncio.set_event_loop(None)

@pytest.fixture
async def async_session():
    """Fixture that provides an async SQLAlchemy session."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=NullPool
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    session = session_factory()
    try:
        await session.begin()
        yield session
    finally:
        await session.rollback()
        await session.close()
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
            self._last_call = asyncio.get_event_loop().time()
            self._call_count = 0
            self._rate_limit_delay = 0.2  # 200ms between calls to ensure test stability
            
        async def generate(self, prompt: str, *args, **kwargs):
            try:
                now = asyncio.get_event_loop().time()
                self._call_count += 1
                
                # Simulate rate limiting
                elapsed = now - self._last_call
                if elapsed < self._rate_limit_delay:
                    await asyncio.sleep(self._rate_limit_delay - elapsed)
                self._last_call = asyncio.get_event_loop().time()
                
                # Generate response based on prompt type and temperature
                temp = kwargs.get('temperature', 0.7)
                if "hierarchical" in prompt.lower():
                    level = 0
                    if "concise" in prompt.lower():
                        level = 1
                    elif "brief" in prompt.lower():
                        level = 2
                    response = f"Summary level {level} (temp={temp}): This is a test summary of the content."
                elif "evaluate" in prompt.lower():
                    response = f'{{"score": 95, "reasoning": "This book is highly relevant (temp={temp})"}}'
                elif "process_epub" in prompt.lower():
                    response = f'{{"status": "success", "book_id": 1, "vector_ids": ["vec123"], "temp": {temp}}}'
                else:
                    # For query agent tests, return temperature-dependent response
                    if "test prompt" in prompt.lower():
                        # Ensure different responses for different temperatures
                        if temp == 0.7:
                            response = {"answer": "Response for temperature 0.7", "citations": [], "confidence": 0.5}
                        elif temp == 0.8:
                            response = {"answer": "Different response for temperature 0.8", "citations": [], "confidence": 0.5}
                        else:
                            response = {"answer": f"Response for temperature {temp:.6f}", "citations": [], "confidence": 0.5}
                    else:
                        variant = hash(f"{prompt}{temp:.6f}") % 1000
                        response = {"answer": f"Response variant {variant} (temp={temp:.6f})", "citations": [], "confidence": 0.5}
                
                # Ensure response is a dict
                if isinstance(response, str):
                    try:
                        import json
                        response = json.loads(response)
                    except:
                        # If it's not JSON, wrap it in our standard response format
                        response = {"answer": response, "citations": [], "confidence": 0.5}
                elif not isinstance(response, dict):
                    response = {"answer": str(response), "citations": [], "confidence": 0.5}
                
                # Ensure we have all required fields
                if not isinstance(response, dict):
                    response = {"answer": str(response), "citations": [], "confidence": 0.5}
                elif "answer" not in response:
                    response = {"answer": str(response), "citations": response.get("citations", []), "confidence": response.get("confidence", 0.5)}
                
                return {
                    "choices": [{
                        "text": response
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
