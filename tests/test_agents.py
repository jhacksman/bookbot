import pytest
from typing import AsyncGenerator, Dict, Any, cast
from bookbot.agents.selection.agent import SelectionAgent
from bookbot.agents.summarization.agent import SummarizationAgent
from bookbot.agents.librarian.agent import LibrarianAgent
from bookbot.agents.query.agent import QueryAgent
from bookbot.utils.venice_client import VeniceConfig
from bookbot.utils.resource_manager import VRAMManager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker, AsyncEngine
from sqlalchemy.engine.base import Engine
from bookbot.database.models import Base

@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    engine: AsyncEngine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    session_maker = async_sessionmaker(
        engine, expire_on_commit=False
    )
    async with session_maker() as session:
        yield session
    await engine.dispose()

@pytest.fixture
def vram_manager():
    return VRAMManager(total_vram=64.0)

@pytest.mark.asyncio
async def test_selection_agent_vram(venice_config: VeniceConfig, vram_manager: VRAMManager):
    agent = SelectionAgent(venice_config, vram_limit=16.0)
    async with vram_manager.allocate("selection_agent", 16.0):
        await agent.initialize()
        assert agent.is_active
        
        test_book = {
            "title": "Deep Learning",
            "author": "Ian Goodfellow",
            "description": "A comprehensive guide to machine learning"
        }
        
        result = await agent.process({"books": [test_book]})
        assert result["status"] == "success"
        assert "selected_books" in result
        assert "evaluations" in result
        
        # Verify VRAM allocation
        allocations = await vram_manager.get_allocations()
        assert "selection_agent" in allocations
        assert allocations["selection_agent"] == 16.0
        
        await agent.cleanup()
        assert not agent.is_active

@pytest.mark.asyncio
async def test_summarization_agent_vram(venice_config: VeniceConfig, vram_manager: VRAMManager):
    agent = SummarizationAgent(venice_config, vram_limit=16.0)
    async with vram_manager.allocate("summarization_agent", 16.0):
        await agent.initialize()
        assert agent.is_active
        
        test_content = """
        Deep learning is a subset of machine learning that uses neural networks
        with multiple layers. These networks can automatically learn representations
        from data without explicit feature engineering.
        """
        
        result = await agent.process({
            "content": test_content,
            "book_id": "test123",
            "title": "Test Book"
        })
        
        assert result["status"] == "success"
        assert "summaries" in result
        assert len(result["summaries"]) == 3  # Default depth
        
        # Verify VRAM allocation
        allocations = await vram_manager.get_allocations()
        assert "summarization_agent" in allocations
        assert allocations["summarization_agent"] == 16.0
        
        await agent.cleanup()
        assert not agent.is_active

@pytest.mark.asyncio
async def test_librarian_agent_vram(venice_config: VeniceConfig, vram_manager: VRAMManager, db_session: AsyncSession):
    agent = LibrarianAgent(venice_config, db_url="sqlite+aiosqlite:///:memory:", vram_limit=16.0)
    async with vram_manager.allocate("librarian_agent", 16.0):
        await agent.initialize()
        assert agent.is_active
        
        test_book = {
            "title": "Test Book",
            "author": "Test Author",
            "content_hash": "abc123",
            "metadata": {"format": "pdf"}
        }
        
        result = await agent.process({
            "action": "add_book",
            "book": test_book
        })
        
        assert result["status"] == "success"
        assert "book_id" in result
        
        # Verify VRAM allocation
        allocations = await vram_manager.get_allocations()
        assert "librarian_agent" in allocations
        assert allocations["librarian_agent"] == 16.0
        
        await agent.cleanup()
        assert not agent.is_active

@pytest.mark.asyncio
async def test_query_agent_vram(venice_config: VeniceConfig, vram_manager: VRAMManager, db_session: AsyncSession):
    agent = QueryAgent(venice_config, db_session, vram_limit=16.0)
    async with vram_manager.allocate("query_agent", 16.0):
        await agent.initialize()
        assert agent.is_active
        
        # Add test content to vector store
        await agent.vector_store.add_texts(
            texts=["This is a test document about AI."],
            metadata=[{"book_id": 1}],
            ids=["test1"]
        )
        
        result = await agent.process({
            "question": "What is this document about?"
        })
        
        assert result["status"] == "success"
        assert "response" in result
        assert "citations" in result
        assert "confidence" in result
        
        # Verify VRAM allocation
        allocations = await vram_manager.get_allocations()
        assert "query_agent" in allocations
        assert allocations["query_agent"] == 16.0
        
        await agent.cleanup()
        assert not agent.is_active
