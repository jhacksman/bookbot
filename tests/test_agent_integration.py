import pytest
import asyncio
from bookbot.agents.selection.agent import SelectionAgent
from bookbot.agents.summarization.agent import SummarizationAgent
from bookbot.agents.librarian.agent import LibrarianAgent
from bookbot.agents.query.agent import QueryAgent
from bookbot.utils.venice_client import VeniceConfig
from bookbot.utils.resource_manager import VRAMManager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker, AsyncEngine
from typing import AsyncGenerator, cast
from bookbot.database.models import Base

@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    engine: AsyncEngine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = async_sessionmaker(
        engine, expire_on_commit=False
    )
    async with async_session() as session:
        yield session
    await engine.dispose()

@pytest.fixture
def vram_manager():
    return VRAMManager(total_vram=64.0)

@pytest.mark.asyncio
async def test_full_pipeline(venice_config, vram_manager, db_session):
    """Test the complete pipeline from book selection to querying."""
    agents = []
    allocations = []
    try:
        # Initialize agents with VRAM allocation - using a flatter structure
        selection_alloc = vram_manager.allocate("selection", 16.0)
        summarization_alloc = vram_manager.allocate("summarization", 16.0)
        librarian_alloc = vram_manager.allocate("librarian", 16.0)
        query_alloc = vram_manager.allocate("query", 16.0)
        
        async with selection_alloc as _:
            async with summarization_alloc as _:
                async with librarian_alloc as _:
                    async with query_alloc as _:
                        # Initialize all agents
                        selection_agent = SelectionAgent(venice_config, vram_limit=16.0)
                        summarization_agent = SummarizationAgent(venice_config, vram_limit=16.0)
                        librarian_agent = LibrarianAgent(venice_config, db_url="sqlite+aiosqlite:///:memory:", vram_limit=16.0)
                        query_agent = QueryAgent(venice_config, db_session, vram_limit=16.0)
                        
                        agents.extend([selection_agent, summarization_agent, librarian_agent, query_agent])
                        
                        # Initialize all agents
                        await asyncio.gather(
                            selection_agent.initialize(),
                            summarization_agent.initialize(),
                            librarian_agent.initialize(),
                            query_agent.initialize()
                        )
                        
                        test_book = {
                            "title": "Deep Learning",
                            "author": "Ian Goodfellow",
                            "description": "A comprehensive guide to deep learning and neural networks",
                            "content": """
                            Deep learning is a subset of machine learning that uses neural networks
                            with multiple layers. These networks can automatically learn representations
                            from data without explicit feature engineering. The depth allows the model
                            to learn hierarchical representations, with each layer building upon the
                            previous ones.
                            """
                        }
                        
                        # Test book selection
                        selection_result = await selection_agent.process({"books": [test_book]})
                        assert selection_result["status"] == "success"
                        assert len(selection_result["selected_books"]) > 0
                        
                        # Test summarization
                        summarization_result = await summarization_agent.process({
                            "content": test_book["content"],
                            "book_id": "test123",
                            "title": test_book["title"]
                        })
                        assert summarization_result["status"] == "success"
                        assert len(summarization_result["summaries"]) == 3
                        
                        # Test adding to library
                        librarian_result = await librarian_agent.process({
                            "action": "add_book",
                            "book": {
                                **test_book,
                                "summaries": summarization_result["summaries"]
                            }
                        })
                        assert librarian_result["status"] == "success"
                        assert "book_id" in librarian_result
                        
                        # Test querying
                        query_result = await query_agent.process({
                            "question": "What is deep learning and how does it relate to neural networks?"
                        })
                        assert query_result["status"] == "success"
                        assert "response" in query_result
                        assert "citations" in query_result
                        assert query_result["confidence"] > 0.0
    finally:
        # Cleanup agents in reverse order
        for agent in reversed(agents):
            try:
                await agent.cleanup()
            except Exception as e:
                print(f"Error cleaning up agent: {e}")

@pytest.mark.asyncio
async def test_vram_limits(venice_config, vram_manager):
    """Test that agents respect VRAM limits when running concurrently."""
    # Allocate all VRAM at once to avoid race conditions
    selection_alloc = vram_manager.allocate("selection", 16.0)
    summarization_alloc = vram_manager.allocate("summarization", 16.0)
    librarian_alloc = vram_manager.allocate("librarian", 16.0)
    query_alloc = vram_manager.allocate("query", 16.0)
    
    async with selection_alloc as _:
        async with summarization_alloc as _:
            async with librarian_alloc as _:
                async with query_alloc as _:
                    # All agents allocated, should be at max VRAM
                    available = await vram_manager.get_available_vram()
                    assert available == 0.0
                    
                    # Try to allocate more VRAM (should fail)
                    with pytest.raises(RuntimeError):
                        async with vram_manager.allocate("extra", 1.0):
                            pass
