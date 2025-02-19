import pytest
from bookbot.agents.selection.agent import SelectionAgent
from bookbot.agents.summarization.agent import SummarizationAgent
from bookbot.agents.librarian.agent import LibrarianAgent
from bookbot.agents.query.agent import QueryAgent
from bookbot.utils.venice_client import VeniceConfig
from bookbot.utils.resource_manager import VRAMManager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from typing import AsyncGenerator
from bookbot.database.models import Base

@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=True)
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
    # Initialize agents
    selection_agent = SelectionAgent(venice_config, vram_limit=16.0)
    summarization_agent = SummarizationAgent(venice_config, vram_limit=16.0)
    librarian_agent = LibrarianAgent(venice_config, db_url="sqlite+aiosqlite:///:memory:", vram_limit=16.0)
    query_agent = QueryAgent(venice_config, db_session, vram_limit=16.0)
    
    await selection_agent.initialize()
    await summarization_agent.initialize()
    await librarian_agent.initialize()
    await query_agent.initialize()
    
    try:
        # Test book selection
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
        # Cleanup
        await selection_agent.cleanup()
        await summarization_agent.cleanup()
        await librarian_agent.cleanup()
        await query_agent.cleanup()

@pytest.mark.asyncio
async def test_vram_limits(venice_config, vram_manager):
    """Test that agents respect VRAM limits when running concurrently."""
    async with vram_manager.allocate("selection", 16.0):
        async with vram_manager.allocate("summarization", 16.0):
            async with vram_manager.allocate("librarian", 16.0):
                async with vram_manager.allocate("query", 16.0):
                    # All agents allocated, should be at max VRAM
                    available = await vram_manager.get_available_vram()
                    assert available == 0.0
                    
                    # Try to allocate more VRAM (should fail)
                    with pytest.raises(RuntimeError):
                        async with vram_manager.allocate("extra", 1.0):
                            pass
