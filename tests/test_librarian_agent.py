import pytest
from bookbot.agents.librarian.agent import LibrarianAgent
from bookbot.utils.venice_client import VeniceConfig

@pytest.mark.asyncio
async def test_librarian_agent_initialization():
    config = VeniceConfig(api_key="test_key")
    agent = LibrarianAgent(config)
    await agent.initialize()
    assert agent.is_active
    await agent.cleanup()
    assert not agent.is_active

@pytest.mark.asyncio
async def test_librarian_agent_add_book():
    config = VeniceConfig(api_key="test_key")
    agent = LibrarianAgent(config)
    await agent.initialize()
    
    test_book = {
        "title": "Test Book",
        "author": "Test Author",
        "content_hash": "abc123",
        "metadata": {"format": "pdf"},
        "vector_id": "vec123"
    }
    
    result = await agent.process({
        "action": "add_book",
        "book": test_book
    })
    
    assert result["status"] == "success"
    assert "book_id" in result
    
    # Verify book was added
    book = await agent.get_book(result["book_id"])
    assert book is not None
    assert book["title"] == test_book["title"]
    assert book["author"] == test_book["author"]
    
    await agent.cleanup()

@pytest.mark.asyncio
async def test_librarian_agent_add_summary():
    config = VeniceConfig(api_key="test_key")
    agent = LibrarianAgent(config)
    await agent.initialize()
    
    # First add a book
    book_result = await agent.process({
        "action": "add_book",
        "book": {"title": "Test Book"}
    })
    
    test_summary = {
        "book_id": book_result["book_id"],
        "level": 0,
        "content": "Test summary content",
        "vector_id": "vec456"
    }
    
    result = await agent.process({
        "action": "add_summary",
        "summary": test_summary
    })
    
    assert result["status"] == "success"
    assert "summary_id" in result
    
    await agent.cleanup()

@pytest.mark.asyncio
async def test_librarian_agent_invalid_action():
    config = VeniceConfig(api_key="test_key")
    agent = LibrarianAgent(config)
    await agent.initialize()
    
    result = await agent.process({
        "action": "invalid_action"
    })
    
    assert result["status"] == "error"
    assert "message" in result
    
    await agent.cleanup()
