import pytest
import asyncio
from bookbot.agents.query.agent import QueryAgent
from bookbot.utils.venice_client import VeniceConfig
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from bookbot.database.models import Base, Book, Summary

@pytest.mark.asyncio
async def test_query_agent_initialization(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = QueryAgent(config, async_session)
    try:
        await agent.initialize()
        assert agent.is_active
        
        # Verify agent state
        assert agent.vram_limit == 16.0
        assert agent.vector_store is not None
        
        # Test cleanup
        await agent.cleanup()
        assert not agent.is_active
    finally:
        if agent.is_active:
            await agent.cleanup()

@pytest.mark.asyncio
async def test_query_agent_empty_query(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = QueryAgent(config, async_session)
    await agent.initialize()
    
    result = await agent.process({})
    assert result["status"] == "error"
    assert "message" in result
    
    await agent.cleanup()

@pytest.mark.asyncio
async def test_query_agent_no_relevant_content(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = QueryAgent(config, async_session)
    await agent.initialize()
    
    result = await agent.process({"question": "What is the meaning of life?"})
    assert result["status"] == "success"
    assert "response" in result
    assert "citations" in result
    assert len(result["citations"]) == 0
    assert result["confidence"] == 0.0
    
    await agent.cleanup()

@pytest.mark.asyncio
async def test_query_agent_with_content(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = QueryAgent(config, async_session)
    try:
        await agent.initialize()
        assert agent.is_active
        
        # Add test book and summary to the database
        async with async_session() as session:
            async with session.begin():
                book = Book(
                    title="Test Book",
                    author="Test Author",
                    content_hash="test123",
                    vector_id="vec123"
                )
                session.add(book)
                await session.flush()
                
                summary = Summary(
                    book_id=book.id,
                    level=0,
                    content="This is a test summary about AI.",
                    vector_id="vec456"
                )
                session.add(summary)
                await session.flush()
        
        # Add vector to store for testing
        await agent.vector_store.add_texts(
            texts=[str(summary.content)],
            metadata=[{"book_id": str(book.id)}],
            ids=[str(summary.vector_id)]
        )
        
        # Test querying
        result = await agent.process({
            "question": "What is this book about?"
        })
        
        assert result["status"] == "success"
        assert "response" in result
        assert "citations" in result
        assert result["confidence"] > 0.0
        
        # Add more content and test again
        await agent.vector_store.add_texts(
            texts=["This is a test summary about AI."],
            metadata=[{"book_id": str(book.id)}],
            ids=["test1"]
        )
        
        result = await agent.process({"question": "What is this book about?"})
        assert result["status"] == "success"
        assert "response" in result
        assert "citations" in result
        assert result["confidence"] > 0.0
    finally:
        if agent.is_active:
            await agent.cleanup()
