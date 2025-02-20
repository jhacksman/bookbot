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
        assert agent.vram_limit == 16.0
        assert agent.vector_store is not None
    finally:
        await agent.cleanup()
        assert not agent.is_active

@pytest.mark.asyncio
async def test_query_agent_empty_query(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = QueryAgent(config, async_session)
    try:
        await agent.initialize()
        result = await agent.process({})
        assert result["status"] == "error"
        assert "message" in result
    finally:
        await agent.cleanup()

@pytest.mark.asyncio
async def test_query_agent_no_relevant_content(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = QueryAgent(config, async_session)
    try:
        await agent.initialize()
        result = await agent.process({"question": "What is the meaning of life?", "context": None})
        assert result["status"] == "success"
        assert "response" in result
        assert "citations" in result
        assert len(result["citations"]) == 0
        assert result["confidence"] == 0.3
    finally:
        await agent.cleanup()

@pytest.mark.asyncio
async def test_query_agent_with_content(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = QueryAgent(config, async_session)
    try:
        await agent.initialize()
        assert agent.is_active

        book = Book(
            title="Test Book",
            author="Test Author",
            content_hash="test123",
            vector_id="vec123"
        )
        async_session.add(book)
        await async_session.flush()
        
        summary = Summary(
            book_id=book.id,
            level=0,
            content="This is a test summary about AI.",
            vector_id="vec456"
        )
        async_session.add(summary)
        await async_session.flush()
        await async_session.commit()
        
        # Add initial texts
        await agent.vector_store.add_texts(
            texts=[summary.content],
            metadata=[{"book_id": str(summary.book_id)}],
            ids=[summary.vector_id]
        )
        await asyncio.sleep(0.1)
        
        result = await agent.process({
            "question": "What is this book about?",
            "context": None
        })
        assert result["status"] == "success"
        assert "response" in result
        assert "citations" in result
        assert result["confidence"] > 0.0
        
        # Add additional test texts
        await agent.vector_store.add_texts(
            texts=["This is a test summary about AI."],
            metadata=[{"book_id": str(summary.book_id)}],
            ids=["test1"]
        )
        await asyncio.sleep(0.1)
        
        result = await agent.process({"question": "What is this book about?", "context": None})
        assert result["status"] == "success"
        assert "response" in result
        assert "citations" in result
        assert result["confidence"] > 0.0
    finally:
        await agent.cleanup()
