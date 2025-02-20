import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch
from bookbot.agents.query.agent import QueryAgent
from bookbot.utils.venice_client import VeniceConfig
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
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
    finally:
        await agent.cleanup()

@pytest.mark.asyncio
async def test_query_preprocessing(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = QueryAgent(config, async_session)
    try:
        await agent.initialize()
        result = await agent.process({"question": "  What   is  this  book   about?\n\r"})
        assert result["status"] == "success"
        assert "response" in result
        assert "citations" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        
        # Test cache hit
        cached_result = await agent.process({"question": "  What   is  this  book   about?\n\r"})
        assert cached_result == result
    finally:
        await agent.cleanup()

@pytest.mark.asyncio
async def test_citation_formatting(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = QueryAgent(config, async_session)
    try:
        await agent.initialize()
        
        # Add test data
        book = Book(
            title="Test Book",
            author="Test Author",
            content_hash="test123",
            vector_id="vec123"
        )
        async_session.add(book)
        await async_session.commit()
        
        # Mock vector store search
        agent.vector_store.similarity_search = AsyncMock(return_value=[{
            "content": "This is a test summary about AI.",
            "metadata": {"book_id": str(book.id)},
            "distance": 0.2
        }])
        
        # Mock Venice response
        mock_response = {
            "choices": [{
                "text": json.dumps({
                    "answer": "Test answer [1]",
                    "citations": ["[1]"],
                    "confidence": 0.8
                })
            }]
        }
        agent.venice.generate = AsyncMock(return_value=mock_response)
        
        result = await agent.process({"question": "What is this book about?"})
        assert result["status"] == "success"
        assert len(result["citations"]) > 0
        
        citation = result["citations"][0]
        assert citation["id"] == "[1]"
        assert citation["book_id"] == str(book.id)
        assert citation["title"] == "Test Book"
        assert citation["author"] == "Test Author"
        assert citation["quoted_text"] == "This is a test summary about AI."
        assert citation["relevance_score"] == 0.8
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
    finally:
        await agent.cleanup()

@pytest.mark.asyncio
async def test_query_error_handling(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = QueryAgent(config, async_session)
    try:
        await agent.initialize()
        
        # Test invalid input
        result = await agent.process({})
        assert result["status"] == "error"
        assert "message" in result
        
        # Test empty question
        result = await agent.process({"question": ""})
        assert result["status"] == "error"
        assert "message" in result
        
        # Test whitespace-only question
        result = await agent.process({"question": "   \n\r\t   "})
        assert result["status"] == "error"
        assert "message" in result
        
        # Test vector store error
        agent.vector_store.similarity_search = AsyncMock(side_effect=Exception("Vector store error"))
        result = await agent.process({"question": "What is this about?"})
        assert result["status"] == "error"
        assert "message" in result
        
        # Test Venice API error
        agent.vector_store.similarity_search = AsyncMock(return_value=[{
            "content": "Test content",
            "metadata": {"book_id": "123"},
            "distance": 0.2
        }])
        agent.venice.generate = AsyncMock(side_effect=Exception("Venice API error"))
        result = await agent.process({"question": "What is this about?"})
        assert result["status"] == "error"
        assert "message" in result
    finally:
        await agent.cleanup()
