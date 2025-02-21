import pytest
import asyncio
from unittest.mock import patch, MagicMock
from bookbot.agents.selection.agent import SelectionAgent
from bookbot.utils.venice_client import VeniceConfig
from bookbot.utils.vector_store import VectorStore

@pytest.mark.asyncio
async def test_selection_agent_initialization(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = SelectionAgent(config, session=async_session)
    assert agent.vram_limit == 16.0
    assert not agent.is_active
    assert isinstance(agent.vector_store, VectorStore)
    assert agent.rate_limiter is not None
    assert agent.cache is not None

@pytest.mark.asyncio
async def test_selection_agent_process_empty_input(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = SelectionAgent(config, session=async_session)
    await agent.initialize()
    
    result = await agent.process({})
    assert result["status"] == "error"
    assert "message" in result
    assert "No books provided" in result["message"]

@pytest.mark.asyncio
async def test_selection_agent_process_success(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = SelectionAgent(config, session=async_session)
    await agent.initialize()
    
    # Mock the Venice client response
    async def mock_generate(*args, **kwargs):
        return {
            "choices": [{
                "text": '{"relevance_score": 35, "technical_score": 25, "recency_score": 12, "expertise_score": 13, "total_score": 85, "reasoning": "Highly relevant AI/ML text", "key_topics": ["deep learning", "neural networks"], "target_audience": "researchers", "prerequisites": ["calculus", "linear algebra"], "recommended_reading_order": 4}'
            }]
        }
    agent.venice.generate = mock_generate
    
    test_books = [{
        "title": "Deep Learning",
        "author": "Ian Goodfellow",
        "description": "Comprehensive guide to deep learning",
        "publication_date": "2016",
        "publisher": "MIT Press",
        "isbn": "9780262035613",
        "language": "en",
        "format": "pdf",
        "content": "Deep learning is a form of machine learning..."
    }]
    
    result = await agent.process({"books": test_books})
    assert result["status"] == "success"
    assert "selected_books" in result
    assert len(result["selected_books"]) > 0
    assert "evaluations" in result
    assert len(result["evaluations"]) > 0
    
    book = result["selected_books"][0]
    assert book["title"] == "Deep Learning"
    assert "evaluation" in book
    assert "total_score" in book
    assert 0 <= book["total_score"] <= 100

@pytest.mark.asyncio
async def test_selection_agent_metadata_extraction(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = SelectionAgent(config, session=async_session)
    await agent.initialize()
    
    test_book = {
        "title": "Machine Learning",
        "author": "Tom Mitchell",
        "description": "Introduction to ML",
        "publication_date": "1997",
        "publisher": "McGraw Hill",
        "isbn": "9780070428072",
        "language": "en",
        "format": "pdf",
        "content": "Sample content"
    }
    
    metadata = await agent.extract_metadata(test_book)
    assert metadata["title"] == "Machine Learning"
    assert metadata["author"] == "Tom Mitchell"
    assert metadata["publication_date"] == "1997"
    assert metadata["publisher"] == "McGraw Hill"
    assert metadata["isbn"] == "9780070428072"
    assert metadata["language"] == "en"
    assert metadata["file_format"] == "pdf"
    assert metadata["content_length"] == len(test_book["content"])

@pytest.mark.asyncio
async def test_selection_agent_evaluation_caching(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = SelectionAgent(config, session=async_session)
    await agent.initialize()
    
    # Mock the Venice client response
    async def mock_generate(*args, **kwargs):
        return {
            "choices": [{
                "text": '{"relevance_score": 35, "technical_score": 25, "recency_score": 12, "expertise_score": 13, "total_score": 85, "reasoning": "Highly relevant AI/ML text", "key_topics": ["deep learning", "neural networks"], "target_audience": "researchers", "prerequisites": ["calculus", "linear algebra"], "recommended_reading_order": 4}'
            }]
        }
    agent.venice.generate = mock_generate
    
    test_book = {
        "title": "Deep Learning",
        "author": "Ian Goodfellow",
        "description": "Test description"
    }
    
    # First evaluation
    eval1 = await agent.evaluate_book(test_book)
    # Second evaluation should use cache
    eval2 = await agent.evaluate_book(test_book)
    
    assert eval1 == eval2

@pytest.mark.asyncio
async def test_selection_agent_vector_storage(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = SelectionAgent(config, session=async_session)
    await agent.initialize()
    
    # Mock the Venice client response
    async def mock_generate(*args, **kwargs):
        return {
            "choices": [{
                "text": '{"relevance_score": 35, "technical_score": 25, "recency_score": 12, "expertise_score": 13, "total_score": 85, "reasoning": "Highly relevant AI/ML text", "key_topics": ["deep learning", "neural networks"], "target_audience": "researchers", "prerequisites": ["calculus", "linear algebra"], "recommended_reading_order": 4}'
            }]
        }
    agent.venice.generate = mock_generate
    
    test_books = [{
        "title": "Deep Learning",
        "author": "Ian Goodfellow",
        "description": "Test description",
        "content": "Test content"
    }]
    
    result = await agent.process({"books": test_books})
    assert result["status"] == "success"
    
    # Verify vectors were stored
    vectors = await agent.vector_store.search(
        "deep learning neural networks",
        limit=1
    )
    assert len(vectors) > 0
    assert vectors[0]["metadata"]["title"] == "Deep Learning"

@pytest.mark.asyncio
async def test_selection_agent_rate_limiting(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = SelectionAgent(config, session=async_session)
    await agent.initialize()
    
    # Mock the Venice client response
    async def mock_generate(*args, **kwargs):
        await asyncio.sleep(0.2)  # Add delay to simulate API call
        return {
            "choices": [{
                "text": '{"relevance_score": 35, "technical_score": 25, "recency_score": 12, "expertise_score": 13, "total_score": 85, "reasoning": "Highly relevant AI/ML text", "key_topics": ["deep learning", "neural networks"], "target_audience": "researchers", "prerequisites": ["calculus", "linear algebra"], "recommended_reading_order": 4}'
            }]
        }
    agent.venice.generate = mock_generate
    
    # Create multiple books to trigger rate limiting
    test_books = [
        {
            "title": f"Book {i}",
            "author": f"Author {i}",
            "description": f"Description {i}"
        }
        for i in range(10)
    ]
    
    start_time = asyncio.get_event_loop().time()
    result = await agent.process({"books": test_books})
    end_time = asyncio.get_event_loop().time()
    
    assert result["status"] == "success"
    # Verify rate limiting added some delay
    assert end_time - start_time >= 1.0  # At least 1 second delay

@pytest.mark.asyncio
async def test_selection_agent_error_handling(async_session):
    config = VeniceConfig(api_key="test_key")
    agent = SelectionAgent(config, session=async_session)
    await agent.initialize()
    
    # Test with invalid book data
    test_books = [{
        "title": None,
        "author": None,
        "description": None
    }]
    
    result = await agent.process({"books": test_books})
    assert result["status"] == "success"  # Should still succeed
    assert "errors" in result
    assert len(result["errors"]) > 0
