import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from bookbot.agents.summarization.agent import SummarizationAgent
from bookbot.utils.venice_client import VeniceConfig
from bookbot.database.models import Book, Summary, SummaryType, SummaryLevel
from sqlalchemy import select

pytestmark = pytest.mark.asyncio

@pytest.mark.asyncio
async def test_chapter_detection():
    config = VeniceConfig(api_key="test_key")
    agent = SummarizationAgent(config, AsyncMock())
    
    test_content = """
Chapter 1
Introduction text here
Some more text

Chapter 2
More content here
And here

CHAPTER 3
Final chapter text

1. Another Section
With content

Book 1
Book content here

Volume 2
Volume content here
"""
    chapters = agent._split_into_chapters(test_content)
    assert len(chapters) == 6
    assert "Chapter 1" in chapters[0]
    assert "Chapter 2" in chapters[1]
    assert "CHAPTER 3" in chapters[2]
    assert "1. Another Section" in chapters[3]
    assert "Book 1" in chapters[4]
    assert "Volume 2" in chapters[5]

@pytest.mark.asyncio
async def test_summary_caching():
    config = VeniceConfig(api_key="test_key")
    agent = SummarizationAgent(config, AsyncMock())
    await agent.initialize()
    
    test_content = "Test book content"
    test_metadata = {"title": "Test Book", "author": "Test Author"}
    
    mock_summary = [{
        "level": SummaryLevel.DETAILED,
        "content": "Detailed summary",
        "vector": [0.1, 0.2, 0.3],
        "vector_id": "vec123"
    }]
    
    with patch.object(agent, 'generate_hierarchical_summary', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_summary
        
        # First call should generate summary
        result1 = await agent.process_book_content(test_content, test_metadata)
        assert mock_generate.called
        assert result1[0]["level"] == SummaryLevel.DETAILED
        
        # Second call should use cache
        mock_generate.reset_mock()
        result2 = await agent.process_book_content(test_content, test_metadata)
        assert not mock_generate.called
        assert result2 == result1

@pytest.mark.asyncio
async def test_rate_limiting():
    config = VeniceConfig(api_key="test_key")
    agent = SummarizationAgent(config, AsyncMock())
    await agent.initialize()
    
    test_content = "Test content"
    mock_response = {
        "choices": [{"text": "Summary"}],
    }
    mock_embedding = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}]
    }
    
    with patch.object(agent.venice, 'generate', new_callable=AsyncMock) as mock_generate, \
         patch.object(agent.venice, 'embed', new_callable=AsyncMock) as mock_embed, \
         patch.object(agent._rate_limiter, 'wait_for_token', new_callable=AsyncMock) as mock_wait:
        mock_generate.return_value = mock_response
        mock_embed.return_value = mock_embedding
        
        await agent.generate_hierarchical_summary(test_content)
        assert mock_wait.called
        assert mock_generate.called
        assert mock_embed.called

@pytest.mark.asyncio
async def test_database_integration(async_session):
    config = VeniceConfig(api_key="test_key")
    
    agent = SummarizationAgent(config, async_session)
    await agent.initialize()
    
    # Create test book
    async with async_session.begin():
        book = Book(
            title="Test Book",
            author="Test Author",
            content_hash="test123",
            vector_id="vec123"
        )
        async_session.add(book)
        await async_session.flush()
        await async_session.refresh(book)
    
    test_content = "Chapter 1\nTest content"
    test_metadata = {
        "book_id": book.id,
        "title": "Test Book",
        "author": "Test Author"
    }
        
    mock_summary = [{
        "level": SummaryLevel.DETAILED,
        "content": "Summary",
        "vector": [0.1, 0.2, 0.3],
        "vector_id": "vec456",
        "summary_type": SummaryType.CHAPTER,
        "chapter_index": 0
    }]
    
    with patch.object(agent, 'generate_hierarchical_summary', new_callable=AsyncMock) as mock_generate:
        mock_generate.return_value = mock_summary
        result = await agent.process({
            "content": test_content,
            "metadata": test_metadata
        })
        
        assert result["status"] == "success"
        assert len(result["summaries"]) > 0
        
        # Verify summary was saved
        await async_session.commit()  # Commit changes
        
        stmt = select(Summary).where(Summary.book_id == book.id)
        result = await async_session.execute(stmt)
        db_summaries = result.scalars().all()
        assert len(db_summaries) > 0
        assert db_summaries[0].level == SummaryLevel.DETAILED
        assert db_summaries[0].summary_type == SummaryType.CHAPTER

@pytest.mark.asyncio
async def test_error_handling():
    config = VeniceConfig(api_key="test_key")
    agent = SummarizationAgent(config, AsyncMock())
    await agent.initialize()
    
    # Test missing content
    result = await agent.process({})
    assert result["status"] == "error"
    assert "No content provided" in result["message"]
    
    # Test processing error
    with patch.object(agent, 'process_book_content', side_effect=Exception("Test error")):
        result = await agent.process({"content": "test", "metadata": {}})
        assert result["status"] == "error"
        assert "Test error" in result["message"]
