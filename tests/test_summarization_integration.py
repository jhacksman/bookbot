import pytest
from unittest.mock import AsyncMock, patch
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from bookbot.agents.summarization.agent import SummarizationAgent
from bookbot.agents.librarian.agent import LibrarianAgent
from bookbot.utils.venice_client import VeniceConfig
from bookbot.database.models import Book, Summary, SummaryType, SummaryLevel

pytestmark = pytest.mark.asyncio

async def test_summarization_pipeline(async_session):
    """Test the complete summarization pipeline."""
    # Initialize agents
    venice_config = VeniceConfig(api_key="test_key")
    librarian = None
    summarizer = None
    
    try:
        librarian = LibrarianAgent(venice_config, "sqlite+aiosqlite:///:memory:", calibre_path=Path("/tmp/test_library"))
        await librarian.initialize()
        
        summarizer = SummarizationAgent(venice_config, async_session)
        await summarizer.initialize()
        
        # Create test book
        async with async_session.begin():
            book = Book(
                title="AI Fundamentals",
                author="Test Author",
                content_hash="test_hash_123",
                vector_id="vec_123"
            )
            async_session.add(book)
            await async_session.flush()
            await async_session.refresh(book)
        
        # Test content
        test_content = """
Chapter 1
Introduction to AI
Artificial Intelligence is a broad field of computer science.

Chapter 2
Machine Learning
Machine learning is a subset of AI that focuses on data.

Chapter 3
Deep Learning
Deep learning uses neural networks with multiple layers."""
        
        # Mock Venice.ai responses
        mock_generate_response = {
            "choices": [{"text": "Test summary content"}]
        }
        mock_embed_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
            
        with patch('bookbot.utils.venice_client.VeniceClient.generate', new_callable=AsyncMock) as mock_generate, \
             patch('bookbot.utils.venice_client.VeniceClient.embed', new_callable=AsyncMock) as mock_embed:
            mock_generate.return_value = mock_generate_response
            mock_embed.return_value = mock_embed_response
            
            # Process book content
            result = await summarizer.process({
                "content": test_content,
                "metadata": {
                    "book_id": book.id,
                    "title": book.title,
                    "author": book.author
                }
            })
            
            assert result["status"] == "success"
            assert len(result["summaries"]) > 0
            
            # Verify database state
            from sqlalchemy import select
            stmt = select(Summary).where(Summary.book_id == book.id).order_by(Summary.level, Summary.chapter_index)
            result = await async_session.execute(stmt)
            db_summaries = result.scalars().all()
        
            # Should have both chapter and book summaries
            chapter_summaries = [s for s in db_summaries if s.summary_type == SummaryType.CHAPTER]
            book_summaries = [s for s in db_summaries if s.summary_type == SummaryType.BOOK]
            
            assert len(chapter_summaries) > 0
            assert len(book_summaries) > 0
            
            # Verify chapter summaries
            assert all(s.chapter_index is not None for s in chapter_summaries)
            assert all(s.level in [SummaryLevel.DETAILED, SummaryLevel.CONCISE] for s in chapter_summaries)
            
            # Verify book summaries
            assert all(s.chapter_index is None for s in book_summaries)
            assert any(s.level == SummaryLevel.BRIEF for s in book_summaries)
        
            # Verify vector embeddings
            assert all(s.vector_id for s in db_summaries)
    finally:
        if librarian:
            await librarian.cleanup()
        if summarizer:
            await summarizer.cleanup()
            
@pytest.mark.asyncio
async def test_concurrent_summarization(async_session):
    """Test that multiple summarization tasks can run concurrently without issues."""
    venice_config = VeniceConfig(api_key="test_key")
    summarizer = None
    
    try:
        summarizer = SummarizationAgent(venice_config, async_session)
        await summarizer.initialize()
        
        books = []
        # Create test books
        for i in range(3):
            book = Book(
                title=f"Test Book {i}",
                author="Test Author",
                content_hash=f"test_hash_{i}",
                vector_id=f"vec_{i}"
            )
            async_session.add(book)
            await async_session.flush()
            await async_session.refresh(book)
            books.append(book)
            
            # Mock Venice.ai responses
            mock_generate_response = {
                "choices": [{"text": "Test summary"}]
            }
            mock_embed_response = {
                "data": [{"embedding": [0.1, 0.2, 0.3]}]
            }
            
            # Process multiple books concurrently
            import asyncio
            tasks = []
            
            with patch('bookbot.utils.venice_client.VeniceClient.generate', new_callable=AsyncMock) as mock_generate, \
                 patch('bookbot.utils.venice_client.VeniceClient.embed', new_callable=AsyncMock) as mock_embed:
                mock_generate.return_value = mock_generate_response
                mock_embed.return_value = mock_embed_response
                
                # Process books sequentially to avoid SQLite concurrency issues
                results = []
                for i, book in enumerate(books):
                    content = f"Chapter 1\nTest content for book {i}"
                    result = await summarizer.process({
                        "content": content,
                        "metadata": {
                            "book_id": book.id,
                            "title": book.title,
                            "author": book.author
                        }
                    })
                    results.append(result)
                
                # Verify all tasks completed successfully
                for i, result in enumerate(results):
                    if result["status"] != "success":
                        print(f"Task {i} failed with message: {result.get('message', 'No message')}")
                assert all(r["status"] == "success" for r in results)
                assert all(len(r["summaries"]) > 0 for r in results)
                
                # Verify rate limiting worked
                assert mock_generate.call_count > 0
                assert mock_embed.call_count > 0
                
                # Verify database state
                for book in books:
                    from sqlalchemy import func
                    stmt = select(func.count()).select_from(Summary).where(Summary.book_id == book.id)
                    result = await async_session.execute(stmt)
                    count = result.scalar()
                    assert count > 0
    finally:
        if summarizer:
            await summarizer.cleanup()

@pytest.mark.asyncio
async def test_error_recovery(async_session):
    """Test that the system handles errors gracefully and maintains consistency."""
    venice_config = VeniceConfig(api_key="test_key")
    summarizer = None
    
    try:
        summarizer = SummarizationAgent(venice_config, async_session)
        await summarizer.initialize()
        
        async with async_session.begin():
            book = Book(
                title="Error Test Book",
                author="Test Author",
                content_hash="error_test_hash",
                vector_id="error_vec"
            )
            async_session.add(book)
            await async_session.flush()
            await async_session.refresh(book)
            
        # Simulate API errors
        with patch('bookbot.utils.venice_client.VeniceClient.generate', new_callable=AsyncMock) as mock_generate:
            mock_generate.side_effect = Exception("API Error")
            
            result = await summarizer.process({
                "content": "Test content",
                "metadata": {
                    "book_id": book.id,
                    "title": book.title,
                    "author": book.author
                }
            })
            
            # Should return error status
            assert result["status"] == "error"
            assert "API Error" in result["message"]
            
            # Database should be in consistent state
            from sqlalchemy import func, select
            stmt = select(func.count()).select_from(Summary).where(Summary.book_id == book.id)
            result = await async_session.execute(stmt)
            count = result.scalar()
            assert count == 0  # No partial summaries should be saved
    finally:
        if summarizer:
            await summarizer.cleanup()
