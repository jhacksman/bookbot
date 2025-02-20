import pytest
import asyncio
from bookbot.agents.query.agent import QueryAgent
from bookbot.utils.venice_client import VeniceConfig
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import text
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
async def test_query_agent_with_content():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=NullPool
    )
    
    # Create a new session factory
    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False
    )
    
    # Use a shared in-memory database with a single connection
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=True,
        poolclass=NullPool,
        connect_args={"check_same_thread": False}
    )
    
    # Create tables and run test in a single transaction
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        # Create session with the same connection
        session = AsyncSession(bind=conn, expire_on_commit=False)
        
        try:
            config = VeniceConfig(api_key="test_key")
            agent = QueryAgent(config, session)
            await agent.initialize()
            assert agent.is_active

            # Add test data within a transaction
            async with session.begin():
                book = Book(
                    title="Test Book",
                    author="Test Author",
                    content_hash="test123",
                    vector_id="vec123"
                )
                session.add(book)
                
                summary = Summary(
                    book_id=book.id,
                    level=0,
                    content="This is a test summary about AI.",
                    vector_id="vec456"
                )
                session.add(summary)
            
            # Add test data to vector store once
            await agent.vector_store.add_texts(
                texts=[summary.content],
                metadata=[{"book_id": str(book.id), "type": "summary", "level": summary.level}],
                ids=[summary.vector_id]
            )
            
            # Wait for vector store to be ready
            await asyncio.sleep(0.1)
            
            # Run query
            result = await agent.process({
                "question": "What is this book about?",
                "context": None
            })
            assert result["status"] == "success"
            assert "response" in result
            assert "citations" in result
            assert isinstance(result["confidence"], float)
            assert 0.0 <= result["confidence"] <= 1.0
        finally:
            await agent.cleanup()
            await session.close()
            await engine.dispose()
