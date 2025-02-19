import pytest
from bookbot.agents.query.agent import QueryAgent
from bookbot.utils.venice_client import VeniceConfig
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from bookbot.database.models import Base, Book, Summary

@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    session_factory = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    session = await session_factory()
    try:
        yield session
    finally:
        await session.close()
        await engine.dispose()

@pytest.mark.asyncio
async def test_query_agent_initialization(db_session):
    config = VeniceConfig(api_key="test_key")
    agent = QueryAgent(config, db_session)
    await agent.initialize()
    assert agent.is_active
    await agent.cleanup()
    assert not agent.is_active

@pytest.mark.asyncio
async def test_query_agent_empty_query(db_session):
    config = VeniceConfig(api_key="test_key")
    agent = QueryAgent(config, db_session)
    await agent.initialize()
    
    result = await agent.process({})
    assert result["status"] == "error"
    assert "message" in result
    
    await agent.cleanup()

@pytest.mark.asyncio
async def test_query_agent_no_relevant_content(db_session):
    config = VeniceConfig(api_key="test_key")
    agent = QueryAgent(config, db_session)
    await agent.initialize()
    
    result = await agent.process({"question": "What is the meaning of life?"})
    assert result["status"] == "success"
    assert "response" in result
    assert "citations" in result
    assert len(result["citations"]) == 0
    assert result["confidence"] == 0.0
    
    await agent.cleanup()

@pytest.mark.asyncio
async def test_query_agent_with_content(db_session):
    config = VeniceConfig(api_key="test_key")
    agent = QueryAgent(config, db_session)
    await agent.initialize()
    
    # Add test book and summary to the database
    book = Book(
        title="Test Book",
        author="Test Author",
        content_hash="test123",
        vector_id="vec123"
    )
    db_session.add(book)
    await db_session.flush()
    
    summary = Summary(
        book_id=book.id,
        level=0,
        content="This is a test summary about AI.",
        vector_id="vec456"
    )
    db_session.add(summary)
    await db_session.commit()
    
    # Add content to vector store
    await agent.vector_store.add_texts(
        texts=["This is a test summary about AI."],
        metadata=[{"book_id": book.id}],
        ids=["test1"]
    )
    
    result = await agent.process({"question": "What is this book about?"})
    assert result["status"] == "success"
    assert "response" in result
    assert "citations" in result
    assert result["confidence"] > 0.0
    
    await agent.cleanup()
